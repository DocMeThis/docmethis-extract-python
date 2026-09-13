# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""AST callgraph - build the call graph between functions.

Two passes:
1. Local pass: reparse each module's AST and visit each function body to collect
   direct call edges. Resolve names in module scope -> imports -> external order.
   Resolve ``self.method()`` through the class MRO and ``super().method()``
   through the parent's MRO.

   Note: reparsing is deliberate; it simplifies the API because no cache passes
   between stages. If measured reparsing becomes a bottleneck, pass a
   ``{file_path: ast.Module}`` cache as a parameter.

2. Global pass: detect direct and indirect recursion (DFS), calculate
   ``callgraph_depth`` for each function (maximum reachable depth in the local
   graph, capped at ``max_depth``), and mark ``depth_exceeded`` edges when a
   target reaches the cap.
"""

from __future__ import annotations

import ast
import dataclasses
import logging
import re
from typing import TYPE_CHECKING, NamedTuple

from docmethis_extract_python.static_extraction.ast_parser import parse_file_with_lines
from docmethis_extract_python.static_extraction.models import (
    CallgraphEdge,
    ClassRecord,
    Confidence,
    FunctionRecord,
    ImportRecord,
    MethodType,
    ModuleRecord,
    ProjectRecord,
    iter_functions,
    name_simple,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Default value, overridable through build_callgraph().
_CALLGRAPH_MAX_DEPTH: int = 5


# FIX #8: express _AstIndex keys through NamedTuple so (lineno, name) fields
# are explicit instead of an anonymous tuple[int, str].
class _AstKey(NamedTuple):
    """AST index key: line number plus function name.

    Attributes
    ----------
    lineno : int
        Line number of the function definition in the source file.
    name : str
        Name of the function the key refers to.

    """

    lineno: int
    name: str


# Type alias for a module's AST-node index.
_AstIndex = dict[_AstKey, ast.FunctionDef | ast.AsyncFunctionDef]

# Regex validating resolvable call forms.
# Accept only qualified Python names (a.b.c); ignore subscripts, lambdas, and
# nested calls. FIX #4: apply this filter in _resolve_call_qualified (ast.Attribute
# calls) as well as the final _process_call fallback.
_DOTTED_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


@dataclasses.dataclass
class _CallgraphContext:
    """Shared context for collecting call edges.

    Group indexes and mapping to avoid an overly long signature.

    Attributes
    ----------
    ast_index : _AstIndex
        Index of AST nodes by their ``_AstKey`` (line number and name).
    mapping : dict[str, str]
        Mapping of simple names to their qualified names.
    index_fn : dict[str, FunctionRecord]
        Mapping of function qualified names to their function records.
    index_cls : dict[str, ClassRecord]
        Mapping of class qualified names to their class records.

    """

    ast_index: _AstIndex
    mapping: dict[str, str]
    index_fn: dict[str, FunctionRecord]
    index_cls: dict[str, ClassRecord]


class _ResolveCallParams(NamedTuple):
    """Common parameters for call resolution.

    Attributes
    ----------
    func : ast.Attribute
        The attribute node representing the called function.
    source : str
        Qualified name of the module or object the call originates from.
    fn_record : FunctionRecord
        Record of the function being called.
    context : _CallgraphContext
        Shared context used to resolve the call target.
    line : int
        Line number of the call in the source file.

    """

    func: ast.Attribute
    source: str
    fn_record: FunctionRecord
    context: _CallgraphContext
    line: int


# ---------------------------------------------------------------------------
# Project index
# ---------------------------------------------------------------------------


def _effective_modules(modules: list[ModuleRecord]) -> list[ModuleRecord]:
    """Exclude ``.pyi`` stubs that duplicate an implementation of the same module.

    ``resolve_module_name()`` ignores extensions: ``pkg/api.py`` and ``pkg/api.pyi`` have the same ``module_name`` and their
    FunctionRecords have the same ``qualified_name``. Indexing both let the later-sorted stub overwrite the implementation. The
    stub AST then mismatched the .py records and replaced real edges with empty stub edges (``...`` bodies). The callgraph
    describes executed code, so the implementation wins.

    Keep a stub without an implementation as the module's only source.

    Parameters
    ----------
    modules : list[ModuleRecord]
        The sequence of module records to filter; stub records for modules that also have an implementation are removed.

    Returns
    -------
    list[ModuleRecord]
        The filtered list of module records, containing the implementation for modules that have one, and retaining stub-only
        modules unchanged.

    """
    implementations = {module.module_name for module in modules if not module.is_stub}
    return [module for module in modules if not (module.is_stub and module.module_name in implementations)]


def _index_functions(project_record: ProjectRecord) -> dict[str, FunctionRecord]:
    """Index ``qualified_name -> FunctionRecord`` across the project.

    Parameters
    ----------
    project_record : ProjectRecord
        The project record whose modules are scanned to build the function index.

    Returns
    -------
    dict[str, FunctionRecord]
        A dictionary mapping each function's qualified name to its corresponding FunctionRecord for all functions in the project.

    """
    return {fn.qualified_name: fn for fn in iter_functions(_effective_modules(project_record.modules))}


def _index_classes(project_record: ProjectRecord) -> dict[str, ClassRecord]:
    """Index ``qualified_name -> ClassRecord`` across the project.

    Parameters
    ----------
    project_record : ProjectRecord
        The project record whose modules are traversed to collect class records.

    Returns
    -------
    dict[str, ClassRecord]
        A dictionary mapping each class's qualified name to its corresponding ClassRecord, covering all classes in the project's
        modules.

    """
    return {class_.qualified_name: class_ for module in _effective_modules(project_record.modules) for class_ in module.classes}


# ---------------------------------------------------------------------------
# Module exports (for resolving internal star imports)
# ---------------------------------------------------------------------------


def _module_exported_names(module: ModuleRecord, tree: ast.Module) -> frozenset[str]:
    """Return names exported by a module.

    Look for ``__all__`` in top-level assignments first. If absent, return public names (without the ``_`` prefix) from the
    ModuleRecord's functions and classes.

    Parameters
    ----------
    module : ModuleRecord
        The module record to inspect for exported names; its functions and classes are used to derive public names when no __all__
        assignment is present.
    tree : ast.Module
        The AST module tree to inspect for top-level __all__ assignments.

    Returns
    -------
    frozenset[str]
        Returns a frozenset of strings containing the names exported by the module. If a top-level __all__ assignment is present,
        its string elements are used; otherwise, the frozenset contains the public names (without a leading underscore) from the
        module's functions and classes.

    """
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
        ):
            value = node.value
            if isinstance(value, (ast.List, ast.Tuple)):
                return frozenset(elt.value for elt in value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str))

    names: set[str] = set()
    for fn in module.functions:
        simple = name_simple(fn.qualified_name)
        if not simple.startswith("_"):
            names.add(simple)
    for cls in module.classes:
        simple = name_simple(cls.qualified_name)
        if not simple.startswith("_"):
            names.add(simple)
    return frozenset(names)


@dataclasses.dataclass(frozen=True)
class _ExportContext:
    """Project view needed to resolve an import to the module defining its symbol.

    The three tables can be computed only after seeing all modules and are always
    consumed together, so group them instead of passing three parallel parameters.

    Attributes
    ----------
    exported_names : dict[str, frozenset[str]]
        Mapping of module names to the set of names they expose.
    origins : dict[str, dict[str, str]]
        Mapping of module names to a map of visible names to their original qualified names.
    known_symbols : frozenset[str]
        Qualified names actually defined in the project, used as the re-export stop condition.

    """

    # module -> names it exposes (``__all__`` or public definitions)
    exported_names: dict[str, frozenset[str]] = dataclasses.field(default_factory=dict)
    # module -> {visible name: original qualified_name}
    origins: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)
    # qualified_names actually defined in the project (re-export stop condition)
    known_symbols: frozenset[str] = frozenset()

    def resolve(self, target_symbol: str) -> str:
        """Follow re-exports to the module that actually defines the symbol.

        ``pkg.f`` (re-exported by ``pkg/__init__.py``) -> ``pkg.api.f``. Without this traversal, the target is absent from indexes
        and the edge is marked external, with the project package itself as the external library.

        Stop when the target is a known project symbol, no further hop is possible (a truly external symbol; return unchanged), or
        an import cycle occurs.

        Parameters
        ----------
        target_symbol : str
            The dotted name of the symbol to resolve, such as 'pkg.f'.

        Returns
        -------
        str
            The resolved target symbol after following re-exports, or the original target_symbol unchanged when it is already
            known, no further hop is possible, or an import cycle is detected.

        """
        seen: set[str] = set()
        while target_symbol not in self.known_symbols and target_symbol not in seen:
            seen.add(target_symbol)
            module_name, _, identifier_name = target_symbol.rpartition(".")
            next_origin = self.origins.get(module_name, {}).get(identifier_name)
            if next_origin is None or next_origin == target_symbol:
                break
            target_symbol = next_origin

        return target_symbol


def _module_origins(module: ModuleRecord) -> dict[str, str]:
    """Return ``local_name -> original qualified_name`` for a module.

    An exported name is not necessarily defined in that module: an ``__init__.py``
    that does ``from .api import f`` and ``__all__ = ["f"]`` exposes ``f`` without owning it. This table maps each visible name to
    where the symbol actually lives, either locally or in the imported module (which may itself be a re-export).

    Parameters
    ----------
    module : ModuleRecord
        The module record to inspect for functions, classes, and imports whose exported names are resolved to their originating
        qualified names.

    Returns
    -------
    dict[str, str]
        A mapping from each visible local name in the module to the fully qualified name of the symbol's actual origin, whether
        defined locally or re-exported from another module.

    """
    origins: dict[str, str] = {}

    for fn in module.functions:
        origins[name_simple(fn.qualified_name)] = fn.qualified_name

    for class_ in module.classes:
        origins[name_simple(class_.qualified_name)] = class_.qualified_name

    for imp in module.imports:
        if imp.name in (None, "*"):
            continue

        target_symbol = _absolutize_module(imp.module, module) or imp.module
        # Local definitions win: `from .x import f` followed by `def f` in the
        # same module refers to the local definition.
        origins.setdefault(imp.alias or imp.name, f"{target_symbol}.{imp.name}")

    return origins


# ---------------------------------------------------------------------------
# Module resolution mapping
# ---------------------------------------------------------------------------


def _absolutize_module(imported_name: str, current_module: ModuleRecord) -> str | None:
    """Convert a relative module import to an absolute name (``.utils`` -> ``pkg.utils``).

    ``ImportRecord.module`` keeps the source spelling (``.utils``, ``..core``), while project indexes use absolute names. Follow
    PEP 328: the anchor is the importer package, the module itself for ``__init__``, otherwise its parent.

    Return the name unchanged for an absolute import, or ``None`` when an import goes above the root package (``from ..x import
    *`` in a top-level module).

    Parameters
    ----------
    imported_name : str
        The module name as written in the import statement, including any leading dots for relative imports (e.g., '.utils' or
        '..core'). For absolute imports, it is the fully qualified module name without leading dots.
    current_module : ModuleRecord
        The module record of the importing module. Its module name and package status (whether the file is an __init__.py) are
        used to compute the anchor package for resolving relative imports.

    Returns
    -------
    str | None
        Returns the absolute module name for a relative import, the original name unchanged for an absolute import, or None when
        the relative import goes above the root package.

    """
    level = len(imported_name) - len(imported_name.lstrip("."))
    if level == 0:
        return imported_name

    suffix = imported_name[level:]
    is_package = current_module.file_path.stem == "__init__"
    package = current_module.module_name if is_package else current_module.module_name.rpartition(".")[0]

    # Move up (level - 1) steps from the anchor package.
    segments = package.rsplit(".", level - 1)
    if not package or len(segments) < level:
        return None

    base = segments[0]
    return f"{base}.{suffix}" if suffix else base


def _apply_star_imports(
    star_imports: list[ImportRecord],
    mapping: dict[str, str],
    current_module: ModuleRecord,
    exports: _ExportContext,
) -> None:
    """Resolve internal star imports and add them to the mapping.

    Absolutize relative imports (``from .utils import *``) before lookup; otherwise they would never match ``exported_names``,
    indexed by absolute module name.

    Re-exported names (present in ``__all__`` of an ``__init__`` but defined elsewhere) are followed to their original module by
    :meth:`_ExportContext.resolve`.

    Existing imports (from explicit imports) are not overwritten because ``setdefault`` does not modify existing keys.

    Parameters
    ----------
    star_imports : list[ImportRecord]
        A list of ImportRecord objects representing star imports (e.g. `from module import *`) to resolve and add to the mapping.
        Each record provides the module name and related import metadata used during star-import resolution.
    mapping : dict[str, str]
        Mapping of identifier names to their fully qualified target paths, updated in place with symbols resolved from internal
        star imports.
    current_module : ModuleRecord
        The module record representing the module in which the star import occurs. It provides the module context needed to
        absolutize relative import paths before lookup and is used to identify the module's file path for logging and symbol
        resolution.
    exports : _ExportContext
        The export context used to resolve re-exported names and to look up the exported names for the star-imported module.

    """
    for imp in star_imports:
        target_symbol = _absolutize_module(imp.module, current_module)
        if target_symbol is not None and target_symbol in exports.exported_names:
            names = exports.exported_names[target_symbol]
            for identifier_name in names:
                mapping.setdefault(identifier_name, exports.resolve(f"{target_symbol}.{identifier_name}"))
            logger.info(
                "Resolved star import in '%s': 'from %s import *' -> '%s', %d symbols.",
                current_module.file_path,
                imp.module,
                target_symbol,
                len(names),
            )
        else:
            logger.debug(
                "Ignored external star import in '%s': 'from %s import *' - symbols cannot be resolved statically.",
                current_module.file_path,
                imp.module,
            )


def _module_mapping(module: ModuleRecord, exports: _ExportContext | None = None) -> dict[str, str]:
    """Build a ``local_name -> qualified_name`` mapping for a module.

    Cover:
    - functions and classes defined in the module (simple name -> qualified_name);
    - methods accessible through ``ClassName.method`` (composite name);
    - explicit imports: alias or name -> ``module.name``, with re-exports followed to origin;
    - internal star imports resolved through ``exports.exported_names`` (lower priority).

    Keep the first value on collision.

    Parameters
    ----------
    module : ModuleRecord
        The module record whose functions, classes, methods, and imports are used to build the local-name-to-qualified-name
        mapping.
    exports : _ExportContext | None = None
        Optional export context used to resolve re-exports and internal star imports; when omitted, an empty context is created.

    Returns
    -------
    dict[str, str]
        A dictionary mapping local names to qualified names for the module, including definitions, methods, explicit imports, and
        resolved internal star imports; on collisions, the first value is retained.

    """
    exports = exports or _ExportContext()
    mapping: dict[str, str] = {}

    def _set(mapping_key: str, item_value: str) -> None:
        if mapping_key in mapping and mapping[mapping_key] != item_value:
            logger.debug(
                "Collision in _module_mapping: '%s' -> '%s' ignored; already mapped to '%s'.",
                mapping_key,
                item_value,
                mapping[mapping_key],
            )
            return
        mapping.setdefault(mapping_key, item_value)

    for fn in module.functions:
        _set(name_simple(fn.qualified_name), fn.qualified_name)
        _set(fn.qualified_name, fn.qualified_name)

    for class_ in module.classes:
        _set(name_simple(class_.qualified_name), class_.qualified_name)
        _set(class_.qualified_name, class_.qualified_name)
        for m in class_.methods:
            _set(f"{name_simple(class_.qualified_name)}.{name_simple(m.qualified_name)}", m.qualified_name)

    # Pass 1 - explicit imports (highest priority over star imports).
    star_imports: list[ImportRecord] = []
    for imp in module.imports:
        if imp.name == "*":
            star_imports.append(imp)
            continue
        # from X import Y [as Z] -> key=alias|Y, value=X.Y
        # import X.Y.Z [as Z] -> key=alias|X, value=key (first segment only)
        # Absolutize X for relative imports ('.utils' -> 'pkg.utils'); otherwise
        # it could not match index_fn, which uses absolute names.
        target_symbol = _absolutize_module(imp.module, module) or imp.module
        mapping_key = imp.alias or imp.name or imp.module.split(".")[0]
        # `from pkg import f` where pkg/__init__ re-exports f from pkg.api: the
        # naive 'pkg.f' is absent from every index and the edge would be external.
        item_value = exports.resolve(f"{target_symbol}.{imp.name}") if imp.name else mapping_key
        _set(mapping_key, item_value)

    # Pass 2 - star imports (after explicit imports to preserve their priority).
    _apply_star_imports(star_imports, mapping, module, exports)

    return mapping


# ---------------------------------------------------------------------------
# Resolve call names
# ---------------------------------------------------------------------------


def _resolve_name(
    identifier_name: str,
    mapping: dict[str, str],
    index_fn: dict[str, FunctionRecord],
) -> tuple[str, bool, str | None]:
    """Resolve a call name to ``(qualified_name, is_external, external_lib)``.

    Strategy:
    1. Direct mapping match.
    2. First segment in the mapping (``os.path.join`` -> look up ``os``).
    3. Fallback: return the name unchanged as external.

    Parameters
    ----------
    identifier_name : str
        The call name to resolve, which may be a simple name or a dotted path such as 'os.path.join'.
    mapping : dict[str, str]
        Mapping used to resolve call names to qualified names. Keys are either complete identifier names or leading name segments;
        values are the replacement qualified names used during resolution.
    index_fn : dict[str, FunctionRecord]
        Mapping of function names to their corresponding FunctionRecord objects, used to determine whether a resolved call target
        is defined within the analyzed codebase or should be treated as external.

    Returns
    -------
    tuple[str, bool, str | None]
        Returns a tuple containing the resolved qualified name, a boolean indicating whether the resolved name is external (not
        present in the local function index), and the external library name as the first dotted segment when external, or None
        otherwise.

    """
    if identifier_name in mapping:
        resolved = mapping[identifier_name]
        is_ext = resolved not in index_fn
        return resolved, is_ext, (resolved.split(".")[0] if is_ext else None)

    parts = identifier_name.split(".")
    if len(parts) > 1 and parts[0] in mapping:
        prefix = mapping[parts[0]]
        resolved = f"{prefix}.{'.'.join(parts[1:])}"
        is_ext = resolved not in index_fn
        return resolved, is_ext, (resolved.split(".")[0] if is_ext else None)

    return identifier_name, True, parts[0] or None


# ---------------------------------------------------------------------------
# Resolve methods through MRO
# ---------------------------------------------------------------------------


def _resolve_method_via_mro(
    method_name: str,
    parent_class: str,
    index_cls: dict[str, ClassRecord],
    index_fn: dict[str, FunctionRecord],
    *,
    skip_first: bool = False,
) -> str | None:
    """Return a method qualified_name by walking the MRO.

    ``skip_first``: True for ``super().method()`` (skip the current class).
    Return ``None`` when the method is absent from the local index.

    Parameters
    ----------
    method_name : str
        Name of the method to resolve, used to build candidate qualified names while walking the MRO.
    parent_class : str
        The name of the parent class whose MRO is traversed to resolve the method.
    index_cls : dict[str, ClassRecord]
        Mapping of class names to their ClassRecord entries, used to look up the parent class and traverse its MRO.
    index_fn : dict[str, FunctionRecord]
        Mapping of qualified function names to FunctionRecord objects, used to look up candidate method definitions while walking
        the MRO.
    skip_first : bool = False
        Keyword-only boolean flag. When True, the first entry in the MRO list (the current class) is skipped, so resolution starts
        from the next ancestor; this is intended for super().method() calls. Defaults to False.

    Returns
    -------
    str | None
        Resolve the qualified name of a method by walking the MRO of the given parent class, optionally skipping the first class
        to support super() calls. Returns the matching method qualified name when found in the function index, or None when the
        method is absent.

    """
    class_ = index_cls.get(parent_class)
    if class_ is None:
        return None

    mro = class_.hierarchy.mro_list if (class_.hierarchy and class_.hierarchy.mro_list) else [parent_class]
    start = 1 if skip_first else 0

    for ancestor in mro[start:]:
        candidate = f"{ancestor}.{method_name}"
        if candidate in index_fn:
            return candidate

    return None


# ---------------------------------------------------------------------------
# Collect calls in a function (iterative AST walk)
# ---------------------------------------------------------------------------


def _first_parameter_name(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Return the first parameter name (``self`` or ``cls``), when present.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The AST node representing a function or async function definition from which the first parameter name is extracted.

    Returns
    -------
    str | None
        Returns the name of the first positional or positional-only parameter of the given function definition, typically 'self'
        or 'cls'; returns None if the function has no such parameters.

    """
    all_args = node.args.posonlyargs + node.args.args
    return all_args[0].arg if all_args else None


def _resolve_call_via_self(
    params: _ResolveCallParams,
    self_name: str | None,
) -> CallgraphEdge | None:
    """Process ``self.method()`` calls.

    Parameters
    ----------
    params : _ResolveCallParams
        The call-resolution parameters that describe the call expression being processed, including the AST node, function record,
        source name, line number, and the module/function index used for method resolution.
    self_name : str | None
        The name of the instance variable used to access the method, typically 'self', or None if the call is not via an instance.

    Returns
    -------
    CallgraphEdge | None
        Resolves a method call expressed as self.method() into a callgraph edge, using MRO lookup when possible and falling back
        to a qualified external reference otherwise; returns None if the call is not a self call or no parent class context
        exists.

    """
    if (
        not isinstance(params.func.value, ast.Name)
        or self_name is None
        or params.func.value.id != self_name
        or params.fn_record.parent_class is None
    ):
        return None

    attr = params.func.attr
    target = _resolve_method_via_mro(attr, params.fn_record.parent_class, params.context.index_cls, params.context.index_fn)
    if target is None:
        target = f"{params.fn_record.parent_class}.{attr}"
        is_ext = target not in params.context.index_fn
        ext_lib = target.split(".")[0] if is_ext else None
    else:
        is_ext = False
        ext_lib = None

    return CallgraphEdge(
        source_qname=params.source,
        target_qname=target,
        call_type="via_self",
        is_external=is_ext,
        external_lib=ext_lib,
        line=params.line,
    )


def _resolve_call_via_super(
    params: _ResolveCallParams,
) -> CallgraphEdge | None:
    """Process ``super().method()`` calls.

    Parameters
    ----------
    params : _ResolveCallParams
        The parameters bundle for resolving a super() call, including the call AST node, the enclosing function record, the static
        extraction context, and the source location.

    Returns
    -------
    CallgraphEdge | None
        Returns a CallgraphEdge for the resolved super() call if the call can be resolved, otherwise None.

    """
    if not (
        isinstance(params.func.value, ast.Call)
        and isinstance(params.func.value.func, ast.Name)
        and params.func.value.func.id == "super"
        and params.fn_record.parent_class is not None
    ):
        return None

    attr = params.func.attr
    target = _resolve_method_via_mro(
        attr,
        params.fn_record.parent_class,
        params.context.index_cls,
        params.context.index_fn,
        skip_first=True,
    )
    if target is None:
        return None

    return CallgraphEdge(
        source_qname=params.source,
        target_qname=target,
        call_type="via_super",
        is_external=False,
        external_lib=None,
        line=params.line,
    )


def _resolve_call_qualified(
    params: _ResolveCallParams,
) -> CallgraphEdge | None:
    """Process qualified calls ``a.b.c()``.

    FIX #4: ``ast.unparse`` can produce unresolvable forms for complex ``ast.Attribute`` nodes (for example, ``obj[0].method``,
    ``(lambda: x).attr``). Apply ``_DOTTED_NAME_RE`` here, consistently with the ``_process_call`` fallback, to avoid polluting
    the graph with these edges.

    Parameters
    ----------
    params : _ResolveCallParams
        Call resolution parameters for the qualified call being processed, containing the function AST node, source qualified
        name, line number, and the resolution context needed to determine the target.

    Returns
    -------
    CallgraphEdge | None
        Returns a CallgraphEdge for the resolved qualified call, or None if the call cannot be unparsed or does not match the
        expected dotted-name pattern.

    """
    try:
        identifier_name = ast.unparse(params.func)
    except ValueError:
        return None

    if not _DOTTED_NAME_RE.match(identifier_name):
        logger.debug(
            "Ignored unresolvable qualified call form: '%s' (source=%s, line=%d)",
            identifier_name,
            params.source,
            params.line,
        )
        return None

    target, is_ext, ext_lib = _resolve_name(identifier_name, params.context.mapping, params.context.index_fn)
    return CallgraphEdge(
        source_qname=params.source,
        target_qname=target,
        call_type="qualified_call",
        is_external=is_ext,
        external_lib=ext_lib,
        line=params.line,
    )


def _process_call(
    call_node: ast.Call,
    source: str,
    self_name: str | None,
    fn_record: FunctionRecord,
    context: _CallgraphContext,
) -> CallgraphEdge | None:
    """Create a ``CallgraphEdge`` from an ``ast.Call`` node.

    Return ``None`` for unusable calls (overly complex expression, bare ``super()`` call, and so on).

    Produced call types:
    - ``via_self``: ``self.method()``
    - ``via_super``: ``super().method()``
    - ``direct``: ``name()`` or an unqualified simple call
    - ``qualified_call``: ``a.b.c()`` or ``ClassName.method()`` (internal or external)

    Known limitation - scope shadowing: resolution checks only module/import mapping, not local or enclosing scopes. A parameter
    or local variable named ``foo`` that shadows an import produces a false internal edge. Fixing this requires full scope
    analysis, outside it.8 (see DEC-009).

    Parameters
    ----------
    call_node : ast.Call
        The AST node representing the function call to process.
    source : str
        The qualified name of the source function or callable being processed; used as the source_qname for the resulting
        CallgraphEdge.
    self_name : str | None
        The name of the self parameter for the enclosing method, used to identify calls made via self, or None if the call is not
        inside a method.
    fn_record : FunctionRecord
        The FunctionRecord representing the function that contains the call being processed, used for resolving call parameters
        and constructing the resulting callgraph edge.
    context : _CallgraphContext
        Context object used during call graph construction to resolve call targets; it carries the module/import mapping and
        function index needed to classify calls as internal or external.

    Returns
    -------
    CallgraphEdge | None
        Returns a CallgraphEdge for the processed call, or None for unusable calls such as overly complex expressions, bare
        super() calls, or unresolvable call forms.

    """
    func = call_node.func
    line = call_node.lineno

    # --- self.method(), super().method(), or qualified_call ---
    if isinstance(func, ast.Attribute):
        params = _ResolveCallParams(func, source, fn_record, context, line)
        return _resolve_call_via_self(params, self_name) or _resolve_call_via_super(params) or _resolve_call_qualified(params)

    # --- Direct call: name(); ignore bare super() (super().__init__ is handled above). ---
    if isinstance(func, ast.Name):
        identifier_name = func.id
        if identifier_name == "super":
            return None
        target, is_ext, ext_lib = _resolve_name(identifier_name, context.mapping, context.index_fn)
        return CallgraphEdge(
            source_qname=source,
            target_qname=target,
            call_type="direct",
            is_external=is_ext,
            external_lib=ext_lib,
            line=line,
        )

    # --- Other forms (subscript, lambda, nested call, and so on). ---
    try:
        identifier_name = ast.unparse(func)
    except ValueError:
        return None

    if not _DOTTED_NAME_RE.match(identifier_name):
        logger.debug(
            "Ignored unresolvable call form: '%s' (source=%s, line=%d)",
            identifier_name,
            source,
            line,
        )
        return None

    target, is_ext, ext_lib = _resolve_name(identifier_name, context.mapping, context.index_fn)
    return CallgraphEdge(
        source_qname=source,
        target_qname=target,
        call_type="direct",
        is_external=is_ext,
        external_lib=ext_lib,
        line=line,
    )


def _collect_calls(
    fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
    fn_record: FunctionRecord,
    context: _CallgraphContext,
) -> list[CallgraphEdge]:
    """Iteratively walk a function body without entering nested scopes.

    Remove duplicate edges (same source, target, and call_type). FIX #3: sort edges by line before deduplication so the first
    occurrence (lowest line) is retained regardless of the stack walk's LIFO order.

    Parameters
    ----------
    fn_node : ast.FunctionDef | ast.AsyncFunctionDef
        The AST node for the function or async function whose body is traversed to collect call edges. It provides the starting
        child nodes for the iterative walk and is used to determine the first parameter name for non-static methods.
    fn_record : FunctionRecord
        FunctionRecord for the function whose body is being walked; supplies the source qualified name and method kind used during
        call-edge resolution.
    context : _CallgraphContext
        Context object that carries shared state and configuration for the callgraph extraction pass. It is used to resolve call
        targets, track method context, and store information needed while collecting call edges from a function body.

    Returns
    -------
    list[CallgraphEdge]
        A list of CallgraphEdge objects for the calls found in the function body, excluding nested scopes; the edges are sorted by
        line and deduplicated by source, target, and call type.

    """
    # @staticmethod has no self/cls. Force None so an ordinary parameter is not
    # treated as self by _resolve_call_via_self.
    self_name = _first_parameter_name(fn_node) if fn_record.method_kind is not MethodType.STATIC_METHOD else None
    source = fn_record.qualified_name
    edges: list[CallgraphEdge] = []

    # Do not enter nested FunctionDef/AsyncFunctionDef/ClassDef nodes so we stay
    # in the current function scope.
    stack: list[ast.AST] = list(ast.iter_child_nodes(fn_node))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call):
            edge = _process_call(node, source, self_name, fn_record, context)
            if edge is not None:
                edges.append(edge)
        stack.extend(ast.iter_child_nodes(node))

    # FIX #3: sort by line before deduplication. LIFO traversal does not
    # guarantee that the first collected edge is on the lowest line; sorting
    # makes behavior deterministic and independent of visit order.
    edges.sort(key=lambda e: e.line)

    # Deduplicate by (source, target, call_type).
    seen: set[tuple[str, str, str]] = set()
    deduplicated_edges: list[CallgraphEdge] = []
    for e in edges:
        mapping_key = (e.source_qname, e.target_qname, e.call_type)
        if mapping_key not in seen:
            seen.add(mapping_key)
            deduplicated_edges.append(e)

    return deduplicated_edges


# ---------------------------------------------------------------------------
# Local pass - collect per module
# ---------------------------------------------------------------------------


def _build_ast_index(tree: ast.Module) -> _AstIndex:
    """Build a ``_AstKey(lineno, name) -> AST node`` index for a module.

    Index all functions definable outside a function body: top-level functions,
    class methods, and functions in control blocks (if/try/for/while/with). Do NOT enter function bodies, so closures (not tracked
    by ExtractionVisitor) are not indexed.

    Parameters
    ----------
    tree : ast.Module
        The AST module node whose top-level and control-block function definitions are indexed.

    Returns
    -------
    _AstIndex
        An _AstIndex mapping each indexed function's _AstKey (line number and name) to its corresponding AST node. The index
        includes top-level functions, class methods, and functions defined in control blocks, but excludes functions nested inside
        other function bodies.

    """
    index: _AstIndex = {}
    stack: list[ast.AST] = list(ast.iter_child_nodes(tree))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            index[_AstKey(node.lineno, node.name)] = node
            # Stop; do not recurse into the body (closures are not tracked).
        else:
            stack.extend(ast.iter_child_nodes(node))
    return index


def _find_node_ast(
    ast_index: _AstIndex,
    line_start: int,
    name_simple: str,
    qualified_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Find a function AST node through the precomputed index.

    Parameters
    ----------
    ast_index : _AstIndex
        The precomputed index used to look up function AST nodes by their line start and simple name.
    line_start : int
        The starting line number of the function definition, used as part of the key to look up the AST node.
    name_simple : str
        Simple (unqualified) name of the function to locate in the AST index.
    qualified_name : str
        Qualified name of the function being searched for, used in diagnostic logging when the AST node cannot be found.

    Returns
    -------
    ast.FunctionDef | ast.AsyncFunctionDef | None
        Returns the matching function or async function AST node if found in the precomputed index; otherwise returns None.

    """
    node = ast_index.get(_AstKey(line_start, name_simple))
    if node is None:
        logger.debug(
            "AST node not found for '%s' (line %d, name '%s') - edges ignored.",
            qualified_name,
            line_start,
            name_simple,
        )
    return node


def _process_function(
    fn: FunctionRecord,
    context: _CallgraphContext,
    callgraph: dict[str, list[CallgraphEdge]],
) -> None:
    """Collect a function's edges and store them in ``callgraph``.

    Parameters
    ----------
    fn : FunctionRecord
        The FunctionRecord representing the function whose call edges are to be collected and stored in the callgraph.
    context : _CallgraphContext
        The callgraph extraction context, which carries the AST index and other shared state used to locate function definitions
        and collect call edges.
    callgraph : dict[str, list[CallgraphEdge]]
        Mapping from function qualified names to lists of CallgraphEdge objects; the collected edges for the current function are
        stored under its qualified name.

    """
    fn_node = _find_node_ast(context.ast_index, fn.line_start, name_simple(fn.qualified_name), fn.qualified_name)
    if fn_node is None:
        callgraph[fn.qualified_name] = []
        return
    edges = _collect_calls(fn_node, fn, context)
    callgraph[fn.qualified_name] = edges


def _local_pass(
    project_record: ProjectRecord,
    index_fn: dict[str, FunctionRecord],
    index_cls: dict[str, ClassRecord],
) -> dict[str, list[CallgraphEdge]]:
    """Collect all direct edges for every project function.

    Return a dict ``{source_qualified_name -> [CallgraphEdge]}``.

    Note: reparse each file instead of reusing the AST from the first static
    extraction. This simplifies the API because no cache passes between stages. If measured reparsing becomes a bottleneck, pass a
    ``{file_path: ast.Module}`` cache as a parameter.

    Parameters
    ----------
    project_record : ProjectRecord
        The ProjectRecord whose modules should be scanned for callgraph edges.
    index_fn : dict[str, FunctionRecord]
        A dictionary mapping each function's qualified name to its FunctionRecord, providing the project-wide function index used
        to resolve call targets and distinguish project-defined symbols during callgraph construction.
    index_cls : dict[str, ClassRecord]
        A dictionary mapping qualified class names to their ClassRecord objects, used to resolve class references and determine
        known project symbols during callgraph construction.

    Returns
    -------
    dict[str, list[CallgraphEdge]]
        A dictionary mapping each project function's qualified name to a list of CallgraphEdge objects representing the direct
        call edges collected for that function.

    """
    callgraph: dict[str, list[CallgraphEdge]] = {}

    # Pre-pass: parse every file once, calculate exported names (needed for star
    # imports), and build origins (needed to follow re-exports). All modules must
    # be seen first. Index by file_path, not module_name, so each ModuleRecord is
    # paired with its own file AST (see _effective_modules for .py/.pyi twins).
    trees: dict[Path, ast.Module] = {}
    exported_names: dict[str, frozenset[str]] = {}
    origins: dict[str, dict[str, str]] = {}

    modules = _effective_modules(project_record.modules)

    for module in modules:
        outcome = parse_file_with_lines(module.file_path)
        if outcome is None:
            logger.warning(
                "Unable to reparse '%s' for callgraph - module ignored.",
                module.file_path,
            )
            for fn in iter_functions([module]):
                callgraph.setdefault(fn.qualified_name, [])
        else:
            tree, _ = outcome
            trees[module.file_path] = tree
            exported_names[module.module_name] = _module_exported_names(module, tree)
            origins[module.module_name] = _module_origins(module)

    exports = _ExportContext(
        exported_names=exported_names,
        origins=origins,
        # Symbols actually defined in the project - re-export stop condition.
        known_symbols=frozenset(index_fn) | frozenset(index_cls),
    )

    # Main pass: build the callgraph with resolved star imports.
    for module in modules:
        tree = trees.get(module.file_path)
        if tree is None:
            continue  # Already handled (unparseable file).

        mapping = _module_mapping(module, exports)
        ast_index = _build_ast_index(tree)
        context = _CallgraphContext(ast_index, mapping, index_fn, index_cls)

        for fn in iter_functions([module]):
            _process_function(fn, context, callgraph)

    return callgraph


# ---------------------------------------------------------------------------
# Global pass - unified DFS analysis
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CallgraphNodeAnalysis:
    """DFS analysis result for a callgraph node.

    Attributes
    ----------
    depth : int
        Maximum reachable depth of the node.
    is_recursive : bool
        True when the node is part of a recursion cycle.

    """

    depth: int
    is_recursive: bool


@dataclasses.dataclass(frozen=True)
class CallgraphAnalysis:
    """Global callgraph DFS analysis result.

    Attributes
    ----------
    nodes : dict[str, CallgraphNodeAnalysis]
        Mapping of node qualified names to their analysis result.
    depth_exceeded_edges : set[tuple[str, str]]
        Set of (source, target) edges whose depth exceeds the maximum.

    """

    nodes: dict[str, CallgraphNodeAnalysis]
    depth_exceeded_edges: set[tuple[str, str]]


@dataclasses.dataclass
class _DfsFrame:
    """Local state for a node during iterative DFS.

    Attributes
    ----------
    node : str
        Qualified name of the current node being visited.
    children : list[str]
        Qualified names of the node's children.
    next_child_index : int
        Index of the next child to process.
    max_child_depth : int
        Maximum depth reached among the node's children.

    """

    node: str
    children: list[str]
    next_child_index: int = 0
    max_child_depth: int = 0


@dataclasses.dataclass
class _DfsState:
    """Mutable state shared during ``_analyze_callgraph`` DFS.

    Attributes
    ----------
    max_depth : int
        Maximum depth the analysis is capped at.
    current_state : dict[str, int]
        Visitation state of each node: 0 unvisited, 1 active, 2 done.
    depths : dict[str, int]
        Computed depth of each node, capped at ``max_depth``.
    recursive_nodes : set[str]
        Nodes found to be part of a recursion cycle.
    depth_exceeded_edges : set[tuple[str, str]]
        ``(source, target)`` edges whose target depth reaches or exceeds ``max_depth``.
    stack : list[_DfsFrame]
        Stack of frames used by the iterative DFS.
    stack_positions : dict[str, int]
        Mapping of node names to their position in the stack.

    """

    max_depth: int
    current_state: dict[str, int] = dataclasses.field(default_factory=dict)
    depths: dict[str, int] = dataclasses.field(default_factory=dict)
    recursive_nodes: set[str] = dataclasses.field(default_factory=set)
    depth_exceeded_edges: set[tuple[str, str]] = dataclasses.field(default_factory=set)
    stack: list[_DfsFrame] = dataclasses.field(default_factory=list)
    stack_positions: dict[str, int] = dataclasses.field(default_factory=dict)


def _process_dfs_postorder(state: _DfsState) -> None:
    """Finalize the stack-head node (postorder) and update its parent.

    Parameters
    ----------
    state : _DfsState
        The DFS state used during postorder processing, containing the node stack, depth mappings, visitation states, stack
        positions, maximum depth, and depth-exceeded edges.

    """
    frame = state.stack[-1]
    state.depths[frame.node] = min(frame.max_child_depth, state.max_depth)
    state.current_state[frame.node] = 2
    state.stack_positions.pop(frame.node, None)
    state.stack.pop()

    if state.stack:
        parent = state.stack[-1]
        parent.max_child_depth = max(parent.max_child_depth, state.depths[frame.node] + 1)
        if state.depths[frame.node] >= state.max_depth:
            state.depth_exceeded_edges.add((parent.node, frame.node))


def _process_dfs_child(
    child: str,
    child_state: int,
    frame: _DfsFrame,
    state: _DfsState,
    local_graph: dict[str, list[str]],
) -> None:
    """Process a child according to its DFS state (unvisited, active, done).

    Parameters
    ----------
    child : str
        The identifier of the child node being processed.
    child_state : int
        The DFS state of the child node, where 0 means unvisited, 1 means active (currently on the DFS stack), and 2 means done
        (fully processed).
    frame : _DfsFrame
        The DFS frame for the current node being processed, used to track traversal state and update the maximum child depth when
        the child has already been fully processed.
    state : _DfsState
        The DFS state object that records each node's visitation status, stack position, recursion set, depth information, and
        edges that exceed the configured maximum depth. It is mutated as the child is processed.
    local_graph : dict[str, list[str]]
        Mapping of graph nodes to their child nodes; used to retrieve the children of the child being processed.

    """
    if child_state == 0:
        state.current_state[child] = 1
        state.stack_positions[child] = len(state.stack)
        state.stack.append(_DfsFrame(node=child, children=local_graph[child]))
    elif child_state == 1:
        for frame_in_cycle in state.stack[state.stack_positions[child] :]:
            state.recursive_nodes.add(frame_in_cycle.node)
    else:
        frame.max_child_depth = max(frame.max_child_depth, state.depths[child] + 1)
        if state.depths[child] >= state.max_depth:
            state.depth_exceeded_edges.add((frame.node, child))


def _analyze_callgraph(
    callgraph: dict[str, list[CallgraphEdge]],
    index_fn: dict[str, FunctionRecord],
    max_depth: int,
) -> CallgraphAnalysis:
    """Analyze recursion, depth, and depth_exceeded in one DFS.

    Return a ``CallgraphAnalysis`` containing:
    - ``nodes``: depth and recursion flag per function;
    - ``depth_exceeded_edges``: ``(source, target)`` pairs to mark.

    Semantics:
    - Only internal targets present in ``index_fn`` participate in the local graph.
    - A back-edge marks a cycle but does not contribute to depth.
    - Depth is capped at ``max_depth``.
    - An edge ``A -> B`` is in ``depth_exceeded_edges`` when ``depth[B] >= max_depth``.
    - Cycle nodes receive depth 0 (absent from ``depths`` when processed as a
      target by their ancestors).

    Parameters
    ----------
    callgraph : dict[str, list[CallgraphEdge]]
        A dictionary mapping each function's qualified name to a list of CallgraphEdge objects representing outgoing calls from
        that function. Only edges whose target is internal and present in index_fn participate in the local graph analysis.
    index_fn : dict[str, FunctionRecord]
        Mapping of qualified function names to their FunctionRecord objects. It defines the set of internal functions that
        participate in the local callgraph analysis; only edges whose target appears in this mapping are considered, and every key
        is treated as a node in the graph.
    max_depth : int
        Maximum depth to cap the analysis at; any function depth reaching or exceeding this value causes the incoming edge to be
        recorded in depth_exceeded_edges.

    Returns
    -------
    CallgraphAnalysis
        Returns a CallgraphAnalysis containing a per-function depth and recursion flag, along with the list of (source, target)
        edges that exceed the maximum depth.

    """
    local_graph: dict[str, list[str]] = {
        source: [e.target_qname for e in edges if not e.is_external and e.target_qname in index_fn]
        for source, edges in callgraph.items()
    }
    for qname in index_fn:
        local_graph.setdefault(qname, [])

    state = _DfsState(max_depth=max_depth)

    for start in index_fn:
        if state.current_state.get(start, 0) != 0:
            continue

        state.current_state[start] = 1
        state.stack_positions[start] = 0
        state.stack.append(_DfsFrame(node=start, children=local_graph[start]))

        while state.stack:
            frame = state.stack[-1]

            if frame.next_child_index >= len(frame.children):
                _process_dfs_postorder(state)
                continue

            child = frame.children[frame.next_child_index]
            frame.next_child_index += 1
            _process_dfs_child(child, state.current_state.get(child, 0), frame, state, local_graph)

    nodes = {
        qname: CallgraphNodeAnalysis(
            depth=state.depths.get(qname, 0),
            is_recursive=(qname in state.recursive_nodes),
        )
        for qname in index_fn
    }
    return CallgraphAnalysis(nodes=nodes, depth_exceeded_edges=state.depth_exceeded_edges)


def _apply_callgraph_analysis(
    callgraph: dict[str, list[CallgraphEdge]],
    index_fn: dict[str, FunctionRecord],
    analysis: CallgraphAnalysis,
) -> None:
    """Project analysis results onto the callgraph and ``FunctionRecord`` objects.

    Parameters
    ----------
    callgraph : dict[str, list[CallgraphEdge]]
        Mapping from qualified function names to their list of callgraph edges.
    index_fn : dict[str, FunctionRecord]
        A dictionary mapping qualified function names to FunctionRecord instances. The function iterates over these records to
        apply callgraph analysis results, updating each record's callgraph edges, depth, recursion flag, and confidence.
    analysis : CallgraphAnalysis
        The analysis to apply, containing callgraph node data and depth-exceeded edge information.

    """
    for qname, fn in index_fn.items():
        node = analysis.nodes[qname]
        edges = callgraph.get(qname, [])
        new_edges = [
            dataclasses.replace(edge, depth_exceeded=True)
            if (not edge.is_external and (qname, edge.target_qname) in analysis.depth_exceeded_edges)
            else edge
            for edge in edges
        ]
        callgraph[qname] = new_edges
        fn.callgraph = new_edges
        fn.callgraph_depth = node.depth
        fn.is_recursive = node.is_recursive
        fn.callgraph_confidence = Confidence.EXPLICIT if new_edges else Confidence.INFERRED_LOW


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_callgraph(
    project_record: ProjectRecord,
    max_depth: int = _CALLGRAPH_MAX_DEPTH,
) -> dict[str, list[CallgraphEdge]]:
    """Build the complete project callgraph in two passes.

    1. Local pass: collect direct edges for each function by parsing AST files
       and resolving call names.
    2. Global pass: detect recursion, calculate depths, mark ``depth_exceeded``,
       and mutate ``FunctionRecord`` objects in place.

    Parameters
    ----------
    project_record : ProjectRecord
        Project to analyze.
    max_depth : int
        Maximum depth of the local callgraph (default: 5). Edges to targets
        reaching this limit are marked ``depth_exceeded``.

    Returns
    -------
    dict[str, list[CallgraphEdge]]
        Returns ``{source_qualified_name -> [CallgraphEdge]}``.

        Also mutates project ``FunctionRecord`` objects **in place**:

        - ``callgraph``: deduplicated direct edges
        - ``callgraph_confidence``: EXPLICIT when edges exist, INFERRED_LOW otherwise
        - ``is_recursive``: True when a member of a cycle
        - ``callgraph_depth``: maximum reachable depth in [0, max_depth]

    """
    index_fn = _index_functions(project_record)
    index_cls = _index_classes(project_record)
    callgraph = _local_pass(project_record, index_fn, index_cls)
    analysis = _analyze_callgraph(callgraph, index_fn, max_depth)
    _apply_callgraph_analysis(callgraph, index_fn, analysis)
    return callgraph
