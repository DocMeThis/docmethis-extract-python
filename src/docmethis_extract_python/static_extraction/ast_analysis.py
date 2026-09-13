# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""AST analysis - parse and extract function and class identities.

Use only the standard ``ast`` module (DEC-001). Log and ignore parsing errors
(DEC-T03).
"""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from docmethis_extract_python.static_extraction.ast_inheritance import (
    DEFAULT_MIXIN_MAX_METHODS,
    extract_class_attributes,
    extract_instance_attributes,
    prepare_class_hierarchy,
)
from docmethis_extract_python.static_extraction.ast_metrics import compute_complexity, detect_io
from docmethis_extract_python.static_extraction.ast_parser import parse_file_with_lines
from docmethis_extract_python.static_extraction.ast_properties import extract_observable_properties
from docmethis_extract_python.static_extraction.exceptions_ast import extract_exceptions
from docmethis_extract_python.static_extraction.models import (
    MISSING_VALUE,
    SPECIAL_VALUE,
    ClassRecord,
    Confidence,
    Decorator,
    FunctionRecord,
    ImportRecord,
    MethodType,
    ModuleRecord,
    Parameter,
    ParameterType,
    PropertyAccessor,
    Provenance,
    ResolvedTypes,
    Signature,
    TypeInfo,
    Visibility,
    name_simple,
)
from docmethis_extract_python.static_extraction.source_code_extraction import (
    build_class_skeleton,
    build_module_skeleton,
    extract_source_function,
)
from docmethis_extract_python.static_extraction.traversal import effective_visibility
from docmethis_extract_python.static_extraction.type_normalization import normalize_type
from docmethis_extract_python.static_extraction.type_resolution import CacheStubs, enrich_resolved_types

logger = logging.getLogger(__name__)

_KNOWN_DECORATORS: frozenset[str] = frozenset(
    {
        "classmethod",
        "staticmethod",
        "property",
        "abstractmethod",
    }
)

# Implicit parameters excluded from ResolvedTypes (not useful to Module 2 docs).
_IMPLICIT_PARAMETERS: frozenset[str] = frozenset({"self", "cls"})


def _format_default_value(default: ast.expr | None) -> str:
    """Return the string representation of a parameter's default value.

    - ``None``: ``MISSING_VALUE``
    - Simple constant: literal representation (for example, ``"42"``, ``"None"``)
    - Complex expression: ``SPECIAL_VALUE``

    Parameters
    ----------
    default : ast.expr | None
        The AST node representing the parameter's default value, or None if the parameter has no default.

    Returns
    -------
    str
        Returns a string representing the parameter default value: MISSING_VALUE if default is None, the literal representation if
        it is a simple constant, or SPECIAL_VALUE for complex expressions.

    """
    if default is None:
        return MISSING_VALUE
    if isinstance(default, ast.Constant):
        return ast.unparse(default)
    return SPECIAL_VALUE


def _extract_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Parameter]:
    """Extract function parameters from its AST node.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The AST node of the function definition from which to extract parameters.

    Returns
    -------
    list[Parameter]
        A list of Parameter objects representing the parameters extracted from the AST node, in the order they appear in the
        function signature.

    """
    args = node.args
    params: list[Parameter] = []

    # Positional-only and positional-or-keyword share right-aligned args.defaults.
    all_posargs = args.posonlyargs + args.args
    pos_default_offset = len(all_posargs) - len(args.defaults)

    for i, arg in enumerate(args.posonlyargs):
        default_idx = i - pos_default_offset
        default = args.defaults[default_idx] if default_idx >= 0 else None
        params.append(
            Parameter(
                name=arg.arg,
                annotation_raw=ast.unparse(arg.annotation) if arg.annotation else MISSING_VALUE,
                default_value=_format_default_value(default),
                kind=ParameterType.POSITIONAL_ONLY,
                line_start=arg.lineno,
            )
        )

    for i, arg in enumerate(args.args):
        default_idx = len(args.posonlyargs) + i - pos_default_offset
        default = args.defaults[default_idx] if default_idx >= 0 else None
        params.append(
            Parameter(
                name=arg.arg,
                annotation_raw=ast.unparse(arg.annotation) if arg.annotation else MISSING_VALUE,
                default_value=_format_default_value(default),
                kind=ParameterType.POSITIONAL_OR_KEYWORD,
                line_start=arg.lineno,
            )
        )

    if args.vararg:
        params.append(
            Parameter(
                name=args.vararg.arg,
                annotation_raw=ast.unparse(args.vararg.annotation) if args.vararg.annotation else MISSING_VALUE,
                default_value=MISSING_VALUE,
                kind=ParameterType.VAR_POSITIONAL,
                line_start=args.vararg.lineno,
            )
        )

    for arg, kw_default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        params.append(
            Parameter(
                name=arg.arg,
                annotation_raw=ast.unparse(arg.annotation) if arg.annotation else MISSING_VALUE,
                default_value=_format_default_value(kw_default),
                kind=ParameterType.KEYWORD_ONLY,
                line_start=arg.lineno,
            )
        )

    if args.kwarg:
        params.append(
            Parameter(
                name=args.kwarg.arg,
                annotation_raw=ast.unparse(args.kwarg.annotation) if args.kwarg.annotation else MISSING_VALUE,
                default_value=MISSING_VALUE,
                kind=ParameterType.VAR_KEYWORD,
                line_start=args.kwarg.lineno,
            )
        )

    return params


def _extract_decorators(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Decorator]:
    """Extract function decorators from its AST node.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The AST node representing a function or async function definition from which decorators are extracted.

    Returns
    -------
    list[Decorator]
        A list of Decorator objects representing each decorator extracted from the function or async function AST node, in source
        order.

    """
    return [
        Decorator(
            full_decorator=ast.unparse(dec),
            is_known=_extract_decorator_name(dec) in _KNOWN_DECORATORS,
        )
        for dec in node.decorator_list
    ]


def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Signature:
    """Build a function Signature from its AST node.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The AST node representing the function or async function definition from which to build the signature.

    Returns
    -------
    Signature
        A Signature object containing the extracted parameters, return annotation, and decorators from the given AST node.

    """
    return Signature(
        parameters=_extract_parameters(node),
        return_annotation=ast.unparse(node.returns) if node.returns else MISSING_VALUE,
        decorators=_extract_decorators(node),
    )


def _imports_from_import_node(node: ast.Import, *, conditional: bool, type_checking_only: bool) -> list[ImportRecord]:
    """Build ImportRecords from an ast.Import node.

    Parameters
    ----------
    node : ast.Import
        The ast.Import node from which to extract import records.
    conditional : bool
        Indicates whether the import is considered conditional.
    type_checking_only : bool
        Indicates whether the import is used only for type checking; this flag is passed through to each generated ImportRecord.

    Returns
    -------
    list[ImportRecord]
        Returns a list of ImportRecord objects, one for each alias in the given ast.Import node, with the module, alias,
        conditional, and type_checking_only fields populated from the node and arguments.

    """
    return [
        ImportRecord(
            module=alias.name,
            name=None,
            alias=alias.asname,
            conditional=conditional,
            type_checking_only=type_checking_only,
        )
        for alias in node.names
    ]


def _imports_from_import_from_node(node: ast.ImportFrom, *, conditional: bool, type_checking_only: bool) -> list[ImportRecord]:
    """Build ImportRecords from an ast.ImportFrom node.

    Parameters
    ----------
    node : ast.ImportFrom
        The ast.ImportFrom node representing the import-from statement to convert into ImportRecord objects.
    conditional : bool
        Whether the import represented by the node is conditional; this value is propagated to each generated ImportRecord.
    type_checking_only : bool
        A boolean flag indicating whether the import statements being processed are type-checking-only imports.

    Returns
    -------
    list[ImportRecord]
        A list of ImportRecord objects representing the imports from the given ast.ImportFrom node.

    """
    module = "." * (node.level or 0) + (node.module or "")
    return [
        ImportRecord(
            module=module,
            name=alias.name,
            alias=alias.asname,
            conditional=conditional,
            type_checking_only=type_checking_only,
        )
        for alias in node.names
    ]


def _imports_from_statements(stmts: list[ast.stmt], *, conditional: bool, type_checking_only: bool) -> list[ImportRecord]:
    """Extract ImportRecords from statements with the given markers.

    Parameters
    ----------
    stmts : list[ast.stmt]
        The list of AST statement nodes to process. Each statement is inspected to determine whether it is an import statement
        (ast.Import or ast.ImportFrom), and any such statements are used to generate ImportRecords.
    conditional : bool
        Whether the import statements being processed are conditional.
    type_checking_only : bool
        Flag indicating whether the imports should be recorded as type-checking-only, used to mark each extracted ImportRecord
        accordingly.

    Returns
    -------
    list[ImportRecord]
        A list of ImportRecord objects extracted from the given statements.

    """
    result: list[ImportRecord] = []
    for stmt in stmts:
        if isinstance(stmt, ast.Import):
            result.extend(_imports_from_import_node(stmt, conditional=conditional, type_checking_only=type_checking_only))
        elif isinstance(stmt, ast.ImportFrom):
            result.extend(_imports_from_import_from_node(stmt, conditional=conditional, type_checking_only=type_checking_only))
    return result


def _is_try_import_error(node: ast.Try) -> bool:
    """Return True when a try block catches ImportError or ModuleNotFoundError.

    Parameters
    ----------
    node : ast.Try
        The AST try node to inspect for handlers that catch ImportError or ModuleNotFoundError.

    Returns
    -------
    bool
        Return True if the try statement contains an exception handler that catches ImportError or ModuleNotFoundError, otherwise
        return False.

    """
    return any(
        handler.type is not None
        and (
            (isinstance(handler.type, ast.Name) and handler.type.id in {"ImportError", "ModuleNotFoundError"})
            or (isinstance(handler.type, ast.Attribute) and handler.type.attr in {"ImportError", "ModuleNotFoundError"})
        )
        for handler in node.handlers
    )


def _is_type_checking_if(node: ast.If) -> bool:
    """Return True when an if tests TYPE_CHECKING (simple or typing.TYPE_CHECKING).

    Parameters
    ----------
    node : ast.If
        The AST If node to inspect.

    Returns
    -------
    bool
        True if the if condition is a TYPE_CHECKING reference, either as a bare name or as an attribute access such as
        typing.TYPE_CHECKING; False otherwise.

    """
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _extract_module_imports(tree: ast.Module) -> tuple[list[ImportRecord], bool]:
    """Extract top-level imports with conditional markers.

    Handle:

    - direct imports
    - ``try/except ImportError`` -> ``conditional=True``
    - ``if TYPE_CHECKING`` -> ``type_checking_only=True``
    - ``from __future__ import annotations`` -> ``future_annotations=True``

    Return ``(imports, future_annotations)``.

    Parameters
    ----------
    tree : ast.Module
        The AST module node representing the parsed source code whose top-level import statements are extracted and analyzed.

    Returns
    -------
    tuple[list[ImportRecord], bool]
        Returns a tuple whose first element is a list of ImportRecord objects representing the top-level imports extracted from
        the module, and whose second element is a boolean flag that is True when the module includes a future annotations import
        and False otherwise.

    """
    imports: list[ImportRecord] = []
    future_annotations = False

    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(_imports_from_import_node(node, conditional=False, type_checking_only=False))

        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__" and any(a.name == "annotations" for a in node.names):
                future_annotations = True
            imports.extend(_imports_from_import_from_node(node, conditional=False, type_checking_only=False))

        elif isinstance(node, ast.Try) and _is_try_import_error(node):
            imports.extend(_imports_from_statements(node.body, conditional=True, type_checking_only=False))

        elif isinstance(node, ast.If) and _is_type_checking_if(node):
            imports.extend(_imports_from_statements(node.body, conditional=False, type_checking_only=True))

    return imports, future_annotations


def _compute_type_confidence(types: ResolvedTypes | None) -> Confidence:
    """Return the overall confidence of resolved types.

    The overall confidence is the lowest confidence among all resolved types. Return ``ABSENT`` when no type was resolved (empty
    dict and missing return).

    Parameters
    ----------
    types : ResolvedTypes | None
        The resolved type information to compute confidence from, containing parameter types and an optional return type; pass
        None when no type information is available.

    Returns
    -------
    Confidence
        The overall confidence of the resolved types, equal to the lowest confidence among all resolved types. Returns ABSENT if
        no types were resolved (no parameters and no return type).

    """
    if types is None:
        return Confidence.ABSENT
    confidences = [ti.confidence for ti in types.parameters.values()]
    if types.return_type is not None:
        confidences.append(types.return_type.confidence)
    if not confidences:
        return Confidence.ABSENT
    if any(c == Confidence.INFERRED_LOW for c in confidences):
        return Confidence.INFERRED_LOW
    if any(c == Confidence.INFERRED_HIGH for c in confidences):
        return Confidence.INFERRED_HIGH
    return Confidence.EXPLICIT


def _build_resolved_types(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ResolvedTypes | None:
    """Build ResolvedTypes from a function's explicit annotations.

    Phase 1 (iteration 2): explicit annotations only, normalized to PEP 585/604. Exclude implicit parameters (``self``, ``cls``).
    Return ``None`` when no annotation is present.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        AST node representing the function or async function definition whose explicit annotations are used to build resolved
        types.

    Returns
    -------
    ResolvedTypes | None
        Builds a ResolvedTypes object from a function's explicit type annotations, normalizing them to PEP 585/604 and excluding
        implicit parameters such as self and cls. Returns None when no parameter or return annotation is present.

    """
    params = _extract_parameters(node)

    parameters: dict[str, TypeInfo] = {}
    for p in params:
        if p.name in _IMPLICIT_PARAMETERS:
            continue
        if p.annotation_raw != MISSING_VALUE:
            parameters[p.name] = TypeInfo(
                type_str=normalize_type(p.annotation_raw),
                confidence=Confidence.EXPLICIT,
                provenance=Provenance.ANNOTATION,
                raw=p.annotation_raw,
            )

    return_: TypeInfo | None = None
    return_raw = ast.unparse(node.returns) if node.returns else MISSING_VALUE
    if return_raw != MISSING_VALUE:
        return_ = TypeInfo(
            type_str=normalize_type(return_raw),
            confidence=Confidence.EXPLICIT,
            provenance=Provenance.ANNOTATION,
            raw=return_raw,
        )

    if not parameters and return_ is None:
        return None

    return ResolvedTypes(parameters=parameters, return_type=return_)


def extract_module_record(  # noqa: PLR0913
    source_path: Path,
    module_name: str,
    *,
    is_stub: bool,
    is_test: bool,
    file_hash: str,
    mixin_max_methods: int = DEFAULT_MIXIN_MAX_METHODS,
) -> ModuleRecord | None:
    """Extract a ModuleRecord from a Python file.

    Return ``None`` when the file cannot be parsed. Progressive fields keep their default values.

    Parameters
    ----------
    source_path : Path
        The filesystem path of the Python source file to parse and extract a ModuleRecord from.
    module_name : str
        The fully qualified name of the module being extracted, used to populate the ModuleRecord and determine visibility.
    is_stub : bool
        Indicates whether the source file is a stub file, typically with a .pyi extension.
    is_test : bool
        Indicates whether the source file is considered a test module.
    file_hash : str
        Hash of the file's contents used to uniquely identify the module version.
    mixin_max_methods : int
        Maximum number of methods a class may have to be classified as a mixin during extraction.

    Returns
    -------
    ModuleRecord | None
        Returns a ModuleRecord when the source file is parsed successfully, or None when the file cannot be parsed.

    """
    outcome = parse_file_with_lines(source_path)
    if outcome is None:
        return None
    tree, lines = outcome

    visitor = ExtractionVisitor(source_path, module_name, lines, mixin_max_methods=mixin_max_methods)
    visitor.visit(tree)
    visitor.finalize()
    imports, future_annotations = _extract_module_imports(tree)

    source_module = build_module_skeleton(
        visitor.top_level_function_nodes,
        visitor.top_level_class_nodes,
        lines,
    )

    return ModuleRecord(
        file_path=source_path.resolve(),
        module_name=module_name,
        functions=visitor.function_records,
        classes=visitor.classes,
        is_stub=is_stub,
        is_test=is_test,
        line_start=1,
        line_end=len(lines),
        visibility=_determine_module_visibility(module_name),
        existing_docstring=ast.get_docstring(tree),
        docstring_line_start=_extract_docstring_line(tree),
        imports=imports,
        future_annotations=future_annotations,
        source_code=source_module,
        file_hash=file_hash,
    )


class ExtractionVisitor(ast.NodeVisitor):
    """AST visitor that extracts functions and classes from a module.

    Maintain context stacks to compute qualified names
    (for example, ``OuterClass.InnerClass.method``).

    Parameters
    ----------
    source_path : Path
        Path to the source file that the visitor will analyze.
    module_name : str
        Name of the module being analyzed, used to associate extracted records with their originating module.
    lines : list[str] | None = None
        Optional list of source code lines to associate with the visitor. When omitted or None, an empty list is used.
    mixin_max_methods : int
        Maximum number of methods a class may define for it to still be treated as a mixin during extraction.

    Attributes
    ----------
    source_path : Path
        Path to the source file being analyzed.
    module_name : str
        Name of the module being analyzed.
    function_records : list[FunctionRecord]
        Extracted function records.
    classes : list[ClassRecord]
        Extracted class records.
    top_level_function_nodes : list[ast.FunctionDef | ast.AsyncFunctionDef]
        Top-level AST function nodes used by build_module_skeleton.
    top_level_class_nodes : list[ast.ClassDef]
        Top-level AST class nodes used by build_module_skeleton.

    """

    def __init__(
        self,
        source_path: Path,
        module_name: str,
        lines: list[str] | None = None,
        mixin_max_methods: int = DEFAULT_MIXIN_MAX_METHODS,
    ) -> None:
        """Initialize the visitor with the file path, module name, and source lines.

        Parameters
        ----------
        source_path : Path
            Path to the source file that the visitor will analyze.
        module_name : str
            Name of the module being analyzed, used to associate extracted records with their originating module.
        lines : list[str] | None = None
            Optional list of source code lines to associate with the visitor. When omitted or None, an empty list is used.
        mixin_max_methods : int
            Maximum number of methods a class may define for it to still be treated as a mixin during extraction.

        """
        self.source_path = source_path
        self.module_name = module_name
        self._lines: list[str] = lines or []
        self._mixin_max_methods = mixin_max_methods
        self.function_records: list[FunctionRecord] = []
        self.classes: list[ClassRecord] = []
        self._class_stack: list[ClassRecord] = []
        self._cache_stubs: CacheStubs = CacheStubs()
        # Top-level AST nodes used by build_module_skeleton.
        self.top_level_function_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.top_level_class_nodes: list[ast.ClassDef] = []
        # it.10: @overload nodes waiting to be attached to an implementation.
        # One buffer stack entry is kept per scope (module, then nested classes),
        # so orphan overloads are finalized while their scope is active.
        self._overload_stack: list[dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]]] = [{}]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a synchronous function definition.

        Parameters
        ----------
        node : ast.FunctionDef
            The AST node representing the synchronous function definition to visit.

        """
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an asynchronous function definition.

        Parameters
        ----------
        node : ast.AsyncFunctionDef
            The AST node representing the asynchronous function definition to visit.

        """
        self._process_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition.

        Parameters
        ----------
        node : ast.ClassDef
            The AST class definition node to visit.

        """
        qualified_name = self._global_qualified_name(node.name)
        is_top_level = not self._class_stack

        record = ClassRecord(
            qualified_name=qualified_name,
            file_path=self.source_path.resolve(),
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            col_start=node.col_offset,
            col_end=node.end_col_offset,
            visibility=self._composed_class_visibility(node.name),
            parent_module=self.module_name,
            existing_docstring=ast.get_docstring(node),
            docstring_line_start=_extract_docstring_line(node),
            source_code=build_class_skeleton(node, self._lines) if self._lines else None,
            hierarchy=prepare_class_hierarchy(node, mixin_max_methods=self._mixin_max_methods),
            class_attributes=extract_class_attributes(node),
            instance_attributes=extract_instance_attributes(node),
        )

        self._class_stack.append(record)
        self._overload_stack.append({})
        self.generic_visit(node)
        self._finalize_overloads()
        self._overload_stack.pop()
        self._class_stack.pop()

        if is_top_level:
            self.top_level_class_nodes.append(node)
        self.classes.append(record)

    def _process_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        overloads: list[ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
    ) -> None:
        """Process a function definition (sync or async).

        ``overloads`` is provided only by :meth:`_finalize_overloads`, which promotes an
        orphan overload to a ``FunctionRecord``. The ``@overload`` short circuit is then bypassed and the existing signatures are
        injected as-is.

        Parameters
        ----------
        node : ast.FunctionDef | ast.AsyncFunctionDef
            The AST node representing the function definition to process, either a synchronous function (ast.FunctionDef) or an
            asynchronous function (ast.AsyncFunctionDef).
        overloads : list[ast.FunctionDef | ast.AsyncFunctionDef] | None = None
            Optional list of overloaded function definition AST nodes associated with this function. When provided by
            _finalize_overloads, it promotes an orphan overload and bypasses the @overload short circuit, injecting existing
            signatures as-is. If None, overloads are either collected from the current overload stack or absent.

        """
        qualified_name = self._global_qualified_name(node.name)

        # Accumulate @overload definitions without creating a FunctionRecord (DEC-023).
        if overloads is None and _is_overload(node):
            self._overload_stack[-1].setdefault(qualified_name, []).append(node)
            return

        in_class = len(self._class_stack) > 0
        parent_class = self._class_stack[-1].qualified_name if in_class else None

        base_types = _build_resolved_types(node)
        parent_classes = [name_simple(class_.qualified_name) for class_ in self._class_stack]
        resolved_types = enrich_resolved_types(
            base_types, node, self.source_path, self.module_name, parent_classes, cache=self._cache_stubs
        )

        exceptions, except_blocks = extract_exceptions(node)
        properties = extract_observable_properties(node)

        record = FunctionRecord(
            qualified_name=qualified_name,
            file_path=self.source_path.resolve(),
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            col_start=node.col_offset,
            col_end=node.end_col_offset,
            method_kind=_determine_method_type(node, in_class=in_class),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            visibility=self._composed_class_visibility(node.name),
            parent_class=parent_class,
            parent_module=self.module_name,
            existing_docstring=ast.get_docstring(node),
            docstring_line_start=_extract_docstring_line(node),
            signature=_build_signature(node),
            signature_confidence=Confidence.EXPLICIT,
            types=resolved_types,
            types_confidence=_compute_type_confidence(resolved_types),
            complexity=compute_complexity(node),
            complexity_confidence=Confidence.EXPLICIT,
            io_effects=detect_io(node),
            io_effects_confidence=Confidence.EXPLICIT,
            source_code=extract_source_function(node, self._lines) if self._lines else None,
            exceptions=exceptions,
            exceptions_confidence=Confidence.EXPLICIT,
            except_blocks=except_blocks,
            observable_properties=properties,
            observable_properties_confidence=Confidence.EXPLICIT if properties else Confidence.ABSENT,
            property_accessor=_determine_property_accessor(node),
        )

        if overloads is None:
            overloads = self._overload_stack[-1].pop(qualified_name, [])

        if overloads:
            record.has_overloads = True
            record.overload_signatures = [_build_signature(overload) for overload in overloads]

        if in_class and self._class_stack:
            self._class_stack[-1].methods.append(record)
        else:
            self.top_level_function_nodes.append(node)
            self.function_records.append(record)

    def _finalize_overloads(self) -> None:
        """Consume overloads in the current scope that no implementation claimed.

        This covers a ``.pyi`` or overload-only module, where no implementation
        empties the buffer, and overloads declared **after** an implementation.
        Per it.10 (phase 1), attach signatures to the last ``FunctionRecord`` with
        the same name. If none exists, promote the last overload to a
        ``FunctionRecord`` so the function is not lost.
        """
        buffer = self._overload_stack[-1]
        remaining = list(buffer.items())
        buffer.clear()

        for qualified_name, nodes in remaining:
            existing = self._last_record(qualified_name)
            if existing is None:
                self._process_function(nodes[-1], overloads=nodes)
                continue
            existing.has_overloads = True
            existing.overload_signatures.extend(_build_signature(node) for node in nodes)

    def _last_record(self, qualified_name: str) -> FunctionRecord | None:
        """Return the last FunctionRecord produced in the current scope for this name.

        Parameters
        ----------
        qualified_name : str
            The qualified name of the function or method whose last FunctionRecord should be returned.

        Returns
        -------
        FunctionRecord | None
            The last FunctionRecord in the current scope whose qualified_name matches the given qualified_name, or None if no
            matching record exists.

        """
        records = self._class_stack[-1].methods if self._class_stack else self.function_records
        for record in reversed(records):
            if record.qualified_name == qualified_name:
                return record
        return None

    def finalize(self) -> None:
        """Finalize extraction; call after ``visit()`` on the module tree.

        Empty the module-level overload buffer. Class buffers are emptied when
        leaving each ``visit_ClassDef``.
        """
        self._finalize_overloads()

    def _composed_class_visibility(self, identifier_name: str) -> Visibility:
        """Return a class's visibility combined with nested-container visibility.

        M1 flattens nested classes into separate records: a public ``Inner`` class
        inside private ``__Outer`` is private.

        Parameters
        ----------
        identifier_name : str
            The class identifier whose composed visibility should be computed.

        Returns
        -------
        Visibility
            Return the effective visibility of a class after combining its own name-based visibility with the visibility of its
            innermost enclosing class, if any. Nested classes are flattened by M1, so a public inner class inside a private outer
            class is considered private.

        """
        symbol_visibility = _determine_visibility(identifier_name)
        if not self._class_stack:
            return symbol_visibility
        return effective_visibility(self._class_stack[-1].visibility, symbol_visibility)

    def _global_qualified_name(self, identifier_name: str) -> str:
        """Return the global qualified name including the module prefix (DEC-022).

        Format: ``{qualified_module}.{local_qualname}``, for example,
        ``"docmethis.orchestrateur.cli.main"``. Methods use their class's already globally qualified name as prefix.

        Parameters
        ----------
        identifier_name : str
            The local identifier name to be prefixed with the module or class qualified name to form the global qualified name.

        Returns
        -------
        str
            The fully qualified global name for the given identifier, consisting of the module prefix (or the current class's
            globally qualified name when inside a class) followed by the identifier name.

        """
        if not self._class_stack:
            return f"{self.module_name}.{identifier_name}"
        return f"{self._class_stack[-1].qualified_name}.{identifier_name}"


def _determine_visibility(identifier_name: str) -> Visibility:
    """Determine visibility from a name's prefix.

    Parameters
    ----------
    identifier_name : str
        The identifier name whose prefix determines the visibility classification.

    Returns
    -------
    Visibility
        The Visibility value corresponding to the identifier name's prefix: PUBLIC for dunder names and names without a leading
        underscore, PRIVATE for names with two leading underscores, and PROTECTED for names with one leading underscore.

    """
    if identifier_name.startswith("__") and identifier_name.endswith("__"):
        # Dunder names (__init__, __str__, etc.) are public.
        return Visibility.PUBLIC

    if identifier_name.startswith("__"):
        return Visibility.PRIVATE

    if identifier_name.startswith("_"):
        return Visibility.PROTECTED

    return Visibility.PUBLIC


def _determine_module_visibility(module_name: str) -> Visibility:
    """Determine module visibility from its qualified segments.

    Parameters
    ----------
    module_name : str
        The module name to analyze, provided as a dotted string of qualified segments.

    Returns
    -------
    Visibility
        Returns the module's visibility: Visibility.PUBLIC when every non-__init__ segment of the module name lacks a leading
        underscore, and Visibility.PROTECTED otherwise.

    """
    for segment in module_name.split("."):
        if segment != "__init__" and segment.startswith("_"):
            return Visibility.PROTECTED

    return Visibility.PUBLIC


def _extract_docstring_line(node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> int | None:
    """Return the actual starting line of a documentable node's docstring.

    Parameters
    ----------
    node : ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        The AST node representing a module, class, function definition, or async function definition whose docstring starting line
        should be extracted.

    Returns
    -------
    int | None
        The actual starting line of the documentable node's docstring, or None if the node has no docstring or its first statement
        is not a docstring expression.

    """
    if not node.body:
        return None

    first_statement = node.body[0]
    if not isinstance(first_statement, ast.Expr):
        return None

    if ast.get_docstring(node) is not None:
        return first_statement.lineno

    return None


def _determine_method_type(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    in_class: bool,
) -> MethodType:
    """Determine method type from decorators and context.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The function definition node (either ast.FunctionDef or ast.AsyncFunctionDef) to inspect for decorators and determine its
        method type.
    in_class : bool
        Whether the function node is defined within a class body.

    Returns
    -------
    MethodType
        Returns a MethodType value that classifies the given function definition as a plain function, class method, static method,
        property, or instance method based on its decorators and whether it is defined inside a class.

    """
    if not in_class:
        return MethodType.FUNCTION

    if _determine_property_accessor(node) is not None:
        return MethodType.PROPERTY

    for decorator in node.decorator_list:
        decorator_name = _extract_decorator_name(decorator)
        if decorator_name == "classmethod":
            return MethodType.CLASS_METHOD

        if decorator_name == "staticmethod":
            return MethodType.STATIC_METHOD

        if decorator_name == "property":
            return MethodType.PROPERTY

    return MethodType.INSTANCE_METHOD


def _determine_property_accessor(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> PropertyAccessor | None:
    """Return the property accessor role declared by a function decorator."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "property":
            return PropertyAccessor.GETTER
        if isinstance(decorator, ast.Attribute) and decorator.attr == "setter":
            return PropertyAccessor.SETTER
    return None


def _extract_decorator_name(node: ast.expr) -> str:
    """Extract the simple name of an AST decorator.

    Parameters
    ----------
    node : ast.expr
        The AST expression node representing the decorator from which to extract the simple name.

    Returns
    -------
    str
        Returns the simple name extracted from the AST decorator node: the identifier for a Name node, the attribute name for an
        Attribute node, the recursively extracted function name for a Call node, or an empty string for any other node type.

    """
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        return node.attr

    if isinstance(node, ast.Call):
        return _extract_decorator_name(node.func)

    return ""


def _is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether the function has the ``@overload`` decorator (DEC-023).

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        AST function definition to inspect for the presence of an @overload decorator.

    Returns
    -------
    bool
        Return whether the function has the @overload decorator (DEC-023).

    """
    return any(_extract_decorator_name(dec) == "overload" for dec in node.decorator_list)
