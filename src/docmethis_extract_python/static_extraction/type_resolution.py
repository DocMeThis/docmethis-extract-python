# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Resolve missing types from stubs and local inference.

Sources in descending priority (explicit annotations are handled upstream):

1. Local stubs (``.pyi``) - ``confidence: explicit``, ``provenance: stub``
2. Typeshed stubs (through ``typeshed-client``) - ``confidence: explicit``, ``provenance: stub``
3. Simple local inference - ``confidence: inferred_low``, ``provenance: local_inference``

A resolved type is never replaced.

Iteration 8 (F16 phase 2) - exception propagation through the callgraph:
``propagate_exceptions_via_callgraph`` walks the callgraph for two levels and
adds uncaught exceptions from local target_qnames to ``FunctionRecord.exceptions``.
"""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path

from docmethis_extract_python.static_extraction.ast_parser import parse_file
from docmethis_extract_python.static_extraction.models import (
    CallgraphEdge,
    Confidence,
    ExceptionRecord,
    FunctionRecord,
    ProjectRecord,
    Provenance,
    ResolvedTypes,
    TypeInfo,
    iter_functions,
)
from docmethis_extract_python.static_extraction.type_normalization import normalize_type

logger = logging.getLogger(__name__)

_IMPLICIT_PARAMETERS: frozenset[str] = frozenset({"self", "cls"})

# Methods exclusive to list / dict for parameter type inference.
_LIST_METHODS: frozenset[str] = frozenset({"append", "extend", "insert", "remove", "sort", "reverse"})
_DICT_METHODS: frozenset[str] = frozenset({"keys", "values", "items", "get", "update", "setdefault", "popitem"})

# isinstance(x, T) attend exactement 2 arguments
_NARGS_ISINSTANCE: int = 2

# Default maximum depth for callgraph exception propagation (DEC-009).
_DEFAULT_MAX_DEPTH: int = 2

# Mapping tables for _type_from_literal.
_NODE_TYPES: dict[type, str] = {ast.List: "list", ast.Dict: "dict", ast.Set: "set", ast.Tuple: "tuple"}
_CONSTANT_TYPES: dict[type, str] = {bool: "bool", int: "int", float: "float", str: "str", type(None): "None"}

# Internal type alias: qualified name -> node list (a list for overloads).
_IndexStub = dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]]

# Sentinel: typeshed context not initialized (distinct from None = failure).
_UNINITIALIZED_CONTEXT: object = object()


# ---------------------------------------------------------------------------
# Stub cache - instantiate once per AST visitor
# ---------------------------------------------------------------------------


class CacheStubs:
    """Cache parsed and indexed stubs on an AST visitor (not global).

    Avoid reparsing and traversing each stub for every visited function.
    Use one instance per ``extract_module_record`` call.
    """

    def __init__(self) -> None:
        """Initialize the empty cache."""
        # None = missing or unreadable stub; dict = built index.
        self._local: dict[Path, _IndexStub | None] = {}
        # None = missing module in typeshed or error; dict = built index.
        self._typeshed: dict[str, _IndexStub | None] = {}
        # _UNINITIALIZED_CONTEXT = not attempted; None = failure; object = valid context.
        self._typeshed_context: object = _UNINITIALIZED_CONTEXT

    def get_local_index(self, stub_path: Path) -> _IndexStub | None:
        """Return the local stub (.pyi) index, parsing it when needed.

        Parameters
        ----------
        stub_path : Path
            Path to the local stub (.pyi) file whose index should be returned, parsing it if not already cached.

        Returns
        -------
        _IndexStub | None
            The parsed local stub index for the given path, or None if the stub could not be loaded.

        """
        if stub_path not in self._local:
            self._local[stub_path] = _load_local_index(stub_path)
        return self._local[stub_path]

    def get_typeshed_index(self, module_name: str) -> _IndexStub | None:
        """Return the typeshed stub index, parsing it when needed.

        Parameters
        ----------
        module_name : str
            The name of the module for which to retrieve the typeshed stub index.

        Returns
        -------
        _IndexStub | None
            Returns the cached typeshed stub index for the specified module, loading and parsing it if not already cached; returns
            None when no index is available.

        """
        if module_name not in self._typeshed:
            if self._typeshed_context is _UNINITIALIZED_CONTEXT:
                self._typeshed_context = _initialize_typeshed_context()
            self._typeshed[module_name] = _load_typeshed_index(module_name, self._typeshed_context)
        return self._typeshed[module_name]


def _build_stub_index(tree: ast.Module) -> _IndexStub:
    """Build a qualified-name -> node index from a stub tree.

    Parameters
    ----------
    tree : ast.Module
        The AST module representing the stub tree to build the index from.

    Returns
    -------
    _IndexStub
        The built index mapping qualified names to AST nodes from the stub tree.

    """
    index: _IndexStub = {}
    _index_stub_body(tree.body, [], index)
    return index


def _index_stub_body(statements: list[ast.stmt], prefix: list[str], index: _IndexStub) -> None:
    """Recursively index functions and classes in a stub body.

    Parameters
    ----------
    statements : list[ast.stmt]
        A list of AST statements from a stub body to recursively index for function and class definitions.
    prefix : list[str]
        A list of names representing the nesting path of enclosing classes or modules that leads to the current stub body. Each
        element is a segment of the fully qualified name prefix, and it is used to construct the mapping key for indexed functions
        and classes.
    index : _IndexStub
        The mutable index object that maps dotted names to lists of AST nodes; it is updated in place as functions and classes are
        encountered.

    """
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mapping_key = ".".join([*prefix, statement.name])
            index.setdefault(mapping_key, []).append(statement)

        elif isinstance(statement, ast.ClassDef):
            _index_stub_body(statement.body, [*prefix, statement.name], index)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def enrich_resolved_types(  # noqa: PLR0913, PLR0917
    base_types: ResolvedTypes | None,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    python_path: Path,
    module_name: str,
    parent_classes: list[str] | None = None,
    cache: CacheStubs | None = None,
) -> ResolvedTypes | None:
    """Enrich types from stubs and then local inference.

    Return ``base_types`` when no additional source provides types. Never replace an already resolved type.

    Parameters
    ----------
    base_types : ResolvedTypes | None
        The initial resolved types to be enriched, or None if no types have been resolved yet.
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The AST node representing the function or async function definition whose resolved types are being enriched.
    python_path : Path
        Filesystem path to the Python source file or package being processed.
    module_name : str
        Name of the module being processed, used to look up typeshed stubs during type enrichment.
    parent_classes : list[str] | None = None
        An optional list of names of the classes that contain the function, providing context for type resolution. Defaults to
        None.
    cache : CacheStubs | None = None
        An optional cache used during stub and typeshed type enrichment to store or retrieve resolved type information; defaults
        to None.

    Returns
    -------
    ResolvedTypes | None
        Returns the enriched resolved types, or None if no types are available. If no additional source provides types, returns
        base_types unchanged; already resolved types are never replaced.

    """
    enclosing_classes = parent_classes or []
    types = _enrich_from_local_stub(base_types, node, python_path, enclosing_classes, cache)
    types = _enrich_from_typeshed(types, node, module_name, enclosing_classes, cache)
    return _enrich_with_local_inference(types, node)


# ---------------------------------------------------------------------------
# Stubs locaux (.pyi)
# ---------------------------------------------------------------------------


def _enrich_from_local_stub(
    existing_types: ResolvedTypes | None,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    python_path: Path,
    parent_classes: list[str],
    cache: CacheStubs | None,
) -> ResolvedTypes | None:
    """Complete types from a local stub (.pyi) when present.

    Parameters
    ----------
    existing_types : ResolvedTypes | None
        The existing resolved types to be enriched with information from a local stub, or None if no types have been resolved yet.
    node : ast.FunctionDef | ast.AsyncFunctionDef
        AST node representing the function or async function definition whose types are being enriched from the local stub.
    python_path : Path
        Path to the Python file for which a local stub (.pyi) should be looked up.
    parent_classes : list[str]
        A list of class names representing the nesting path from the module level to the function being processed, used to
        construct the stub lookup key.
    cache : CacheStubs | None
        Optional cache of stub files used to look up the local stub index for the given Python path.

    Returns
    -------
    ResolvedTypes | None
        Enriches the given resolved types by looking up a local stub (.pyi) file for the target function and merging any matching
        stub candidates into the existing types. Returns the merged resolved types when a stub match is found, otherwise returns
        the existing types unchanged.

    """
    stub_path = python_path.with_suffix(".pyi")
    index = cache.get_local_index(stub_path) if cache is not None else _load_local_index(stub_path)
    if index is None:
        return existing_types

    mapping_key = ".".join([*parent_classes, node.name])
    candidates = index.get(mapping_key, [])
    if not candidates:
        return existing_types

    return _merge_stub_candidates(existing_types, candidates)


def _load_local_index(stub_path: Path) -> _IndexStub | None:
    """Load and index a local stub without caching.

    Parameters
    ----------
    stub_path : Path
        Path to the local stub file that should be loaded and indexed.

    Returns
    -------
    _IndexStub | None
        An _IndexStub representing the indexed stub, or None if the stub file does not exist or cannot be parsed.

    """
    if not stub_path.exists():
        return None
    tree = parse_file(stub_path)
    if tree is None:
        return None
    return _build_stub_index(tree)


def _merge_from_stub(
    existing_types: ResolvedTypes | None,
    stub_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ResolvedTypes | None:
    """Build or enrich ResolvedTypes from a stub node.

    Parameters
    ----------
    existing_types : ResolvedTypes | None
        The ResolvedTypes object to enrich with type information extracted from the stub node, or None if no resolved types exist
        yet.
    stub_node : ast.FunctionDef | ast.AsyncFunctionDef
        The stub function definition node from which parameter and return type information is extracted and merged into the
        existing resolved types.

    Returns
    -------
    ResolvedTypes | None
        Build or enrich ResolvedTypes by merging type information from a stub function or async function definition into existing
        resolved types. Returns a new ResolvedTypes instance when the stub adds any parameter or return type annotations,
        otherwise returns the existing_types object unchanged.

    """
    parameters: dict[str, TypeInfo] = dict(existing_types.parameters) if existing_types else {}
    return_: TypeInfo | None = existing_types.return_type if existing_types else None

    args = stub_node.args
    all_parameters = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if args.vararg:
        all_parameters = [*all_parameters, args.vararg]
    if args.kwarg:
        all_parameters = [*all_parameters, args.kwarg]

    changed = False
    for arg in all_parameters:
        if arg.arg in _IMPLICIT_PARAMETERS or arg.arg in parameters:
            continue
        if arg.annotation is not None:
            raw = ast.unparse(arg.annotation)
            parameters[arg.arg] = TypeInfo(
                type_str=normalize_type(raw),
                confidence=Confidence.EXPLICIT,
                provenance=Provenance.STUB,
                raw=raw,
            )
            changed = True

    if return_ is None and stub_node.returns is not None:
        raw = ast.unparse(stub_node.returns)
        return_ = TypeInfo(
            type_str=normalize_type(raw),
            confidence=Confidence.EXPLICIT,
            provenance=Provenance.STUB,
            raw=raw,
        )
        changed = True

    if not changed:
        return existing_types

    return ResolvedTypes(parameters=parameters, return_type=return_)


def _merge_stub_candidates(
    existing_types: ResolvedTypes | None,
    candidates: list[ast.FunctionDef | ast.AsyncFunctionDef],
) -> ResolvedTypes | None:
    """Return the merge with the first candidate that actually enriches types.

    With overloads, iterate until the first real contribution. If none contributes (all types are already resolved), return
    ``existing_types`` unchanged.

    Parameters
    ----------
    existing_types : ResolvedTypes | None
        The already resolved types to be enriched by merging with a stub candidate, or None if no types have been resolved yet.
    candidates : list[ast.FunctionDef | ast.AsyncFunctionDef]
        A sequence of stub function or async function definitions to consider for merging, in priority order; the first candidate
        that meaningfully contributes to the resolved types is used.

    Returns
    -------
    ResolvedTypes | None
        The first merged result that differs from existing_types, or existing_types itself when no candidate adds type
        information.

    """
    for candidate in candidates:
        outcome = _merge_from_stub(existing_types, candidate)
        if outcome is not existing_types:
            return outcome
    return existing_types


# ---------------------------------------------------------------------------
# Stubs typeshed (via typeshed-client)
# ---------------------------------------------------------------------------


def _enrich_from_typeshed(
    existing_types: ResolvedTypes | None,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    module_name: str,
    parent_classes: list[str],
    cache: CacheStubs | None,
) -> ResolvedTypes | None:
    """Complete types from typeshed through typeshed-client when available.

    Parameters
    ----------
    existing_types : ResolvedTypes | None
        The currently resolved type information for the function, if any; used as the base result and returned unchanged when
        typeshed data is unavailable or no matching candidates are found.
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The function or async function definition whose types are to be enriched from typeshed.
    module_name : str
        Name of the module whose typeshed index should be consulted for type enrichment.
    parent_classes : list[str]
        Names of the parent classes in which the function is nested, used to construct the mapping key for the typeshed index.
    cache : CacheStubs | None
        Optional cache of typeshed stubs used to avoid reinitializing the typeshed context. When provided, the function retrieves
        the typeshed index from this cache; otherwise, it initializes a fresh typeshed context.

    Returns
    -------
    ResolvedTypes | None
        The enriched resolved types after merging available typeshed stub candidates into the existing types, or the original
        existing_types unchanged when no typeshed index or matching candidates are available.

    """
    if cache is not None:
        index = cache.get_typeshed_index(module_name)
    else:
        ctx = _initialize_typeshed_context()
        index = _load_typeshed_index(module_name, ctx)

    if index is None:
        return existing_types

    mapping_key = ".".join([*parent_classes, node.name])
    candidates = index.get(mapping_key, [])
    if not candidates:
        return existing_types

    return _merge_stub_candidates(existing_types, candidates)


def _initialize_typeshed_context() -> object:
    """Initialize the typeshed search context, or return None when unavailable.

    Returns
    -------
    object
        The initialized typeshed search context, or None if the typeshed_client library is unavailable or initialization fails.

    """
    try:
        import typeshed_client  # type: ignore # noqa: PGH003, PLC0415

        search_path = [Path(path) for path in sys.path if path]
        return typeshed_client.get_search_context(search_path=search_path)
    except ImportError:
        return None  # missing library - expected behavior
    except Exception:
        logger.warning("Unable to initialize typeshed context", exc_info=True)
        return None


def _load_typeshed_index(module_name: str, context: object) -> _IndexStub | None:
    """Load and index a typeshed stub without caching.

    Parameters
    ----------
    module_name : str
        Name of the module whose typeshed stub should be loaded and indexed.
    context : object
        An object used as the search context when locating the typeshed stub file for the given module. It is passed directly to
        typeshed_client.get_stub_file. If None, the function returns None without attempting to load a stub.

    Returns
    -------
    _IndexStub | None
        Returns an _IndexStub representing the indexed typeshed stub, or None if the stub could not be loaded, parsed, or indexed.

    """
    if context is None:
        return None
    try:
        import typeshed_client  # type: ignore # noqa: PGH003, PLC0415

        stub_path = typeshed_client.get_stub_file(module_name, search_context=context)
    except ImportError:
        return None  # missing library - expected behavior
    except Exception:
        logger.debug("Typeshed error for module %s", module_name, exc_info=True)
        return None
    if stub_path is None:
        return None
    tree = parse_file(Path(stub_path))
    if tree is None:
        return None
    return _build_stub_index(tree)


# ---------------------------------------------------------------------------
# Simple local inference
# ---------------------------------------------------------------------------


def _analyze_body(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    targets: set[str],
) -> _BodyInferrer:
    """Run the inferrer over the function body (excluding the docstring).

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        AST node representing the function whose body is to be analyzed; must be a FunctionDef or AsyncFunctionDef.
    targets : set[str]
        Set of target names whose types are inferred while analyzing the function body.

    Returns
    -------
    _BodyInferrer
        An inferrer that has processed the function's body statements, excluding any leading docstring.

    """
    body = node.body
    first_statement = body[0] if body else None
    if (
        isinstance(first_statement, ast.Expr)
        and isinstance(first_statement.value, ast.Constant)
        and isinstance(first_statement.value.value, str)
    ):
        body = body[1:]
    inferrer = _BodyInferrer(targets)
    for statement in body:
        inferrer.visit(statement)
    return inferrer


def _build_inference_result(
    existing_types: ResolvedTypes | None,
    parameter_inferences: dict[str, str],
    inferred_return: TypeInfo | None,
) -> ResolvedTypes | None:
    """Return enriched ResolvedTypes, or existing_types when no types are added.

    Parameters
    ----------
    existing_types : ResolvedTypes | None
        The existing resolved type information to enrich, or None if no prior types are available.
    parameter_inferences : dict[str, str]
        Mapping of parameter names to their inferred type strings, used to enrich the resolved types.
    inferred_return : TypeInfo | None
        The inferred return type for the function being analyzed, or None if no return type was inferred.

    Returns
    -------
    ResolvedTypes | None
        The enriched ResolvedTypes containing the merged parameter inferences and inferred return type. If no parameter inferences
        are provided and no return type is inferred, the original existing_types is returned unchanged (which may be None).

    """
    if not parameter_inferences and inferred_return is None:
        return existing_types
    parameters: dict[str, TypeInfo] = dict(existing_types.parameters) if existing_types else {}
    for identifier_name, type_str in parameter_inferences.items():
        parameters[identifier_name] = TypeInfo(
            type_str=type_str,
            confidence=Confidence.INFERRED_LOW,
            provenance=Provenance.LOCAL_INFERENCE,
            raw=None,
        )
    return_ = (existing_types.return_type if existing_types else None) or inferred_return
    return ResolvedTypes(parameters=parameters, return_type=return_)


def _enrich_with_local_inference(
    existing_types: ResolvedTypes | None,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ResolvedTypes | None:
    """Complete types through local inference on the function body.

    Parameters
    ----------
    existing_types : ResolvedTypes | None
        Existing resolved type information to be enriched, or None if no prior resolution is available.
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The AST node representing the function definition whose body is analyzed for local type inference.

    Returns
    -------
    ResolvedTypes | None
        Returns the enriched ResolvedTypes object with locally inferred parameter and return types merged into the existing type
        information, or None when there are no existing types and no inferences to report.

    """
    all_parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    if node.args.vararg:
        all_parameters = [*all_parameters, node.args.vararg]
    if node.args.kwarg:
        all_parameters = [*all_parameters, node.args.kwarg]

    resolved = set(existing_types.parameters) if existing_types else set()
    targets = {
        parameter.arg
        for parameter in all_parameters
        if parameter.arg not in _IMPLICIT_PARAMETERS and parameter.arg not in resolved
    }
    return_already_resolved = existing_types is not None and existing_types.return_type is not None

    if not targets and return_already_resolved:
        return existing_types

    inferrer = _analyze_body(node, targets)
    parameter_inferences = inferrer.inferences

    type_environment: dict[str, str] = {}
    if existing_types:
        type_environment.update({identifier_name: info.type_str for identifier_name, info in existing_types.parameters.items()})
    type_environment.update(parameter_inferences)
    type_environment.update(inferrer.local_types)

    inferred_return: TypeInfo | None = None
    if not return_already_resolved:
        return_type = _infer_return_type(node, type_environment)
        if return_type:
            inferred_return = TypeInfo(
                type_str=return_type,
                confidence=Confidence.INFERRED_LOW,
                provenance=Provenance.LOCAL_INFERENCE,
                raw=None,
            )

    return _build_inference_result(existing_types, parameter_inferences, inferred_return)


class _BodyInferrer(ast.NodeVisitor):
    """Analyze a function body to infer parameter and variable types.

    Parameters
    ----------
    targets : set[str]
        Set of parameter names for which type inference should be performed.

    Attributes
    ----------
    local_types : dict[str, str]
        Mapping of variable names to their inferred type strings.
    inferences : dict[str, str]
        Mapping of target parameter names to their inferred type strings.

    """

    def __init__(self, targets: set[str]) -> None:
        """Initialize the inferrer with target parameter names.

        Parameters
        ----------
        targets : set[str]
            Set of parameter names for which type inference should be performed.

        """
        self._targets = targets
        self.inferences: dict[str, str] = {}
        self.local_types: dict[str, str] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Do not enter nested functions.

        Parameters
        ----------
        node : ast.FunctionDef
            The AST node representing a function definition to be visited; nested function definitions are not entered.

        """

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Do not enter nested coroutines.

        Parameters
        ----------
        node : ast.AsyncFunctionDef
            The AST node representing the async function definition being visited.

        """

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Do not enter lambdas.

        Parameters
        ----------
        node : ast.Lambda
            The AST node of type ast.Lambda that is currently being visited; the visitor does not traverse into it.

        """

    def visit_Assign(self, node: ast.Assign) -> None:
        """Detect ``x = []`` / ``x = {}`` / ``x = ''`` and ``x[k] = v``.

        Parameters
        ----------
        node : ast.Assign
            The ast.Assign node representing the assignment statement to analyze.

        """
        if len(node.targets) != 1:
            self.generic_visit(node)
            return

        target = node.targets[0]

        if isinstance(target, ast.Name):
            type_init = _type_from_literal(node.value)
            if type_init and target.id in self._targets:
                self.inferences.setdefault(target.id, type_init)
            elif type_init:
                self.local_types[target.id] = type_init

        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            if target.value.id in self._targets:
                self.inferences.setdefault(target.value.id, "dict")

        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Detect ``x += '...'`` -> ``x: str``.

        Parameters
        ----------
        node : ast.AugAssign
            The augmented assignment node to visit.

        """
        if (
            isinstance(node.target, ast.Name)
            and node.target.id in self._targets
            and isinstance(node.op, ast.Add)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            self.inferences.setdefault(node.target.id, "str")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Detect ``for _ in x:`` -> ``x: Iterable``.

        Parameters
        ----------
        node : ast.For
            The AST node representing the for loop to visit.

        """
        if isinstance(node.iter, ast.Name) and node.iter.id in self._targets:
            self.inferences.setdefault(node.iter.id, "Iterable")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect ``isinstance(x, T)`` and characteristic method calls.

        Parameters
        ----------
        node : ast.Call
            The AST node representing the function call to visit.

        """
        # isinstance(x, T) -> x: T
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) == _NARGS_ISINSTANCE
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in self._targets
        ):
            name_type = _type_from_isinstance(node.args[1])
            if name_type:
                self.inferences.setdefault(node.args[0].id, name_type)

        # x.method(...) -> infer x's type from the method.
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            name_var = node.func.value.id
            method = node.func.attr
            if name_var in self._targets:
                if method in _LIST_METHODS:
                    self.inferences.setdefault(name_var, "list")
                elif method in _DICT_METHODS:
                    self.inferences.setdefault(name_var, "dict")

        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Infer return type
# ---------------------------------------------------------------------------


class _ReturnCollector(ast.NodeVisitor):
    """Collect direct Return nodes from a function without entering nested functions.

    Attributes
    ----------
    returns : list[ast.Return]
        The Return nodes collected from the visited function.

    """

    def __init__(self) -> None:
        """Initialize the collector."""
        self.returns: list[ast.Return] = []

    def visit_Return(self, node: ast.Return) -> None:
        """Record the Return node.

        Parameters
        ----------
        node : ast.Return
            The AST Return node to record.

        """
        self.returns.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Do not enter nested functions.

        Parameters
        ----------
        node : ast.FunctionDef
            The AST node representing a function definition to visit.

        """

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Do not enter nested coroutines.

        Parameters
        ----------
        node : ast.AsyncFunctionDef
            The abstract syntax tree node representing the async function definition to visit.

        """

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Do not enter lambdas.

        Parameters
        ----------
        node : ast.Lambda
            The ast.Lambda node that is being visited; the visitor does not descend into it.

        """


def _infer_return_type(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    types_env: dict[str, str],
) -> str | None:
    """Infer the return type from direct return statements.

    Return ``None`` when it cannot be determined with confidence.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        AST node representing the function or async function whose return type is to be inferred.
    types_env : dict[str, str]
        Mapping of variable names to their type annotations, used to resolve the types of expressions appearing in return
        statements.

    Returns
    -------
    str | None
        The inferred return type as a string, or None when it cannot be determined with confidence.

    """
    collector = _ReturnCollector()
    for stmt in node.body:
        collector.visit(stmt)
    returns = collector.returns

    if not returns:
        return None

    return_types: list[str] = []
    for ret in returns:
        if ret.value is None:
            return_types.append("None")
        else:
            t = _infer_expression_type(ret.value, types_env)
            if t is None:
                return None  # At least one unknown return; abort.
            return_types.append(t)

    # Implicit path: when the last statement is neither return nor raise, the
    # function may reach the end of the body and implicitly return None.
    if node.body and not isinstance(node.body[-1], (ast.Return, ast.Raise)):
        return_types.append("None")

    unique = set(return_types)
    return next(iter(unique)) if len(unique) == 1 else None


def _infer_expression_type(syntax_node: ast.expr, types_env: dict[str, str]) -> str | None:
    """Infer an expression type from the type environment.

    Parameters
    ----------
    syntax_node : ast.expr
        AST expression node whose type is to be inferred from the type environment.
    types_env : dict[str, str]
        A dictionary mapping variable names to type strings, representing the type environment used to resolve the types of name
        expressions.

    Returns
    -------
    str | None
        Returns the inferred type as a string, or None when the expression type cannot be determined.

    """
    type_literal = _type_from_literal(syntax_node)
    if type_literal:
        return type_literal

    if isinstance(syntax_node, ast.Name) and syntax_node.id in types_env:
        return types_env[syntax_node.id]

    if isinstance(syntax_node, ast.BinOp) and isinstance(syntax_node.op, (ast.Add, ast.Sub, ast.Mult)):
        left = _infer_expression_type(syntax_node.left, types_env)
        right = _infer_expression_type(syntax_node.right, types_env)
        if left == right and left in {"int", "float", "str"}:
            return left

    return None


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------


def _type_from_literal(syntax_node: ast.expr) -> str | None:
    """Return the Python type of an AST literal, or ``None`` when unknown.

    Parameters
    ----------
    syntax_node : ast.expr
        The AST expression node whose literal type should be resolved.

    Returns
    -------
    str | None
        Return the Python type represented by an AST literal, or None when the type cannot be determined.

    """
    type_node = _NODE_TYPES.get(type(syntax_node))
    if type_node:
        return type_node
    if isinstance(syntax_node, ast.Constant):
        # bool before int because bool is a subclass of int.
        return _CONSTANT_TYPES.get(type(syntax_node.value))
    return None


def _type_from_isinstance(syntax_node: ast.expr) -> str | None:
    """Extract a type name from isinstance's second argument.

    Return ``None`` for unions ``(int, str)`` (ambiguous type).

    Parameters
    ----------
    syntax_node : ast.expr
        The AST expression node to inspect, representing the second argument passed to isinstance().

    Returns
    -------
    str | None
        Returns the extracted type name as a string when the isinstance second argument is a simple name or attribute; returns
        None for unions or other ambiguous expressions.

    """
    if isinstance(syntax_node, ast.Name):
        return syntax_node.id
    if isinstance(syntax_node, ast.Attribute):
        return syntax_node.attr
    return None


# ---------------------------------------------------------------------------
# Exception propagation through the callgraph (it.8 phase 4)
# ---------------------------------------------------------------------------


def _index_functions_project(project_record: ProjectRecord) -> dict[str, FunctionRecord]:
    """Index ``qualified_name -> FunctionRecord`` across the project.

    Parameters
    ----------
    project_record : ProjectRecord
        The project record containing the modules to scan for functions.

    Returns
    -------
    dict[str, FunctionRecord]
        A mapping of qualified_name to FunctionRecord for every function indexed from the project's modules.

    """
    return {fn.qualified_name: fn for fn in iter_functions(project_record.modules)}


_CATCH_ALL_TYPES: frozenset[str] = frozenset({"Exception", "BaseException"})
"""Python hierarchy roots that catch all derived exceptions."""


def _is_caught(exc_type: str, caught: frozenset[str]) -> bool:
    """Return True when an exception type is covered by the caller's except blocks.

    Match exactly (no inheritance hierarchy; known false negatives). The ``"*"`` sentinel means bare except (catches everything).

    Parameters
    ----------
    exc_type : str
        The exception type string to check for coverage by the caught exception types.
    caught : frozenset[str]
        Set of exception type names covered by the caller's except blocks; may contain the '*' sentinel to represent a bare except
        clause.

    Returns
    -------
    bool
        True if the exception type is exactly present in the caught set, or if the caught set contains the bare-except sentinel;
        False otherwise.

    """
    return exc_type in caught or "*" in caught


def _is_caught_at(fn: FunctionRecord, exc_type: str, line: int) -> bool:
    """Return whether an exception is caught at a specific source line.

    Handler ranges are used when available. Older records may not contain a
    range, in which case the handler remains applicable for compatibility.
    A handler that re-raises does not absorb the exception.

    Parameters
    ----------
    fn : FunctionRecord
        Function containing the handlers.
    exc_type : str
        Exception type to match.
    line : int
        Source line where the exception is raised or the call is made.

    Returns
    -------
    bool
        True when a non-reraising handler covering ``line`` catches the type.

    """
    for block in fn.except_blocks or []:
        if block.is_reraised:
            continue
        if block.try_start and block.try_end and not block.try_start <= line <= block.try_end:
            continue
        caught = (
            frozenset({"*"})
            if not block.caught_types or _CATCH_ALL_TYPES.intersection(block.caught_types)
            else frozenset(block.caught_types)
        )
        if _is_caught(exc_type, caught):
            return True
    return False


_ExceptionIdentity = tuple[str, str | None, bool, bool, str | None, Provenance]


def _exception_identity(
    exception: ExceptionRecord,
) -> _ExceptionIdentity:
    """Return the stable identity used for propagated exception aggregation."""
    condition = exception.condition.expression if exception.condition is not None else None
    return (
        exception.exception_type,
        exception.message,
        exception.is_reraise,
        exception.chained_from,
        condition,
        exception.provenance,
    )


def exception_escapes(function_record: FunctionRecord, exception: ExceptionRecord) -> bool:
    """Return whether an exception record escapes its owning function.

    Propagated records already represent an exception that escaped the source
    callable. Local records are checked against the owning function's handlers.
    """
    if exception.provenance is Provenance.CALLGRAPH_PROPAGATION:
        return True
    return not _is_caught_at(function_record, exception.exception_type, exception.raise_line)


def _add_propagated_exception(
    fn: FunctionRecord,
    source: ExceptionRecord,
    origin_line: int,
    existing: set[_ExceptionIdentity],
) -> None:
    """Add one propagated record without discarding source facts.

    Conditions and literal messages are copied from the source record. The
    caller line is used as the propagated record's location because no source
    argument mapping exists at this stage.

    Parameters
    ----------
    fn : FunctionRecord
        The function record to which the propagated exception is added; it is mutated in place.
    source : ExceptionRecord
        Source exception record to propagate.
    origin_line : int
        Line number of the caller's callsite where the propagated exception becomes observable.
    existing : set[_ExceptionIdentity]
        Stable identities already recorded for the current caller.

    """
    propagated = ExceptionRecord(
        exception_type=source.exception_type,
        message=source.message,
        raise_line=origin_line,
        is_reraise=source.is_reraise,
        chained_from=source.chained_from,
        condition=source.condition,
        provenance=Provenance.CALLGRAPH_PROPAGATION,
    )
    identity = _exception_identity(propagated)
    if identity in existing:
        return
    if fn.exceptions is None:
        fn.exceptions = []
    fn.exceptions.append(propagated)
    existing.add(identity)
    if fn.exceptions_confidence == Confidence.ABSENT:
        fn.exceptions_confidence = Confidence.INFERRED_HIGH


def propagate_exceptions_via_callgraph(
    project_record: ProjectRecord,
    callgraph: dict[str, list[CallgraphEdge]],
    max_depth: int = _DEFAULT_MAX_DEPTH,
) -> None:
    """Enrich ``FunctionRecord.exceptions`` through callgraph propagation.

    For each function ``f``, walk local target_qnames for two levels (DEC-009)
    from a pre-propagation snapshot and add uncaught exceptions with:
    - ``confidence = INFERRED_HIGH`` (updated on ``fn.exceptions_confidence``)
    - ``provenance = CALLGRAPH_PROPAGATION`` (implicitly added here)

    Rules:
    - Consider only local target_qnames (``is_external = False``).
    - Match exceptions and except blocks exactly (no Python hierarchy).
    - A bare ``except`` (``caught_types = []``) stops propagation.
    - Match a handler only when its ``try`` range covers the raise or call line.
    - Preserve distinct conditions, messages, and local versus propagated provenance.
    - Do not let a recursive path re-propagate the caller's own records.

    Known limitations:
    - Exceptions raised in lambdas or nested comprehensions in target_qname are
      not captured by static extraction and are therefore not propagated.
    - Exception inheritance beyond the built-in catch-all roots is not
      considered.

    Parameters
    ----------
    project_record : ProjectRecord
        The project record whose function exception data will be enriched through callgraph propagation.
    callgraph : dict[str, list[CallgraphEdge]]
        Mapping from each caller function's qualified name to a list of CallgraphEdge objects representing outgoing call edges
        from that caller to its callees.
    max_depth : int
        Maximum depth of the callgraph to traverse when propagating exceptions. Direct callees are always considered; second-level
        callees are considered only when max_depth is at least the default depth used by the function.

    """
    index_fn = _index_functions_project(project_record)
    source_exceptions = {qname: tuple(fn.exceptions or []) for qname, fn in index_fn.items()}

    for caller_qname in sorted(index_fn):
        fn = index_fn[caller_qname]
        existing = {_exception_identity(exception) for exception in fn.exceptions or []}
        caller_edges = sorted(
            (
                edge
                for edge in callgraph.get(caller_qname, [])
                if not edge.is_external and edge.target_qname in index_fn and edge.target_qname != caller_qname
            ),
            key=lambda edge: (edge.line, edge.target_qname, edge.call_type, edge.external_lib or ""),
        )

        for edge1 in caller_edges:
            target1 = index_fn[edge1.target_qname]
            for exception in source_exceptions[edge1.target_qname]:
                if exception_escapes(target1, exception) and not _is_caught_at(fn, exception.exception_type, edge1.line):
                    _add_propagated_exception(fn, exception, edge1.line, existing)

            if max_depth < _DEFAULT_MAX_DEPTH:
                continue

            callee_edges = sorted(
                (
                    edge
                    for edge in callgraph.get(edge1.target_qname, [])
                    if not edge.is_external
                    and edge.target_qname in index_fn
                    and edge.target_qname not in {caller_qname, edge1.target_qname}
                ),
                key=lambda edge: (edge.line, edge.target_qname, edge.call_type, edge.external_lib or ""),
            )
            for edge2 in callee_edges:
                target2 = index_fn[edge2.target_qname]
                for exception in source_exceptions[edge2.target_qname]:
                    if (
                        exception_escapes(target2, exception)
                        and not _is_caught_at(target1, exception.exception_type, edge2.line)
                        and not _is_caught_at(fn, exception.exception_type, edge1.line)
                    ):
                        _add_propagated_exception(fn, exception, edge1.line, existing)
