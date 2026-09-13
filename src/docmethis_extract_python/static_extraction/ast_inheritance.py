# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Analyze inheritance, C3 MRO, and implicit protocols.

Two call stages:
- During parsing: ``prepare_class_hierarchy()`` and ``extract_*_attributes``
  are called by ``ast_analysis.ExtractionVisitor.visit_ClassDef``.
- Post-processing: ``analyze_inheritance_project()`` is called by
  ``__main__.analyze_project`` after parsing, once the class index is available.
"""

from __future__ import annotations

import ast
import logging
import sys
from collections import deque
from typing import TYPE_CHECKING

from docmethis_extract_python.static_extraction.models import (
    ClassHierarchy,
    ClassRecord,
    Confidence,
    ImplicitProtocol,
    MethodOrigin,
    ModuleRecord,
    ProjectRecord,
    name_simple,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Name of the Python hierarchy root, used as a sentinel instead of magic strings.
_PYTHON_ROOT = "object"

# Reserve three quarters of the limit for the calling environment; cap at 200
# for reasonable hierarchies. Computed once when the module loads.
_MRO_MAX_DEPTH: int = min(200, sys.getrecursionlimit() // 4)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class InvalidProtocolOrderError(AssertionError):
    """Raised when protocol order violates the specificity invariant.

    Parameters
    ----------
    more_specific : str
        The name of the protocol that is more specific and should precede the more general protocol in the ordering.
    more_general : str
        The name of the protocol that is considered more general and must appear after the more specific protocol in the
        ``_PROTOCOLES`` order.

    """

    def __init__(self, more_specific: str, more_general: str) -> None:
        """Initializes the exception with a message describing the invalid protocol order.

        Parameters
        ----------
        more_specific : str
            The name of the protocol that is more specific and should precede the more general protocol in the ordering.
        more_general : str
            The name of the protocol that is considered more general and must appear after the more specific protocol in the
            _PROTOCOLES order.

        Examples
        --------
        obj.__init__(more_specific=some_text, more_general=some_text)

        """
        super().__init__(
            f"Invalid _PROTOCOLES order: '{more_specific}' (more specific) must precede '{more_general}' (more general).",
        )


# ---------------------------------------------------------------------------
# Implicit protocols - map required dunders to protocol names.
# Order matters: most specific to least specific.
# Invariant: a protocol P_a whose dunders are a superset of P_b's must appear
# BEFORE P_b in this list. Example: Iterator before Iterable.
# ---------------------------------------------------------------------------

_PROTOCOLS: list[tuple[frozenset[str], str]] = [
    (frozenset({"__iter__", "__next__"}), "Iterator"),
    (frozenset({"__aiter__", "__anext__"}), "AsyncIterator"),
    (frozenset({"__enter__", "__exit__"}), "ContextManager"),
    (frozenset({"__aenter__", "__aexit__"}), "AsyncContextManager"),
    (frozenset({"__getitem__", "__len__"}), "Mapping"),
    (frozenset({"__iter__"}), "Iterable"),
    (frozenset({"__contains__"}), "Container"),
    (frozenset({"__len__"}), "Sized"),
    (frozenset({"__getitem__"}), "Sequence"),
    (frozenset({"__call__"}), "Callable"),
    (frozenset({"__hash__"}), "Hashable"),
    (frozenset({"__eq__", "__lt__"}), "Comparable"),
    (frozenset({"__await__"}), "Awaitable"),
]


def _detect_violations_order(
    protocols: list[tuple[frozenset[str], str]],
) -> list[tuple[str, str]]:
    """Return pairs (more_specific, more_general) whose order is invalid.

    A protocol i whose dunders are a strict subset of a later protocol j
    violates the specificity invariant: protocol j should appear first.

    Return an empty list when the order is correct.

    This replaces the misleading ``_ordre_protocoles_valide`` name because the function returns violations, not a validity
    boolean.

    Parameters
    ----------
    protocols : list[tuple[frozenset[str], str]]
        Sequence of protocols to validate, where each item is a tuple containing a frozenset of dunder method names and the
        protocol name; the order of this list is checked for specificity violations.

    Returns
    -------
    list[tuple[str, str]]
        Returns a list of pairs (more_specific, more_general) that are out of order, where the first element is the protocol that
        should appear earlier and the second is the protocol that appears earlier in the input. The list is empty when the
        protocol order is valid.

    """
    violations: list[tuple[str, str]] = []
    for i, (dunders_i, name_i) in enumerate(protocols):
        for j, (dunders_j, name_j) in enumerate(protocols):
            # Violation: i appears before j (i < j), but i is less specific
            # than j (dunders_i < dunders_j). j should precede i.
            if i < j and dunders_i < dunders_j:
                violations.append((name_j, name_i))
    return violations


def _verify_order_protocols() -> None:
    """Verify protocol order by raising an error for any detected violations.

    Raises
    ------
    InvalidProtocolOrderError
        Explicitly raised.

    """
    violations = _detect_violations_order(_PROTOCOLS)
    for name_j, name_i in violations:
        raise InvalidProtocolOrderError(name_j, name_i)


_verify_order_protocols()


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _walk_no_inner_scope(node: ast.AST) -> Iterator[ast.AST]:
    """Iteratively walk without entering nested functions, lambdas, or classes.

    Use an explicit stack (``collections.deque``) instead of recursion to avoid Python's recursion limit on deeply nested ASTs
    (generated code, templates).

    Parameters
    ----------
    node : ast.AST
        The root AST node from which the iterative traversal starts.

    Returns
    -------
    Iterator[ast.AST]
        An iterator over the AST nodes visited, excluding nodes inside nested functions, lambdas, and classes.

    """
    stack: deque[ast.AST] = deque([node])
    while stack:
        current = stack.pop()
        yield current
        for child in ast.iter_child_nodes(current):
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
            ):
                continue
            stack.append(child)


# ---------------------------------------------------------------------------
# AST extraction - called during parsing (visit_ClassDef)
# ---------------------------------------------------------------------------


def extract_bases_raw(node: ast.ClassDef) -> list[str]:
    """Return unresolved base names (positional bases only, not metaclass).

    Parameters
    ----------
    node : ast.ClassDef
        AST class definition node whose base classes are to be extracted.

    Returns
    -------
    list[str]
        A list of unresolved base names extracted from the positional bases of the class definition; metaclass arguments are not
        included.

    """
    return [ast.unparse(base) for base in node.bases]


def is_abstract_class(node: ast.ClassDef) -> bool:
    """Return True when the class inherits ABC/ABCMeta or contains @abstractmethod.

    Known limitation: a composite decorator such as ``@some_wrapper(abstractmethod)``
    is not detected because unparsing sees the outer decorator. Only direct ``@abstractmethod`` and ``@abc.abstractmethod``
    decorators are recognized.

    Parameters
    ----------
    node : ast.ClassDef
        The AST class definition node to inspect for abstractness.

    Returns
    -------
    bool
        True if the class is abstract (inherits ABC/ABCMeta or contains a direct @abstractmethod decorator), otherwise False.

    """
    abstract_names = {"ABC", "abc.ABC", "ABCMeta", "abc.ABCMeta"}
    for base in node.bases:
        if ast.unparse(base) in abstract_names:
            return True
    for keyword in node.keywords:
        if keyword.arg == "metaclass" and ast.unparse(keyword.value) in abstract_names:
            return True
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in item.decorator_list:
                if ast.unparse(dec) in ("abstractmethod", "abc.abstractmethod"):
                    return True
    return False


# "Few methods" threshold for the mixin heuristic, defaulting to
# ``ConfigurationDocmethis.mixin_max_methods`` (overridable in [tool.docmethis]).
# Real mixins usually expose one to three methods; 5 is a conservative bound
# that avoids classifying large utility *Mixin classes as pure mixins.
DEFAULT_MIXIN_MAX_METHODS: int = 5


# ---------------------------------------------------------------------------
# Helpers - analyze the __init__ body (phase 7a it.10)
# ---------------------------------------------------------------------------


def _is_super_init_call(statement: ast.stmt) -> bool:
    """Return True when statement is `super().__init__(...)` or `super(A, self).__init__()`.

    Parameters
    ----------
    statement : ast.stmt
        The AST statement node to inspect for a super().__init__ call.

    Returns
    -------
    bool
        True if the statement is a call to super().__init__(...) or super(A, self).__init__(); False otherwise.

    """
    if not isinstance(statement, ast.Expr):
        return False
    call = statement.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr != "__init__":
        return False
    inner = func.value
    return isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "super"


def _is_inert_statement(statement: ast.stmt) -> bool:
    """Return True when a statement does nothing: `pass`, a docstring, or `...`.

    The latter two share an AST form, `Expr(Constant)`, hence the grouping.

    Parameters
    ----------
    statement : ast.stmt
        The AST statement node to check for inertness.

    Returns
    -------
    bool
        Return True if the given AST statement is inert, meaning it has no runtime effect: a `pass` statement, a docstring
        expression, or an ellipsis (`...`) expression. Return False otherwise.

    """
    if isinstance(statement, ast.Pass):
        return True
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Constant):
        return False
    item_value = statement.value.value
    return isinstance(item_value, str) or item_value is ...


def _is_simple_init(method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True when __init__ is simple enough for a class to remain a mixin candidate.

    Allowed body: docstring, `self.attr = ...` assignments, a
    `super().__init__(...)` call, `pass`, or `...`. Any other statement (loop, condition, try, external call) returns False.

    Parameters
    ----------
    method : ast.FunctionDef | ast.AsyncFunctionDef
        The AST node of the __init__ method to inspect, either a function definition or an async function definition.

    Returns
    -------
    bool
        Returns True if the given __init__ method has a body consisting only of allowed statements (docstring, self attribute
        assignments, a super().__init__() call, pass, or ...); otherwise returns False.

    """
    all_args = method.args.posonlyargs + method.args.args
    self_name = all_args[0].arg if all_args else "self"

    for stmt in method.body:
        if _is_inert_statement(stmt):
            continue
        if _is_super_init_call(stmt):
            continue
        if isinstance(stmt, ast.Assign) and all(_is_self_attribute(t, self_name) for t in stmt.targets):
            continue
        if isinstance(stmt, ast.AnnAssign) and _is_self_attribute(stmt.target, self_name):
            continue
        return False
    return True


def is_mixin(node: ast.ClassDef, mixin_max_methods: int = DEFAULT_MIXIN_MAX_METHODS) -> bool:
    """Mixin heuristic: name ending in 'Mixin', simple or absent __init__, and few methods.

    All three criteria are required (it.8 spec, refined in it.10 phase 7):
    1. Name ending in ``Mixin``.
    2. No ``__init__``, or an ``__init__`` limited to ``self.attr = ...``
       assignments and/or a ``super().__init__()`` call.
    3. Method count <= ``mixin_max_methods``.

    Parameters
    ----------
    node : ast.ClassDef
        The AST class definition node to inspect for mixin characteristics.
    mixin_max_methods : int
        Maximum number of methods a class may define to still be considered a mixin.

    Returns
    -------
    bool
        True if the class node satisfies the mixin heuristic (name ends in 'Mixin', simple or absent __init__, and method count
        within the limit); False otherwise.

    """
    if not node.name.endswith("Mixin"):
        return False

    methods = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]

    init = next((m for m in methods if m.name == "__init__"), None)
    if init is not None and not _is_simple_init(init):
        logger.debug(
            "Class '%s' excluded from mixin detection: complex __init__.",
            node.name,
        )
        return False

    if len(methods) > mixin_max_methods:
        logger.debug(
            "Class '%s' excluded from mixin detection: %d methods > threshold %d.",
            node.name,
            len(methods),
            mixin_max_methods,
        )
        return False

    return True


def extract_class_attributes(node: ast.ClassDef) -> list[str]:
    """Extract names of class attributes defined directly in the body.

    Parameters
    ----------
    node : ast.ClassDef
        The AST class definition node from which to extract class attribute names.

    Returns
    -------
    list[str]
        A list of the names of class attributes assigned directly in the class body, excluding methods and nested classes, in
        source order.

    """
    attrs: list[str] = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            attrs.append(item.target.id)
        elif isinstance(item, ast.Assign):
            attrs.extend(target.id for target in item.targets if isinstance(target, ast.Name))
    return attrs


def _is_self_attribute(target: ast.expr, self_name: str) -> bool:
    """Return True when ``target`` is a ``self.x`` access (structure only).

    Parameters
    ----------
    target : ast.expr
        The AST expression node to inspect; the function checks whether it represents an attribute access on the instance name.
    self_name : str
        The name of the receiver variable (for example, 'self' or 'cls') that the target's value must match to be considered a
        self attribute access.

    Returns
    -------
    bool
        True if target is a self.x access (structure only), False otherwise.

    """
    return isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == self_name


def _targets(stmt: ast.stmt) -> list[ast.expr]:
    """Return the assignment targets for an AST statement, or an empty list if the statement is not an assignment.

    Parameters
    ----------
    stmt : ast.stmt
        The AST statement node to inspect for assignment targets.

    Returns
    -------
    list[ast.expr]
        Returns a list of AST expression nodes representing the assignment targets of the given statement. For an ast.Assign node,
        this is the statement's targets list; for an ast.AnnAssign node, it is a list containing the single target; for any other
        statement type, an empty list is returned.

    """
    if isinstance(stmt, ast.Assign):
        return stmt.targets
    if isinstance(stmt, ast.AnnAssign):
        return [stmt.target]
    return []


def extract_instance_attributes(node: ast.ClassDef) -> list[str]:
    """Extract instance attribute names (self.x = ...) from methods.

    Walk each method body without entering nested functions. Treat the first parameter of each method as ``self``.

    Note: an attribute present both in a class annotation and assigned in a method
    appears in both lists (``extract_class_attributes`` and this one). The caller is responsible for deduplication;
    ``ClassRecord`` is the recommended entry point for handling it once.

    Parameters
    ----------
    node : ast.ClassDef
        The class definition node whose methods are scanned for instance attribute assignments.

    Returns
    -------
    list[str]
        A list of instance attribute names found on self in the class's methods, in order of first discovery and with duplicates
        removed.

    """
    attrs: list[str] = []
    seen: set[str] = set()

    for item in node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        all_args = item.args.posonlyargs + item.args.args
        if not all_args:
            continue
        self_name = all_args[0].arg

        for stmt_node in _walk_no_inner_scope(item):
            for target in _targets(stmt_node):
                if _is_self_attribute(target, self_name) and target.attr not in seen:
                    attrs.append(target.attr)
                    seen.add(target.attr)

    return attrs


def prepare_class_hierarchy(node: ast.ClassDef, mixin_max_methods: int = DEFAULT_MIXIN_MAX_METHODS) -> ClassHierarchy:
    """Build a partial ClassHierarchy from an AST node.

    ``direct_parents`` contains raw (unresolved) names. ``mro_list`` is empty and filled by ``analyze_inheritance_project`` during
    post-processing.

    Parameters
    ----------
    node : ast.ClassDef
        The AST class definition node to analyze.
    mixin_max_methods : int
        Maximum number of methods a class may have to be considered a mixin; used when determining whether the class is a mixin.

    Returns
    -------
    ClassHierarchy
        A partial ClassHierarchy built from the AST node, with direct_parents set to raw base names, mro_list empty, and
        is_abstract and is_mixin flags computed from the node.

    """
    return ClassHierarchy(
        direct_parents=extract_bases_raw(node),
        mro_list=[],
        is_abstract=is_abstract_class(node),
        is_mixin=is_mixin(node, mixin_max_methods=mixin_max_methods),
    )


# ---------------------------------------------------------------------------
# Helpers - extract a method name
# ---------------------------------------------------------------------------


def _name_method(m: object) -> str:
    """Return a method's simple name from its record.

    Prefer a non-empty ``name`` attribute; otherwise use the last segment of ``qualified_name``. Raise ``AttributeError`` when
    neither attribute exists, which signals an unexpected type instead of silently returning an empty string.

    Parameters
    ----------
    m : object
        The method record from which to extract the simple name. Expected to have a 'name' attribute and a 'qualified_name'
        attribute.

    Returns
    -------
    str
        The simple name of the method as a string, taken from the non-empty 'name' attribute if present, otherwise from the last
        segment of 'qualified_name'.

    """
    name: str | None = getattr(m, "name", None)
    if name:
        return name
    return name_simple(m.qualified_name)


# ---------------------------------------------------------------------------
# Protocol detection - usable at any stage
# ---------------------------------------------------------------------------


def detect_protocols(
    class_: ClassRecord,
    index_classes: dict[str, ClassRecord],
) -> list[ImplicitProtocol]:
    """Detect structurally implicit protocols.

    Collect dunders on the class **and** its ancestors through the MRO (excluding ``object``). A subclass inheriting ``__iter__``
    and ``__next__`` from a parent is therefore recognized as ``Iterator``. Numeric protocols are excluded (DEC-008).

    Parameters
    ----------
    class_ : ClassRecord
        The class record to inspect for structurally implicit protocols.
    index_classes : dict[str, ClassRecord]
        Mapping of qualified class names to ClassRecord objects, used to look up ancestor classes when collecting inherited dunder
        methods through the MRO.

    Returns
    -------
    list[ImplicitProtocol]
        A list of ImplicitProtocol instances, one for each structurally detected protocol whose required dunder methods are
        present on the class or its ancestors.

    """
    methods: set[str] = {_name_method(m) for m in class_.methods}

    mro = class_.hierarchy.mro_list if (class_.hierarchy and class_.hierarchy.mro_list) else []
    for parent_qname in mro[1:]:  # [0] = the class itself
        if parent_qname == _PYTHON_ROOT:
            continue
        parent = index_classes.get(parent_qname)
        if parent is not None:
            methods.update(_name_method(m) for m in parent.methods)

    return [
        ImplicitProtocol(name=protocol_name, dunder_methods=sorted(dunders))
        for dunders, protocol_name in _PROTOCOLS
        if dunders.issubset(methods)
    ]


# ---------------------------------------------------------------------------
# Method classification - new / overridden (it.8)
# ---------------------------------------------------------------------------


def classify_methods(
    class_: ClassRecord,
    index_classes: dict[str, ClassRecord],
) -> None:
    """Classify each method of ``class_`` by its origin in the hierarchy.

    For each method in ``class_.methods``:

    - ``NEW``: no MRO parent defines a method with the same name.
    - ``OVERRIDDEN``: at least one MRO parent defines a method with the same name.

    For each method defined by a parent and absent from the current class, add a copied ``FunctionRecord`` to ``class_.methods``
    with:

    - ``method_origin = INHERITED``
    - ``parent_class`` updated to the current class qualified_name (the method is
      made visible on this class even when defined elsewhere).

    Add only the nearest parent's method in the MRO (the first parent defining it in MRO order). Insertion is stable: own methods
    keep their order and inherited methods are appended.

    Functions outside classes (``parent_class is None``) are not changed. ``class_.hierarchy.mro_list`` must already be calculated
    (call after ``_compute_mro_project``).

    Parameters
    ----------
    class_ : ClassRecord
        The ClassRecord instance to process. Its methods are classified as NEW or OVERRIDDEN, and inherited methods are populated
        on it. Its hierarchy.mro_list must already be computed.
    index_classes : dict[str, ClassRecord]
        Maps qualified class names to ClassRecord instances, used to look up parent classes when walking the MRO during method
        classification.

    """
    if class_.hierarchy is None or not class_.hierarchy.mro_list:
        return

    own_names: set[str] = {_name_method(m) for m in class_.methods}

    # --- Step 1: mark NEW / OVERRIDDEN on own methods ---
    names_parents: set[str] = set()
    for parent_qname in class_.hierarchy.mro_list[1:]:
        parent = index_classes.get(parent_qname)
        if parent is None:
            continue
        for m in parent.methods:
            names_parents.add(_name_method(m))

    for method in class_.methods:
        method_name = _name_method(method)
        method.method_origin = MethodOrigin.OVERRIDDEN if method_name in names_parents else MethodOrigin.NEW

    # --- Step 2: populate inherited_methods (nearest parent defining the method) ---
    # Store in ClassRecord.inherited_methods separately from methods to preserve
    # the invariant "methods = methods defined on this class" and avoid duplicate
    # qualified_names across the project.
    import copy  # noqa: PLC0415 - local import avoids a potential circular import

    class_.inherited_methods = []
    already_seen: set[str] = set(own_names)
    for parent_qname in class_.hierarchy.mro_list[1:]:
        parent = index_classes.get(parent_qname)
        if parent is None:
            continue
        for m in parent.methods:
            method_name = _name_method(m)
            if method_name in already_seen:
                continue
            inherited_method = copy.copy(m)
            inherited_method.method_origin = MethodOrigin.INHERITED
            class_.inherited_methods.append(inherited_method)
            already_seen.add(method_name)


# ---------------------------------------------------------------------------
# Post-processing - resolve MRO and protocols across the project
# ---------------------------------------------------------------------------


def _build_index_classes(project_record: ProjectRecord) -> dict[str, ClassRecord]:
    """Build a ``qualified_name -> ClassRecord`` index across the project.

    Parameters
    ----------
    project_record : ProjectRecord
        The ProjectRecord instance that holds the project's modules and their classes, used as the source for building the class
        index.

    Returns
    -------
    dict[str, ClassRecord]
        A dictionary mapping each class's qualified name to its corresponding ClassRecord.

    """
    return {class_.qualified_name: class_ for module in project_record.modules for class_ in module.classes}


def _build_mapping_names(module: ModuleRecord) -> dict[str, str]:
    """Build a ``simple_name_or_alias -> qualified_name`` mapping for a module.

    Combine classes local to the module and imported names with aliases.

    For ``import X.Y.Z``, only ``X.Y.Z.Cls`` is accessible through ``X.Y.Z``, not ``X.Cls``. The mapping reflects this behavior.

    Key collisions (two classes with the same short name or conflicting import aliases) are detected and logged at debug level.

    Parameters
    ----------
    module : ModuleRecord
        The ModuleRecord representing the module whose local classes and imports are combined into the
        simple-name-to-qualified-name mapping.

    Returns
    -------
    dict[str, str]
        A dictionary mapping each accessible simple name or import alias to the fully qualified name it resolves to for the
        module.

    """
    mapping: dict[str, str] = {}

    def _set_mapping(mapping_key: str, item_value: str, analysis_context: str) -> None:
        if mapping_key in mapping:
            if mapping[mapping_key] == item_value:
                return
            logger.debug(
                "Name collision in module '%s' mapping: key '%s' mapped to '%s', overwritten by '%s' (%s).",
                module.module_name,
                mapping_key,
                mapping[mapping_key],
                item_value,
                analysis_context,
            )
        mapping[mapping_key] = item_value

    for class_ in module.classes:
        simple = name_simple(class_.qualified_name)
        # Add the simple key (for example, "Foo") only for top-level classes;
        # otherwise a nested class and a top-level class with the same name collide.
        # Nested classes remain reachable in two ways:
        # - qualified reference ("Outer.Inner") through the two-step lookup in
        #   _resolve_name_base;
        # - bare reference from an enclosing scope ("Inner" in Outer) through
        #   _resolve_name_base_lexical, which checks scopes before this mapping.
        if module.module_name + "." + simple == class_.qualified_name:
            _set_mapping(simple, class_.qualified_name, "local class")
        # Always register the qualified key (direct fqn lookup).
        _set_mapping(class_.qualified_name, class_.qualified_name, "local class (fqn)")

    for imp in module.imports:
        # `from X import *` introduces no usable name here. Registering it would
        # create the literal '*' key, causing two star imports in one module to
        # collide on a phantom key. Star-imported bases remain unresolved as before,
        # without polluting the mapping or reporting a nonexistent collision.
        if imp.name == "*":
            continue

        # from X import Y [as Z] -> key=alias|Y, value=X.Y
        # import X.Y.Z [as Z]   -> key=alias|X.Y.Z, value=X.Y.Z
        mapping_key = imp.alias or imp.name or imp.module
        _set_mapping(mapping_key, f"{imp.module}.{imp.name}" if imp.name else imp.module, "import")

    return mapping


def _resolve_name_base(identifier_name: str, mapping: dict[str, str]) -> str:
    """Resolve a base name to its qualified_name.

    1. Direct match.
    2. Qualified first segment (``pkg.Cls`` -> look up ``pkg`` and rebuild).
    3. Fallback: return the name unchanged (external or built-in class).

    Built-in types (``list``, ``dict``, ``Exception``, and so on) are not in the index and use the fallback. They are returned
    unchanged and treated as leaves ``[name, _PYTHON_ROOT]`` by ``_mro_recursive``.

    This mapping is flat and only knows module scope. For a nested class, use :func:`_resolve_name_base_lexical`, which checks
    enclosing scopes first.

    Parameters
    ----------
    identifier_name : str
        The name of a base class to resolve. It may be a simple identifier or a dotted qualified name. This string is used as the
        lookup key in the mapping and is returned unchanged when no resolution is found.
    mapping : dict[str, str]
        A flat mapping from names to fully qualified names, used to resolve base class names; it only contains module-scope
        entries.

    Returns
    -------
    str
        The resolved qualified name for the given base identifier. If the identifier or its first segment is found in the mapping,
        the mapped value is returned, with the remainder of the identifier appended when a qualified name is rebuilt. If no match
        is found, the original identifier is returned unchanged.

    """
    if identifier_name in mapping:
        return mapping[identifier_name]
    first_segment = identifier_name.split(".", maxsplit=1)[0]
    if first_segment in mapping:
        return mapping[first_segment] + identifier_name[len(first_segment) :]
    return identifier_name


def _enclosing_scopes(qualified_name: str, module_name: str) -> list[str]:
    """Return lexical scopes around a class, nearest to farthest.

    ``mod.Outer.Inner.Cls`` -> ``["mod.Outer.Inner", "mod.Outer", "mod"]``.

    The lower bound is the module: segments above it are packages, not namespaces
    that could contain the base class. Including them would resolve a same-named class from a neighboring `__init__.py`.

    Parameters
    ----------
    qualified_name : str
        The qualified name of the class for which to compute enclosing scopes, as a dotted string.
    module_name : str
        Name of the module that contains the qualified name; used as the lower bound when building enclosing scopes.

    Returns
    -------
    list[str]
        The lexical scopes around the class, from nearest to farthest, ending with the module name.

    """
    prefix = module_name + "."
    if not qualified_name.startswith(prefix):
        return [module_name]

    segments = qualified_name[len(prefix) :].split(".")
    scopes = [module_name + "." + ".".join(segments[:i]) for i in range(len(segments) - 1, 0, -1)]
    scopes.append(module_name)
    return scopes


def _resolve_name_base_lexical(
    identifier_name: str,
    class_: ClassRecord,
    mapping: dict[str, str],
    index: dict[str, ClassRecord],
) -> str:
    """Resolve a base name according to Python lexical scope.

    ``class Child(Base)`` written in ``Outer`` refers to ``Outer.Base`` when it
    exists: the class body is the active namespace while the base is evaluated.
    A nested class therefore shadows its module-level namesake.

    Try scopes from nearest to farthest, then fall back to the module's flat mapping (top-level classes, imports, built-ins).

    Parameters
    ----------
    identifier_name : str
        The name of the base class as written in the class definition; this identifier is resolved by searching enclosing lexical
        scopes and then falling back to the module-level mapping.
    class_ : ClassRecord
        The class record for the class whose base name is being resolved; its qualified name and parent module define the lexical
        scopes to search.
    mapping : dict[str, str]
        A dictionary that maps each available name to its resolved fully qualified name in the module's flat namespace; used as
        the final fallback after lexical scopes are exhausted.
    index : dict[str, ClassRecord]
        Mapping of qualified class names to ClassRecord instances; used to look up candidates when resolving a base name in
        lexical scope.

    Returns
    -------
    str
        The resolved fully qualified base name for the identifier, either from the nearest enclosing scope that contains it or
        from the module-level fallback resolution.

    """
    for scope in _enclosing_scopes(class_.qualified_name, class_.parent_module):
        candidate = f"{scope}.{identifier_name}"
        if candidate in index:
            return candidate

    return _resolve_name_base(identifier_name, mapping)


def _c3_merge(sequences: list[list[str]]) -> list[str] | None:
    """Standard C3 merge algorithm.

    Return ``None`` when the MRO is inconsistent (unresolvable cycle).

    Copy input lists defensively at the start; this function does not mutate caller-owned sequences.

    Parameters
    ----------
    sequences : list[list[str]]
        A list of lists of strings representing the sequences to merge using the C3 linearization algorithm. Each inner list is a
        class lineage or MRO sequence; empty sequences are ignored.

    Returns
    -------
    list[str] | None
        The merged C3 linearization as a list of class names, or None if the MRO is inconsistent and cannot be resolved.

    """
    sequences = [list(s) for s in sequences if s]
    result: list[str] = []

    while True:
        sequences = [s for s in sequences if s]
        if not sequences:
            return result
        for seq in sequences:
            head = seq[0]
            if not any(head in s[1:] for s in sequences):
                result.append(head)
                for s in sequences:
                    if s and s[0] == head:
                        del s[0]
                break
        else:
            logger.warning(
                "Inconsistent C3 MRO - unable to linearize from node: %s",
                sequences[0][0] if sequences[0] else "?",
            )
            return None


def _mro_recursive(
    identifier_name: str,
    index: dict[str, ClassRecord],
    cache: dict[str, list[str] | None],
    source_path: frozenset[str] = frozenset(),
    depth: int = 0,
) -> list[str] | None:
    # Consult the cache only outside the current path so a real cycle is not hidden.
    """Recursively compute the C3 method resolution order for a class, handling inheritance cycles and depth limits.

    Parameters
    ----------
    identifier_name : str
        The name of the class or type whose method resolution order is being computed.
    index : dict[str, ClassRecord]
        A mapping of class names to ClassRecord objects, used as the lookup table for class hierarchy information while computing
        the method resolution order.
    cache : dict[str, list[str] | None]
        A memoization dictionary mapping an identifier name to its already computed MRO list, or None when MRO calculation failed
        or was truncated; it is consulted and updated only when the identifier is not in the current source path so that genuine
        inheritance cycles are not masked.
    source_path : frozenset[str]
        Set of class names currently on the inheritance path being explored, used to detect cycles and control caching.
    depth : int = 0
        The current recursion depth used to limit how deeply the MRO calculation may proceed before it is truncated.

    Returns
    -------
    list[str] | None
        Returns the computed method resolution order (MRO) as a list of class names for the given identifier, or None if the MRO
        cannot be determined due to an inheritance cycle or because the maximum recursion depth was exceeded. For classes not
        present in the index, or classes without direct parents, a default MRO containing the class name followed by the Python
        root is returned.

    """
    if identifier_name not in source_path and identifier_name in cache:
        return cache[identifier_name]

    result: list[str] | None

    if depth > _MRO_MAX_DEPTH:
        logger.warning(
            "MRO depth limit reached (%d) for class: %s - calculation truncated.",
            _MRO_MAX_DEPTH,
            identifier_name,
        )
        result = None
    elif identifier_name == _PYTHON_ROOT:
        result = [_PYTHON_ROOT]
    elif identifier_name in source_path:
        logger.warning("Inheritance cycle detected for class: %s", identifier_name)
        result = None
    elif identifier_name not in index:
        result = [identifier_name, _PYTHON_ROOT]
    else:
        class_ = index[identifier_name]
        if class_.hierarchy is None or not class_.hierarchy.direct_parents:
            result = [identifier_name, _PYTHON_ROOT]
        else:
            new_path = source_path | {identifier_name}
            parents = class_.hierarchy.direct_parents
            parent_mros: list[list[str]] = []
            for p in parents:
                mro_p = _mro_recursive(p, index, cache, new_path, depth + 1)
                if mro_p is None:
                    result = None
                    break
                parent_mros.append(mro_p)
            else:
                result = _c3_merge([[identifier_name], *parent_mros, list(parents)])

    # Cache only outside the current path.
    if identifier_name not in source_path:
        cache[identifier_name] = result

    return result


def resolve_mro(
    class_: ClassRecord,
    index: dict[str, ClassRecord],
    cache: dict[str, list[str] | None] | None = None,
) -> list[str] | None:
    """Calculate a class's C3 MRO.

    Return a list beginning with ``class_.qualified_name`` and ending with ``_PYTHON_ROOT``, or ``None`` for an inconsistent
    hierarchy.

    ``cache`` shares memoization across successive calls (typically for all project classes), avoiding recalculation for common
    nodes. When ``None``, create a local cache, matching the previous signature's behavior.

    Classes absent from the index (external or built-in) are treated as leaves ``[name, _PYTHON_ROOT]``.

    Cycle detection uses the current branch path (``frozenset``), not global state shared between branches. This handles diamond
    inheritance without false positives.

    A depth limit (``_MRO_MAX_DEPTH``) protects against pathologically deep hierarchies.

    Parameters
    ----------
    class_ : ClassRecord
        The ClassRecord instance representing the class for which the C3 MRO should be computed.
    index : dict[str, ClassRecord]
        Mapping of qualified class names to their ClassRecord definitions, used to look up base classes and determine which
        classes are absent from the project.
    cache : dict[str, list[str] | None] | None = None
        An optional mapping used to share memoized MRO results across successive calls. Keys are class qualified names and values
        are either the resolved MRO list or None for an inconsistent hierarchy. When None, the function creates a local cache for
        the duration of the call.

    Returns
    -------
    list[str] | None
        A list of class names in C3 linearization order, starting with the class's qualified name and ending with _PYTHON_ROOT, or
        None if the hierarchy is inconsistent.

    """
    if cache is None:
        cache = {}
    return _mro_recursive(class_.qualified_name, index, cache)


def _resolve_parents_project(
    project_record: ProjectRecord,
    index: dict[str, ClassRecord],
) -> None:
    """Step 2 - resolve raw parent names to qualified names.

    A parent already present in the index (a valid qualified name) or equal to ``_PYTHON_ROOT`` is not resolved again, making
    repeated calls on the same ``ProjectRecord`` idempotent.

    Build ``index`` before calling this function.

    The former ``deja_resolues`` parameter was removed. The condition ``p in index or p == _PYTHON_ROOT`` fully protects against
    double resolution, making the external set unnecessary and the signature simpler.

    Parameters
    ----------
    project_record : ProjectRecord
        The ProjectRecord whose raw parent names are resolved in place.
    index : dict[str, ClassRecord]
        Mapping of qualified class names to ClassRecord objects used to resolve parent names; parents already present in this
        index are kept unchanged.

    """
    for module in project_record.modules:
        mapping = _build_mapping_names(module)
        for class_ in module.classes:
            if class_.hierarchy is None:
                continue
            class_.hierarchy.direct_parents = [
                p if (p in index or p == _PYTHON_ROOT) else _resolve_name_base_lexical(p, class_, mapping, index)
                for p in class_.hierarchy.direct_parents
            ]


def _compute_mro_project(project_record: ProjectRecord, index: dict[str, ClassRecord]) -> None:
    """Step 3 - calculate the C3 MRO for each class.

    Share the MRO cache across all project classes to avoid recalculating common nodes (the previous per-class cache had quadratic
    complexity).

    Parameters
    ----------
    project_record : ProjectRecord
        The ProjectRecord whose modules and classes are processed to compute and store each class's C3 MRO.
    index : dict[str, ClassRecord]
        Mapping of class qualified names to their ClassRecord objects, used to resolve base classes when computing the C3
        linearization (MRO) for each project class.

    """
    cache: dict[str, list[str] | None] = {}
    for module in project_record.modules:
        for class_ in module.classes:
            if class_.hierarchy is None:
                continue
            mro = resolve_mro(class_, index, cache=cache)
            if mro is None:
                logger.warning(
                    "MRO cannot be calculated for '%s' - hierarchy_confidence will become LOW.",
                    class_.qualified_name,
                )
                class_.hierarchy.mro_list = []
            else:
                class_.hierarchy.mro_list = mro


def _detect_protocols_project(project_record: ProjectRecord, index_classes: dict[str, ClassRecord]) -> None:
    """Step 4 - detect implicit protocols and update hierarchy_confidence.

    Parameters
    ----------
    project_record : ProjectRecord
        The project record whose modules and classes are scanned to detect implicit protocols and update hierarchy confidence
        values.
    index_classes : dict[str, ClassRecord]
        A mapping of class names to their corresponding ClassRecord objects, used to resolve class relationships when detecting
        implicit protocols across the project.

    """
    for module in project_record.modules:
        for class_ in module.classes:
            class_.protocols = detect_protocols(class_, index_classes)
            if class_.hierarchy is None:
                class_.hierarchy_confidence = Confidence.ABSENT
            elif not class_.hierarchy.mro_list:
                class_.hierarchy_confidence = Confidence.INFERRED_LOW
            else:
                class_.hierarchy_confidence = Confidence.EXPLICIT


def analyze_inheritance_project(project_record: ProjectRecord) -> None:
    """Post-process inheritance across the project.

    1. Build the class index (needed for step 2 idempotence).
    2. Resolve raw parent names to qualified names.
    3. Calculate each class's C3 MRO (shared project cache).
    4. Detect implicit protocols and update ``hierarchy_confidence``.

    Mutate ``ClassRecord`` objects in place.

    Idempotence: build the index first so ``_resolve_parents_project`` can detect
    already resolved names and avoid resolving them again through ``p in index or p == _PYTHON_ROOT``.

    Protocol detection (step 4) consults the MRO to include inherited dunders. It applies to all classes, including those without
    ``hierarchy`` (only local methods are then used).

    Parameters
    ----------
    project_record : ProjectRecord
        The project-level record containing the modules and classes to analyze; its class records are mutated in place during
        inheritance post-processing.

    """
    index = _build_index_classes(project_record)
    _resolve_parents_project(project_record, index)
    _compute_mro_project(project_record, index)
    for module in project_record.modules:
        for class_ in module.classes:
            classify_methods(class_, index)
    _detect_protocols_project(project_record, index)
