# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Statically link tests to production functions (iteration 6, Module 1).

Resolve function calls in test files to production functions through pure AST
analysis without running tests.

Supported frameworks: pytest and unittest (DEC-010).
Mechanism: import-alias resolution plus name matching (DEC-011).
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

from docmethis_extract_python.static_extraction.ast_parser import parse_file_with_lines
from docmethis_extract_python.static_extraction.models import (
    MISSING_VALUE,
    Confidence,
    FunctionRecord,
    ImportRecord,
    ModuleRecord,
    Parameter,
    ParameterType,
    TestCoverage,
    TestReference,
    Visibility,
    iter_functions,
    name_simple,
)

logger = logging.getLogger(__name__)

__all__ = ["link_test_functions"]

# Parameter kinds that do not consume a call argument.
_VARIADIC_KINDS: frozenset[ParameterType] = frozenset({ParameterType.VAR_POSITIONAL, ParameterType.VAR_KEYWORD})
_IMPLICIT_PARAMETERS: frozenset[str] = frozenset({"self", "cls"})


# ---------------------------------------------------------------------------
# Disambiguation helpers
# ---------------------------------------------------------------------------


def _call_argument_count(call_node: ast.Call) -> int | None:
    """Return call argument count, or None when unpacking makes arity unknown.

    ``f(*args)`` and ``f(**kwargs)`` hide the actual argument count. Literal
    counting would yield 1 (the ``Starred`` node) and 0 (``kw.arg`` is None),
    producing false arity that could reject the right candidate under criterion
    2. Prefer unknown arity so the criterion is skipped rather than falsified.
    """
    if any(isinstance(arg, ast.Starred) for arg in call_node.args):
        return None
    if any(kw.arg is None for kw in call_node.keywords):
        return None
    return len(call_node.args) + len(call_node.keywords)


def _effective_parameters(fn: FunctionRecord) -> list[Parameter] | None:
    """Return parameters consuming a call argument, or None without a signature.

    Exclude variadics (which impose no bound) and implicit `self`/`cls`
    (never passed explicitly). Both arity counters derive from this list, so
    minimum arity cannot exceed maximum arity.
    """
    if fn.signature is None:
        return None
    return [p for p in fn.signature.parameters if p.kind not in _VARIADIC_KINDS and p.name not in _IMPLICIT_PARAMETERS]


def _count_non_variadic_parameters(fn: FunctionRecord) -> int | None:
    """Return parameters excluding *args/**kwargs/self/cls, or None without a signature.

    This is the function's **maximum** arity when it is not variadic.
    """
    params = _effective_parameters(fn)
    return None if params is None else len(params)


def _count_required_parameters(fn: FunctionRecord) -> int | None:
    """Return minimum arity: parameters without defaults, excluding variadics/self/cls."""
    params = _effective_parameters(fn)
    return None if params is None else sum(p.default_value == MISSING_VALUE for p in params)


def _is_variadic(fn: FunctionRecord) -> bool:
    """Return whether a function declares *args or **kwargs (unbounded maximum arity)."""
    return fn.signature is not None and any(p.kind in _VARIADIC_KINDS for p in fn.signature.parameters)


def _common_prefix_length(module_a: str, module_b: str) -> int:
    """Return the number of common initial segments in two module names.

    Stop at the first divergence. Counting all equal segments would include
    later coincidences ('a.x.c' vs 'a.b.c' -> 2 instead of 1), incorrectly
    treating candidates with different proximity as equal.
    """
    length = 0
    for x, y in zip(module_a.split("."), module_b.split("."), strict=False):
        if x != y:
            break
        length += 1
    return length


def _arity_compatible(fn: FunctionRecord, n_args: int) -> bool:
    """Return whether ``fn`` can accept a call with ``n_args`` arguments.

    Compare against ``[required, total]`` rather than total alone:
    ``foo(a, b=1)`` accepts both ``foo(1)`` and ``foo(1, 2)``. A variadic
    function has no upper bound. A missing signature is incompatible.
    """
    required = _count_required_parameters(fn)
    maximum = _count_non_variadic_parameters(fn)
    if required is None or maximum is None:
        return False
    return required <= n_args and (_is_variadic(fn) or n_args <= maximum)


# ---------------------------------------------------------------------------
# Production index
# ---------------------------------------------------------------------------


@dataclass
class _IndexProduction:
    """Index production functions for static call resolution."""

    by_full_key: dict[str, FunctionRecord] = field(default_factory=dict)
    # "module_name.qualified_name" -> FunctionRecord

    by_simple_name: dict[str, list[FunctionRecord]] = field(default_factory=dict)
    # Last qualified_name component -> [FunctionRecord].

    @staticmethod
    def _in_module(fn: FunctionRecord, prefix: str) -> bool:
        return fn.parent_module == prefix or fn.parent_module.startswith(prefix + ".")

    @staticmethod
    def _resolve_contextually(
        candidates: list[FunctionRecord],
        analysis_context: str,
        test_module: str,
        n_args: int | None = None,
    ) -> FunctionRecord | None:
        """Disambiguate candidates by call context using four ordered criteria.

        Apply in order until one candidate remains:
        1. The function parent_module is test_module or its prefix.
        2. Arity compatibility: n_args falls within the signature interval.
        3. The only public candidate (visibility = PUBLIC).
        4. Longest common prefix between parent_module and test_module.
        """
        remaining = list(candidates)

        # Criterion 1 - candidate parent_module equals or prefixes test_module.
        filtered = [fn for fn in remaining if fn.parent_module == test_module or test_module.startswith(fn.parent_module + ".")]
        if len(filtered) == 1:
            return filtered[0]
        if filtered:
            remaining = filtered

        # Criterion 2 - arity compatibility (known n_args).
        if n_args is not None:
            filtered = [fn for fn in remaining if _arity_compatible(fn, n_args)]
            if len(filtered) == 1:
                return filtered[0]
            if filtered:
                remaining = filtered

        # Criterion 3 - only public candidate.
        public_candidates = [fn for fn in remaining if fn.visibility == Visibility.PUBLIC]
        if len(public_candidates) == 1:
            return public_candidates[0]
        if public_candidates:
            remaining = public_candidates

        # Criterion 4 - module proximity: longest common prefix.
        max_prefix = max(_common_prefix_length(fn.parent_module, test_module) for fn in remaining)
        filtered = [fn for fn in remaining if _common_prefix_length(fn.parent_module, test_module) == max_prefix]
        if len(filtered) == 1:
            return filtered[0]

        logger.warning(
            "Ambiguous resolution for %r: %d candidates after disambiguation; ignored",
            analysis_context,
            len(filtered),
        )
        return None

    @staticmethod
    def _resolve_unique(
        candidates: list[FunctionRecord],
        analysis_context: str,
        test_module: str = "",
        n_args: int | None = None,
    ) -> FunctionRecord | None:
        if len(candidates) > 1:
            if test_module:
                return _IndexProduction._resolve_contextually(candidates, analysis_context, test_module, n_args)
            logger.warning("Ambiguous resolution for %r: %d candidates; ignored", analysis_context, len(candidates))
            return None
        return candidates[0] if candidates else None

    def find_by_direct_target(
        self,
        target_symbol: str,
        test_module: str = "",
        n_args: int | None = None,
    ) -> FunctionRecord | None:
        """Resolve a "module.name" target (from ``from X import Y``).

        Search by exact key first, then by package and simple-name match.
        """
        if target_symbol in self.by_full_key:
            return self.by_full_key[target_symbol]
        parts = target_symbol.rsplit(".", 1)
        if len(parts) != 2:  # noqa: PLR2004
            return None
        package, identifier_name = parts
        candidates = [fn for fn in self.by_simple_name.get(identifier_name, []) if self._in_module(fn, package)]
        return self._resolve_unique(candidates, target_symbol, test_module, n_args)

    def find_by_unique_simple_name(
        self,
        identifier_name: str,
        test_module: str = "",
        n_args: int | None = None,
    ) -> FunctionRecord | None:
        """Return the FunctionRecord when a simple name is unambiguous."""
        return self._resolve_unique(self.by_simple_name.get(identifier_name, []), identifier_name, test_module, n_args)

    def find_in_module(
        self,
        module_prefix: str,
        identifier_name: str,
        test_module: str = "",
        n_args: int | None = None,
    ) -> FunctionRecord | None:
        """Find a function named ``identifier_name`` in module or package ``module_prefix``."""
        candidates = [fn for fn in self.by_simple_name.get(identifier_name, []) if self._in_module(fn, module_prefix)]
        return self._resolve_unique(candidates, f"{module_prefix}.{identifier_name}", test_module, n_args)

    def find_method(
        self,
        module_prefix: str,
        cls_name: str,
        method_name: str,
        test_module: str = "",
        n_args: int | None = None,
    ) -> FunctionRecord | None:
        """Find method ``cls_name.method_name`` in module or package ``module_prefix``."""
        key_qname = f"{cls_name}.{method_name}"
        candidates = [
            fn
            for fn in self.by_simple_name.get(method_name, [])
            if fn.qualified_name == key_qname and self._in_module(fn, module_prefix)
        ]
        return self._resolve_unique(candidates, f"{module_prefix}.{key_qname}", test_module, n_args)


def _build_index_production(modules: list[ModuleRecord]) -> _IndexProduction:
    """Build the production-function index from all modules."""
    index = _IndexProduction()
    modules_prod = [m for m in modules if not m.is_test and not m.is_stub]
    for fn in iter_functions(modules_prod):
        _indexer_function(fn, fn.parent_module, index)
    return index


def _indexer_function(fn: FunctionRecord, module_name: str, index: _IndexProduction) -> None:
    """Add a FunctionRecord to the index."""
    mapping_key = f"{module_name}.{fn.qualified_name}"
    index.by_full_key[mapping_key] = fn
    index.by_simple_name.setdefault(name_simple(fn.qualified_name), []).append(fn)


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------


def _build_alias_maps(
    imports: list[ImportRecord],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build alias maps from a module's imports.

    Return ``(direct_aliases, module_aliases)``:

    - ``direct_aliases``: local_name -> ``"module.name"`` (``from X import Y [as Z]``)
    - ``module_aliases``: local_prefix -> module (``import X [as Y]``)
    """
    direct_aliases: dict[str, str] = {}
    module_aliases: dict[str, str] = {}
    for imp in imports:
        if imp.type_checking_only:
            continue
        if imp.name is None:
            local = imp.alias or imp.module.split(".")[-1]
            module_aliases[local] = imp.module
        else:
            local = imp.alias or imp.name
            direct_aliases[local] = f"{imp.module}.{imp.name}"
    return direct_aliases, module_aliases


# ---------------------------------------------------------------------------
# Collect calls in a test body
# ---------------------------------------------------------------------------


class _CallCollector(ast.NodeVisitor):
    """Collect all ast.Call nodes in an AST subtree."""

    def __init__(self) -> None:
        """Initialize the call list."""
        self.calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Record the call and continue visiting."""
        self.calls.append(node)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Test-framework detection
# ---------------------------------------------------------------------------


def _is_unittest_class(node: ast.ClassDef) -> bool:
    """Return True when a class inherits unittest.TestCase."""
    return any("TestCase" in ast.unparse(base) for base in node.bases)


def _detect_framework(node: ast.ClassDef | None, imports: list[ImportRecord]) -> str:
    """Infer the test framework from the class and module imports.

    Priority: unittest (TestCase inheritance) > pytest (pytest import) > unknown.
    """
    if node is not None and _is_unittest_class(node):
        return "unittest"
    if any(imp.module == "pytest" or imp.module.startswith("pytest.") for imp in imports):
        return "pytest"
    return "unknown"


# ---------------------------------------------------------------------------
# Main visitor
# ---------------------------------------------------------------------------


class _TestLinksVisitor(ast.NodeVisitor):
    """Visit a test file and collect links from calls to production functions.

    Maintain a class-context stack to build qualified test-function names
    (for example, ``TestHelpers.test_bounds``).
    """

    def __init__(
        self,
        module_record: ModuleRecord,
        index: _IndexProduction,
    ) -> None:
        """Initialize the visitor with the test module and production index."""
        self._module = module_record
        self._index = index
        self._direct_aliases, self._module_aliases = _build_alias_maps(module_record.imports)
        self._class_stack: list[tuple[str, str]] = []  # (class_name, framework)
        self.links: list[tuple[str, TestReference]] = []  # (production function key, reference)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a test class while maintaining framework context."""
        framework = _detect_framework(node, self._module.imports)
        self._class_stack.append((node.name, framework))
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a synchronous function definition."""
        self._process_function_node(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an asynchronous function definition."""
        self._process_function_node(node)

    def _process_function_node(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Process a function node: analyze calls for tests, otherwise continue visiting."""
        if not node.name.startswith("test"):
            self.generic_visit(node)
            return

        if self._class_stack:
            cls_name, framework = self._class_stack[-1]
            test_qname = f"{cls_name}.{node.name}"
        else:
            framework = _detect_framework(None, self._module.imports)
            test_qname = node.name

        collector = _CallCollector()
        for stmt in node.body:
            collector.visit(stmt)

        for call_node in collector.calls:
            fn = self._resolve_call(call_node)
            if fn is None:
                continue
            ref = TestReference(
                test_qualified_name=test_qname,
                test_module=self._module.module_name,
                test_file=self._module.file_path,
                test_line=node.lineno,
                framework=framework,
                call_line=call_node.lineno,
            )
            fn_key = f"{fn.parent_module}.{fn.qualified_name}"
            self.links.append((fn_key, ref))

    def _resolve_call(self, call_node: ast.Call) -> FunctionRecord | None:
        """Resolve an ast.Call node to a production FunctionRecord."""
        test_module = self._module.module_name
        n_args = _call_argument_count(call_node)
        func = call_node.func
        if isinstance(func, ast.Name):
            return self._resolve_name(func.id, test_module, n_args)
        if isinstance(func, ast.Attribute):
            return self._resolve_attribute(func, test_module, n_args)
        return None

    def _resolve_name(self, identifier_name: str, test_module: str, n_args: int | None) -> FunctionRecord | None:
        """Resolve a direct call ``f(...)``."""
        if identifier_name in self._direct_aliases:
            return self._index.find_by_direct_target(self._direct_aliases[identifier_name], test_module, n_args)
        return self._index.find_by_unique_simple_name(identifier_name, test_module, n_args)

    def _resolve_attribute(self, attr_node: ast.Attribute, test_module: str, n_args: int | None) -> FunctionRecord | None:
        """Resolve an attribute call ``obj.f(...)`` or ``obj.Cls.f(...)``."""
        item_value = attr_node.value
        attr = attr_node.attr

        if isinstance(item_value, ast.Name) and item_value.id in self._module_aliases:
            module = self._module_aliases[item_value.id]
            return self._index.find_in_module(module, attr, test_module, n_args)

        if (
            isinstance(item_value, ast.Attribute)
            and isinstance(item_value.value, ast.Name)
            and item_value.value.id in self._module_aliases
        ):
            module = self._module_aliases[item_value.value.id]
            return self._index.find_method(module, item_value.attr, attr, test_module, n_args)

        return None


# ---------------------------------------------------------------------------
# Enrich FunctionRecords
# ---------------------------------------------------------------------------


def _populate_test_coverage(
    fn: FunctionRecord,
    module_name: str,
    links: dict[str, list[TestReference]],
) -> None:
    """Populate a FunctionRecord test_coverage field from the link mapping."""
    fn_key = f"{module_name}.{fn.qualified_name}"
    raw_references = links.get(fn_key, [])

    if not raw_references:
        fn.test_coverage = TestCoverage()
        fn.test_coverage_confidence = Confidence.ABSENT
        return

    seen: set[tuple[str, str]] = set()
    unique_references: list[TestReference] = []
    for ref in raw_references:
        mapping_key = (ref.test_qualified_name, ref.test_module)
        if mapping_key not in seen:
            seen.add(mapping_key)
            unique_references.append(ref)

    fn.test_coverage = TestCoverage(
        test_references=unique_references,
        is_directly_tested=True,
        test_count=len(unique_references),
    )
    fn.test_coverage_confidence = Confidence.INFERRED_HIGH


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def link_test_functions(modules: list[ModuleRecord]) -> None:
    """Enrich production FunctionRecords with static TestCoverage.

    Use pure AST analysis of test modules to detect production functions called
    directly by test functions. Mutate FunctionRecords in place. Always run after
    loading the cache because results are not saved in the per-module cache.

    Parameters
    ----------
    modules : list[ModuleRecord]
        Complete list of project modules (tests and production combined).

    """
    index = _build_index_production(modules)
    links: dict[str, list[TestReference]] = {}

    for module in modules:
        if not module.is_test:
            continue
        outcome = parse_file_with_lines(module.file_path)
        if outcome is None:
            logger.warning("Unable to reparse test file: %s", module.file_path)
            continue
        tree, _ = outcome
        visitor = _TestLinksVisitor(module, index)
        visitor.visit(tree)
        for fn_key, ref in visitor.links:
            links.setdefault(fn_key, []).append(ref)

    modules_prod = [m for m in modules if not m.is_test and not m.is_stub]
    for fn in iter_functions(modules_prod):
        _populate_test_coverage(fn, fn.parent_module, links)
