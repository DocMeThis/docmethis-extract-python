# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Integrated validation of iteration 1.10, phase 8.

Unit tests for phases 1 through 7 verify each fix in isolation on manually built
``ModuleRecord`` objects. This module does the opposite: it runs a fixture project
through the real pipeline and verifies the metrics announced by ``iteration_10.md``.

The goal is to cover propagation between pipeline stages. A fix can be correct in
isolation and then be neutralized by the next stage, as happened between phase 5
mapping and inheritance resolution.

Scope: the complete static pipeline (extraction, inheritance, callgraph, and
test-to-function links). Running the fixture's actual test suite is excluded because
it is covered by ``test_coverage_diagnostics.py`` and adds no propagation coverage.

The iteration plan proposed a separate robustness test directory, which does not exist
in this repository. This file follows the actual source-tree test convention.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import pytest

from docmethis_extract_python.static_extraction.__main__ import analyze_project
from docmethis_extract_python.static_extraction.callgraph_ast import build_callgraph
from docmethis_extract_python.static_extraction.configuration import ConfigurationDocmethis
from docmethis_extract_python.static_extraction.dynamic_analysis.test_analysis import link_test_functions

if TYPE_CHECKING:
    from docmethis_extract_python.static_extraction.models import CallgraphEdge, ClassRecord, FunctionRecord, ProjectRecord

# ---------------------------------------------------------------------------
# Fixture project: one file per fix, with covered phases annotated.
# ---------------------------------------------------------------------------

PYPROJECT = '[project]\nname = "fixture"\nversion = "0.1.0"\n'

# Phase 4: the package re-exports a symbol it does not define.
PKG_INIT = '"""Fixture package."""\n\nfrom pkg.api import process\n\n__all__ = ["process"]\n'

# Phase 1: @overload declarations must merge into the implementation.
PKG_API = '''"""Public API."""

from typing import overload

__all__ = ["process"]


@overload
def process(value: int) -> int: ...
@overload
def process(value: str) -> str: ...
def process(value):
    """Single implementation for both overloads."""
    return value
'''

# Phase 2: two same-named ``format_value`` functions, distinguishable only by arity.
PKG_UTILS = '"""Utilities."""\n\n\ndef format_value(text):\n    """One parameter."""\n    return text.strip()\n'
PKG_INTERNAL = '"""Internal utilities."""\n\n\ndef format_value(a, b, c):\n    """Three parameters."""\n    return f"{a}{b}{c}"\n'

# Phase 3: function defined in a control block (previously never indexed).
PKG_CONDITIONAL = '''"""Conditional definition."""

import sys

if sys.version_info >= (3, 0):

    def by_version():
        """Defined inside an if statement."""
        return "recent"
'''

# Phases 5 and 7: nested/top-level name collisions, lexical inheritance, and a simple mixin.
PKG_MODELS = '''"""Models."""


class Inner:
    """Top-level name matching a nested class."""


class Container:
    """Contains nested classes that inherit from one another."""

    class Base:
        """Nested base class."""

        def greet(self):
            """Method to inherit."""
            return "hello"

    class Child(Base):
        """Inherits from a nested sibling by its bare name."""

    class Inner:
        """Shadows the module-level Inner class."""

    class Daughter(Inner):
        """Must inherit from Container.Inner, not the top-level Inner."""


class TraceMixin:
    """Mixin with a trivial __init__; it must not be excluded."""

    def __init__(self):
        self.trace = []

    def trace(self):
        """Trace the call."""
        return "trace"
'''

# Phase 4: internal re-export consumer; the edge must remain intra-project.
PKG_SERVICE = '''"""Service."""

from pkg import *


def execute(value):
    """Call process through the package re-export."""
    return process(value)
'''

TEST_API = '''"""API tests."""

from pkg import *


def test_process():
    assert process(1) == 1
'''

# Two star imports: ``format_value`` is ambiguous, so only arity can decide (phase 2).
TEST_UTILS = '''"""Utility tests."""

from pkg.internal import *
from pkg.utils import *


def test_format_value():
    assert format_value(" x ") == "x"
'''

TEST_MODELS = '''"""Model tests."""

import pkg.models


def test_child():
    assert pkg.models.Container.Child().greet() == "hello"
'''

FIXTURE_FILES = {
    "pyproject.toml": PYPROJECT,
    "pkg/__init__.py": PKG_INIT,
    "pkg/api.py": PKG_API,
    "pkg/utils.py": PKG_UTILS,
    "pkg/internal.py": PKG_INTERNAL,
    "pkg/conditional.py": PKG_CONDITIONAL,
    "pkg/models.py": PKG_MODELS,
    "pkg/service.py": PKG_SERVICE,
    "tests/test_api.py": TEST_API,
    "tests/test_utils.py": TEST_UTILS,
    "tests/test_models.py": TEST_MODELS,
}


class _Analysis(NamedTuple):
    """Pipeline result for the fixture project, plus its emitted logs."""

    project_record: ProjectRecord
    callgraph: dict[str, list[CallgraphEdge]]
    logs: list[logging.LogRecord]


class _LogCollector(logging.Handler):
    """Capture docmethis logs without relying on propagation to pytest's handler."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture(scope="module")
def analysis(tmp_path_factory: pytest.TempPathFactory) -> _Analysis:
    """Write the fixture project and run it through the complete static pipeline."""
    root = tmp_path_factory.mktemp("fixture")
    for relative_path, payload in FIXTURE_FILES.items():
        target_path = root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(payload, encoding="utf-8")

    collector = _LogCollector()
    docmethis_logger = logging.getLogger("docmethis")
    initial_level = docmethis_logger.level
    docmethis_logger.addHandler(collector)
    docmethis_logger.setLevel(logging.DEBUG)

    try:
        project_record = analyze_project(
            root,
            ConfigurationDocmethis(full_reanalysis=True),
            skip_dynamic=True,
            write_cache=False,
        )
        callgraph = build_callgraph(project_record)
        link_test_functions(project_record.modules)
    finally:
        docmethis_logger.removeHandler(collector)
        docmethis_logger.setLevel(initial_level)

    return _Analysis(project_record=project_record, callgraph=callgraph, logs=collector.records)


def _function(project_record: ProjectRecord, qualified_name: str) -> FunctionRecord:
    for module in project_record.modules:
        for fn in module.functions:
            if fn.qualified_name == qualified_name:
                return fn
    msg = f"function {qualified_name!r} is missing from the fixture project"
    raise AssertionError(msg)


def _class(project_record: ProjectRecord, qualified_name: str) -> ClassRecord:
    for module in project_record.modules:
        for class_ in module.classes:
            if class_.qualified_name == qualified_name:
                return class_
    msg = f"class {qualified_name!r} is missing from the fixture project"
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Metric 1: zero occurrences of the seven warnings.
# ---------------------------------------------------------------------------


class TestWarningMetric:
    """The ``iteration_10.md`` criterion: no known or new warnings."""

    def test_no_warning_on_fixture_project(self, analysis: _Analysis) -> None:
        """The criterion is global: no warnings, not just the seven known ones."""
        warnings = [record for record in analysis.logs if record.levelno >= logging.WARNING]

        assert warnings == [], [f"{record.name}: {record.getMessage()}" for record in warnings]

    @pytest.mark.parametrize(
        "pattern",
        [
            "Ambiguous resolution",
            "AST node not found",
            "Star import",
            "Name collision",
            "mixin detection",
            "@overload",
            "coverage_mapper",
        ],
    )
    def test_pattern_absent_at_all_levels(self, analysis: _Analysis, pattern: str) -> None:
        """These messages are debug-only and must not be emitted at all."""
        emitted = [record.getMessage() for record in analysis.logs if pattern in record.getMessage()]

        assert emitted == []


# ---------------------------------------------------------------------------
# Metric 2: test-to-function mapping.
# ---------------------------------------------------------------------------


class TestTestLinkMetric:
    """The ``iteration_10.md`` criterion requires a resolution rate above 95%."""

    # Production functions actually called by the fixture tests.
    EXPECTED_TARGETS = ("pkg.api.process", "pkg.utils.format_value")

    def test_resolution_rate(self, analysis: _Analysis) -> None:
        resolved = [
            identifier_name
            for identifier_name in self.EXPECTED_TARGETS
            if _function(analysis.project_record, identifier_name).test_coverage is not None
        ]
        rate = len(resolved) / len(self.EXPECTED_TARGETS)

        assert rate > 0.95, f"resolved: {resolved}"

    def test_same_name_disambiguated_by_arity(self, analysis: _Analysis) -> None:
        """``format_value(" x ")`` selects pkg.utils, not pkg.internal (three parameters)."""
        utils = _function(analysis.project_record, "pkg.utils.format_value")
        internal = _function(analysis.project_record, "pkg.internal.format_value")

        assert utils.test_coverage is not None
        assert utils.test_coverage.is_directly_tested is True
        assert internal.test_coverage is None or not internal.test_coverage.is_directly_tested

    def test_call_via_star_import_reexport(self, analysis: _Analysis) -> None:
        """``from pkg import *`` followed by ``process(1)`` resolves to pkg.api."""
        process = _function(analysis.project_record, "pkg.api.process")

        assert process.test_coverage is not None
        assert process.test_coverage.is_directly_tested is True


# ---------------------------------------------------------------------------
# Metric 3: intra-project callgraph.
# ---------------------------------------------------------------------------


class TestCallgraphMetric:
    """The ``iteration_10.md`` criterion requires more intra-project edges."""

    def test_reexport_edge_remains_internal(self, analysis: _Analysis) -> None:
        """``pkg.service.execute`` calls ``process`` imported by star from the package."""
        edges = analysis.callgraph.get("pkg.service.execute", [])
        targets = {edge.target_qname: edge for edge in edges}

        assert "pkg.api.process" in targets, f"found targets: {sorted(targets)}"
        assert targets["pkg.api.process"].is_external is False

    def _externally_marked_project_targets(self, analysis: _Analysis) -> set[str]:
        return {
            edge.target_qname
            for edges in analysis.callgraph.values()
            for edge in edges
            if edge.is_external and edge.external_lib == "pkg"
        }

    def test_no_project_function_marked_external(self, analysis: _Analysis) -> None:
        """The fixture package must not appear as an external library."""
        function_records = {fn.qualified_name for module in analysis.project_record.modules for fn in module.functions}
        incorrectly_external = self._externally_marked_project_targets(analysis) & function_records

        assert incorrectly_external == set()

    @pytest.mark.xfail(
        reason=(
            "Known defect outside iteration 1.10 scope: _resolve_name checks `resolved not in index_fn`, "
            "but index_fn contains only functions. A constructor call for a project class is therefore "
            "classified as external. index_cls is already carried by _CallgraphContext but is not used "
            "by this test; fixing it changes edge semantics for propagate_exceptions_via_callgraph, so "
            "the trade-off remains open."
        ),
        strict=True,
    )
    def test_project_constructor_is_not_external(self, analysis: _Analysis) -> None:
        """A test call to ``pkg.models.Container.Child()`` is incorrectly marked external."""
        assert self._externally_marked_project_targets(analysis) == set()

    def test_function_in_if_block_is_indexed(self, analysis: _Analysis) -> None:
        """Phase 3: the function exists and has a callgraph entry."""
        _function(analysis.project_record, "pkg.conditional.by_version")

        assert "pkg.conditional.by_version" in analysis.callgraph


# ---------------------------------------------------------------------------
# Metric 4: inheritance tree.
# ---------------------------------------------------------------------------


class TestInheritanceMetric:
    """The ``iteration_10.md`` criterion requires no collisions and relevant mixins."""

    def test_nested_sibling_inheritance(self, analysis: _Analysis) -> None:
        """``class Child(Base)`` in Container resolves to Container.Base."""
        child_class = _class(analysis.project_record, "pkg.models.Container.Child")

        assert child_class.hierarchy is not None
        assert child_class.hierarchy.direct_parents == ["pkg.models.Container.Base"]
        assert "greet" in {method.qualified_name.rsplit(".", 1)[-1] for method in child_class.inherited_methods}

    def test_nested_class_masks_top_level_name(self, analysis: _Analysis) -> None:
        """``class Daughter(Inner)`` in Container resolves to Container.Inner."""
        daughter = _class(analysis.project_record, "pkg.models.Container.Daughter")

        assert daughter.hierarchy is not None
        assert daughter.hierarchy.direct_parents == ["pkg.models.Container.Inner"]

    def test_trivial_mixin_init_is_preserved(self, analysis: _Analysis) -> None:
        """Phase 7: a trivial __init__ does not exclude the mixin."""
        mixin = _class(analysis.project_record, "pkg.models.TraceMixin")

        assert mixin.hierarchy is not None
        assert mixin.hierarchy.is_mixin is True


# ---------------------------------------------------------------------------
# Metric 5: overloads.
# ---------------------------------------------------------------------------


class TestOverloadMetric:
    """The ``iteration_10.md`` criterion requires has_overloads and overload_signatures."""

    def test_overloads_merge_into_implementation(self, analysis: _Analysis) -> None:
        process = _function(analysis.project_record, "pkg.api.process")

        assert process.has_overloads is True
        assert len(process.overload_signatures) == 2

    def test_single_record_per_name(self, analysis: _Analysis) -> None:
        """DEC-023: overloads do not create a competing FunctionRecord."""
        api = next(module for module in analysis.project_record.modules if module.module_name == "pkg.api")
        names = [fn.qualified_name for fn in api.functions]

        assert names.count("pkg.api.process") == 1
