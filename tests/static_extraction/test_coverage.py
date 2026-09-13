# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Unit tests for coverage.py mapping to FunctionRecord (iteration 9)."""

from __future__ import annotations

from pathlib import Path

from docmethis_extract_python.static_extraction.dynamic_analysis.coverage import (
    _compute_function_ratio,
    _enrich_function,
    _function_affected_by_failure,
    _lines_by_file,
    map_coverage,
)
from docmethis_extract_python.static_extraction.dynamic_analysis.test_runner import ExecutionResult
from docmethis_extract_python.static_extraction.models import (
    ClassRecord,
    Confidence,
    FunctionRecord,
    MethodType,
    ModuleRecord,
    Visibility,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fn(
    qname: str = "function",
    file_path: Path = Path("/proj/pkg/mod.py"),
    line_start: int = 10,
    line_end: int = 20,
    parent_module: str = "pkg.mod",
) -> FunctionRecord:
    return FunctionRecord(
        qualified_name=qname,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        col_start=0,
        col_end=0,
        method_kind=MethodType.FUNCTION,
        visibility=Visibility.PUBLIC,
        parent_class=None,
        parent_module=parent_module,
        existing_docstring=None,
    )


def _module(
    *,
    file_path: Path = Path("/proj/pkg/mod.py"),
    module_name: str = "pkg.mod",
    is_test: bool = False,
    functions: list[FunctionRecord] | None = None,
    classes: list[ClassRecord] | None = None,
) -> ModuleRecord:
    return ModuleRecord(
        file_path=file_path,
        module_name=module_name,
        is_test=is_test,
        functions=functions or [],
        classes=classes or [],
    )


def _result(coverage_json: dict | None = None, failing_tests: list[str] | None = None) -> ExecutionResult:
    return ExecutionResult(
        success=True,
        framework="pytest",
        returncode=0,
        stdout="",
        stderr="",
        coverage_json=coverage_json,
        failing_tests=failing_tests or [],
    )


def _coverage_json(file_path: str, executed: list[int], missing: list[int]) -> dict:
    return {"files": {file_path: {"executed_lines": executed, "missing_lines": missing}}}


# ---------------------------------------------------------------------------
# _lines_by_file
# ---------------------------------------------------------------------------


def test_lines_by_file_structure() -> None:
    """The coverage.py JSON is converted to {Path: (executed, missing)}."""
    data = _coverage_json("/foo.py", [1, 2, 5], [3, 4])
    outcome = _lines_by_file(data)
    source_path = Path("/foo.py").resolve()
    assert source_path in outcome
    executes, missing_items = outcome[source_path]
    assert executes == {1, 2, 5}
    assert missing_items == {3, 4}


def test_lines_by_file_empty_files() -> None:
    """A report without files produces an empty dict."""
    assert _lines_by_file({"files": {}}) == {}


def test_lines_by_file_multiple_files() -> None:
    """Multiple files are all present in the result."""
    data = {
        "files": {
            "/a.py": {"executed_lines": [1], "missing_lines": []},
            "/b.py": {"executed_lines": [], "missing_lines": [2]},
        }
    }
    outcome = _lines_by_file(data)
    assert len(outcome) == 2


# ---------------------------------------------------------------------------
# _compute_function_ratio
# ---------------------------------------------------------------------------


def test_compute_ratio_all_covered() -> None:
    """All function lines covered gives ratio 1.0."""
    fn = _fn(line_start=1, line_end=5)
    executes = {1, 2, 3, 4, 5}
    _, uncovered, ratio = _compute_function_ratio(fn, executes, set())
    assert ratio == 1.0
    assert uncovered == []


def test_compute_ratio_none_covered() -> None:
    """No lines covered gives ratio 0.0."""
    fn = _fn(line_start=1, line_end=5)
    missing_items = {1, 2, 3, 4, 5}
    covered, _, ratio = _compute_function_ratio(fn, set(), missing_items)
    assert ratio == 0.0
    assert covered == []


def test_compute_ratio_half_covered() -> None:
    """Half the lines covered gives ratio 0.5."""
    fn = _fn(line_start=1, line_end=4)
    _, _, ratio = _compute_function_ratio(fn, {1, 2}, {3, 4})
    assert ratio == 0.5


def test_compute_ratio_ignores_lines_outside_range() -> None:
    """Lines outside [line_start, line_end] are not counted."""
    fn = _fn(line_start=10, line_end=15)
    # Lines 1-9 and 20-25 do not concern this function.
    covered, uncovered, _ = _compute_function_ratio(fn, {1, 5, 10, 11}, {3, 12})
    assert sorted(covered) == [10, 11]
    assert sorted(uncovered) == [12]


def test_compute_ratio_empty_range_returns_zero() -> None:
    """No executed or missing lines in the range gives ratio 0.0, not ZeroDivisionError."""
    fn = _fn(line_start=50, line_end=60)
    _, _, ratio = _compute_function_ratio(fn, {1, 2}, {3, 4})
    assert ratio == 0.0


# ---------------------------------------------------------------------------
# _function_affected_by_failure
# ---------------------------------------------------------------------------


def test_function_affected_by_failure_simple_name() -> None:
    """Return True when a test node id contains the function's simple name."""
    fn = _fn(qname="pkg.module.function")
    _module()
    assert _function_affected_by_failure(fn, ["tests/test_mod.py::test_function"]) is True


def test_function_affected_by_failure_file_name() -> None:
    """Return True when a node id contains the source file stem."""
    fn = _fn(file_path=Path("/proj/pkg/utils.py"), qname="pkg.utils.other_function")
    _module(file_path=Path("/proj/pkg/utils.py"))
    assert _function_affected_by_failure(fn, ["tests/test_utils.py::test_other"]) is True


def test_function_affected_by_failure_empty_list() -> None:
    """An empty failing-test list returns False."""
    fn = _fn()
    _module()
    assert _function_affected_by_failure(fn, []) is False


def test_function_affected_by_failure_unrelated() -> None:
    """Unrelated node ids return False."""
    fn = _fn(qname="pkg.mod.function", file_path=Path("/proj/pkg/mod.py"))
    _module(file_path=Path("/proj/pkg/mod.py"))
    assert _function_affected_by_failure(fn, ["tests/test_other.py::test_other"]) is False


# ---------------------------------------------------------------------------
# _enrich_function
# ---------------------------------------------------------------------------


def test_enrich_function_has_any_test_true() -> None:
    """Covered lines in the range set has_any_test=True and INFERRED_HIGH confidence."""
    fn = _fn(line_start=1, line_end=5)
    _enrich_function(fn, {1, 2, 3}, {4, 5}, [])
    assert fn.test_coverage is not None
    assert fn.test_coverage.has_any_test is True
    assert fn.test_coverage_confidence == Confidence.INFERRED_HIGH


def test_enrich_function_has_any_test_false() -> None:
    """No covered lines set has_any_test=False and leaves confidence ABSENT."""
    fn = _fn(line_start=1, line_end=5)
    _enrich_function(fn, set(), {1, 2, 3, 4, 5}, [])
    assert fn.test_coverage is not None
    assert fn.test_coverage.has_any_test is False
    assert fn.test_coverage_confidence == Confidence.ABSENT


def test_enrich_function_has_failing_test_true() -> None:
    """Covered lines plus a related failing test set has_failing_test=True."""
    fn = _fn(qname="pkg.mod.function", line_start=10, line_end=20, file_path=Path("/proj/pkg/mod.py"))
    _enrich_function(fn, {10, 11}, {12}, ["tests/test_mod.py::test_function"])
    assert fn.test_coverage is not None
    assert fn.test_coverage.has_failing_test is True


def test_enrich_function_has_failing_test_false_when_uncovered() -> None:
    """An uncovered function has_failing_test=False even when tests fail."""
    fn = _fn(qname="pkg.mod.function", line_start=10, line_end=20)
    _enrich_function(fn, set(), {10, 11, 12}, ["tests/test_mod.py::test_function"])
    assert fn.test_coverage is not None
    assert fn.test_coverage.has_failing_test is False


def test_enrich_function_creates_missing_test_coverage() -> None:
    """test_coverage=None initially is created by _enrich_function."""
    fn = _fn(line_start=1, line_end=3)
    assert fn.test_coverage is None
    _enrich_function(fn, {1}, {2, 3}, [])
    assert fn.test_coverage is not None


# ---------------------------------------------------------------------------
# map_coverage
# ---------------------------------------------------------------------------


def test_map_coverage_without_json() -> None:
    """coverage_json=None leaves FunctionRecords unchanged."""
    fn = _fn(line_start=1, line_end=5)
    mod = _module(functions=[fn])
    map_coverage(_result(coverage_json=None), [mod])
    assert fn.test_coverage is None


def test_map_coverage_excludes_test_modules() -> None:
    """Modules with is_test=True are ignored."""
    fn = _fn(line_start=1, line_end=5)
    mod = _module(is_test=True, functions=[fn])
    cov = _coverage_json(str(Path("/proj/pkg/mod.py").resolve()), [1, 2], [3])
    map_coverage(_result(cov), [mod])
    assert fn.test_coverage is None


def test_map_coverage_file_missing_from_report() -> None:
    """An unreferenced file in coverage.json leaves its FunctionRecord unchanged."""
    fn = _fn(file_path=Path("/proj/pkg/mod.py"), line_start=1, line_end=5)
    mod = _module(file_path=Path("/proj/pkg/mod.py"), functions=[fn])
    cov = _coverage_json("/other/file.py", [1, 2], [3])
    map_coverage(_result(cov), [mod])
    assert fn.test_coverage is None


def test_map_coverage_enriches_functions(tmp_path: Path) -> None:
    """Production functions are enriched with coverage."""
    source_file = tmp_path / "mod.py"
    fn = _fn(file_path=source_file, line_start=1, line_end=5)
    mod = _module(file_path=source_file, functions=[fn])
    cov = _coverage_json(str(source_file.resolve()), [1, 2, 3], [4, 5])
    map_coverage(_result(cov), [mod])
    assert fn.test_coverage is not None
    assert fn.test_coverage.has_any_test is True
    assert fn.test_coverage.ratio == 0.6


def test_map_coverage_enriches_methods(tmp_path: Path) -> None:
    """Class methods are enriched as well."""
    source_file = tmp_path / "mod.py"
    method = _fn(qname="Class.method", file_path=source_file, line_start=10, line_end=14)
    cls = ClassRecord(
        qualified_name="MaClasse",
        file_path=source_file,
        line_start=1,
        line_end=20,
        col_start=0,
        col_end=0,
        visibility=Visibility.PUBLIC,
        parent_module="pkg.mod",
        existing_docstring=None,
        methods=[method],
    )
    mod = _module(file_path=source_file, classes=[cls])
    cov = _coverage_json(str(source_file.resolve()), [10, 11], [12, 13, 14])
    map_coverage(_result(cov), [mod])
    assert method.test_coverage is not None
    assert method.test_coverage.has_any_test is True
