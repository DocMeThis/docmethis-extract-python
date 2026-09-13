# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Map coverage.py data to FunctionRecord (iteration 1.9).

Parse the JSON report produced by coverage.py and enrich FunctionRecords with
TestCoverage dynamic fields: covered_lines, uncovered_lines, ratio,
has_any_test, and has_failing_test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from docmethis_extract_python.static_extraction.models import (
    Confidence,
    FunctionRecord,
    ModuleRecord,
    TestCall,
    TestCoverage,
    iter_functions,
    name_simple,
)

if TYPE_CHECKING:
    from docmethis_extract_python.static_extraction.dynamic_analysis.test_runner import ExecutionResult

__all__ = ["link_call_functions", "map_coverage"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extract covered lines by file from coverage.py JSON
# ---------------------------------------------------------------------------


def _lines_by_file(coverage_json: dict) -> dict[Path, tuple[set[int], set[int]]]:
    """Extract the {absolute_path: (covered_lines, uncovered_lines)} mapping.

    The coverage.py JSON has this structure: {
      "files": {
        "/abs/path/to/file.py": {
          "executed_lines": [1, 2, 5, ...],
          "missing_lines": [3, 4, 6, ...],
          ...
        }
      }
    }

    Parameters
    ----------
    coverage_json : dict
        Coverage data in the JSON format emitted by coverage.py, with a top-level 'files' key whose values map each absolute file
        path to a record containing 'executed_lines' and 'missing_lines' lists.

    Returns
    -------
    dict[Path, tuple[set[int], set[int]]]
        Extract the mapping from absolute file paths to a tuple containing the set of covered lines and the set of uncovered lines
        from a coverage.py JSON report.

    """
    outcome: dict[Path, tuple[set[int], set[int]]] = {}
    files = coverage_json.get("files", {})
    for path_str, data in files.items():
        source_path = Path(path_str).resolve()
        executes = set(data.get("executed_lines", []))
        missing_items = set(data.get("missing_lines", []))
        outcome[source_path] = (executes, missing_items)

    return outcome


# ---------------------------------------------------------------------------
# Calculate a ratio for a line range
# ---------------------------------------------------------------------------


def _compute_function_ratio(
    fn: FunctionRecord,
    executes: set[int],
    missing_items: set[int],
) -> tuple[list[int], list[int], float]:
    """Return (covered_lines, uncovered_lines, ratio) for a FunctionRecord range.

    Consider only lines in [line_start, line_end].

    Parameters
    ----------
    fn : FunctionRecord
        Function record whose line_start and line_end define the line range used to compute coverage; only lines within this range
        are considered.
    executes : set[int]
        Set of integer line numbers that were executed during dynamic analysis, used to determine which lines within the
        function's range are covered.
    missing_items : set[int]
        Set of line numbers that were not executed during coverage analysis, used to identify uncovered lines within the
        function's range.

    Returns
    -------
    tuple[list[int], list[int], float]
        A tuple containing the sorted covered line numbers, the sorted uncovered line numbers, and the coverage ratio as covered
        lines divided by total executable lines, or 0.0 when there are no executable lines.

    """
    line_range = set(range(fn.line_start, fn.line_end + 1))
    fn_executes = sorted(line_range & executes)
    fn_missing = sorted(line_range & missing_items)
    total_executable = len(fn_executes) + len(fn_missing)
    ratio = len(fn_executes) / total_executable if total_executable > 0 else 0.0
    return fn_executes, fn_missing, ratio


# ---------------------------------------------------------------------------
# Enrich a FunctionRecord
# ---------------------------------------------------------------------------


def _enrich_function(
    fn: FunctionRecord,
    executes: set[int],
    missing_items: set[int],
    failing_tests: list[str],
) -> None:
    """Populate fn.test_coverage dynamic fields from coverage data.

    Parameters
    ----------
    fn : FunctionRecord
        The FunctionRecord instance to enrich with coverage data; its test_coverage attributes are updated in place.
    executes : set[int]
        Set of line numbers executed during test runs, used to determine covered lines for the function.
    missing_items : set[int]
        Set of line numbers that were not executed by any test, used to determine uncovered lines and the coverage ratio.
    failing_tests : list[str]
        A list of test identifiers for tests that are currently failing; used to determine whether this function is affected by a
        failing test via a name-based heuristic.

    """
    covered, uncovered, ratio = _compute_function_ratio(fn, executes, missing_items)

    has_any_test = len(covered) > 0

    # A test is "failing for this function" when its node id contains the function
    # name or the module is involved. This is a simple name-based heuristic.
    has_failing_test = has_any_test and _function_affected_by_failure(fn, failing_tests)

    if fn.test_coverage is None:
        fn.test_coverage = TestCoverage()

    fn.test_coverage.covered_lines = covered
    fn.test_coverage.uncovered_lines = uncovered
    fn.test_coverage.ratio = ratio
    fn.test_coverage.has_any_test = has_any_test
    fn.test_coverage.has_failing_test = has_failing_test

    if has_any_test:
        fn.test_coverage_confidence = Confidence.INFERRED_HIGH
    elif fn.test_coverage_confidence == Confidence.ABSENT:
        fn.test_coverage_confidence = Confidence.ABSENT


def _function_affected_by_failure(
    fn: FunctionRecord,
    failing_tests: list[str],
) -> bool:
    """Return True when a failing test may concern this function.

    Criterion: the failing test node id contains the function's simple name or
    the source module's file name.

    Parameters
    ----------
    fn : FunctionRecord
        The function record whose simple name or source module file name is matched against failing test node IDs.
    failing_tests : list[str]
        List of test node identifiers for tests that are currently failing, used to determine whether the function is related to
        any of those failures.

    Returns
    -------
    bool
        True if any failing test node id contains the function's simple name or the source module's file name; False otherwise.

    """
    if not failing_tests:
        return False

    file_name = fn.file_path.stem

    return any(name_simple(fn.qualified_name) in node_id or file_name in node_id for node_id in failing_tests)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def map_coverage(
    outcome: ExecutionResult,
    modules: list[ModuleRecord],
) -> None:
    """Enrich FunctionRecords with dynamic coverage data.

    Use the coverage.py JSON report in ``outcome.coverage_json`` to populate
    ``TestCoverage`` dynamic fields (covered_lines, uncovered_lines, ratio,
    has_any_test, has_failing_test) on each production-module FunctionRecord.

    Return without modifying FunctionRecords when the JSON report is absent
    (tests not run or coverage disabled).

    Parameters
    ----------
    outcome : ExecutionResult
        Test execution result (from execute_tests()).
    modules : list[ModuleRecord]
        Complete list of project modules.

    """
    if outcome.coverage_json is None:
        diagnostic = outcome.coverage_diagnostic
        if diagnostic is not None and "timeout" in diagnostic:
            logger.warning("coverage_mapper - coverage lost due to timeout: %s.", diagnostic)
        else:
            logger.debug(
                "coverage_mapper - no JSON report available%s.",
                f" ({diagnostic})" if diagnostic else " (unknown cause or coverage disabled)",
            )
        return

    lines_by_file = _lines_by_file(outcome.coverage_json)

    for module in modules:
        if module.is_test or module.is_stub:
            continue

        source_path = Path(module.file_path).resolve()
        if source_path not in lines_by_file:
            logger.debug("coverage_mapper - file absent from report: %s", source_path)
            continue

        executes, missing_items = lines_by_file[source_path]

        for fn in iter_functions([module]):
            _enrich_function(fn, executes, missing_items, outcome.failing_tests)


# ---------------------------------------------------------------------------
# Enrich from collected calls (it.9 phase 2)
# ---------------------------------------------------------------------------


def link_call_functions(raw_calls: list[dict], modules: list[ModuleRecord]) -> None:
    """Enrich production FunctionRecords with dynamically collected TestCalls.

    Map each raw entry (a dict from the plugin's JSON output) to its FunctionRecord
    through qualified_name, then append it to TestCoverage.calls.
    Mutate FunctionRecords in place.

    Parameters
    ----------
    raw_calls : list[dict]
        List of dicts from the appels.json file produced by call_collector.
    modules : list[ModuleRecord]
        Complete list of project modules (tests and production combined).

    """
    # Index qualified_name -> FunctionRecord (production only).
    modules_prod = [m for m in modules if not m.is_test and not m.is_stub]
    index = {fn.qualified_name: fn for fn in iter_functions(modules_prod)}

    for raw_call in raw_calls:
        try:
            invocation = TestCall(
                function_name=raw_call["function_name"],
                test_source=raw_call["test_source"],
                args=raw_call.get("args", {}),
                result=raw_call.get("result"),
                exception=raw_call.get("exception"),
            )
        except (KeyError, TypeError):
            logger.warning("link_call_functions - invalid entry ignored: %s", raw_call)
            continue

        fn = index.get(invocation.function_name)
        if fn is None:
            logger.debug("link_call_functions - unknown function: %s", invocation.function_name)
            continue

        if fn.test_coverage is None:
            fn.test_coverage = TestCoverage()

        fn.test_coverage.calls.append(invocation)
