# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Generate example snippets from collected TestCalls (it.9 phase 3).

Transform TestCalls (phase 2) into UsageExamples (doctest or raw format) after
quality filtering: simple arguments, serializable result, and passing test.
Detect contradictions (same call, different results).
"""

from __future__ import annotations

import logging
from typing import Any

from docmethis_extract_python.static_extraction.models import (
    Confidence,
    FunctionRecord,
    ModuleRecord,
    TestCall,
    UsageExample,
    iter_functions,
    name_simple,
)

__all__ = ["generate_examples_from_calls"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------


def _is_simple_value(item_value: object) -> bool:
    """Return True when a value is a simple literal usable in a snippet.

    A value is simple when it is a JSON primitive (int, float, bool, None), a string without parentheses/angle brackets (which
    suggest a complex object repr()), or a homogeneous collection of simple values.

    Parameters
    ----------
    item_value : object
        The value to check for simple-literal eligibility.

    Returns
    -------
    bool
        True if the value is a simple literal usable in a snippet; False otherwise.

    """
    if isinstance(item_value, bool | int | float | type(None)):
        return True

    if isinstance(item_value, str):
        # Exclude object reprs ("MyObj(x=1)") and sentinels ("<non_serializable>").
        return not item_value.startswith("<") and "(" not in item_value

    if isinstance(item_value, list):
        return all(_is_simple_value(v) for v in item_value)

    if isinstance(item_value, dict):
        return all(_is_simple_value(k) and _is_simple_value(v) for k, v in item_value.items())

    return False


def _is_eligible_call(invocation: TestCall, exclude_failed_tests: list[str]) -> bool:
    """Return True when a call can produce a quality example.

    Conditions:
    - The source test did not fail.
    - The function did not raise an exception.
    - The result is serializable.
    - All arguments are simple literals.

    Parameters
    ----------
    invocation : TestCall
        The invocation to evaluate for eligibility, containing the test source, exception, result, and arguments.
    exclude_failed_tests : list[str]
        List of test source identifiers for tests that should be excluded from eligibility because they are considered failed.

    Returns
    -------
    bool
        True if the invocation is eligible to produce a quality example; otherwise, False.

    """
    if invocation.test_source in exclude_failed_tests:
        return False

    if invocation.exception is not None:
        return False

    if invocation.result == "<non_serializable>":
        return False

    return all(_is_simple_value(v) for v in invocation.args.values())


# ---------------------------------------------------------------------------
# Snippet-part constructors
# ---------------------------------------------------------------------------


def _generate_setup(fn: FunctionRecord) -> str:
    """Return the minimal import needed to reproduce the example in isolation.

    Parameters
    ----------
    fn : FunctionRecord
        The function record for which the minimal import statement is generated.

    Returns
    -------
    str
        The minimal import statement needed to reproduce the example in isolation.

    """
    if fn.parent_class:
        return f"from {fn.parent_module} import {fn.parent_class}"
    name_short = name_simple(fn.qualified_name)
    return f"from {fn.parent_module} import {name_short}"


def _generate_call(name_short: str, args: dict[str, Any]) -> str:
    """Return the call expression as valid Python source.

    Parameters
    ----------
    name_short : str
        The short name of the callable to appear as the function name in the generated call expression.
    args : dict[str, Any]
        A dictionary mapping parameter names to their values, used to construct the call expression.

    Returns
    -------
    str
        The call expression as valid Python source.

    """
    return f"{name_short}({', '.join(repr(v) for v in args.values())})"


def _format_result(result: object) -> str:
    """Return the return value as valid Python source.

    Parameters
    ----------
    result : object
        The object to be formatted as a valid Python source representation.

    Returns
    -------
    str
        A string containing the Python source representation of the input result, produced by repr().

    """
    return repr(result)


def _choose_format(result_str: str) -> str:
    """Return 'doctest' when the result fits on one line, otherwise 'raw'.

    Parameters
    ----------
    result_str : str
        The string representation of the result to inspect for line breaks.

    Returns
    -------
    str
        Returns 'doctest' when the result string contains no newline character; otherwise returns 'raw'.

    """
    return "doctest" if "\n" not in result_str else "raw"


# ---------------------------------------------------------------------------
# Build a UsageExample
# ---------------------------------------------------------------------------


def _build_example(invocation: TestCall, fn: FunctionRecord, exclude_failed_tests: list[str]) -> UsageExample | None:
    """Try to build a UsageExample from a TestCall.

    Return None when the call fails the quality filter.

    Parameters
    ----------
    invocation : TestCall
        The TestCall to convert into a UsageExample.
    fn : FunctionRecord
        The FunctionRecord of the target function for which the usage example is being built.
    exclude_failed_tests : list[str]
        A list of test names to exclude from example generation; invocations from these tests are ignored when building the
        example.

    Returns
    -------
    UsageExample | None
        Returns a UsageExample built from the given TestCall if the call passes the quality filter; otherwise, returns None.

    """
    if not _is_eligible_call(invocation, exclude_failed_tests):
        return None

    name_short = name_simple(fn.qualified_name)
    setup = _generate_setup(fn)
    call = _generate_call(name_short, invocation.args)
    result_str = _format_result(invocation.result)
    fmt = _choose_format(result_str)

    return UsageExample(
        function=fn.qualified_name,
        test_source=invocation.test_source,
        setup=setup,
        call=call,
        result=result_str,
        provenance="test",
        confidence="high",
        format=fmt,
        contradiction_suspected=False,
    )


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------


def _detect_contradictions(examples: list[UsageExample]) -> None:
    """Mark contradiction_suspected=True when identical calls have different results.

    Parameters
    ----------
    examples : list[UsageExample]
        A list of UsageExample objects representing usage snippets to analyze. The function inspects each example's call and
        result attributes; when multiple examples share the same call but produce different results, the corresponding examples
        are marked as contradictory.

    """
    by_call: dict[str, set[str]] = {}
    for example in examples:
        by_call.setdefault(example.call, set()).add(example.result)

    for example in examples:
        if len(by_call.get(example.call, set())) > 1:
            example.contradiction_suspected = True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _process_function(fn: FunctionRecord, exclude_failed_tests: list[str]) -> None:
    """Processes a function's test coverage to generate usage examples, filtering failed tests and detecting contradictions.

    Parameters
    ----------
    fn : FunctionRecord
        The function record to process, whose test coverage calls are used to generate usage examples and whose examples are
        updated in place.
    exclude_failed_tests : list[str]
        A list of test names or identifiers for failed tests that should be excluded when building usage examples from test
        coverage.

    """
    if fn.test_coverage is None or not fn.test_coverage.calls:
        return

    examples: list[UsageExample] = []
    for invocation in fn.test_coverage.calls:
        example = _build_example(invocation, fn, exclude_failed_tests)
        if example is not None:
            examples.append(example)

    if not examples:
        return

    _detect_contradictions(examples)
    fn.examples = examples
    fn.examples_confidence = Confidence.INFERRED_HIGH
    logger.debug("snippets - generated %d example(s) for %s", len(examples), fn.qualified_name)


def generate_examples_from_calls(
    modules: list[ModuleRecord],
    exclude_failed_tests: list[str],
) -> None:
    """Generate UsageExamples from TestCalls and attach them to FunctionRecords.

    For each production FunctionRecord with TestCalls, apply the quality filter
    and generate doctest or raw UsageExamples. Detect contradictions between
    examples for the same function. Mutate FunctionRecords in place.

    Parameters
    ----------
    modules : list[ModuleRecord]
        Complete list of project modules.
    exclude_failed_tests : list[str]
        Node ids of failed tests (from ExecutionResult.failing_tests).
        Calls from those tests are excluded.

    """
    modules_prod = [m for m in modules if not m.is_test and not m.is_stub]
    for fn in iter_functions(modules_prod):
        _process_function(fn, exclude_failed_tests)
