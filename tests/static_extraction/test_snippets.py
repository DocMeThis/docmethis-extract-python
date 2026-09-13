# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for snippets.py - generating UsageExample from TestCall (iteration 9, phase 3)."""

from __future__ import annotations

from pathlib import Path

from docmethis_extract_python.static_extraction.dynamic_analysis.snippets import (
    _build_example,
    _choose_format,
    _detect_contradictions,
    _format_result,
    _generate_call,
    _generate_setup,
    _is_eligible_call,
    _is_simple_value,
    generate_examples_from_calls,
)
from docmethis_extract_python.static_extraction.models import (
    Confidence,
    FunctionRecord,
    MethodType,
    ModuleRecord,
    UsageExample,
    Visibility,
)
from docmethis_extract_python.static_extraction.models import (
    TestCall as _TestCall,
)
from docmethis_extract_python.static_extraction.models import (
    TestCoverage as _TestCoverage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fn(
    qname: str = "pkg.mod.function",
    parent_module: str = "pkg.mod",
    parent_class: str | None = None,
    file_path: Path = Path("/proj/pkg/mod.py"),
    calls: list[_TestCall] | None = None,
) -> FunctionRecord:
    fn = FunctionRecord(
        qualified_name=qname,
        file_path=file_path,
        line_start=1,
        line_end=10,
        col_start=0,
        col_end=0,
        method_kind=MethodType.FUNCTION,
        visibility=Visibility.PUBLIC,
        parent_class=parent_class,
        parent_module=parent_module,
        existing_docstring=None,
    )
    if calls:
        fn.test_coverage = _TestCoverage(calls=calls)
    return fn


def _module(
    *,
    functions: list[FunctionRecord] | None = None,
    is_test: bool = False,
) -> ModuleRecord:
    return ModuleRecord(
        file_path=Path("/proj/pkg/mod.py"),
        module_name="pkg.mod",
        is_test=is_test,
        functions=functions or [],
    )


def _call(
    args: dict | None = None,
    result: object = 42,
    exception: str | None = None,
    test_source: str = "tests/test_mod.py::test_function",
) -> _TestCall:
    return _TestCall(
        function_name="pkg.mod.function",
        test_source=test_source,
        args=args if args is not None else {"x": 1, "y": 2},
        result=result,
        exception=exception,
    )


# ---------------------------------------------------------------------------
# _is_simple_value
# ---------------------------------------------------------------------------


def test_simple_value_int() -> None:
    assert _is_simple_value(42) is True


def test_simple_value_float() -> None:
    assert _is_simple_value(3.14) is True


def test_simple_value_bool() -> None:
    assert _is_simple_value(item_value=True) is True


def test_simple_value_none() -> None:
    assert _is_simple_value(None) is True


def test_simple_value_ordinary_string() -> None:
    assert _is_simple_value("hello") is True


def test_simple_value_string_with_parentheses() -> None:
    """A string containing '(' indicates a repr of a complex object."""
    assert _is_simple_value("MyObj(x=1)") is False


def test_simple_value_non_serializable_string() -> None:
    assert _is_simple_value("<non_serializable>") is False


def test_simple_value_simple_list() -> None:
    assert _is_simple_value([1, 2, 3]) is True


def test_simple_value_complex_list() -> None:
    assert _is_simple_value([1, "MyObj()", 3]) is False


def test_simple_value_simple_dict() -> None:
    assert _is_simple_value({"a": 1}) is True


def test_simple_value_complex_dict() -> None:
    assert _is_simple_value({"a": "MyObj()"}) is False


# ---------------------------------------------------------------------------
# _is_eligible_call
# ---------------------------------------------------------------------------


def test_eligible_call_nominal_case() -> None:
    assert _is_eligible_call(_call(), exclude_failed_tests=[]) is True


def test_eligible_call_failed_test() -> None:
    invocation = _call(test_source="tests/t.py::test_x")
    assert _is_eligible_call(invocation, exclude_failed_tests=["tests/t.py::test_x"]) is False


def test_eligible_call_raised_exception() -> None:
    assert _is_eligible_call(_call(exception="ValueError"), exclude_failed_tests=[]) is False


def test_eligible_call_non_serializable_result() -> None:
    assert _is_eligible_call(_call(result="<non_serializable>"), exclude_failed_tests=[]) is False


def test_eligible_call_complex_argument() -> None:
    assert _is_eligible_call(_call(args={"obj": "MyFixture()"}), exclude_failed_tests=[]) is False


def test_eligible_call_without_args() -> None:
    """A call without arguments is eligible."""
    assert _is_eligible_call(_call(args={}), exclude_failed_tests=[]) is True


# ---------------------------------------------------------------------------
# _generate_setup
# ---------------------------------------------------------------------------


def test_generate_setup_function_free() -> None:
    fn = _fn(qname="pkg.mod.function", parent_module="pkg.mod")
    assert _generate_setup(fn) == "from pkg.mod import function"


def test_generate_setup_method() -> None:
    """For a method, import the class rather than the method."""
    fn = _fn(qname="pkg.mod.Class.method", parent_module="pkg.mod", parent_class="Class")
    assert _generate_setup(fn) == "from pkg.mod import Class"


def test_generate_setup_deep_module() -> None:
    fn = _fn(qname="a.b.c.fn", parent_module="a.b.c")
    assert _generate_setup(fn) == "from a.b.c import fn"


# ---------------------------------------------------------------------------
# _generate_call
# ---------------------------------------------------------------------------


def test_generate_call_int_args() -> None:
    assert _generate_call("add", {"a": 1, "b": 2}) == "add(1, 2)"


def test_generate_call_str_args() -> None:
    assert _generate_call("greet", {"name": "Alice"}) == "greet('Alice')"


def test_generate_call_without_args() -> None:
    assert _generate_call("do_something", {}) == "do_something()"


def test_generate_call_none_arg() -> None:
    assert _generate_call("fn", {"x": None}) == "fn(None)"


# ---------------------------------------------------------------------------
# _format_result
# ---------------------------------------------------------------------------


def test_format_result_int() -> None:
    assert _format_result(3) == "3"


def test_format_result_str() -> None:
    assert _format_result("hello") == "'hello'"


def test_format_result_none() -> None:
    assert _format_result(None) == "None"


def test_format_result_list() -> None:
    assert _format_result([1, 2]) == "[1, 2]"


# ---------------------------------------------------------------------------
# _choose_format
# ---------------------------------------------------------------------------


def test_choose_format_simple() -> None:
    assert _choose_format("3") == "doctest"


def test_choose_format_multiline() -> None:
    assert _choose_format("line1\nline2") == "raw"


# ---------------------------------------------------------------------------
# _build_example
# ---------------------------------------------------------------------------


def test_build_example_nominal_case() -> None:
    """An eligible call produces a valid UsageExample."""
    fn = _fn()
    invocation = _call(args={"a": 1, "b": 2}, result=3)
    ex = _build_example(invocation, fn, exclude_failed_tests=[])
    assert ex is not None
    assert ex.call == "function(1, 2)"
    assert ex.result == "3"
    assert ex.setup == "from pkg.mod import function"
    assert ex.provenance == "test"
    assert ex.confidence == "high"
    assert ex.format == "doctest"
    assert ex.contradiction_suspected is False


def test_build_example_failed_test() -> None:
    fn = _fn()
    invocation = _call(test_source="tests/t.py::test_x")
    assert _build_example(invocation, fn, exclude_failed_tests=["tests/t.py::test_x"]) is None


def test_build_example_exception() -> None:
    fn = _fn()
    assert _build_example(_call(exception="ValueError"), fn, exclude_failed_tests=[]) is None


def test_build_example_complex_argument() -> None:
    fn = _fn()
    assert _build_example(_call(args={"obj": "MyFixture()"}), fn, exclude_failed_tests=[]) is None


def test_build_example_non_serializable_result() -> None:
    fn = _fn()
    assert _build_example(_call(result="<non_serializable>"), fn, exclude_failed_tests=[]) is None


def test_build_example_valid_none_result() -> None:
    """An explicit None result, not an exception, is a valid example."""
    fn = _fn()
    ex = _build_example(_call(args={}, result=None), fn, exclude_failed_tests=[])
    assert ex is not None
    assert ex.result == "None"


# ---------------------------------------------------------------------------
# _detect_contradictions
# ---------------------------------------------------------------------------


def test_detect_contradictions_calls_identical_different_results() -> None:
    """Two examples with the same call but different results set contradiction_suspected=True."""
    _ = _fn()
    ex1 = UsageExample("f", "t1", "s", "fn(1)", "1", "test", "high", "doctest")
    ex2 = UsageExample("f", "t2", "s", "fn(1)", "2", "test", "high", "doctest")
    _detect_contradictions([ex1, ex2])
    assert ex1.contradiction_suspected is True
    assert ex2.contradiction_suspected is True


def test_detect_contradictions_calls_identical_same_result() -> None:
    """The same call and result do not create a contradiction."""
    ex1 = UsageExample("f", "t1", "s", "fn(1)", "1", "test", "high", "doctest")
    ex2 = UsageExample("f", "t2", "s", "fn(1)", "1", "test", "high", "doctest")
    _detect_contradictions([ex1, ex2])
    assert ex1.contradiction_suspected is False
    assert ex2.contradiction_suspected is False


def test_detect_contradictions_calls_differents() -> None:
    """Different calls do not create a contradiction even with different results."""
    ex1 = UsageExample("f", "t1", "s", "fn(1)", "1", "test", "high", "doctest")
    ex2 = UsageExample("f", "t2", "s", "fn(2)", "2", "test", "high", "doctest")
    _detect_contradictions([ex1, ex2])
    assert ex1.contradiction_suspected is False
    assert ex2.contradiction_suspected is False


# ---------------------------------------------------------------------------
# generate_examples_from_calls - integration
# ---------------------------------------------------------------------------


def test_generate_examples_from_calls_nominal_case() -> None:
    """A FunctionRecord with eligible TestCalls receives examples."""
    invocation = _call(args={"a": 1, "b": 2}, result=3)
    fn = _fn(calls=[invocation])
    mod = _module(functions=[fn])
    generate_examples_from_calls([mod], exclude_failed_tests=[])
    assert fn.examples is not None
    assert len(fn.examples) == 1
    assert fn.examples[0].call == "function(1, 2)"
    assert fn.examples_confidence == Confidence.INFERRED_HIGH


def test_generate_examples_from_calls_without_calls() -> None:
    """A FunctionRecord without TestCalls keeps examples=None."""
    fn = _fn()
    mod = _module(functions=[fn])
    generate_examples_from_calls([mod], exclude_failed_tests=[])
    assert fn.examples is None
    assert fn.examples_confidence == Confidence.ABSENT


def test_generate_examples_from_calls_all_filtered() -> None:
    """When all TestCalls are filtered, examples remains None."""
    invocation = _call(exception="ValueError")
    fn = _fn(calls=[invocation])
    mod = _module(functions=[fn])
    generate_examples_from_calls([mod], exclude_failed_tests=[])
    assert fn.examples is None


def test_generate_examples_from_calls_ignores_test_modules() -> None:
    """Modules with is_test=True are ignored."""
    invocation = _call(args={"a": 1}, result=1)
    fn = _fn(calls=[invocation])
    mod = _module(is_test=True, functions=[fn])
    generate_examples_from_calls([mod], exclude_failed_tests=[])
    assert fn.examples is None


def test_generate_examples_from_calls_detects_contradiction() -> None:
    """Two calls with the same call and different results are marked."""
    call_one = _call(args={"x": 1}, result=1, test_source="t::test_a")
    call_two = _call(args={"x": 1}, result=2, test_source="t::test_b")
    fn = _fn(calls=[call_one, call_two])
    mod = _module(functions=[fn])
    generate_examples_from_calls([mod], exclude_failed_tests=[])
    assert fn.examples is not None
    assert all(ex.contradiction_suspected for ex in fn.examples)


def test_generate_examples_from_calls_excludes_failed_tests() -> None:
    """Calls from failed tests are excluded."""
    invocation = _call(test_source="tests/t.py::test_failure")
    fn = _fn(calls=[invocation])
    mod = _module(functions=[fn])
    generate_examples_from_calls([mod], exclude_failed_tests=["tests/t.py::test_failure"])
    assert fn.examples is None
