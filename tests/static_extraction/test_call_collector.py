# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for call_collector (iteration 9, phase 2) and coverage linking."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from docmethis_extract_python.static_extraction.dynamic_analysis.call_collector import (
    _build_args_dict,
    _is_test_module,
    _serialize_value,
)
from docmethis_extract_python.static_extraction.dynamic_analysis.coverage import link_call_functions
from docmethis_extract_python.static_extraction.models import (
    FunctionRecord,
    MethodType,
    ModuleRecord,
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
    file_path: Path = Path("/proj/pkg/mod.py"),
    line_start: int = 1,
    line_end: int = 10,
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
) -> ModuleRecord:
    return ModuleRecord(
        file_path=file_path,
        module_name=module_name,
        is_test=is_test,
        functions=functions or [],
    )


def _raw_call(
    function_name: str = "pkg.mod.function",
    test_source: str = "tests/test_mod.py::test_function",
    args: dict | None = None,
    result: object = 42,
    exception: str | None = None,
) -> dict:
    return {
        "function_name": function_name,
        "test_source": test_source,
        "args": args or {"x": 1},
        "result": result,
        "exception": exception,
    }


# ---------------------------------------------------------------------------
# _serialize_value
# ---------------------------------------------------------------------------


def test_serialize_value_int() -> None:
    """An integer is returned directly."""
    assert _serialize_value(42) == 42


def test_serialize_value_string() -> None:
    """A string is returned directly."""
    assert _serialize_value("hello") == "hello"


def test_serialize_value_simple_list() -> None:
    """A JSON-compatible list is returned directly."""
    assert _serialize_value([1, 2, 3]) == [1, 2, 3]


def test_serialize_value_dict() -> None:
    """A JSON-compatible dict is returned directly."""
    assert _serialize_value({"a": 1}) == {"a": 1}


def test_serialize_value_non_serializable_returns_repr() -> None:
    """A non-JSON-serializable object returns its repr()."""

    class _Object:
        def __repr__(self) -> str:
            return "Object()"

    outcome = _serialize_value(_Object())
    assert outcome == "Object()"


def test_serialize_value_none() -> None:
    """None is returned directly (JSON null)."""
    assert _serialize_value(None) is None


# ---------------------------------------------------------------------------
# _build_args_dict
# ---------------------------------------------------------------------------


def test_build_args_dict_positional() -> None:
    """Positional arguments are mapped to parameter names."""

    def function(x: int, y: int) -> int:
        return x + y

    outcome = _build_args_dict(function, (1, 2), {})
    assert outcome == {"x": 1, "y": 2}


def test_build_args_dict_keyword_args() -> None:
    """Keyword arguments are included."""

    def function(x: int, y: int = 0) -> int:
        return x + y

    outcome = _build_args_dict(function, (1,), {"y": 99})
    assert outcome == {"x": 1, "y": 99}


def test_build_args_dict_more_args_than_parameters() -> None:
    """When there are more args than parameters, fall back to 'arg{i}'."""

    def function() -> None:
        pass

    outcome = _build_args_dict(function, (1, 2), {})
    assert "arg0" in outcome
    assert "arg1" in outcome


def test_build_args_dict_non_serializable_value() -> None:
    """A non-serializable value is converted to repr or '<non_serializable>'."""

    class _Opaque:
        def __repr__(self) -> str:
            return "Opaque()"

    def function(obj: object) -> None:
        pass

    outcome = _build_args_dict(function, (_Opaque(),), {})
    assert outcome["obj"] == "Opaque()"


# ---------------------------------------------------------------------------
# _is_test_module
# ---------------------------------------------------------------------------


def test_is_test_module_test_prefix() -> None:
    assert _is_test_module("myproject.tests.test_foo", "/proj/tests/test_foo.py") is True


def test_is_test_module_test_suffix() -> None:
    assert _is_test_module("myproject.foo_test", "/proj/foo_test.py") is True


def test_is_test_module_conftest() -> None:
    assert _is_test_module("conftest", "/proj/conftest.py") is True


def test_is_test_module_production() -> None:
    assert _is_test_module("myproject.utils", "/proj/myproject/utils.py") is False


# ---------------------------------------------------------------------------
# link_call_functions
# ---------------------------------------------------------------------------


def test_link_call_functions_enriches_function_record() -> None:
    """A call matching a FunctionRecord is added to its calls."""
    fn = _fn(qname="pkg.mod.function")
    mod = _module(functions=[fn])
    raw_calls = [_raw_call(function_name="pkg.mod.function", result=42)]
    link_call_functions(raw_calls, [mod])
    assert fn.test_coverage is not None
    assert len(fn.test_coverage.calls) == 1
    invocation = fn.test_coverage.calls[0]
    assert invocation.function_name == "pkg.mod.function"
    assert invocation.result == 42


def test_link_call_functions_ignores_unknown_qname() -> None:
    """A call with an unknown function_name is ignored without raising."""
    fn = _fn(qname="pkg.mod.other_function")
    mod = _module(functions=[fn])
    raw_calls = [_raw_call(function_name="pkg.mod.missing")]
    link_call_functions(raw_calls, [mod])
    assert fn.test_coverage is None


def test_link_call_functions_creates_missing_test_coverage() -> None:
    """When test_coverage is None, create it before adding the call."""
    fn = _fn(qname="pkg.mod.function")
    assert fn.test_coverage is None
    mod = _module(functions=[fn])
    link_call_functions([_raw_call()], [mod])
    assert fn.test_coverage is not None


def test_link_call_functions_excludes_test_modules() -> None:
    """Modules with is_test=True are excluded from the index."""
    fn = _fn(qname="pkg.mod.function")
    mod = _module(is_test=True, functions=[fn])
    link_call_functions([_raw_call()], [mod])
    assert fn.test_coverage is None


def test_link_call_functions_adds_multiple_calls() -> None:
    """Multiple calls to the same function are all added."""
    fn = _fn(qname="pkg.mod.function")
    mod = _module(functions=[fn])
    bruts = [
        _raw_call(test_source="tests/t.py::test_a", result=1),
        _raw_call(test_source="tests/t.py::test_b", result=2),
    ]
    link_call_functions(bruts, [mod])
    assert len(fn.test_coverage.calls) == 2


def test_link_call_functions_records_exception() -> None:
    """A call with an exception is recorded correctly."""
    fn = _fn(qname="pkg.mod.function")
    mod = _module(functions=[fn])
    raw_call = _raw_call(result=None, exception="ValueError")
    link_call_functions([raw_call], [mod])
    assert fn.test_coverage.calls[0].exception == "ValueError"


def test_link_call_functions_ignores_invalid_entry() -> None:
    """A raw entry without 'function_name' is ignored without raising."""
    fn = _fn(qname="pkg.mod.function")
    mod = _module(functions=[fn])
    link_call_functions([{"test_source": "test"}], [mod])  # no function_name
    assert fn.test_coverage is None


def test_link_call_functions_preserves_existing_calls() -> None:
    """Existing test_coverage.calls are not overwritten."""
    fn = _fn(qname="pkg.mod.function")
    fn.test_coverage = _TestCoverage(calls=[_TestCall("pkg.mod.function", "t::t_old", {}, 0, None)])
    mod = _module(functions=[fn])
    link_call_functions([_raw_call(test_source="t::t_new", result=99)], [mod])
    assert len(fn.test_coverage.calls) == 2


# ---------------------------------------------------------------------------
# Plugin integration test - subprocess execution on a small project.
# ---------------------------------------------------------------------------


def _create_mini_project(tmp_path: Path) -> None:
    """Create a minimal project with a function and a test that calls it."""
    src = tmp_path / "mypkg"
    src.mkdir()
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "calculator.py").write_text(
        textwrap.dedent("""\
            def add(a, b):
                return a + b
        """),
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_calculator.py").write_text(
        textwrap.dedent("""\
            from mypkg.calculator import add

            def test_add():
                assert add(1, 2) == 3
        """),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""\
            [tool.pytest.ini_options]
            testpaths = ["tests"]
        """),
        encoding="utf-8",
    )


def test_plugin_collects_direct_call(tmp_path: Path) -> None:
    """The plugin writes appels.json with the direct call from the test."""
    _create_mini_project(tmp_path)
    calls_file = tmp_path / "appels.json"

    env_extra = {
        "DOCMETHIS_APPELS_OUTPUT": str(calls_file),
        "DOCMETHIS_PROJECT_ROOT": str(tmp_path),
        "PYTHONPATH": str(tmp_path),
    }
    env = {**os.environ, **env_extra}

    proc = subprocess.run(  # noqa: PLW1510
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "docmethis_extract_python.static_extraction.dynamic_analysis.call_collector",
            "--tb=short",
            "-q",
            str(tmp_path / "tests"),
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        timeout=60,
    )

    assert calls_file.exists(), f"appels.json missing. stderr={proc.stderr}"
    calls = json.loads(calls_file.read_text(encoding="utf-8"))
    assert len(calls) >= 1
    invocation = calls[0]
    assert "add" in invocation["function_name"]
    assert invocation["result"] == 3
    assert invocation["exception"] is None
    assert invocation["args"] == {"a": 1, "b": 2}


def test_plugin_no_instrument(tmp_path: Path) -> None:
    """With DOCMETHIS_NO_INSTRUMENT=1, the calls file is not created."""
    _create_mini_project(tmp_path)
    calls_file = tmp_path / "appels.json"

    env = {
        **os.environ,
        "DOCMETHIS_APPELS_OUTPUT": str(calls_file),
        "DOCMETHIS_PROJECT_ROOT": str(tmp_path),
        "DOCMETHIS_NO_INSTRUMENT": "1",
        "PYTHONPATH": str(tmp_path),
    }

    subprocess.run(  # noqa: PLW1510
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "docmethis_extract_python.static_extraction.dynamic_analysis.call_collector",
            "-q",
            str(tmp_path / "tests"),
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        timeout=60,
    )

    assert not calls_file.exists(), "appels.json must not be created with --no-instrument"
