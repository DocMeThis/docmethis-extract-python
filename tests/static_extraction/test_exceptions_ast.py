# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Unit tests for exception extraction: raises and except blocks."""

from __future__ import annotations

import ast
import textwrap

from docmethis_extract_python.static_extraction.exceptions_ast import extract_exceptions

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _parse_func(src: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Return the first FunctionDef or AsyncFunctionDef found in src."""
    tree = ast.parse(textwrap.dedent(src))
    ast.fix_missing_locations(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return node
    msg = "No function found"
    raise ValueError(msg)


# Tests for raise X("msg").


def test_raise_simple_with_message() -> None:
    src = """
    def f(x):
        raise ValueError("x must be positive")
    """
    excs, blocs = extract_exceptions(_parse_func(src))
    assert len(excs) == 1
    assert excs[0].exception_type == "ValueError"
    assert excs[0].message == "x must be positive"
    assert excs[0].is_reraise is False
    assert excs[0].chained_from is False
    assert excs[0].condition is None
    assert blocs == []


def test_raise_without_message() -> None:
    src = """
    def f():
        raise TypeError
    """
    excs, _ = extract_exceptions(_parse_func(src))
    assert len(excs) == 1
    assert excs[0].exception_type == "TypeError"
    assert excs[0].message is None


def test_raise_without_argument() -> None:
    """A bare raise outside except is treated as a reraise without crashing."""
    src = """
    def f():
        raise
    """
    excs, _ = extract_exceptions(_parse_func(src))
    assert len(excs) == 1
    assert excs[0].is_reraise is True
    assert excs[0].exception_type == "<unknown>"


# Tests for raise ... from e.


def test_raise_chained() -> None:
    src = """
    def f():
        try:
            pass
        except OSError as e:
            raise RuntimeError("wrapped") from e
    """
    excs, _ = extract_exceptions(_parse_func(src))
    assert any(e.chained_from for e in excs)
    chained = next(e for e in excs if e.chained_from)
    assert chained.exception_type == "RuntimeError"


# Tests for reraise inside except.


def test_reraise_inside_except() -> None:
    src = """
    def f():
        try:
            pass
        except ValueError:
            raise
    """
    excs, blocs = extract_exceptions(_parse_func(src))
    reraises = [e for e in excs if e.is_reraise]
    assert len(reraises) == 1
    assert reraises[0].exception_type == "ValueError"

    assert len(blocs) == 1
    assert blocs[0].is_reraised is True
    assert blocs[0].is_swallowed is False


# Tests for raised conditions.


def test_raised_condition_direct_guard() -> None:
    src = """
    def f(x):
        if x < 0:
            raise ValueError("negative")
    """
    excs, _ = extract_exceptions(_parse_func(src))
    assert len(excs) == 1
    assert excs[0].condition is not None
    assert "x < 0" in excs[0].condition.expression


def test_condition_for_multi_statement_guard() -> None:
    """A direct guard keeps its condition when setup precedes the raise."""
    src = """
    def f(x):
        if x < 0:
            x = abs(x)
            raise ValueError("negative")
    """
    excs, _ = extract_exceptions(_parse_func(src))
    assert len(excs) == 1
    assert excs[0].condition is not None
    assert excs[0].condition.expression == "x < 0"


def test_nested_guard_conditions_are_combined() -> None:
    src = """
    def f(x, y):
        if x < 0:
            if y == 0:
                raise ValueError("invalid")
    """
    excs, _ = extract_exceptions(_parse_func(src))
    assert len(excs) == 1
    assert excs[0].condition is not None
    assert excs[0].condition.expression == "(x < 0) and (y == 0)"


def test_else_guard_condition_is_negated() -> None:
    src = """
    def f(x):
        if x >= 0:
            return x
        else:
            raise ValueError("negative")
    """
    excs, _ = extract_exceptions(_parse_func(src))
    assert len(excs) == 1
    assert excs[0].condition is not None
    assert excs[0].condition.expression == "not (x >= 0)"


def test_no_condition_is_inferred_through_loop_or_try() -> None:
    src = """
    def f(items, enabled):
        if enabled:
            for item in items:
                if item < 0:
                    raise ValueError("negative")
        try:
            raise KeyError("missing")
        except KeyError:
            raise RuntimeError("wrapped")
    """
    excs, _ = extract_exceptions(_parse_func(src))
    assert {exc.exception_type for exc in excs} == {"ValueError", "KeyError", "RuntimeError"}
    assert all(exc.condition is None for exc in excs)


def test_dynamic_or_partial_messages_are_not_reported() -> None:
    src = """
    def f(value):
        raise ValueError(f"invalid: {value}")
        raise TypeError("invalid", value)
    """
    excs, _ = extract_exceptions(_parse_func(src))
    assert [exc.message for exc in excs] == [None, None]


# Tests for except blocks - swallowed.


def test_except_swallowed_pass() -> None:
    src = """
    def f():
        try:
            pass
        except Exception:
            pass
    """
    _, blocs = extract_exceptions(_parse_func(src))
    assert len(blocs) == 1
    assert blocs[0].is_swallowed is True
    assert blocs[0].is_reraised is False


def test_except_swallowed_ellipsis() -> None:
    src = """
    def f():
        try:
            pass
        except Exception:
            ...
    """
    _, blocs = extract_exceptions(_parse_func(src))
    assert len(blocs) == 1
    assert blocs[0].is_swallowed is True


def test_except_not_swallowed_when_reraised() -> None:
    src = """
    def f():
        try:
            pass
        except Exception:
            raise
    """
    _, blocs = extract_exceptions(_parse_func(src))
    assert blocs[0].is_swallowed is False
    assert blocs[0].is_reraised is True


# Tests for caught types.


def test_multiple_caught_types() -> None:
    src = """
    def f():
        try:
            pass
        except (ValueError, TypeError):
            pass
    """
    _, blocs = extract_exceptions(_parse_func(src))
    assert set(blocs[0].caught_types) == {"ValueError", "TypeError"}


def test_bare_except() -> None:
    src = """
    def f():
        try:
            pass
        except:
            pass
    """
    _, blocs = extract_exceptions(_parse_func(src))
    assert blocs[0].caught_types == []
    assert blocs[0].is_swallowed is True


# Tests for nested functions - isolation.


def test_raise_in_nested_function_is_ignored() -> None:
    src = """
    def f():
        def inner():
            raise ValueError("inside inner")
    """
    excs, _ = extract_exceptions(_parse_func(src))
    assert excs == []


# Tests for a function without raise or except.


def test_function_empty() -> None:
    src = """
    def f():
        return 42
    """
    excs, blocs = extract_exceptions(_parse_func(src))
    assert excs == []
    assert blocs == []


# Tests for a populated raise_line.


def test_raise_line_is_populated() -> None:
    src = """\
def f(x):
    if x < 0:
        raise ValueError("neg")
"""
    excs, _ = extract_exceptions(_parse_func(src))
    assert excs[0].raise_line == 3
