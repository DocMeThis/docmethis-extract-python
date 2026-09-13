# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Unit tests for observable properties: preconditions, postconditions, and invariants."""

from __future__ import annotations

import ast
import textwrap

from docmethis_extract_python.static_extraction.ast_properties import extract_observable_properties
from docmethis_extract_python.static_extraction.exceptions_ast import extract_exceptions
from docmethis_extract_python.static_extraction.models import Confidence

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _parse_func(src: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(textwrap.dedent(src))
    ast.fix_missing_locations(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return node
    msg = "No function found"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Preconditions - strict preamble.
# ---------------------------------------------------------------------------


def test_precondition_assert_explicit() -> None:
    src = """
    def f(x):
        assert x > 0
        return x * 2
    """
    props = extract_observable_properties(_parse_func(src))
    precs = [p for p in props if p.category == "precondition"]
    assert len(precs) == 1
    assert "x > 0" in precs[0].raw_expression
    assert precs[0].confidence == Confidence.EXPLICIT


def test_precondition_if_raise_guard() -> None:
    src = """
    def f(x):
        if x < 0:
            raise ValueError("negative")
        return x
    """
    props = extract_observable_properties(_parse_func(src))
    precs = [p for p in props if p.category == "precondition"]
    assert len(precs) == 1
    assert precs[0].confidence == Confidence.EXPLICIT
    assert precs[0].related_exception == "ValueError"
    assert "x < 0" in precs[0].raw_expression


def test_precondition_with_assignment_in_preamble() -> None:
    """Assignments in the preamble do not interrupt guard detection."""
    src = """
    def f(x, y):
        z = x + y
        assert z > 0
        return z
    """
    props = extract_observable_properties(_parse_func(src))
    precs = [p for p in props if p.category == "precondition"]
    assert len(precs) == 1
    assert precs[0].confidence == Confidence.EXPLICIT


def test_precondition_inferred_low_outside_preamble() -> None:
    """A guard after the strict preamble has INFERRED_LOW confidence.

    The preamble stops at the first statement that is neither a guard nor an assignment.
    A function call (ast.Expr) breaks the preamble; the following guard is INFERRED_LOW.
    """
    src = """
    def f(x):
        result = x * 2
        print(result)
        if x < 0:
            raise ValueError("negative")
        return result
    """
    props = extract_observable_properties(_parse_func(src))
    precs = [p for p in props if p.category == "precondition"]
    assert len(precs) == 1
    assert precs[0].confidence == Confidence.INFERRED_LOW


# ---------------------------------------------------------------------------
# related_exception link.
# ---------------------------------------------------------------------------


def test_related_exception_link() -> None:
    src = """
    def f(x):
        if not isinstance(x, int):
            raise TypeError("expected int")
        return x + 1
    """
    props = extract_observable_properties(_parse_func(src))
    precs = [p for p in props if p.category == "precondition"]
    assert precs[0].related_exception == "TypeError"


# ---------------------------------------------------------------------------
# Postconditions
# ---------------------------------------------------------------------------


def test_postcondition_assert_end() -> None:
    """An assert in the final statements and outside the preamble is a postcondition.

    The preamble stops after the initial assignment because a function call breaks the sequence.
    """
    src = """
    def f(x):
        result = x * 2
        do_side_effect(result)
        process(result)
        finalize(result)
        assert result > 0
        return result
    """
    props = extract_observable_properties(_parse_func(src))
    posts = [p for p in props if p.category == "postcondition"]
    assert len(posts) == 1
    assert posts[0].confidence == Confidence.EXPLICIT


def test_postcondition_if_raise_end() -> None:
    """An if-raise in the final statements and outside the preamble is a postcondition.

    The preamble stops at the first function call, which breaks the contiguous sequence.
    """
    src = """
    def f(x):
        result = compute(x)
        process(result)
        result2 = finalize(result)
        if result2 is None:
            raise RuntimeError("expected result")
        return result2
    """
    props = extract_observable_properties(_parse_func(src))
    posts = [p for p in props if p.category == "postcondition"]
    assert len(posts) == 1
    assert posts[0].confidence == Confidence.INFERRED_LOW
    assert posts[0].related_exception == "RuntimeError"


# ---------------------------------------------------------------------------
# Invariants.
# ---------------------------------------------------------------------------


def test_invariant_assert_in_middle() -> None:
    src = """
    def f(items):
        assert items
        for item in items:
            assert item is not None
        return items
    """
    props = extract_observable_properties(_parse_func(src))
    invs = [p for p in props if p.category == "invariant"]
    assert any("item is not None" in p.raw_expression for p in invs)


def test_invariant_assert_in_loop() -> None:
    src = """
    def process(data):
        result = []
        for x in data:
            assert x >= 0
            result.append(x)
        return result
    """
    props = extract_observable_properties(_parse_func(src))
    invs = [p for p in props if p.category == "invariant"]
    assert len(invs) >= 1
    assert all(p.confidence == Confidence.EXPLICIT for p in invs)


# ---------------------------------------------------------------------------
# Function without assert or raise.
# ---------------------------------------------------------------------------


def test_function_without_assert_or_raise() -> None:
    src = """
    def f(x):
        return x * 2
    """
    props = extract_observable_properties(_parse_func(src))
    assert props == []


# ---------------------------------------------------------------------------
# Nested functions - isolation.
# ---------------------------------------------------------------------------


def test_assert_in_nested_function_is_ignored() -> None:
    src = """
    def f():
        def inner():
            assert False
        return 42
    """
    props = extract_observable_properties(_parse_func(src))
    assert props == []


# ---------------------------------------------------------------------------
# Integration - raises and linked preconditions.
# ---------------------------------------------------------------------------


def test_integration_raise_and_linked_precondition() -> None:
    """Raise ValueError in if x < 0 detects and links the exception and precondition."""
    src = """
    def f(x):
        if x < 0:
            raise ValueError("x must be non-negative")
        return x
    """
    node = _parse_func(src)
    excs, _ = extract_exceptions(node)
    props = extract_observable_properties(node)

    assert len(excs) == 1
    assert excs[0].exception_type == "ValueError"
    assert excs[0].condition is not None

    precs = [p for p in props if p.category == "precondition"]
    assert len(precs) == 1
    assert precs[0].related_exception == "ValueError"
