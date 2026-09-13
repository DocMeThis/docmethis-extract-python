# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for _build_ast_index and _find_node_ast (phase 3, iteration 1.10).

Fixed root cause: _build_ast_index recursed only into ClassDef, leaving functions
defined in control blocks (if/try/for/while/with) at module or class level out of the index.
"""

from __future__ import annotations

import ast
import logging
import textwrap
from typing import TYPE_CHECKING

from docmethis_extract_python.static_extraction.callgraph_ast import (
    _AstKey,
    _build_ast_index,
    _find_node_ast,
)

if TYPE_CHECKING:
    import pytest


def _parse(code: str) -> ast.Module:
    return ast.parse(textwrap.dedent(code))


# ---------------------------------------------------------------------------
# Tests - _build_ast_index
# ---------------------------------------------------------------------------


class TestBuildAstIndex:
    """Verify that every function outside a function body is indexed."""

    def test_function_toplevel(self) -> None:
        tree = _parse("""
            def foo():
                pass
        """)
        index = _build_ast_index(tree)
        fn = tree.body[0]
        assert _AstKey(fn.lineno, "foo") in index

    def test_class_method(self) -> None:
        tree = _parse("""
            class MyClass:
                def method(self):
                    pass
        """)
        index = _build_ast_index(tree)
        fn = tree.body[0].body[0]
        assert _AstKey(fn.lineno, "method") in index

    def test_function_in_if(self) -> None:
        tree = _parse("""
            if True:
                def conditional_func():
                    pass
        """)
        index = _build_ast_index(tree)
        fn = tree.body[0].body[0]
        assert _AstKey(fn.lineno, "conditional_func") in index

    def test_function_in_try(self) -> None:
        tree = _parse("""
            try:
                def try_func():
                    pass
            except Exception:
                pass
        """)
        index = _build_ast_index(tree)
        fn = tree.body[0].body[0]
        assert _AstKey(fn.lineno, "try_func") in index

    def test_function_in_for(self) -> None:
        tree = _parse("""
            for _ in range(1):
                def loop_func():
                    pass
        """)
        index = _build_ast_index(tree)
        fn = tree.body[0].body[0]
        assert _AstKey(fn.lineno, "loop_func") in index

    def test_function_in_with(self) -> None:
        tree = _parse("""
            with open("f") as f:
                def with_func():
                    pass
        """)
        index = _build_ast_index(tree)
        fn = tree.body[0].body[0]
        assert _AstKey(fn.lineno, "with_func") in index

    def test_method_in_nested_class(self) -> None:
        tree = _parse("""
            class Outer:
                class Inner:
                    def inner_method(self):
                        pass
        """)
        index = _build_ast_index(tree)
        fn = tree.body[0].body[0].body[0]
        assert _AstKey(fn.lineno, "inner_method") in index

    def test_closure_is_not_indexed(self) -> None:
        """A closure defined inside a function body is not indexed."""
        tree = _parse("""
            def outer():
                def closure():
                    pass
        """)
        index = _build_ast_index(tree)
        # "outer" must be indexed, while "closure" must not be.
        outer = tree.body[0]
        closure = outer.body[0]
        assert _AstKey(outer.lineno, "outer") in index
        assert _AstKey(closure.lineno, "closure") not in index

    def test_multiple_functions_on_different_lines(self) -> None:
        tree = _parse("""
            def alpha():
                pass

            def beta():
                pass
        """)
        index = _build_ast_index(tree)
        alpha, beta = tree.body[0], tree.body[1]
        assert _AstKey(alpha.lineno, "alpha") in index
        assert _AstKey(beta.lineno, "beta") in index


# ---------------------------------------------------------------------------
# Tests - _find_node_ast
# ---------------------------------------------------------------------------


class TestFindNodeAst:
    """Verify _find_node_ast lookup and log level behavior."""

    def test_node_found(self) -> None:
        tree = _parse("""
            def foo():
                pass
        """)
        index = _build_ast_index(tree)
        fn_node = tree.body[0]
        result = _find_node_ast(index, fn_node.lineno, "foo", "mod.foo")
        assert result is fn_node

    def test_missing_node_returns_none(self) -> None:
        index: dict = {}
        result = _find_node_ast(index, 99, "inexistant", "mod.inexistant")
        assert result is None

    def test_missing_node_logs_debug_not_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        index: dict = {}
        with caplog.at_level(logging.DEBUG, logger="docmethis_extract_python.static_extraction.callgraph_ast"):
            _find_node_ast(index, 1, "missing", "mod.missing")
        debug_msgs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("missing" in r.message for r in debug_msgs)
        assert not warning_msgs
