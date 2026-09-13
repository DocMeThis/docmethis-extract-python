# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for source_code_extraction.py (iteration 1.5)."""

from __future__ import annotations

import ast
import json
import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docmethis_extract_python.static_extraction.models import ProjectRecord

from docmethis_extract_python.static_extraction.models import deserialize_project_record, serialize_project
from docmethis_extract_python.static_extraction.source_code_extraction import (
    build_class_skeleton,
    build_module_skeleton,
    extract_source_function,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_fn(source: str) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, list[str]]:
    """Parse a function definition and return (node, lines)."""
    tree = ast.parse(textwrap.dedent(source))
    node = tree.body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    return node, textwrap.dedent(source).splitlines()


def _parse_class(source: str) -> tuple[ast.ClassDef, list[str]]:
    """Parse a class definition and return (node, lines)."""
    tree = ast.parse(textwrap.dedent(source))
    node = tree.body[0]
    assert isinstance(node, ast.ClassDef)
    return node, textwrap.dedent(source).splitlines()


def _parse_module(source: str) -> tuple[list, list, list[str]]:
    """Parse a module and return (top-level functions, top-level classes, lines)."""
    src = textwrap.dedent(source)
    tree = ast.parse(src)
    function_records = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    return function_records, classes, src.splitlines()


# ---------------------------------------------------------------------------
# extract_source_function
# ---------------------------------------------------------------------------


class TestExtractSourceFunction:  # noqa: D101
    def test_function_simple(self) -> None:
        src = "def f(x):\n    return x\n"
        node, lines = _parse_fn(src)
        result = extract_source_function(node, lines)
        assert result == "def f(x):\n    return x"

    def test_function_multilignes(self) -> None:
        src = "def f(\n    x: int,\n    y: int,\n) -> int:\n    return x + y\n"
        node, lines = _parse_fn(src)
        result = extract_source_function(node, lines)
        assert result.startswith("def f(")
        assert "return x + y" in result
        assert "x: int," in result

    def test_function_with_closure(self) -> None:
        src = "def f(x):\n    def inner():\n        pass\n    return inner\n"
        node, lines = _parse_fn(src)
        result = extract_source_function(node, lines)
        assert "def inner():" in result
        assert "return inner" in result

    def test_async_function(self) -> None:
        src = "async def f():\n    pass\n"
        node, lines = _parse_fn(src)
        result = extract_source_function(node, lines)
        assert result.startswith("async def f():")

    def test_indentation_fidelity(self) -> None:
        src = "class C:\n    def m(self):\n        return 1\n"
        tree = ast.parse(src)
        node = tree.body[0].body[0]
        lines = src.splitlines()
        result = extract_source_function(node, lines)
        assert result.startswith("    def m(self):")
        assert "        return 1" in result


# ---------------------------------------------------------------------------
# build_class_skeleton
# ---------------------------------------------------------------------------


class TestBuildClassSkeleton:  # noqa: D101
    def test_attributes_conserves(self) -> None:
        src = """\
        class Point:
            x: int
            y: int
            def distance(self) -> float:
                import math
                return math.sqrt(self.x**2 + self.y**2)
        """
        node, lines = _parse_class(src)
        result = build_class_skeleton(node, lines)
        assert "x: int" in result
        assert "y: int" in result
        assert "import math" not in result
        assert "..." in result

    def test_method_body_is_replaced(self) -> None:
        src = """\
        class Foo:
            def method(self):
                x = 1
                y = 2
                return x + y
        """
        node, lines = _parse_class(src)
        result = build_class_skeleton(node, lines)
        assert "x = 1" not in result
        assert "return x + y" not in result
        assert "..." in result

    def test_method_oneliner(self) -> None:
        src = """\
        class Foo:
            def bar(self): return 42
        """
        node, lines = _parse_class(src)
        result = build_class_skeleton(node, lines)
        assert "def bar(...): ..." in result
        assert "return 42" not in result

    def test_multiline_signature_is_preserved(self) -> None:
        src = """\
        class Foo:
            def method(
                self,
                x: int,
                y: int,
            ) -> int:
                return x + y
        """
        node, lines = _parse_class(src)
        result = build_class_skeleton(node, lines)
        assert "def method(" in result
        assert "x: int," in result
        assert "y: int," in result
        assert "return x + y" not in result
        assert "..." in result

    def test_slots_are_preserved(self) -> None:
        src = """\
        class Foo:
            __slots__ = ("x", "y")
            def method(self): pass
        """
        node, lines = _parse_class(src)
        result = build_class_skeleton(node, lines)
        assert "__slots__" in result

    def test_attributes_after_method_are_preserved(self) -> None:
        src = """\
        class Foo:
            def method(self):
                pass
            x: int = 0
        """
        node, lines = _parse_class(src)
        result = build_class_skeleton(node, lines)
        assert "x: int = 0" in result

    def test_method_decorator_is_preserved(self) -> None:
        src = """\
        class Foo:
            @property
            def value(self):
                return self._value
        """
        node, lines = _parse_class(src)
        result = build_class_skeleton(node, lines)
        assert "@property" in result
        assert "return self._value" not in result
        assert "..." in result

    def test_class_without_methods(self) -> None:
        src = """\
        class Empty:
            x: int
            y: str = "hello"
        """
        node, lines = _parse_class(src)
        result = build_class_skeleton(node, lines)
        assert "x: int" in result
        assert 'y: str = "hello"' in result


# ---------------------------------------------------------------------------
# build_module_skeleton
# ---------------------------------------------------------------------------


class TestBuildModuleSkeleton:  # noqa: D101
    def test_imports_are_preserved(self) -> None:
        src = """\
        import os
        import sys

        def main():
            print(os.getcwd())
        """
        function_records, classes, lines = _parse_module(src)
        result = build_module_skeleton(function_records, classes, lines)
        assert "import os" in result
        assert "import sys" in result
        assert "print(os.getcwd())" not in result

    def test_top_level_function_body_is_stripped(self) -> None:
        src = """\
        def main():
            x = 1
            return x
        """
        function_records, classes, lines = _parse_module(src)
        result = build_module_skeleton(function_records, classes, lines)
        assert "x = 1" not in result
        assert "return x" not in result
        assert "..." in result

    def test_constant_is_preserved(self) -> None:
        src = """\
        VERSION = "1.0"

        def main():
            pass
        """
        function_records, classes, lines = _parse_module(src)
        result = build_module_skeleton(function_records, classes, lines)
        assert 'VERSION = "1.0"' in result

    def test_class_is_reduced_to_skeleton(self) -> None:
        src = """\
        class Foo:
            x: int
            def method(self):
                return 42
        """
        function_records, classes, lines = _parse_module(src)
        result = build_module_skeleton(function_records, classes, lines)
        assert "x: int" in result
        assert "return 42" not in result
        assert "..." in result

    def test_module_with_only_imports(self) -> None:
        src = "import os\nimport sys\n"
        function_records, classes, lines = _parse_module(src)
        result = build_module_skeleton(function_records, classes, lines)
        assert result == "import os\nimport sys"

    def test_top_level_oneliner_function(self) -> None:
        src = "def f(x): return x\n"
        function_records, classes, lines = _parse_module(src)
        result = build_module_skeleton(function_records, classes, lines)
        assert "def f(...): ..." in result
        assert "return x" not in result


# ---------------------------------------------------------------------------
# Portion integration.
# ---------------------------------------------------------------------------


class TestPortionIntegration:  # noqa: D101
    def test_module_source_code_non_none(self, result_portion: ProjectRecord) -> None:
        for module in result_portion.modules:
            assert module.source_code is not None, f"source_code missing from {module.file_path}"
            assert isinstance(module.source_code, str)

    def test_function_source_code_non_none(self, result_portion: ProjectRecord) -> None:
        for module in result_portion.modules:
            for fn in module.functions:
                assert fn.source_code is not None, f"source_code missing from {fn.qualified_name}"
            for class_ in module.classes:
                for method in class_.methods:
                    assert method.source_code is not None, f"source_code missing from {method.qualified_name}"

    def test_class_source_code_non_none(self, result_portion: ProjectRecord) -> None:
        for module in result_portion.modules:
            for class_ in module.classes:
                assert class_.source_code is not None, f"source_code missing from {class_.qualified_name}"

    def test_function_source_starts_with_def(self, result_portion: ProjectRecord) -> None:
        for module in result_portion.modules:
            for fn in module.functions:
                stripped = fn.source_code.lstrip()
                assert stripped.startswith(("def ", "async def ")), (
                    f"{fn.qualified_name}: source_code starts with {stripped[:30]!r}"
                )

    def test_round_trip_json(self, result_portion: ProjectRecord) -> None:
        json_str = serialize_project(result_portion)
        project_two = deserialize_project_record(json.loads(json_str))
        for m1, m2 in zip(result_portion.modules, project_two.modules, strict=True):
            assert m1.source_code == m2.source_code
            for f1, f2 in zip(m1.functions, m2.functions, strict=True):
                assert f1.source_code == f2.source_code
            for c1, c2 in zip(m1.classes, m2.classes, strict=True):
                assert c1.source_code == c2.source_code
