# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for global qualified_name correction (iteration 1.6, DEC-022/DEC-023)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from docmethis_extract_python.static_extraction.ast_analysis import extract_module_record
from docmethis_extract_python.static_extraction.traversal import resolve_module_name


def _write_py(source_path: Path, payload: str) -> Path:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(payload, encoding="utf-8")
    return source_path


class TestQualifiedNameGlobal:
    """Verify that qualified_name includes the module prefix (DEC-022)."""

    def test_function_toplevel(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "def foo(): pass\n")
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.functions[0].qualified_name == "mod.foo"

    def test_class_method(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "class MyClass:\n    def method(self): pass\n")
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        class_ = record.classes[0]
        assert class_.qualified_name == "mod.MyClass"
        assert class_.methods[0].qualified_name == "mod.MyClass.method"

    def test_two_main_functions_in_different_modules(self, tmp_path: Path) -> None:
        """Two main() functions in distinct modules have different qualified_name values."""
        f1 = _write_py(tmp_path / "mod_a.py", "def main(): pass\n")
        f2 = _write_py(tmp_path / "mod_b.py", "def main(): pass\n")
        rec_a = extract_module_record(f1, resolve_module_name(f1, tmp_path), is_stub=False, is_test=False, file_hash="aaa")
        rec_b = extract_module_record(f2, resolve_module_name(f2, tmp_path), is_stub=False, is_test=False, file_hash="bbb")
        assert rec_a is not None
        assert rec_b is not None
        qn_a = rec_a.functions[0].qualified_name
        qn_b = rec_b.functions[0].qualified_name
        assert qn_a == "mod_a.main"
        assert qn_b == "mod_b.main"
        assert qn_a != qn_b

    def test_init_prefix_without_dunder(self, tmp_path: Path) -> None:
        """An __init__.py file produces a module prefix without '.__init__'."""
        f = _write_py(tmp_path / "mypkg" / "__init__.py", "def setup(): pass\n")
        module_name = resolve_module_name(f, tmp_path)
        assert module_name == "mypkg"
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.functions[0].qualified_name == "mypkg.setup"

    def test_class_qualified_name_is_global(self, tmp_path: Path) -> None:
        """ClassRecord.qualified_name includes the module prefix."""
        f = _write_py(tmp_path / "pkg" / "sub.py", "class Widget:\n    pass\n")
        module_name = resolve_module_name(f, tmp_path)
        assert module_name == "pkg.sub"
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.classes[0].qualified_name == "pkg.sub.Widget"

    def test_nested_class(self, tmp_path: Path) -> None:
        """A nested class and its methods have correct qualified_name values."""
        code = "class Outer:\n    class Inner:\n        def method(self): pass\n"
        f = _write_py(tmp_path / "mod.py", code)
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        outer = next(c for c in record.classes if c.qualified_name == "mod.Outer")
        inner = next(c for c in record.classes if "Inner" in c.qualified_name)
        assert inner.qualified_name == "mod.Outer.Inner"
        assert inner.methods[0].qualified_name == "mod.Outer.Inner.method"
        _ = outer  # Keep the assertion that Outer exists explicit.


class TestOverloadIgnore:
    """Verify that @overload functions are ignored (DEC-023)."""

    def test_overload_absent_from_function_records(self, tmp_path: Path) -> None:
        """@overload definitions do not produce FunctionRecords."""
        code = (
            "from typing import overload\n"
            "@overload\n"
            "def process(x: int) -> int: ...\n"
            "@overload\n"
            "def process(x: str) -> str: ...\n"
            "def process(x):\n"
            "    return x\n"
        )
        f = _write_py(tmp_path / "mod.py", code)
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert len(record.functions) == 1
        assert record.functions[0].qualified_name == "mod.process"

    def test_qualified_overload_is_ignored(self, tmp_path: Path) -> None:
        """Qualified typing.overload definitions are ignored too."""
        code = "import typing\n@typing.overload\ndef process(x: int) -> int: ...\ndef process(x):\n    return x\n"
        f = _write_py(tmp_path / "mod.py", code)
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert len(record.functions) == 1
