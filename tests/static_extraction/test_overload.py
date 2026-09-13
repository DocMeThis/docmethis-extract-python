# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for @overload handling (iteration 1.10)."""

from __future__ import annotations

import dataclasses
import textwrap
from typing import TYPE_CHECKING

from docmethis_extract_python.static_extraction.ast_analysis import extract_module_record

if TYPE_CHECKING:
    from pathlib import Path
from docmethis_extract_python.static_extraction.models import MISSING_VALUE, deserialize_function_record
from docmethis_extract_python.static_extraction.traversal import resolve_module_name


def _write_py(source_path: Path, payload: str) -> Path:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(payload, encoding="utf-8")
    return source_path


class TestOverload:
    """Verify extraction of @overload signatures into FunctionRecord."""

    def test_three_overloads_one_implementation(self, tmp_path: Path) -> None:
        """Three @overload definitions plus one implementation produce one FunctionRecord with three signatures."""
        code = (
            "from typing import overload\n"
            "@overload\n"
            "def process(x: int) -> int: ...\n"
            "@overload\n"
            "def process(x: str) -> str: ...\n"
            "@overload\n"
            "def process(x: bytes) -> bytes: ...\n"
            "def process(x):\n"
            "    return x\n"
        )
        f = _write_py(tmp_path / "mod.py", code)
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")

        assert record is not None
        assert len(record.functions) == 1, "One implementation -> one FunctionRecord"
        fn = record.functions[0]
        assert fn.qualified_name == "mod.process"
        assert fn.has_overloads is True
        assert len(fn.overload_signatures) == 3

    def test_overload_parameter_types_are_preserved(self, tmp_path: Path) -> None:
        """Parameter types from each overload are captured."""
        code = (
            "from typing import overload\n"
            "@overload\n"
            "def convert(x: int) -> str: ...\n"
            "@overload\n"
            "def convert(x: float) -> str: ...\n"
            "def convert(x):\n"
            "    return str(x)\n"
        )
        f = _write_py(tmp_path / "mod.py", code)
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")

        assert record is not None
        fn = record.functions[0]
        assert fn.has_overloads is True
        annotations = [sig.parameters[0].annotation_raw for sig in fn.overload_signatures]
        assert "int" in annotations
        assert "float" in annotations

    def test_overload_return_annotations_are_preserved(self, tmp_path: Path) -> None:
        """The return type of each overload is captured in source order.

        In the nominal `@overload` case, parameters are identical and only the return varies.
        Without `return_annotation`, overloads would be indistinguishable to Module 2.
        """
        code = textwrap.dedent("""\
            from typing import overload
            @overload
            def get(key: str) -> int: ...
            @overload
            def get(key: str) -> list[str]: ...
            @overload
            def get(key: str) -> None: ...
            def get(key):
                return None
        """)
        f = _write_py(tmp_path / "mod.py", code)
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")

        assert record is not None
        fn = record.functions[0]
        returns = [sig.return_annotation for sig in fn.overload_signatures]
        assert returns == ["int", "list[str]", "None"]
        # The implementation is unannotated; its signature must not inherit overloads.
        assert fn.signature.return_annotation == MISSING_VALUE

    def test_missing_overload_return_uses_missing_value(self, tmp_path: Path) -> None:
        """An overload without a return annotation uses MISSING_VALUE, not an empty string."""
        code = textwrap.dedent("""\
            from typing import overload
            @overload
            def brut(x: int): ...
            @overload
            def brut(x: str) -> str: ...
            def brut(x):
                return x
        """)
        f = _write_py(tmp_path / "mod.py", code)
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")

        assert record is not None
        returns = [sig.return_annotation for sig in record.functions[0].overload_signatures]
        assert returns == [MISSING_VALUE, "str"]

    def test_overload_decorator_is_preserved_in_signature(self, tmp_path: Path) -> None:
        """Each overload signature preserves its `@overload` decorator."""
        code = textwrap.dedent("""\
            from typing import overload
            @overload
            def f(x: int) -> int: ...
            @overload
            def f(x: str) -> str: ...
            def f(x):
                return x
        """)
        f = _write_py(tmp_path / "mod.py", code)
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")

        assert record is not None
        fn = record.functions[0]
        for sig in fn.overload_signatures:
            assert [dec.full_decorator for dec in sig.decorators] == ["overload"]
        assert fn.signature.decorators == [], "The implementation must not carry @overload"

    def test_no_overloads(self, tmp_path: Path) -> None:
        """An ordinary function has has_overloads=False and overload_signatures=[]."""
        f = _write_py(tmp_path / "mod.py", "def simple(x: int) -> int:\n    return x\n")
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")

        assert record is not None
        fn = record.functions[0]
        assert fn.has_overloads is False
        assert fn.overload_signatures == []

    def test_method_with_overloads(self, tmp_path: Path) -> None:
        """@overload also works for class methods."""
        code = (
            "from typing import overload\n"
            "class Parser:\n"
            "    @overload\n"
            "    def parse(self, data: str) -> list: ...\n"
            "    @overload\n"
            "    def parse(self, data: bytes) -> list: ...\n"
            "    def parse(self, data):\n"
            "        return []\n"
        )
        f = _write_py(tmp_path / "mod.py", code)
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")

        assert record is not None
        assert len(record.classes) == 1
        methods = record.classes[0].methods
        assert len(methods) == 1, "One implementation -> one method"
        m = methods[0]
        assert m.has_overloads is True
        assert len(m.overload_signatures) == 2

    def test_overloads_do_not_leak_between_modules(self, tmp_path: Path) -> None:
        """@overload signatures from one module do not leak into another."""
        code_a = (
            "from typing import overload\n"
            "@overload\n"
            "def helper(x: int) -> int: ...\n"
            "@overload\n"
            "def helper(x: str) -> str: ...\n"
            "def helper(x):\n"
            "    return x\n"
        )
        code_b = "def helper(x):\n    return x\n"
        fa = _write_py(tmp_path / "mod_a.py", code_a)
        fb = _write_py(tmp_path / "mod_b.py", code_b)

        rec_a = extract_module_record(fa, resolve_module_name(fa, tmp_path), is_stub=False, is_test=False, file_hash="aaa")
        rec_b = extract_module_record(fb, resolve_module_name(fb, tmp_path), is_stub=False, is_test=False, file_hash="bbb")

        assert rec_a is not None
        assert rec_b is not None
        assert rec_a.functions[0].has_overloads is True
        assert rec_b.functions[0].has_overloads is False
        assert rec_b.functions[0].overload_signatures == []

    def test_overload_only_module(self, tmp_path: Path) -> None:
        """Without an implementation, the last overload is promoted."""
        code = textwrap.dedent("""\
            from typing import overload
            @overload
            def load(x: int) -> int: ...
            @overload
            def load(x: str) -> str: ...
        """)
        f = _write_py(tmp_path / "mod.py", code)
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")

        assert record is not None
        assert len(record.functions) == 1, "Orphan overloads produce one FunctionRecord"
        fn = record.functions[0]
        assert fn.qualified_name == "mod.load"
        assert fn.has_overloads is True
        assert len(fn.overload_signatures) == 2

    def test_pyi_stub_overload_only(self, tmp_path: Path) -> None:
        """A .pyi stub without an implementation preserves its overload signatures."""
        code = textwrap.dedent("""\
            from typing import overload
            @overload
            def dumps(obj: bytes) -> str: ...
            @overload
            def dumps(obj: str) -> bytes: ...
        """)
        f = _write_py(tmp_path / "mod.pyi", code)
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=True, is_test=False, file_hash="abc")

        assert record is not None
        assert len(record.functions) == 1
        signatures = record.functions[0].overload_signatures
        assert [sig.parameters[0].annotation_raw for sig in signatures] == ["bytes", "str"]
        assert [sig.return_annotation for sig in signatures] == ["str", "bytes"]
        # The promoted overload carries its own signature, including the return.
        assert record.functions[0].signature.return_annotation == "bytes"

    def test_method_overload_only(self, tmp_path: Path) -> None:
        """Method overloads without an implementation remain attached to their class."""
        code = textwrap.dedent("""\
            from typing import overload
            class Reader:
                @overload
                def read(self, n: int) -> bytes: ...
                @overload
                def read(self, n: None) -> bytes: ...
        """)
        f = _write_py(tmp_path / "mod.pyi", code)
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=True, is_test=False, file_hash="abc")

        assert record is not None
        assert record.functions == [], "A method must not be promoted to module level"
        methods = record.classes[0].methods
        assert len(methods) == 1
        assert methods[0].qualified_name == "mod.Reader.read"
        assert methods[0].has_overloads is True
        assert len(methods[0].overload_signatures) == 2

    def test_implementation_before_overloads(self, tmp_path: Path) -> None:
        """When the implementation precedes overloads, overloads attach to it."""
        code = textwrap.dedent("""\
            from typing import overload
            def process(x):
                return x
            @overload
            def process(x: int) -> int: ...
            @overload
            def process(x: str) -> str: ...
        """)
        f = _write_py(tmp_path / "mod.py", code)
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")

        assert record is not None
        assert len(record.functions) == 1
        fn = record.functions[0]
        assert fn.has_overloads is True
        assert len(fn.overload_signatures) == 2

    def test_two_functions_overload_only(self, tmp_path: Path) -> None:
        """Multiple groups of orphan overloads do not mix."""
        code = textwrap.dedent("""\
            from typing import overload
            @overload
            def a(x: int) -> int: ...
            @overload
            def b(x: str) -> str: ...
            @overload
            def a(x: bytes) -> bytes: ...
        """)
        f = _write_py(tmp_path / "mod.py", code)
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")

        assert record is not None
        by_name = {fn.qualified_name: fn for fn in record.functions}
        assert set(by_name) == {"mod.a", "mod.b"}
        assert len(by_name["mod.a"].overload_signatures) == 2
        assert len(by_name["mod.b"].overload_signatures) == 1

    def test_cache_compatibility_with_missing_fields(self, tmp_path: Path) -> None:
        """Deserializing a cache without has_overloads/overload_signatures does not raise."""
        f = _write_py(tmp_path / "mod.py", "def foo(): pass\n")
        module_name = resolve_module_name(f, tmp_path)
        record = extract_module_record(f, module_name, is_stub=False, is_test=False, file_hash="abc")
        assert record is not None

        data = dataclasses.asdict(record.functions[0])
        # Simulate a cache produced before iteration 10 by removing new fields.
        data.pop("has_overloads", None)
        data.pop("overload_signatures", None)

        restored = deserialize_function_record(data)
        assert restored.has_overloads is False
        assert restored.overload_signatures == []
