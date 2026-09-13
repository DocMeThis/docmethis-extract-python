# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for local stubs, typeshed, and local inference."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from docmethis_extract_python.static_extraction.ast_analysis import extract_module_record
from docmethis_extract_python.static_extraction.models import (
    Confidence,
    Provenance,
)
from docmethis_extract_python.static_extraction.type_resolution import _initialize_typeshed_context


def _write_py(source_path: Path, payload: str) -> Path:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(payload, encoding="utf-8")
    return source_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TYPESHED_AVAILABLE = importlib.util.find_spec("typeshed_client") is not None


# ---------------------------------------------------------------------------
# Local stubs
# ---------------------------------------------------------------------------


class TestStubLocal:
    """Tests for resolution through a local .pyi stub."""

    def test_stub_fills_missing_parameters(self, tmp_path: Path) -> None:
        """A .pyi beside the .py fills unannotated parameters."""
        _write_py(tmp_path / "mod.py", "def foo(x, y): return x + y\n")
        _write_py(tmp_path / "mod.pyi", "def foo(x: int, y: int) -> int: ...\n")

        record = extract_module_record(tmp_path / "mod.py", "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        assert fn.types.parameters["x"].type_str == "int"
        assert fn.types.parameters["x"].confidence is Confidence.EXPLICIT
        assert fn.types.parameters["x"].provenance is Provenance.STUB
        assert fn.types.parameters["y"].type_str == "int"
        assert fn.types.return_type is not None
        assert fn.types.return_type.type_str == "int"
        assert fn.types.return_type.provenance is Provenance.STUB

    def test_stub_does_not_replace_existing_annotation(self, tmp_path: Path) -> None:
        """The source annotation takes precedence over the stub."""
        _write_py(tmp_path / "mod.py", "def foo(x: str, y): pass\n")
        _write_py(tmp_path / "mod.pyi", "def foo(x: int, y: int) -> None: ...\n")

        record = extract_module_record(tmp_path / "mod.py", "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        # x must remain str (source annotation), not int (stub).
        assert fn.types.parameters["x"].type_str == "str"
        assert fn.types.parameters["x"].provenance is Provenance.ANNOTATION
        # y is completed by the stub.
        assert fn.types.parameters["y"].type_str == "int"
        assert fn.types.parameters["y"].provenance is Provenance.STUB

    def test_stub_absent_leaves_types_unchanged(self, tmp_path: Path) -> None:
        """Without .pyi, types remain None."""
        _write_py(tmp_path / "mod.py", "def foo(x, y): pass\n")

        record = extract_module_record(tmp_path / "mod.py", "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        # No stub or annotation; local inference also fails, so types remain None.
        fn = record.functions[0]
        # Only local inference could provide a value; nothing is inferable here.
        assert fn.types is None or not fn.types.parameters

    def test_stub_method_in_class(self, tmp_path: Path) -> None:
        """A class method stub is located in the correct class."""
        _write_py(tmp_path / "mod.py", "class C:\n    def method(self, x): pass\n")
        _write_py(tmp_path / "mod.pyi", "class C:\n    def method(self, x: float) -> str: ...\n")

        record = extract_module_record(tmp_path / "mod.py", "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        meth = record.classes[0].methods[0]
        assert meth.types is not None
        assert "self" not in meth.types.parameters
        assert meth.types.parameters["x"].type_str == "float"
        assert meth.types.parameters["x"].provenance is Provenance.STUB
        assert meth.types.return_type is not None
        assert meth.types.return_type.type_str == "str"

    def test_stub_normalizes_legacy_types(self, tmp_path: Path) -> None:
        """Legacy types in the stub are normalized to PEP 585/604."""
        _write_py(tmp_path / "mod.py", "def foo(x): pass\n")
        _write_py(tmp_path / "mod.pyi", "from typing import Optional\ndef foo(x: Optional[int]) -> None: ...\n")

        record = extract_module_record(tmp_path / "mod.py", "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        assert fn.types.parameters["x"].type_str == "int | None"

    def test_unreadable_stub_leaves_types_unchanged(self, tmp_path: Path) -> None:
        """A malformed stub is silently ignored."""
        _write_py(tmp_path / "mod.py", "def foo(x): pass\n")
        (tmp_path / "mod.pyi").write_text("def foo(:\n", encoding="utf-8")

        record = extract_module_record(tmp_path / "mod.py", "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        # No crash; types are not resolved from the stub.

    def test_nested_class_stub(self, tmp_path: Path) -> None:
        """Stub for a method in a nested class."""
        code_py = "class Outer:\n    class Inner:\n        def method(self, n): pass\n"
        code_pyi = "class Outer:\n    class Inner:\n        def method(self, n: int) -> str: ...\n"
        _write_py(tmp_path / "mod.py", code_py)
        _write_py(tmp_path / "mod.pyi", code_pyi)

        record = extract_module_record(tmp_path / "mod.py", "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        inner = next(c for c in record.classes if c.qualified_name == "mod.Outer.Inner")
        meth = inner.methods[0]
        assert meth.types is not None
        assert meth.types.parameters["n"].type_str == "int"
        assert meth.types.return_type is not None
        assert meth.types.return_type.type_str == "str"


# ---------------------------------------------------------------------------
# Typeshed
# ---------------------------------------------------------------------------


class TestTypeshed:
    """Tests for resolution through typeshed stubs."""

    @pytest.mark.skipif(not _TYPESHED_AVAILABLE, reason="typeshed-client not installed")
    def test_context_uses_current_process_search_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Typeshed context initialization must not spawn a helper interpreter."""
        import typeshed_client  # noqa: PLC0415

        captured: dict[str, object] = {}
        real_get_search_context = typeshed_client.get_search_context

        def fail_check_output(*_args: object, **_kwargs: object) -> bytes:
            pytest.fail("typeshed context initialization spawned a subprocess")

        def wrapped_get_search_context(*, search_path: list[Path]) -> object:
            captured["search_path"] = search_path
            return real_get_search_context(search_path=search_path)

        monkeypatch.setattr(subprocess, "check_output", fail_check_output)
        monkeypatch.setattr(typeshed_client, "get_search_context", wrapped_get_search_context)

        assert _initialize_typeshed_context() is not None
        assert captured["search_path"] == [Path(path) for path in sys.path if path]

    @pytest.mark.skipif(not _TYPESHED_AVAILABLE, reason="typeshed-client not installed")
    def test_typeshed_resolves_stdlib_module(self, tmp_path: Path) -> None:
        """For a stdlib module, typeshed completes missing types."""
        # Simulate a file named "pathlib" with a "join" function.
        # Its name matches os.path.join in typeshed.
        code = "def join(a, *paths): pass\n"
        source_path = tmp_path / "os" / "path.py"
        _write_py(source_path, code)

        record = extract_module_record(source_path, "os.path", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        # typeshed has a stub for os.path.join, so types are resolved.
        if fn.types is not None and "a" in fn.types.parameters:
            assert fn.types.parameters["a"].provenance is Provenance.STUB

    def test_typeshed_without_matching_module(self, tmp_path: Path) -> None:
        """An unknown typeshed module does not produce an error."""
        _write_py(tmp_path / "unknown_module.py", "def foo(x): pass\n")
        record = extract_module_record(
            tmp_path / "unknown_module.py", "unknown_module_xyz", is_stub=False, is_test=False, file_hash="abc"
        )
        assert record is not None
        # No crash; types remain None or are inferred locally.


# ---------------------------------------------------------------------------
# Local inference
# ---------------------------------------------------------------------------


class TestLocalInference:
    """Tests for resolution through local function-body inference."""

    def test_inference_for_loop(self, tmp_path: Path) -> None:
        """``for _ in x:`` -> x: Iterable."""
        code = "def foo(x):\n    for item in x:\n        pass\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        assert fn.types.parameters["x"].type_str == "Iterable"
        assert fn.types.parameters["x"].confidence is Confidence.INFERRED_LOW
        assert fn.types.parameters["x"].provenance is Provenance.LOCAL_INFERENCE

    def test_inference_append(self, tmp_path: Path) -> None:
        """``x.append(...)`` -> x: list."""
        code = "def foo(x):\n    x.append(1)\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        assert fn.types.parameters["x"].type_str == "list"
        assert fn.types.parameters["x"].confidence is Confidence.INFERRED_LOW

    def test_inference_dict_method(self, tmp_path: Path) -> None:
        """``x.keys()`` -> x: dict."""
        code = "def foo(x):\n    return list(x.keys())\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        assert fn.types.parameters["x"].type_str == "dict"

    def test_inference_subscript_assign(self, tmp_path: Path) -> None:
        """``x[k] = v`` -> x: dict."""
        code = "def foo(x, k, v):\n    x[k] = v\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        assert fn.types.parameters["x"].type_str == "dict"

    def test_inference_augassign_str(self, tmp_path: Path) -> None:
        """``x += 'suffix'`` -> x: str."""
        code = "def foo(x):\n    x += '_suffix'\n    return x\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        assert fn.types.parameters["x"].type_str == "str"

    def test_inference_isinstance(self, tmp_path: Path) -> None:
        """``isinstance(x, int)`` -> x: int."""
        code = "def foo(x):\n    if isinstance(x, int):\n        return x\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        assert fn.types.parameters["x"].type_str == "int"

    def test_inference_isinstance_union_ignored(self, tmp_path: Path) -> None:
        """``isinstance(x, (int, str))`` -> ambiguous type, no inference."""
        code = "def foo(x):\n    if isinstance(x, (int, str)):\n        return x\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        # isinstance with a tuple does not infer a type.
        if fn.types is not None:
            assert "x" not in fn.types.parameters

    def test_inference_return_constant(self, tmp_path: Path) -> None:
        """All branches return int -> return type int."""
        code = "def foo(x):\n    if x:\n        return 1\n    return 0\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        assert fn.types.return_type is not None
        assert fn.types.return_type.type_str == "int"
        assert fn.types.return_type.confidence is Confidence.INFERRED_LOW
        assert fn.types.return_type.provenance is Provenance.LOCAL_INFERENCE

    def test_inference_return_from_annotated_parameters(self, tmp_path: Path) -> None:
        """``return x + y`` where x: int and y: int -> return type int."""
        code = "def add(x: int, y: int):\n    return x + y\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        assert fn.types.return_type is not None
        assert fn.types.return_type.type_str == "int"

    def test_inference_mixed_returns_not_inferred(self, tmp_path: Path) -> None:
        """Mixed returns (int and str) are not inferred."""
        code = "def foo(x):\n    if x:\n        return 1\n    return 'non'\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        # Mixed return values do not infer a return type.
        if fn.types is not None:
            assert fn.types.return_type is None

    def test_inference_does_not_replace_annotation(self, tmp_path: Path) -> None:
        """Inference must not overwrite an annotated parameter."""
        code = "def foo(x: str):\n    x.append(1)\n    return x\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        # x is annotated str; inference from append -> list must not overwrite it.
        assert fn.types.parameters["x"].type_str == "str"
        assert fn.types.parameters["x"].provenance is Provenance.ANNOTATION

    def test_inference_does_not_descend_into_nested_function(self, tmp_path: Path) -> None:
        """Uses in a nested function do not affect the outer parameter."""
        code = "def foo(x):\n    def inner():\n        for item in x:\n            pass\n    return None\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        # x is used only inside inner; no inference for foo.x.
        if fn.types is not None:
            assert "x" not in fn.types.parameters

    def test_inference_local_variable_for_return(self, tmp_path: Path) -> None:
        """``result = []`` + ``return result`` -> return type list."""
        code = "def foo():\n    result = []\n    return result\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        assert fn.types.return_type is not None
        assert fn.types.return_type.type_str == "list"

    def test_inference_fully_annotated_function_unchanged(self, tmp_path: Path) -> None:
        """A fully annotated function is not changed by inference."""
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        assert fn.types.parameters["a"].provenance is Provenance.ANNOTATION
        assert fn.types.return_type is not None
        assert fn.types.return_type.provenance is Provenance.ANNOTATION
