# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for AST parsing and ModuleRecord extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docmethis_extract_python.static_extraction.ast_analysis import extract_module_record
from docmethis_extract_python.static_extraction.ast_parser import parse_file
from docmethis_extract_python.static_extraction.models import (
    MISSING_VALUE,
    SPECIAL_VALUE,
    Confidence,
    ImportRecord,
    MethodType,
    ParameterType,
    Provenance,
    Visibility,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_py(source_path: Path, payload: str) -> Path:
    """Write Python code to a file and return its path."""
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(payload, encoding="utf-8")
    return source_path


class TestParseFile:
    """Tests for AST parsing."""

    def test_valid_file(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "ok.py", "def foo(): pass\n")
        tree = parse_file(f)
        assert tree is not None

    def test_file_invalid_syntax(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "broken.py", "def foo(:\n")
        tree = parse_file(f)
        assert tree is None

    def test_file_empty(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "empty.py", "")
        tree = parse_file(f)
        assert tree is not None

    def test_file_with_bom(self, tmp_path: Path) -> None:
        f = tmp_path / "bom.py"
        f.write_bytes(b"\xef\xbb\xbfdef foo(): pass\n")
        tree = parse_file(f)
        assert tree is not None


class TestExtractModuleRecord:
    """Tests for ModuleRecord extraction."""

    def test_function_simple(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "def function():\n    pass\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert len(record.functions) == 1
        assert record.functions[0].qualified_name == "mod.function"
        assert record.functions[0].method_kind is MethodType.FUNCTION
        assert record.functions[0].visibility is Visibility.PUBLIC

    def test_class_with_methods(self, tmp_path: Path) -> None:
        code = (
            "class Class:\n"
            "    def method(self):\n"
            "        pass\n"
            "\n"
            "    @classmethod\n"
            "    def cls_method(cls):\n"
            "        pass\n"
            "\n"
            "    @staticmethod\n"
            "    def static_method():\n"
            "        pass\n"
        )
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert len(record.classes) == 1
        class_ = record.classes[0]
        assert class_.qualified_name == "mod.Class"
        assert len(class_.methods) == 3

        names = {m.qualified_name: m.method_kind for m in class_.methods}
        assert names["mod.Class.method"] is MethodType.INSTANCE_METHOD
        assert names["mod.Class.cls_method"] is MethodType.CLASS_METHOD
        assert names["mod.Class.static_method"] is MethodType.STATIC_METHOD

    def test_qualified_name_nested_class(self, tmp_path: Path) -> None:
        code = "class Outer:\n    class Inner:\n        def method(self):\n            pass\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        # Inner is a nested class.
        inner_classes = [c for c in record.classes if "Inner" in c.qualified_name]
        assert len(inner_classes) == 1
        inner = inner_classes[0]
        assert inner.qualified_name == "mod.Outer.Inner"
        assert len(inner.methods) == 1
        assert inner.methods[0].qualified_name == "mod.Outer.Inner.method"

    def test_visibility(self, tmp_path: Path) -> None:
        code = "def public(): pass\ndef _protected(): pass\ndef __private(): pass\ndef __dunder__(): pass\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        vis = {fn.qualified_name: fn.visibility for fn in record.functions}
        assert vis["mod.public"] is Visibility.PUBLIC
        assert vis["mod._protected"] is Visibility.PROTECTED
        assert vis["mod.__private"] is Visibility.PRIVATE
        # Dunder names are public.
        assert vis["mod.__dunder__"] is Visibility.PUBLIC

    def test_extracts_docstrings(self, tmp_path: Path) -> None:
        code = '"""Module docstring."""\n\ndef foo():\n    """Function docstring."""\n'
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.existing_docstring == "Module docstring."
        assert record.functions[0].existing_docstring == "Function docstring."

    def test_async_function(self, tmp_path: Path) -> None:
        code = "async def coroutine(): pass\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert len(record.functions) == 1
        assert record.functions[0].qualified_name == "mod.coroutine"
        assert record.functions[0].is_async is True

    def test_sync_function_is_async_false(self, tmp_path: Path) -> None:
        code = "def function(): pass\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.functions[0].is_async is False

    def test_property(self, tmp_path: Path) -> None:
        code = "class C:\n    @property\n    def name(self):\n        return self._name\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.classes[0].methods[0].method_kind is MethodType.PROPERTY
        assert record.classes[0].methods[0].property_accessor.value == "getter"

    def test_property_setter_has_distinct_accessor_role(self, tmp_path: Path) -> None:
        code = """class C:
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
"""
        f = _write_py(tmp_path / "mod.py", code)

        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")

        assert record is not None
        accessors = [(method.method_kind, method.property_accessor.value) for method in record.classes[0].methods]
        assert accessors == [(MethodType.PROPERTY, "getter"), (MethodType.PROPERTY, "setter")]

    def test_invalid_file_returns_none(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "bad.py", "def foo(:\n")
        record = extract_module_record(f, "bad", is_stub=False, is_test=False, file_hash="abc")
        assert record is None

    def test_file_empty(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "empty.py", "")
        record = extract_module_record(f, "empty", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.functions == []
        assert record.classes == []

    def test_marks_stub_and_test(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.pyi", "def foo(): ...\n")
        record = extract_module_record(f, "mod", is_stub=True, is_test=False, file_hash="abc")
        assert record is not None
        assert record.is_stub is True

    def test_preserves_file_hash(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "x = 1\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="sha256hash")
        assert record is not None
        assert record.file_hash == "sha256hash"

    def test_positions(self, tmp_path: Path) -> None:
        code = "def foo():\n    pass\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.line_start == 1
        assert fn.line_end == 2
        assert fn.col_start == 0

    def test_portion(self, root_portion: Path) -> None:
        """Extract the portion fixture without crashing."""
        files_py = list(root_portion.rglob("*.py"))
        assert len(files_py) > 0
        errors = 0
        total_functions = 0
        total_classes = 0
        for source_file in files_py:
            record = extract_module_record(
                source_file,
                source_file.stem,
                is_stub=False,
                is_test=False,
                file_hash="test",
            )
            if record is None:
                errors += 1
                continue
            total_functions += len(record.functions)
            total_classes += len(record.classes)
        # No more than 10% errors.
        assert errors < len(files_py) * 0.1
        # Functions and classes were found.
        assert total_functions > 10
        assert total_classes > 0


class TestSignatures:
    """Tests for signature extraction (parameters, annotations, defaults)."""

    def test_annotated_return(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "def foo() -> int:\n    pass\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        sig = record.functions[0].signature
        assert sig is not None
        assert sig.return_annotation == "int"
        assert record.functions[0].signature_confidence is Confidence.EXPLICIT

    def test_return_absent(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "def foo():\n    pass\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.functions[0].signature is not None
        assert record.functions[0].signature.return_annotation == MISSING_VALUE

    def test_complex_annotation(self, tmp_path: Path) -> None:
        code = "from typing import Optional, List\ndef foo(x: Optional[List[int]]) -> None:\n    pass\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        sig = record.functions[0].signature
        assert sig is not None
        assert sig.parameters[0].annotation_raw == "Optional[List[int]]"
        assert sig.return_annotation == "None"

    def test_positional_only_parameter(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "def foo(x, /, y):\n    pass\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        params = record.functions[0].signature.parameters
        assert params[0].name == "x"
        assert params[0].kind is ParameterType.POSITIONAL_ONLY
        assert params[1].name == "y"
        assert params[1].kind is ParameterType.POSITIONAL_OR_KEYWORD

    def test_keyword_only_parameter(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "def foo(*, x: int):\n    pass\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        params = record.functions[0].signature.parameters
        assert len(params) == 1
        assert params[0].name == "x"
        assert params[0].kind is ParameterType.KEYWORD_ONLY

    def test_variadic_parameters(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "def foo(*args, **kwargs):\n    pass\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        params = record.functions[0].signature.parameters
        kinds = {p.name: p.kind for p in params}
        assert kinds["args"] is ParameterType.VAR_POSITIONAL
        assert kinds["kwargs"] is ParameterType.VAR_KEYWORD

    def test_multiline_signature_parameter_lines(self, tmp_path: Path) -> None:
        f = _write_py(
            tmp_path / "mod.py",
            "def foo(\n    x: int,\n    /,\n    y: str,\n    *args: object,\n    z: bool,\n    **kwargs: object,\n):\n    pass\n",
        )
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None

        lines = {parameter.name: parameter.line_start for parameter in record.functions[0].signature.parameters}

        assert lines == {"x": 2, "y": 4, "args": 5, "z": 6, "kwargs": 7}

    def test_default_simple(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "def foo(x=42, y='hello'):\n    pass\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        params = record.functions[0].signature.parameters
        assert params[0].default_value == "42"
        assert params[1].default_value == "'hello'"

    def test_default_absent(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "def foo(x):\n    pass\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.functions[0].signature.parameters[0].default_value == MISSING_VALUE

    def test_default_special(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "def foo(x=list()):\n    pass\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.functions[0].signature.parameters[0].default_value == SPECIAL_VALUE

    def test_default_none(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "def foo(x=None):\n    pass\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.functions[0].signature.parameters[0].default_value == "None"

    def test_annotation_absent(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "def foo(x):\n    pass\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.functions[0].signature.parameters[0].annotation_raw == MISSING_VALUE


class TestDecorators:
    """Tests for decorator extraction."""

    def test_simple_decorator(self, tmp_path: Path) -> None:
        code = "class C:\n    @property\n    def x(self):\n        return self._x\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        decs = record.classes[0].methods[0].signature.decorators
        assert len(decs) == 1
        assert decs[0].full_decorator == "property"
        assert decs[0].is_known is True

    def test_known_decorator(self, tmp_path: Path) -> None:
        code = (
            "class C:\n"
            "    @classmethod\n    def a(cls): pass\n"
            "    @staticmethod\n    def b(): pass\n"
            "    @property\n    def c(self): return None\n"
        )
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        for method in record.classes[0].methods:
            assert all(d.is_known for d in method.signature.decorators)

    def test_unknown_decorator(self, tmp_path: Path) -> None:
        code = "def custom_decorator(f): return f\n\n@custom_decorator\ndef foo(): pass\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        function_records = {fn.qualified_name: fn for fn in record.functions}
        decs = function_records["mod.foo"].signature.decorators
        assert len(decs) == 1
        assert decs[0].is_known is False

    def test_decorator_with_args(self, tmp_path: Path) -> None:
        code = "import functools\n\n@functools.lru_cache(maxsize=128)\ndef foo(): pass\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        decs = record.functions[0].signature.decorators
        assert len(decs) == 1
        assert decs[0].full_decorator == "functools.lru_cache(maxsize=128)"

    def test_stacked_decorators(self, tmp_path: Path) -> None:
        code = "def a(f): return f\ndef b(f): return f\n\n@a\n@b\ndef foo(): pass\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        function_records = {fn.qualified_name: fn for fn in record.functions}
        decs = function_records["mod.foo"].signature.decorators
        assert len(decs) == 2
        assert decs[0].full_decorator == "a"
        assert decs[1].full_decorator == "b"


class TestImports:
    """Tests for direct import extraction."""

    def test_import_simple(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "import os\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert ImportRecord(module="os", name=None, alias=None) in record.imports

    def test_import_multiple(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "import os, sys\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        modules = [imp.module for imp in record.imports]
        assert "os" in modules
        assert "sys" in modules

    def test_from_import(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "from pathlib import Path\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert ImportRecord(module="pathlib", name="Path", alias=None) in record.imports

    def test_from_import_alias(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "from typing import Optional as Opt\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert ImportRecord(module="typing", name="Optional", alias="Opt") in record.imports

    def test_relative_import(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "from . import utils\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert ImportRecord(module=".", name="utils", alias=None) in record.imports

    def test_import_inside_function_is_ignored(self, tmp_path: Path) -> None:
        code = "def foo():\n    import os\n    return os.getcwd()\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.imports == []

    def test_import_simple_conditional_false(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "import os\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        imp = record.imports[0]
        assert imp.conditional is False
        assert imp.type_checking_only is False


class TestConditionalImports:
    """Tests for conditional imports and TYPE_CHECKING (iteration 2)."""

    def test_try_except_import_error(self, tmp_path: Path) -> None:
        code = "try:\n    import ujson\nexcept ImportError:\n    ujson = None\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert len(record.imports) == 1
        imp = record.imports[0]
        assert imp.module == "ujson"
        assert imp.conditional is True
        assert imp.type_checking_only is False

    def test_try_except_module_not_found(self, tmp_path: Path) -> None:
        code = "try:\n    import rapidjson\nexcept ModuleNotFoundError:\n    rapidjson = None\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.imports[0].conditional is True

    def test_try_without_import_error_is_ignored(self, tmp_path: Path) -> None:
        code = "try:\n    x = int('a')\nexcept ValueError:\n    x = 0\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.imports == []

    def test_if_type_checking(self, tmp_path: Path) -> None:
        code = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from pathlib import Path\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        tc_imports = [imp for imp in record.imports if imp.type_checking_only]
        assert len(tc_imports) == 1
        assert tc_imports[0].module == "pathlib"
        assert tc_imports[0].name == "Path"

    def test_qualified_typing_type_checking(self, tmp_path: Path) -> None:
        code = "import typing\nif typing.TYPE_CHECKING:\n    import mymodule\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        tc_imports = [imp for imp in record.imports if imp.type_checking_only]
        assert len(tc_imports) == 1
        assert tc_imports[0].module == "mymodule"

    def test_future_annotations(self, tmp_path: Path) -> None:
        code = "from __future__ import annotations\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.future_annotations is True

    def test_future_annotations_false_without_import(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path / "mod.py", "import os\n")
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.future_annotations is False

    def test_future_annotations_alongside_normal_import(self, tmp_path: Path) -> None:
        code = "from __future__ import annotations\nimport os\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        assert record.future_annotations is True
        modules = [imp.module for imp in record.imports]
        assert "__future__" in modules
        assert "os" in modules


class TestResolvedTypes:
    """Tests for resolving types from annotations (iteration 2)."""

    def test_scenario_one_fully_annotated_function(self, tmp_path: Path) -> None:
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        assert fn.types_confidence is Confidence.EXPLICIT
        assert fn.types.parameters["a"].type_str == "int"
        assert fn.types.parameters["a"].confidence is Confidence.EXPLICIT
        assert fn.types.parameters["a"].provenance is Provenance.ANNOTATION
        assert fn.types.parameters["a"].raw == "int"
        assert fn.types.parameters["b"].type_str == "int"
        assert fn.types.return_type is not None
        assert fn.types.return_type.type_str == "int"

    def test_scenario_two_normalizes_legacy_annotations(self, tmp_path: Path) -> None:
        code = "from typing import Optional, List, Dict, Any\ndef foo(x: Optional[List[int]]) -> Dict[str, Any]:\n    pass\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        assert fn.types.parameters["x"].type_str == "list[int] | None"
        assert fn.types.parameters["x"].raw == "Optional[List[int]]"
        assert fn.types.return_type is not None
        assert fn.types.return_type.type_str == "dict[str, Any]"
        assert fn.types.return_type.raw == "Dict[str, Any]"

    def test_scenario_three_unannotated_types_are_none(self, tmp_path: Path) -> None:
        code = "def bar(x, y):\n    pass\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is None
        assert fn.types_confidence is Confidence.ABSENT

    def test_excludes_self_from_resolved_types(self, tmp_path: Path) -> None:
        code = "class C:\n    def method(self, x: int) -> str:\n        pass\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        meth = record.classes[0].methods[0]
        assert meth.types is not None
        assert "self" not in meth.types.parameters
        assert "x" in meth.types.parameters

    def test_partial_annotation(self, tmp_path: Path) -> None:
        code = "def foo(a: int, b) -> None:\n    pass\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        fn = record.functions[0]
        assert fn.types is not None
        assert "a" in fn.types.parameters
        assert "b" not in fn.types.parameters
        assert fn.types.return_type is not None
        assert fn.types.return_type.type_str == "None"


class TestLOC:
    """Tests for lines-of-code calculation (LOC)."""

    def test_loc_function_simple(self, tmp_path: Path) -> None:
        code = "def foo():\n    x = 1\n    return x\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        complexity = record.functions[0].complexity
        assert complexity is not None
        assert complexity.loc >= 1
        assert record.functions[0].complexity_confidence is Confidence.EXPLICIT

    def test_loc_excludes_docstring(self, tmp_path: Path) -> None:
        code_with_docstring = 'def foo():\n    """Docstring."""\n    return 1\n'
        code_without_docstring = "def foo():\n    return 1\n"
        file_with_docstring = _write_py(tmp_path / "with_docstring.py", code_with_docstring)
        file_without_docstring = _write_py(tmp_path / "without_docstring.py", code_without_docstring)
        record_with_docstring = extract_module_record(
            file_with_docstring, "with_docstring", is_stub=False, is_test=False, file_hash="abc"
        )
        record_without_docstring = extract_module_record(
            file_without_docstring, "without_docstring", is_stub=False, is_test=False, file_hash="abc"
        )
        assert record_with_docstring is not None
        assert record_without_docstring is not None
        assert record_with_docstring.functions[0].complexity.loc == record_without_docstring.functions[0].complexity.loc

    def test_loc_method(self, tmp_path: Path) -> None:
        code = "class C:\n    def method(self):\n        x = 1\n        y = 2\n        return x + y\n"
        f = _write_py(tmp_path / "mod.py", code)
        record = extract_module_record(f, "mod", is_stub=False, is_test=False, file_hash="abc")
        assert record is not None
        complexity = record.classes[0].methods[0].complexity
        assert complexity is not None
        assert complexity.loc >= 3
