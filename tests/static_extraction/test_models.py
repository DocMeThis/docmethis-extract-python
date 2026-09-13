# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for the models module: enums, dataclasses, and serialization."""

from __future__ import annotations

import json
from pathlib import Path

from docmethis_extract_python.static_extraction.models import (
    ClassRecord,
    Confidence,
    FunctionRecord,
    MethodType,
    ModuleRecord,
    ParameterType,
    ProjectRecord,
    PropertyAccessor,
    Provenance,
    Visibility,
    deserialize_class_record,
    deserialize_function_record,
    deserialize_project_record,
    serialize_project,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_function_record(**kwargs) -> FunctionRecord:
    """Create a FunctionRecord with defaults for identity fields."""
    defaults = {
        "qualified_name": "module.my_function",
        "file_path": Path("/projet/module.py"),
        "line_start": 10,
        "line_end": 20,
        "col_start": 0,
        "col_end": 0,
        "method_kind": MethodType.FUNCTION,
        "is_async": False,
        "visibility": Visibility.PUBLIC,
        "parent_class": None,
        "parent_module": "module",
        "existing_docstring": None,
    }
    defaults.update(kwargs)
    return FunctionRecord(**defaults)


def _make_class_record(**kwargs) -> ClassRecord:
    """Create a ClassRecord with default values."""
    defaults = {
        "qualified_name": "module.MyClass",
        "file_path": Path("/projet/module.py"),
        "line_start": 1,
        "line_end": 50,
        "col_start": 0,
        "col_end": 0,
        "visibility": Visibility.PUBLIC,
        "parent_module": "module",
        "existing_docstring": "Class docstring.",
    }
    defaults.update(kwargs)
    return ClassRecord(**defaults)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    """Test enum values and serialization."""

    def test_confidence_values(self) -> None:
        assert Confidence.EXPLICIT.value == "explicit"
        assert Confidence.INFERRED_HIGH.value == "inferred_high"
        assert Confidence.INFERRED_LOW.value == "inferred_low"
        assert Confidence.ABSENT.value == "absent"

    def test_provenance_values(self) -> None:
        assert Provenance.ANNOTATION.value == "annotation"
        assert Provenance.STUB.value == "stub"
        assert Provenance.FUZZING.value == "fuzzing"

    def test_type_method_values(self) -> None:
        assert MethodType.FUNCTION.value == "function"
        assert MethodType.INSTANCE_METHOD.value == "instance_method"
        assert MethodType.CLASS_METHOD.value == "classmethod"
        assert MethodType.STATIC_METHOD.value == "staticmethod"
        assert MethodType.PROPERTY.value == "property"

    def test_property_accessor_values(self) -> None:
        assert PropertyAccessor.GETTER.value == "getter"
        assert PropertyAccessor.SETTER.value == "setter"

    def test_visibility_values(self) -> None:
        assert Visibility.PUBLIC.value == "public"
        assert Visibility.PROTECTED.value == "protected"
        assert Visibility.PRIVATE.value == "private"

    def test_type_parameter_values(self) -> None:
        assert ParameterType.POSITIONAL_ONLY.value == "positional_only"
        assert ParameterType.VAR_KEYWORD.value == "var_keyword"

    def test_confidence_from_value(self) -> None:
        assert Confidence("explicit") is Confidence.EXPLICIT
        assert Confidence("absent") is Confidence.ABSENT

    def test_provenance_from_value(self) -> None:
        assert Provenance("annotation") is Provenance.ANNOTATION


# ---------------------------------------------------------------------------
# FunctionRecord tests
# ---------------------------------------------------------------------------


class TestFunctionRecord:
    """Test FunctionRecord construction and default values."""

    def test_identity_fields(self) -> None:
        record = _make_function_record()
        assert record.qualified_name == "module.my_function"
        assert record.file_path == Path("/projet/module.py")
        assert record.method_kind is MethodType.FUNCTION
        assert record.is_async is False
        assert record.visibility is Visibility.PUBLIC

    def test_is_async_true(self) -> None:
        record = _make_function_record(is_async=True)
        assert record.is_async is True

    def test_progressive_fields_absent_by_default(self) -> None:
        record = _make_function_record()
        assert record.signature is None
        assert record.signature_confidence is Confidence.ABSENT
        assert record.types is None
        assert record.types_confidence is Confidence.ABSENT
        assert record.complexity is None
        assert record.profiling is None
        assert record.profiling_confidence is Confidence.ABSENT

    def test_cache_fields_default(self) -> None:
        record = _make_function_record()
        assert record.from_cache is False
        assert record.file_hash == ""
        assert record.analysis_timestamp == ""

    def test_parent_class_none_for_free_function(self) -> None:
        record = _make_function_record(parent_class=None)
        assert record.parent_class is None

    def test_parent_class_for_method(self) -> None:
        record = _make_function_record(
            qualified_name="module.MyClass.method",
            parent_class="MyClass",
            method_kind=MethodType.INSTANCE_METHOD,
        )
        assert record.parent_class == "MyClass"

    def test_existing_docstring(self) -> None:
        record = _make_function_record(existing_docstring="Existing documentation.")
        assert record.existing_docstring == "Existing documentation."


# ---------------------------------------------------------------------------
# ClassRecord tests
# ---------------------------------------------------------------------------


class TestClassRecord:
    """Test ClassRecord with nested methods."""

    def test_construction_empty(self) -> None:
        record = _make_class_record()
        assert record.methods == []
        assert record.hierarchy is None
        assert record.hierarchy_confidence is Confidence.ABSENT
        assert record.protocols == []

    def test_with_methods(self) -> None:
        method = _make_function_record(
            qualified_name="module.MyClass.method",
            parent_class="MyClass",
            method_kind=MethodType.INSTANCE_METHOD,
        )
        record = _make_class_record(methods=[method])
        assert len(record.methods) == 1
        assert record.methods[0].qualified_name == "module.MyClass.method"


# ---------------------------------------------------------------------------
# ModuleRecord tests
# ---------------------------------------------------------------------------


class TestModuleRecord:
    """Test ModuleRecord."""

    def test_construction_empty(self) -> None:
        record = ModuleRecord(
            file_path=Path("/projet/module.py"),
            module_name="module",
        )
        assert record.functions == []
        assert record.classes == []
        assert record.is_stub is False
        assert record.is_test is False

    def test_stub_and_test(self) -> None:
        record = ModuleRecord(
            file_path=Path("/projet/module.pyi"),
            module_name="module",
            is_stub=True,
        )
        assert record.is_stub is True

        record_test = ModuleRecord(
            file_path=Path("/projet/test_module.py"),
            module_name="test_module",
            is_test=True,
        )
        assert record_test.is_test is True


# ---------------------------------------------------------------------------
# ProjectRecord tests
# ---------------------------------------------------------------------------


class TestProjectRecord:
    """Test ProjectRecord."""

    def test_construction_empty(self) -> None:
        record = ProjectRecord(
            project_root=Path("/projet"),
            project_name="my_project",
        )
        assert record.modules == []
        assert record.quality_report is None
        assert record.project_version == ""


# ---------------------------------------------------------------------------
# Serialization/deserialization tests
# ---------------------------------------------------------------------------


class TestSerialisation:
    """Test JSON serialization round trips."""

    def test_function_record_roundtrip(self) -> None:
        original = _make_function_record(
            existing_docstring="My docstring.",
            file_hash="abc123",
            method_kind=MethodType.PROPERTY,
            property_accessor=PropertyAccessor.SETTER,
        )
        data = json.loads(
            serialize_project(
                ProjectRecord(
                    project_root=Path("/projet"),
                    project_name="test",
                    modules=[
                        ModuleRecord(
                            file_path=Path("/projet/module.py"),
                            module_name="module",
                            functions=[original],
                        ),
                    ],
                ),
            )
        )
        fn_data = data["modules"][0]["functions"][0]
        restored = deserialize_function_record(fn_data)

        assert restored.qualified_name == original.qualified_name
        assert restored.file_path == original.file_path
        assert restored.method_kind is original.method_kind
        assert restored.property_accessor is PropertyAccessor.SETTER
        assert restored.visibility is original.visibility
        assert restored.signature_confidence is Confidence.ABSENT
        assert restored.file_hash == "abc123"

    def test_class_record_roundtrip(self) -> None:
        method = _make_function_record(
            qualified_name="module.Cls.meth",
            parent_class="Cls",
            method_kind=MethodType.INSTANCE_METHOD,
        )
        original = _make_class_record(methods=[method])

        data = json.loads(
            serialize_project(
                ProjectRecord(
                    project_root=Path("/projet"),
                    project_name="test",
                    modules=[
                        ModuleRecord(
                            file_path=Path("/projet/module.py"),
                            module_name="module",
                            classes=[original],
                        ),
                    ],
                ),
            )
        )
        cls_data = data["modules"][0]["classes"][0]
        restored = deserialize_class_record(cls_data)

        assert restored.qualified_name == original.qualified_name
        assert len(restored.methods) == 1
        assert restored.methods[0].qualified_name == "module.Cls.meth"

    def test_project_record_roundtrip(self) -> None:
        original = ProjectRecord(
            project_root=Path("/projet"),
            project_name="my_project",
            project_version="1.0.0",
            modules=[
                ModuleRecord(
                    file_path=Path("/projet/a.py"),
                    module_name="a",
                    functions=[_make_function_record()],
                ),
            ],
        )
        json_str = serialize_project(original)
        data = json.loads(json_str)
        restored = deserialize_project_record(data)

        assert restored.project_root == original.project_root
        assert restored.project_name == "my_project"
        assert len(restored.modules) == 1
        assert len(restored.modules[0].functions) == 1

    def test_path_serializes_as_posix(self) -> None:
        record = _make_function_record(file_path=Path("C:/Users/test/module.py"))
        project_record = ProjectRecord(
            project_root=Path("C:/Users/test"),
            project_name="test",
            modules=[
                ModuleRecord(
                    file_path=Path("C:/Users/test/module.py"),
                    module_name="module",
                    functions=[record],
                ),
            ],
        )
        json_str = serialize_project(project_record)
        data = json.loads(json_str)
        # No backslash in JSON.
        assert "\\" not in data["project_root"]
        assert "\\" not in data["modules"][0]["file_path"]

    def test_enum_serializes_as_value(self) -> None:
        project_record = ProjectRecord(
            project_root=Path("/projet"),
            project_name="test",
            modules=[
                ModuleRecord(
                    file_path=Path("/projet/m.py"),
                    module_name="m",
                    functions=[_make_function_record()],
                ),
            ],
        )
        data = json.loads(serialize_project(project_record))
        fn = data["modules"][0]["functions"][0]
        assert fn["method_kind"] == "function"
        assert fn["visibility"] == "public"
        assert fn["signature_confidence"] == "absent"

    def test_valid_json(self) -> None:
        project_record = ProjectRecord(
            project_root=Path("/projet"),
            project_name="test",
        )
        json_str = serialize_project(project_record)
        # Must not raise an exception.
        json.loads(json_str)
