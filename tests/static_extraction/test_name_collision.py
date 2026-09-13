# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for name-collision resolution in inheritance mapping (phase 5, iteration 1.10).

Root cause: _build_mapping_names added the simple key (name_simple) for every class,
including nested classes. If "pkg.mod.Outer.Inner" and "pkg.mod.Inner" coexist,
both have name_simple="Inner", causing a collision and a corrupted hierarchy.

Fix: the simple key is added only for top-level classes. Nested classes are accessible
through the two-step _resolve_name_base lookup or _resolve_name_base_lexical (a bare
reference from an enclosing scope). The latter fixes a regression that treated a base
inherited from a nested sibling as external. See TestLexicalResolution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from docmethis_extract_python.static_extraction.ast_inheritance import (
    _build_mapping_names,
    _enclosing_scopes,
    _resolve_name_base,
    _resolve_name_base_lexical,
)
from docmethis_extract_python.static_extraction.models import (
    ClassRecord,
    ImportRecord,
    ModuleRecord,
    Visibility,
)

if TYPE_CHECKING:
    import pytest


def _cls(qualified_name: str, parent_module: str) -> ClassRecord:
    return ClassRecord(
        qualified_name=qualified_name,
        file_path=Path("/dummy.py"),
        line_start=1,
        line_end=10,
        col_start=0,
        col_end=0,
        visibility=Visibility.PUBLIC,
        parent_module=parent_module,
        existing_docstring=None,
    )


def _module(module_name: str, classes: list[ClassRecord], imports: list[ImportRecord] | None = None) -> ModuleRecord:
    return ModuleRecord(
        file_path=Path(f"/pkg/{module_name.replace('.', '/')}.py"),
        module_name=module_name,
        classes=classes,
        imports=imports or [],
    )


# ---------------------------------------------------------------------------
# Tests - _build_mapping_names
# ---------------------------------------------------------------------------


class TestBuildMappingNames:
    """Verify construction of the name -> qualified_name mapping."""

    def test_top_level_class_adds_simple_key(self) -> None:
        cls = _cls("pkg.mod.Foo", "pkg.mod")
        module = _module("pkg.mod", [cls])
        mapping = _build_mapping_names(module)
        assert mapping.get("Foo") == "pkg.mod.Foo"

    def test_nested_class_does_not_add_simple_key(self) -> None:
        """A nested class must not pollute the mapping with its simple key."""
        outer = _cls("pkg.mod.Outer", "pkg.mod")
        inner = _cls("pkg.mod.Outer.Inner", "pkg.mod")
        module = _module("pkg.mod", [outer, inner])
        mapping = _build_mapping_names(module)
        # "Inner" must not be in the mapping (it would be ambiguous with pkg.mod.Inner).
        assert "Inner" not in mapping or mapping.get("Inner") == "pkg.mod.Inner"
        # qualified_name must always be present.
        assert mapping.get("pkg.mod.Outer.Inner") == "pkg.mod.Outer.Inner"

    def test_no_collision_between_top_level_and_nested_same_simple_name(self, caplog: pytest.LogCaptureFixture) -> None:
        """A top-level and nested class with the same simple name must not warn."""
        top = _cls("pkg.mod.Inner", "pkg.mod")  # top-level Inner
        nested = _cls("pkg.mod.Outer.Inner", "pkg.mod")  # nested Inner
        outer = _cls("pkg.mod.Outer", "pkg.mod")
        module = _module("pkg.mod", [top, outer, nested])

        with caplog.at_level(logging.WARNING, logger="docmethis_extract_python.static_extraction.ast_inheritance"):
            mapping = _build_mapping_names(module)

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warning_records, f"Unexpected warning: {warning_records}"
        # The simple key "Inner" points to the top-level class.
        assert mapping.get("Inner") == "pkg.mod.Inner"
        # The nested class remains accessible by qualified_name.
        assert mapping.get("pkg.mod.Outer.Inner") == "pkg.mod.Outer.Inner"

    def test_two_different_top_level_classes_do_not_collide(self) -> None:
        """Two top-level classes with different names do not collide."""
        cls_a = _cls("pkg.mod.Alpha", "pkg.mod")
        cls_b = _cls("pkg.mod.Beta", "pkg.mod")
        module = _module("pkg.mod", [cls_a, cls_b])
        mapping = _build_mapping_names(module)
        assert mapping.get("Alpha") == "pkg.mod.Alpha"
        assert mapping.get("Beta") == "pkg.mod.Beta"

    def test_qualified_name_is_always_present(self) -> None:
        """The qualified_name key is always registered (top-level or nested)."""
        nested = _cls("pkg.mod.Outer.Inner", "pkg.mod")
        outer = _cls("pkg.mod.Outer", "pkg.mod")
        module = _module("pkg.mod", [outer, nested])
        mapping = _build_mapping_names(module)
        assert mapping.get("pkg.mod.Outer") == "pkg.mod.Outer"
        assert mapping.get("pkg.mod.Outer.Inner") == "pkg.mod.Outer.Inner"

    def test_two_star_imports_do_not_collide(self, caplog: pytest.LogCaptureFixture) -> None:
        """Two `from X import *` imports must not collide on the literal '*' key."""
        module = _module(
            "pkg.mod",
            [],
            [
                ImportRecord(module="pkg.a", name="*", alias=None),
                ImportRecord(module="pkg.b", name="*", alias=None),
            ],
        )

        with caplog.at_level(logging.DEBUG, logger="docmethis_extract_python.static_extraction.ast_inheritance"):
            mapping = _build_mapping_names(module)

        assert "*" not in mapping
        assert not [r for r in caplog.records if "Collision" in r.getMessage()]

    def test_import_class_collision_logs_debug_not_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """An import colliding with a local class logs debug, not warning."""
        cls = _cls("pkg.mod.Foo", "pkg.mod")
        imp = ImportRecord(module="other", name="Foo", alias=None)
        module = _module("pkg.mod", [cls], imports=[imp])

        with caplog.at_level(logging.WARNING, logger="docmethis_extract_python.static_extraction.ast_inheritance"):
            _build_mapping_names(module)

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warning_records


# ---------------------------------------------------------------------------
# Tests - _resolve_name_base with nested classes.
# ---------------------------------------------------------------------------


class TestResolveNestedBaseName:
    """Verify that nested classes resolve through the two-step lookup."""

    def test_nested_class_resolves_via_outer(self) -> None:
        """'Outer.Inner' -> look up 'Outer' -> reconstruct 'pkg.mod.Outer.Inner'."""
        outer = _cls("pkg.mod.Outer", "pkg.mod")
        inner = _cls("pkg.mod.Outer.Inner", "pkg.mod")
        module = _module("pkg.mod", [outer, inner])
        mapping = _build_mapping_names(module)

        result = _resolve_name_base("Outer.Inner", mapping)
        assert result == "pkg.mod.Outer.Inner"

    def test_top_level_class_resolves_directly(self) -> None:
        """'Foo' -> direct lookup -> 'pkg.mod.Foo'."""
        cls = _cls("pkg.mod.Foo", "pkg.mod")
        module = _module("pkg.mod", [cls])
        mapping = _build_mapping_names(module)

        result = _resolve_name_base("Foo", mapping)
        assert result == "pkg.mod.Foo"

    def test_unknown_name_is_returned_unchanged(self) -> None:
        """An unknown name is returned unchanged (external class)."""
        module = _module("pkg.mod", [])
        mapping = _build_mapping_names(module)
        result = _resolve_name_base("ExternalLib", mapping)
        assert result == "ExternalLib"


# ---------------------------------------------------------------------------
# Tests - lexical base scope (phase 5 regression test).
# ---------------------------------------------------------------------------


class TestEnclosingScopes:
    """Scope traversal stops at the module and never reaches the package."""

    def test_nested_class(self) -> None:
        assert _enclosing_scopes("pkg.mod.Outer.Cls", "pkg.mod") == ["pkg.mod.Outer", "pkg.mod"]

    def test_doubly_nested_class(self) -> None:
        scopes = _enclosing_scopes("pkg.mod.A.B.Cls", "pkg.mod")
        assert scopes == ["pkg.mod.A.B", "pkg.mod.A", "pkg.mod"]

    def test_top_level_class(self) -> None:
        assert _enclosing_scopes("pkg.mod.Cls", "pkg.mod") == ["pkg.mod"]

    def test_does_not_reach_package(self) -> None:
        """'pkg.other_module' is not a lexical scope of 'pkg.mod.Cls'."""
        assert "pkg" not in _enclosing_scopes("pkg.mod.Outer.Cls", "pkg.mod")


class TestLexicalResolution:
    """`class Child(Base)` in `Outer` refers to `Outer.Base`, not an external class."""

    def test_nested_sibling_resolves(self) -> None:
        """Regression case: Base has no simple key in the mapping."""
        outer = _cls("pkg.mod.Outer", "pkg.mod")
        base = _cls("pkg.mod.Outer.Base", "pkg.mod")
        child = _cls("pkg.mod.Outer.Child", "pkg.mod")
        module = _module("pkg.mod", [outer, base, child])
        mapping = _build_mapping_names(module)
        index = {c.qualified_name: c for c in module.classes}

        assert "Base" not in mapping
        assert _resolve_name_base_lexical("Base", child, mapping, index) == "pkg.mod.Outer.Base"

    def test_nested_class_shadows_top_level_homonym(self) -> None:
        """The closest scope wins: Other.Inner beats the module-level Inner."""
        inner_top = _cls("pkg.mod.Inner", "pkg.mod")
        other = _cls("pkg.mod.Other", "pkg.mod")
        inner_nested = _cls("pkg.mod.Other.Inner", "pkg.mod")
        child = _cls("pkg.mod.Other.Child", "pkg.mod")
        module = _module("pkg.mod", [inner_top, other, inner_nested, child])
        mapping = _build_mapping_names(module)
        index = {c.qualified_name: c for c in module.classes}

        assert _resolve_name_base_lexical("Inner", child, mapping, index) == "pkg.mod.Other.Inner"
        # From module scope, the same name resolves to the top-level class.
        other_top_child = _cls("pkg.mod.OtherChild", "pkg.mod")
        assert _resolve_name_base_lexical("Inner", other_top_child, mapping, index) == "pkg.mod.Inner"

    def test_top_level_unchanged(self) -> None:
        """No regression on the nominal top-level inheritance path."""
        base = _cls("pkg.mod.Base", "pkg.mod")
        child = _cls("pkg.mod.Child", "pkg.mod")
        module = _module("pkg.mod", [base, child])
        mapping = _build_mapping_names(module)
        index = {c.qualified_name: c for c in module.classes}

        assert _resolve_name_base_lexical("Base", child, mapping, index) == "pkg.mod.Base"

    def test_fallback_import_and_external(self) -> None:
        """Without a scoped homonym, fall back to the flat mapping (imports, built-ins)."""
        child = _cls("pkg.mod.Outer.Child", "pkg.mod")
        module = _module("pkg.mod", [child], [ImportRecord(module="lib", name="Base", alias=None)])
        mapping = _build_mapping_names(module)
        index = {c.qualified_name: c for c in module.classes}

        assert _resolve_name_base_lexical("Base", child, mapping, index) == "lib.Base"
        assert _resolve_name_base_lexical("Exception", child, mapping, index) == "Exception"

    def test_homonym_in_other_package_module_is_ignored(self) -> None:
        """A class from a neighboring module is not in lexical scope."""
        child = _cls("pkg.mod.Outer.Child", "pkg.mod")
        module = _module("pkg.mod", [child])
        mapping = _build_mapping_names(module)
        # 'pkg.Base' exists in the package __init__ but is not lexically visible.
        index = {child.qualified_name: child, "pkg.Base": _cls("pkg.Base", "pkg")}

        assert _resolve_name_base_lexical("Base", child, mapping, index) == "Base"
