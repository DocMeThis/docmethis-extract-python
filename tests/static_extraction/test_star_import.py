# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for internal star-import resolution (phase 4, iteration 1.10).

Verify that `from X import *` resolves when X is internal to the project,
using __all__ when defined and public ModuleRecord names otherwise.
"""

from __future__ import annotations

import ast
import logging
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from docmethis_extract_python.static_extraction.callgraph_ast import (
    _ExportContext,
    _module_exported_names,
    _module_mapping,
    _module_origins,
)
from docmethis_extract_python.static_extraction.models import (
    FunctionRecord,
    ImportRecord,
    MethodType,
    ModuleRecord,
    Visibility,
)

if TYPE_CHECKING:
    import pytest


def _parse(code: str) -> ast.Module:
    return ast.parse(textwrap.dedent(code))


def _fn(qname: str, parent_module: str, line: int = 1) -> FunctionRecord:
    return FunctionRecord(
        qualified_name=qname,
        file_path=Path("/dummy.py"),
        line_start=line,
        line_end=line + 2,
        col_start=0,
        col_end=0,
        method_kind=MethodType.FUNCTION,
        visibility=Visibility.PUBLIC,
        parent_class=None,
        parent_module=parent_module,
        existing_docstring=None,
    )


def _module(
    module_name: str,
    functions: list[FunctionRecord],
    imports: list[ImportRecord] | None = None,
    *,
    file_path: Path | None = None,
) -> ModuleRecord:
    return ModuleRecord(
        file_path=file_path or Path(f"/pkg/{module_name.replace('.', '/')}.py"),
        module_name=module_name,
        functions=functions,
        imports=imports or [],
    )


# ---------------------------------------------------------------------------
# Tests - _module_exported_names
# ---------------------------------------------------------------------------


class TestModuleExportedNames:
    """Verify extraction of names exported by a module."""

    def test_without_all_returns_public_names(self) -> None:
        fn_pub = _fn("pkg.mod.foo", "pkg.mod")
        fn_priv = _fn("pkg.mod._bar", "pkg.mod")
        module = _module("pkg.mod", [fn_pub, fn_priv])
        tree = _parse("def foo(): pass\ndef _bar(): pass\n")
        names = _module_exported_names(module, tree)
        assert "foo" in names
        assert "_bar" not in names

    def test_with_all_returns_only_all(self) -> None:
        fn_a = _fn("pkg.mod.alpha", "pkg.mod")
        fn_b = _fn("pkg.mod.beta", "pkg.mod")
        fn_c = _fn("pkg.mod.gamma", "pkg.mod")
        module = _module("pkg.mod", [fn_a, fn_b, fn_c])
        tree = _parse('__all__ = ["alpha", "beta"]\ndef alpha(): pass\ndef beta(): pass\ndef gamma(): pass\n')
        names = _module_exported_names(module, tree)
        assert names == frozenset({"alpha", "beta"})
        assert "gamma" not in names

    def test_all_tuple(self) -> None:
        fn_a = _fn("pkg.mod.alpha", "pkg.mod")
        module = _module("pkg.mod", [fn_a])
        tree = _parse('__all__ = ("alpha",)\ndef alpha(): pass\n')
        names = _module_exported_names(module, tree)
        assert names == frozenset({"alpha"})

    def test_module_empty(self) -> None:
        module = _module("pkg.mod", [])
        tree = _parse("")
        names = _module_exported_names(module, tree)
        assert names == frozenset()


# ---------------------------------------------------------------------------
# Tests - _module_mapping with internal star imports.
# ---------------------------------------------------------------------------


class TestStarImportMapping:
    """Verify star-import resolution in _module_mapping."""

    def _make_export_context(self, module_name: str, names: set[str]) -> _ExportContext:
        return _ExportContext(exported_names={module_name: frozenset(names)})

    def test_internal_star_import_resolves(self) -> None:
        """From pkg.utils import * maps the three symbols from pkg.utils."""
        imp = ImportRecord(module="pkg.utils", name="*", alias=None)
        module = _module("pkg.main", [], imports=[imp])
        export_context = self._make_export_context("pkg.utils", {"foo", "bar", "baz"})

        mapping = _module_mapping(module, export_context)

        assert mapping.get("foo") == "pkg.utils.foo"
        assert mapping.get("bar") == "pkg.utils.bar"
        assert mapping.get("baz") == "pkg.utils.baz"

    def test_external_star_import_is_unresolved(self, caplog: pytest.LogCaptureFixture) -> None:
        """From external_lib import * is unresolved and logs debug, not warning."""
        imp = ImportRecord(module="external_lib", name="*", alias=None)
        module = _module("pkg.main", [], imports=[imp])
        export_context = _ExportContext()  # pkg.main only, not external_lib.

        with caplog.at_level(logging.DEBUG, logger="docmethis_extract_python.static_extraction.callgraph_ast"):
            mapping = _module_mapping(module, export_context)

        assert "external_lib" not in str(mapping.values())
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warning_records

    def test_star_import_without_export_context(self, caplog: pytest.LogCaptureFixture) -> None:
        """Without an export context (None), star import logs debug, not warning."""
        imp = ImportRecord(module="pkg.utils", name="*", alias=None)
        module = _module("pkg.main", [], imports=[imp])

        with caplog.at_level(logging.DEBUG, logger="docmethis_extract_python.static_extraction.callgraph_ast"):
            mapping = _module_mapping(module, None)

        assert not mapping  # nothing resolved
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warning_records

    def test_explicit_import_takes_priority_over_star(self) -> None:
        """An explicit import takes priority over the same name from a star import."""
        # The explicit import comes after the star import in the file.
        star_imp = ImportRecord(module="pkg.utils", name="*", alias=None)
        explicit_imp = ImportRecord(module="pkg.other", name="foo", alias=None)
        module = _module("pkg.main", [], imports=[star_imp, explicit_imp])
        export_context = self._make_export_context("pkg.utils", {"foo"})

        mapping = _module_mapping(module, export_context)

        # The explicit import (pkg.other.foo) must win over the star import (pkg.utils.foo).
        assert mapping.get("foo") == "pkg.other.foo"

    def test_internal_star_import_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """A resolved internal star import emits an info log."""
        imp = ImportRecord(module="pkg.utils", name="*", alias=None)
        module = _module("pkg.main", [], imports=[imp])
        export_context = self._make_export_context("pkg.utils", {"foo", "bar"})

        with caplog.at_level(logging.INFO, logger="docmethis_extract_python.static_extraction.callgraph_ast"):
            _module_mapping(module, export_context)

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("pkg.utils" in r.message for r in info_records)

    def test_relative_star_import_resolves(self) -> None:
        """From .utils import * in pkg.main is absolutized to pkg.utils."""
        imp = ImportRecord(module=".utils", name="*", alias=None)
        module = _module("pkg.main", [], imports=[imp], file_path=Path("/pkg/main.py"))
        export_context = self._make_export_context("pkg.utils", {"foo", "bar"})

        mapping = _module_mapping(module, export_context)

        assert mapping.get("foo") == "pkg.utils.foo"
        assert mapping.get("bar") == "pkg.utils.bar"

    def test_relative_star_import_from_init(self) -> None:
        """In __init__.py, the anchor is the package itself, not its parent."""
        imp = ImportRecord(module=".utils", name="*", alias=None)
        module = _module("pkg", [], imports=[imp], file_path=Path("/pkg/__init__.py"))
        export_context = self._make_export_context("pkg.utils", {"foo"})

        mapping = _module_mapping(module, export_context)

        assert mapping.get("foo") == "pkg.utils.foo"

    def test_relative_star_import_level_two(self) -> None:
        """From ..core import * in pkg.sub.mod resolves to pkg.core."""
        imp = ImportRecord(module="..core", name="*", alias=None)
        module = _module("pkg.sub.mod", [], imports=[imp], file_path=Path("/pkg/sub/mod.py"))
        export_context = self._make_export_context("pkg.core", {"foo"})

        mapping = _module_mapping(module, export_context)

        assert mapping.get("foo") == "pkg.core.foo"

    def test_single_dot_star_import(self) -> None:
        """From . import * resolves to the current package without a module suffix."""
        imp = ImportRecord(module=".", name="*", alias=None)
        module = _module("pkg.main", [], imports=[imp], file_path=Path("/pkg/main.py"))
        export_context = self._make_export_context("pkg", {"helper"})

        mapping = _module_mapping(module, export_context)

        assert mapping.get("helper") == "pkg.helper"

    def test_relative_star_import_above_package(self, caplog: pytest.LogCaptureFixture) -> None:
        """A relative import above the root is unresolved and emits no warning."""
        imp = ImportRecord(module="..autre", name="*", alias=None)
        module = _module("main", [], imports=[imp], file_path=Path("/main.py"))
        export_context = self._make_export_context("other", {"foo"})

        with caplog.at_level(logging.DEBUG, logger="docmethis_extract_python.static_extraction.callgraph_ast"):
            mapping = _module_mapping(module, export_context)

        assert "foo" not in mapping
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_relative_explicit_import_resolves(self) -> None:
        """From .utils import helper resolves to pkg.utils.helper, not .utils.helper."""
        imp = ImportRecord(module=".utils", name="helper", alias=None)
        module = _module("pkg.main", [], imports=[imp], file_path=Path("/pkg/main.py"))

        mapping = _module_mapping(module, None)

        assert mapping.get("helper") == "pkg.utils.helper"

    def test_relative_explicit_import_with_alias(self) -> None:
        """The alias remains the key; only the target is absolutized."""
        imp = ImportRecord(module="..core", name="run", alias="lancer")
        module = _module("pkg.sub.mod", [], imports=[imp], file_path=Path("/pkg/sub/mod.py"))

        mapping = _module_mapping(module, None)

        assert mapping.get("lancer") == "pkg.core.run"

    def test_star_import_depth_one(self) -> None:
        """Depth one only: star-import re-exports are not transitive."""
        # B does 'from C import *', but calculate B's exports only from its own functions,
        # not from C.
        fn_in_b = _fn("pkg.b.b_func", "pkg.b")
        _module("pkg.b", [fn_in_b])
        # Pass B's exported names = {"b_func"}, not C's functions.
        export_context = _ExportContext(exported_names={"pkg.b": frozenset({"b_func"})})

        imp = ImportRecord(module="pkg.b", name="*", alias=None)
        module_a = _module("pkg.a", [], imports=[imp])

        mapping = _module_mapping(module_a, export_context)
        assert mapping.get("b_func") == "pkg.b.b_func"
        # c_func (hypothetical re-export) is NOT in the mapping.
        assert "c_func" not in mapping


# ---------------------------------------------------------------------------
# Tests - re-exports (an exported name is not necessarily defined in its export module).
# ---------------------------------------------------------------------------


class TestReexports:
    """`pkg/__init__.py` does `from .api import f` + `__all__ = ["f"]`.

    Without following the origin, target 'pkg.f' does not exist in any index and the edge
    is classified as external with the project package as the external library.
    """

    INIT = ImportRecord(module="pkg.api", name="f", alias=None)

    def _package_exports(self) -> _ExportContext:
        """Package pkg re-exports f, which is actually defined in pkg.api."""
        init = _module("pkg", [], imports=[self.INIT], file_path=Path("/pkg/__init__.py"))
        api = _module("pkg.api", [_fn("pkg.api.f", "pkg.api")])
        return _ExportContext(
            exported_names={"pkg": frozenset({"f"}), "pkg.api": frozenset({"f"})},
            origins={"pkg": _module_origins(init), "pkg.api": _module_origins(api)},
            known_symbols=frozenset({"pkg.api.f"}),
        )

    def test_star_import_follows_reexport(self) -> None:
        imp = ImportRecord(module="pkg", name="*", alias=None)
        module = _module("conso", [], imports=[imp])

        mapping = _module_mapping(module, self._package_exports())

        assert mapping.get("f") == "pkg.api.f"

    def test_explicit_import_follows_reexport(self) -> None:
        """The same defect affected `from pkg import f`, which targeted 'pkg.f'."""
        imp = ImportRecord(module="pkg", name="f", alias=None)
        module = _module("conso", [], imports=[imp])

        mapping = _module_mapping(module, self._package_exports())

        assert mapping.get("f") == "pkg.api.f"

    def test_locally_defined_symbol_is_unchanged(self) -> None:
        """Do not follow an origin when the symbol is defined in the imported module."""
        imp = ImportRecord(module="pkg.api", name="f", alias=None)
        module = _module("conso", [], imports=[imp])

        mapping = _module_mapping(module, self._package_exports())

        assert mapping.get("f") == "pkg.api.f"

    def test_external_symbol_is_unchanged(self) -> None:
        """An external target is returned unchanged without looping."""
        imp = ImportRecord(module="requests", name="get", alias=None)
        module = _module("conso", [], imports=[imp])

        mapping = _module_mapping(module, self._package_exports())

        assert mapping.get("get") == "requests.get"

    def test_reexport_chain_is_followed(self) -> None:
        """The pkg -> pkg.api -> pkg.api.impl chain is followed to the end."""
        init = _module("pkg", [], imports=[ImportRecord(module="pkg.api", name="f", alias=None)])
        api = _module("pkg.api", [], imports=[ImportRecord(module="pkg.api.impl", name="f", alias=None)])
        impl = _module("pkg.api.impl", [_fn("pkg.api.impl.f", "pkg.api.impl")])
        exports = _ExportContext(
            exported_names={"pkg": frozenset({"f"})},
            origins={
                "pkg": _module_origins(init),
                "pkg.api": _module_origins(api),
                "pkg.api.impl": _module_origins(impl),
            },
            known_symbols=frozenset({"pkg.api.impl.f"}),
        )

        module = _module("conso", [], imports=[ImportRecord(module="pkg", name="*", alias=None)])
        mapping = _module_mapping(module, exports)

        assert mapping.get("f") == "pkg.api.impl.f"

    def test_reexport_cycle_does_not_loop(self) -> None:
        """Two mutually importing modules stop without an infinite loop."""
        a = _module("pkg.a", [], imports=[ImportRecord(module="pkg.b", name="f", alias=None)])
        b = _module("pkg.b", [], imports=[ImportRecord(module="pkg.a", name="f", alias=None)])
        exports = _ExportContext(
            exported_names={"pkg.a": frozenset({"f"})},
            origins={"pkg.a": _module_origins(a), "pkg.b": _module_origins(b)},
            known_symbols=frozenset(),  # f is not defined anywhere.
        )

        module = _module("conso", [], imports=[ImportRecord(module="pkg.a", name="*", alias=None)])
        mapping = _module_mapping(module, exports)

        assert mapping.get("f") in {"pkg.a.f", "pkg.b.f"}

    def test_local_definition_takes_priority_over_import(self) -> None:
        """`from .x import f` followed by `def f` in the same module: the definition wins."""
        mod = _module(
            "pkg.mixte",
            [_fn("pkg.mixte.f", "pkg.mixte")],
            imports=[ImportRecord(module="pkg.autre", name="f", alias=None)],
        )

        assert _module_origins(mod)["f"] == "pkg.mixte.f"
