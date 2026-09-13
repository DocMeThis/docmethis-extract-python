# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Unit tests for classify_imports."""

from __future__ import annotations

import textwrap
from pathlib import Path

from docmethis_extract_python.static_extraction.models import ImportRecord, ModuleRecord, ProjectRecord
from docmethis_extract_python.static_extraction.traversal import (
    _classify_import,
    _collect_used_names,
    _imported_local_name,
    _load_third_party_dependencies,
    classify_imports,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _imp(module: str, name: str | None = None, alias: str | None = None, *, type_checking_only: bool = False) -> ImportRecord:
    return ImportRecord(module=module, name=name, alias=alias, type_checking_only=type_checking_only)


def _module_with_imports(
    imports: list[ImportRecord],
    file_path: Path = Path("mod.py"),
    module_name: str = "mod",
) -> ModuleRecord:
    return ModuleRecord(file_path=file_path, module_name=module_name, imports=imports)


def _project(module: ModuleRecord, project_root: Path = Path()) -> ProjectRecord:
    return ProjectRecord(project_root=project_root, project_name="test", modules=[module])


# ---------------------------------------------------------------------------
# Tests _classify_import
# ---------------------------------------------------------------------------


class TestClassifyImport:
    """Unit tests for _classify_import."""

    def _third_party(self) -> frozenset[str]:
        return frozenset({"requests", "httpx", "numpy"})

    def test_stdlib(self) -> None:
        imp = _imp("os")
        assert _classify_import(imp, self._third_party()) == "stdlib"

    def test_stdlib_sous_module(self) -> None:
        imp = _imp("os.path")
        assert _classify_import(imp, self._third_party()) == "stdlib"

    def test_stdlib_from(self) -> None:
        imp = _imp("collections", name="deque")
        assert _classify_import(imp, self._third_party()) == "stdlib"

    def test_third_party(self) -> None:
        imp = _imp("requests")
        assert _classify_import(imp, self._third_party()) == "third_party"

    def test_third_party_sous_module(self) -> None:
        imp = _imp("requests.adapters")
        assert _classify_import(imp, self._third_party()) == "third_party"

    def test_local(self) -> None:
        imp = _imp("mypackage.utils")
        assert _classify_import(imp, self._third_party()) == "local"

    def test_relative_is_always_local(self) -> None:
        imp = _imp(".utils")
        assert _classify_import(imp, self._third_party()) == "local"

    def test_double_dot_relative_is_local(self) -> None:
        imp = _imp("..models")
        assert _classify_import(imp, self._third_party()) == "local"


# ---------------------------------------------------------------------------
# Tests _imported_local_name
# ---------------------------------------------------------------------------


class TestImportedLocalName:
    """Unit tests for _imported_local_name."""

    def test_import_module_simple(self) -> None:
        imp = _imp("os")
        assert _imported_local_name(imp) == "os"

    def test_import_module_qualifie(self) -> None:
        imp = _imp("os.path")
        assert _imported_local_name(imp) == "os"

    def test_import_module_with_alias(self) -> None:
        imp = _imp("numpy", alias="np")
        assert _imported_local_name(imp) == "np"

    def test_from_import(self) -> None:
        imp = _imp("os.path", name="join")
        assert _imported_local_name(imp) == "join"

    def test_from_import_with_alias(self) -> None:
        imp = _imp("os.path", name="join", alias="pjoin")
        assert _imported_local_name(imp) == "pjoin"

    def test_star_import(self) -> None:
        imp = _imp("os", name="*")
        assert _imported_local_name(imp) == "*"


# ---------------------------------------------------------------------------
# Tests _collect_used_names
# ---------------------------------------------------------------------------


class TestCollectUsedNames:
    """Unit tests for _collect_used_names."""

    def test_names_simples(self, tmp_path: Path) -> None:
        src = textwrap.dedent("""\
            import os
            x = os.getcwd()
            y = len(x)
        """)
        f = tmp_path / "mod.py"
        f.write_text(src)
        names = _collect_used_names(f)
        assert "os" in names
        assert "x" in names
        assert "len" in names

    def test_missing_file(self, tmp_path: Path) -> None:
        names = _collect_used_names(tmp_path / "ghost.py")
        assert names == frozenset()


# ---------------------------------------------------------------------------
# Tests _load_third_party_dependencies
# ---------------------------------------------------------------------------


class TestLoadThirdPartyDependencies:
    """Unit tests for _load_third_party_dependencies."""

    def test_lit_dependencies(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["httpx>=0.28", "requests"]\n')
        deps = _load_third_party_dependencies(tmp_path)
        assert "httpx" in deps
        assert "requests" in deps

    def test_normalizes_hyphens(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["some-lib>=1.0"]\n')
        deps = _load_third_party_dependencies(tmp_path)
        assert "some_lib" in deps

    def test_lit_dependency_groups(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = []\n[dependency-groups]\ndev = ["pytest>=8.0"]\n')
        deps = _load_third_party_dependencies(tmp_path)
        assert "pytest" in deps

    def test_without_pyproject(self, tmp_path: Path) -> None:
        deps = _load_third_party_dependencies(tmp_path)
        assert deps == frozenset()


# ---------------------------------------------------------------------------
# Tests classify_imports (integration)
# ---------------------------------------------------------------------------


class TestClassifierImports:
    """Integration tests for classify_imports."""

    def test_classification_stdlib(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("import os\nos.getcwd()\n")
        imp = _imp("os")
        module = _module_with_imports([imp], file_path=f)
        project_record = _project(module, project_root=tmp_path)

        classify_imports(project_record)

        assert imp.classification == "stdlib"

    def test_classifies_third_party(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["httpx>=0.28"]\n')
        f = tmp_path / "mod.py"
        f.write_text("import httpx\nhttpx.get('/')\n")
        imp = _imp("httpx")
        module = _module_with_imports([imp], file_path=f)
        project_record = _project(module, project_root=tmp_path)

        classify_imports(project_record)

        assert imp.classification == "third_party"

    def test_classifies_local(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("from myapp import utils\nutils.run()\n")
        imp = _imp("myapp", name="utils")
        module = _module_with_imports([imp], file_path=f)
        project_record = _project(module, project_root=tmp_path)

        classify_imports(project_record)

        assert imp.classification == "local"

    def test_used_import_is_not_marked_unused(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("import os\nos.getcwd()\n")
        imp = _imp("os")
        module = _module_with_imports([imp], file_path=f)
        project_record = _project(module, project_root=tmp_path)

        classify_imports(project_record)

        assert imp.is_unused is False

    def test_unused_import_is_marked_unused(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("import os\n\ndef foo():\n    pass\n")
        imp = _imp("os")
        module = _module_with_imports([imp], file_path=f)
        project_record = _project(module, project_root=tmp_path)

        classify_imports(project_record)

        assert imp.is_unused is True

    def test_used_from_import(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("from os.path import join\nx = join('a', 'b')\n")
        imp = _imp("os.path", name="join")
        module = _module_with_imports([imp], file_path=f)
        project_record = _project(module, project_root=tmp_path)

        classify_imports(project_record)

        assert imp.is_unused is False

    def test_unused_from_import(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("from os.path import join\n\ndef foo():\n    pass\n")
        imp = _imp("os.path", name="join")
        module = _module_with_imports([imp], file_path=f)
        project_record = _project(module, project_root=tmp_path)

        classify_imports(project_record)

        assert imp.is_unused is True

    def test_used_alias(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("import numpy as np\nnp.array([1])\n")
        imp = _imp("numpy", alias="np")
        module = _module_with_imports([imp], file_path=f)
        project_record = _project(module, project_root=tmp_path)

        classify_imports(project_record)

        assert imp.is_unused is False

    def test_unused_alias(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("import numpy as np\n\ndef foo():\n    pass\n")
        imp = _imp("numpy", alias="np")
        module = _module_with_imports([imp], file_path=f)
        project_record = _project(module, project_root=tmp_path)

        classify_imports(project_record)

        assert imp.is_unused is True

    def test_star_import_is_never_unused(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("from os import *\n")
        imp = _imp("os", name="*")
        module = _module_with_imports([imp], file_path=f)
        project_record = _project(module, project_root=tmp_path)

        classify_imports(project_record)

        assert imp.is_unused is False

    def test_type_checking_import_is_never_unused(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("from __future__ import annotations\nfrom myapp.types import MyType\n")
        imp = _imp("myapp.types", name="MyType", type_checking_only=True)
        module = _module_with_imports([imp], file_path=f)
        project_record = _project(module, project_root=tmp_path)

        classify_imports(project_record)

        assert imp.is_unused is False

    def test_relative_import_is_local(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("from . import utils\nutils.run()\n")
        imp = _imp(".utils")
        module = _module_with_imports([imp], file_path=f)
        project_record = _project(module, project_root=tmp_path)

        classify_imports(project_record)

        assert imp.classification == "local"
