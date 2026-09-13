# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for Python file discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docmethis_extract_python.static_extraction.configuration import ConfigurationDocmethis
from docmethis_extract_python.static_extraction.traversal import (
    discover_files,
    is_stub_file,
    is_test_file,
    resolve_module_name,
)

if TYPE_CHECKING:
    from pathlib import Path


def _create_file(source_path: Path) -> Path:
    """Create an empty file and its parent directories."""
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("", encoding="utf-8")
    return source_path


class TestDiscoverFiles:
    """Tests for recursive file discovery."""

    def test_basic_discovery(self, tmp_path: Path) -> None:
        _create_file(tmp_path / "a.py")
        _create_file(tmp_path / "b.py")
        _create_file(tmp_path / "pkg" / "c.py")
        config = ConfigurationDocmethis()
        files = discover_files(tmp_path, config)
        names = [f.name for f in files]
        assert "a.py" in names
        assert "b.py" in names
        assert "c.py" in names

    def test_ignore_non_python_files(self, tmp_path: Path) -> None:
        _create_file(tmp_path / "a.py")
        _create_file(tmp_path / "readme.md")
        _create_file(tmp_path / "data.json")
        config = ConfigurationDocmethis()
        files = discover_files(tmp_path, config)
        names = [f.name for f in files]
        assert "a.py" in names
        assert "readme.md" not in names
        assert "data.json" not in names

    def test_discover_pyi(self, tmp_path: Path) -> None:
        _create_file(tmp_path / "module.pyi")
        config = ConfigurationDocmethis()
        files = discover_files(tmp_path, config)
        assert any(f.name == "module.pyi" for f in files)

    def test_discover_pyw(self, tmp_path: Path) -> None:
        _create_file(tmp_path / "script.pyw")
        config = ConfigurationDocmethis()
        files = discover_files(tmp_path, config)
        assert any(f.name == "script.pyw" for f in files)

    def test_exclude_pycache(self, tmp_path: Path) -> None:
        _create_file(tmp_path / "__pycache__" / "module.py")
        _create_file(tmp_path / "ok.py")
        config = ConfigurationDocmethis()
        files = discover_files(tmp_path, config)
        names = [f.name for f in files]
        assert "ok.py" in names
        assert "module.py" not in names

    def test_exclude_venv(self, tmp_path: Path) -> None:
        _create_file(tmp_path / ".venv" / "lib" / "site.py")
        _create_file(tmp_path / "main.py")
        config = ConfigurationDocmethis()
        files = discover_files(tmp_path, config)
        names = [f.name for f in files]
        assert "main.py" in names
        assert "site.py" not in names

    def test_exclude_pattern_file(self, tmp_path: Path) -> None:
        _create_file(tmp_path / "model_pb2.py")
        _create_file(tmp_path / "real.py")
        config = ConfigurationDocmethis()
        files = discover_files(tmp_path, config)
        names = [f.name for f in files]
        assert "real.py" in names
        assert "model_pb2.py" not in names

    def test_exclude_setup_py(self, tmp_path: Path) -> None:
        _create_file(tmp_path / "setup.py")
        _create_file(tmp_path / "main.py")
        config = ConfigurationDocmethis()
        files = discover_files(tmp_path, config)
        names = [f.name for f in files]
        assert "main.py" in names
        assert "setup.py" not in names

    def test_extra_exclusion_patterns(self, tmp_path: Path) -> None:
        _create_file(tmp_path / "migrations" / "0001.py")
        _create_file(tmp_path / "main.py")
        config = ConfigurationDocmethis(extra_exclude_patterns=("migrations/",))
        files = discover_files(tmp_path, config)
        names = [f.name for f in files]
        assert "main.py" in names
        assert "0001.py" not in names

    def test_absolute_paths(self, tmp_path: Path) -> None:
        _create_file(tmp_path / "a.py")
        config = ConfigurationDocmethis()
        files = discover_files(tmp_path, config)
        for f in files:
            assert f.is_absolute()

    def test_sorted_results(self, tmp_path: Path) -> None:
        _create_file(tmp_path / "z.py")
        _create_file(tmp_path / "a.py")
        _create_file(tmp_path / "m.py")
        config = ConfigurationDocmethis()
        files = discover_files(tmp_path, config)
        assert files == sorted(files)

    def test_portion(self, root_portion: Path) -> None:
        config = ConfigurationDocmethis()
        files = discover_files(root_portion, config)
        # The portion fixture contains multiple Python files.
        assert len(files) > 5
        # All files exist.
        for f in files:
            assert f.exists()


class TestIsTestFile:
    """Tests for test-file detection."""

    def test_test_prefix(self, tmp_path: Path) -> None:
        source_path = tmp_path / "test_foo.py"
        assert is_test_file(source_path, tmp_path) is True

    def test_test_suffix(self, tmp_path: Path) -> None:
        source_path = tmp_path / "foo_test.py"
        assert is_test_file(source_path, tmp_path) is True

    def test_tests_suffix(self, tmp_path: Path) -> None:
        source_path = tmp_path / "foo_tests.py"
        assert is_test_file(source_path, tmp_path) is True

    def test_conftest(self, tmp_path: Path) -> None:
        source_path = tmp_path / "conftest.py"
        assert is_test_file(source_path, tmp_path) is True

    def test_in_tests_directory(self, tmp_path: Path) -> None:
        source_path = tmp_path / "tests" / "helpers.py"
        assert is_test_file(source_path, tmp_path) is True

    def test_in_test_directory(self, tmp_path: Path) -> None:
        source_path = tmp_path / "test" / "helpers.py"
        assert is_test_file(source_path, tmp_path) is True

    def test_file_normal(self, tmp_path: Path) -> None:
        source_path = tmp_path / "models.py"
        assert is_test_file(source_path, tmp_path) is False

    def test_normal_file_in_src(self, tmp_path: Path) -> None:
        source_path = tmp_path / "src" / "models.py"
        assert is_test_file(source_path, tmp_path) is False


class TestIsStubFile:
    """Tests for stub-file detection."""

    def test_pyi(self, tmp_path: Path) -> None:
        assert is_stub_file(tmp_path / "module.pyi") is True

    def test_py(self, tmp_path: Path) -> None:
        assert is_stub_file(tmp_path / "module.py") is False


class TestResolveModuleName:
    """Tests for module-name resolution."""

    def test_file_simple(self, tmp_path: Path) -> None:
        assert resolve_module_name(tmp_path / "interval.py", tmp_path) == "interval"

    def test_file_in_package(self, tmp_path: Path) -> None:
        assert resolve_module_name(tmp_path / "portion" / "interval.py", tmp_path) == "portion.interval"

    def test_init(self, tmp_path: Path) -> None:
        assert resolve_module_name(tmp_path / "portion" / "__init__.py", tmp_path) == "portion"

    def test_nested_init(self, tmp_path: Path) -> None:
        source_path = tmp_path / "portion" / "io" / "__init__.py"
        assert resolve_module_name(source_path, tmp_path) == "portion.io"

    def test_nested_module_depth(self, tmp_path: Path) -> None:
        source_path = tmp_path / "a" / "b" / "c" / "d.py"
        assert resolve_module_name(source_path, tmp_path) == "a.b.c.d"
