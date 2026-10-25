# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for Python file discovery."""

from __future__ import annotations

import subprocess
from pathlib import Path

from docmethis_extract_python.static_extraction.configuration import ConfigurationDocmethis
from docmethis_extract_python.static_extraction.traversal import (
    discover_files,
    is_stub_file,
    is_test_file,
    resolve_module_name,
)


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

    def test_gitignore_excludes_ignored_untracked_files_but_keeps_tracked_files(self, tmp_path: Path) -> None:
        """Git exclusions apply to untracked files, not to files already in the index.

        Parameters
        ----------
        tmp_path : Path
            A test method that operates on the directory given by tmp_path. It writes a .gitignore file excluding ignored.py and
            forced.py into that directory, creates visible.py, ignored.py, and forced.py there, initializes a Git repository in
            it, adds .gitignore and force-adds forced.py to the index, then calls discover_files(tmp_path,
            ConfigurationDocmethis()). It asserts that the returned relative paths equal {Path("visible.py"), Path("forced.py")}.
        """
        (tmp_path / ".gitignore").write_text("ignored.py\nforced.py\n", encoding="utf-8")
        _create_file(tmp_path / "visible.py")
        _create_file(tmp_path / "ignored.py")
        _create_file(tmp_path / "forced.py")
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "-f", "forced.py"], cwd=tmp_path, check=True, capture_output=True)

        files = {path.relative_to(tmp_path) for path in discover_files(tmp_path, ConfigurationDocmethis())}

        assert files == {Path("visible.py"), Path("forced.py")}

    def test_git_discovery_from_nested_root_uses_repository_paths(self, tmp_path: Path) -> None:
        """A nested analysis root does not include sibling files from the worktree.

        Parameters
        ----------
        tmp_path : Path
            The test creates files under tmp_path and a nested project directory, initializes a git repository at tmp_path, then
            calls discover_files on the nested project directory and asserts that only main.py is discovered, excluding generated
            and sibling files.
        """
        project = tmp_path / "project"
        _create_file(project / "main.py")
        _create_file(project / "generated" / "ignored.py")
        _create_file(tmp_path / "sibling.py")
        (tmp_path / ".gitignore").write_text("project/generated/\n", encoding="utf-8")
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True, capture_output=True)

        files = {path.relative_to(project) for path in discover_files(project, ConfigurationDocmethis())}

        assert files == {Path("main.py")}

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
