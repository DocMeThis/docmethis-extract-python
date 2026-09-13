# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Unit tests for test_runner: framework detection, output parsing, and coverage JSON (iteration 9)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from docmethis_extract_python.static_extraction.dynamic_analysis.test_runner import (
    _coverage_json_command,
    _detect_framework,
    _extract_failing_tests_pytest,
    _extract_failing_tests_unittest,
    _pytest_command,
    _read_coverage_json,
    _unittest_command,
    execute_tests,
)

if TYPE_CHECKING:
    import pytest

# ---------------------------------------------------------------------------
# _detect_framework
# ---------------------------------------------------------------------------


def test_detect_framework_pytest_ini(tmp_path: Path) -> None:
    """The presence of pytest.ini gives framework 'pytest'."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    assert _detect_framework(tmp_path) == "pytest"


def test_detect_framework_pyproject_with_pytest_tool(tmp_path: Path) -> None:
    """pyproject.toml with [tool.pytest.ini_options] gives framework 'pytest'."""
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    assert _detect_framework(tmp_path) == "pytest"


def test_detect_framework_setup_cfg(tmp_path: Path) -> None:
    """setup.cfg with [tool:pytest] gives framework 'pytest'."""
    (tmp_path / "setup.cfg").write_text("[tool:pytest]\naddopts = -v\n", encoding="utf-8")
    assert _detect_framework(tmp_path) == "pytest"


def test_detect_framework_fallback_unittest(tmp_path: Path) -> None:
    """Without pytest config and with unittest imports, return 'unittest'."""
    (tmp_path / "test_exemple.py").write_text("import unittest\n", encoding="utf-8")
    assert _detect_framework(tmp_path) == "unittest"


def test_detect_framework_pyproject_without_pytest_tool(tmp_path: Path) -> None:
    """pyproject.toml without [tool.pytest] and with unittest imports gives 'unittest'."""
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 130\n", encoding="utf-8")
    (tmp_path / "test_exemple.py").write_text("import unittest\n", encoding="utf-8")
    assert _detect_framework(tmp_path) == "unittest"


# ---------------------------------------------------------------------------
# _pytest_command
# ---------------------------------------------------------------------------


def test_pytest_command_includes_coverage_file(tmp_path: Path) -> None:
    """The pytest command contains --cov-report=json:{coverage_file}."""
    coverage_file = tmp_path / "coverage.json"
    root = tmp_path / "project"
    root.mkdir()
    cmd = _pytest_command(root, coverage_file)
    assert any(f"--cov-report=json:{coverage_file}" in arg for arg in cmd)


def test_pytest_command_omits_cov_config_without_pyproject(tmp_path: Path) -> None:
    """--cov-config is omitted when pyproject.toml is absent."""
    coverage_file = tmp_path / "cov.json"
    root = tmp_path / "project"
    root.mkdir()
    cmd = _pytest_command(root, coverage_file)
    assert not any("--cov-config" in arg for arg in cmd)


def test_pytest_command_includes_cov_config_with_pyproject(tmp_path: Path) -> None:
    """--cov-config is included when pyproject.toml exists at the root."""
    coverage_file = tmp_path / "cov.json"
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    cmd = _pytest_command(tmp_path, coverage_file)
    assert any("--cov-config" in arg for arg in cmd)


# ---------------------------------------------------------------------------
# _extract_failing_tests_pytest
# ---------------------------------------------------------------------------


def test_extract_failing_tests_pytest_empty() -> None:
    """Output without a FAILED line gives an empty list."""
    assert _extract_failing_tests_pytest("1 passed in 0.1s") == []


def test_extract_failing_tests_pytest_one_failure() -> None:
    """A FAILED line produces the correct node id."""
    stdout = "FAILED tests/test_foo.py::test_bar\n1 failed in 0.5s"
    assert _extract_failing_tests_pytest(stdout) == ["tests/test_foo.py::test_bar"]


def test_extract_failing_tests_pytest_with_reason() -> None:
    """'FAILED node_id - AssertionError' returns only the node id."""
    stdout = "FAILED tests/test_foo.py::test_bar - AssertionError: expected 1\n"
    assert _extract_failing_tests_pytest(stdout) == ["tests/test_foo.py::test_bar"]


def test_extract_failing_tests_pytest_multiple() -> None:
    """Multiple FAILED lines produce all node ids."""
    stdout = "FAILED tests/a.py::test_x\nFAILED tests/b.py::test_y\n2 failed"
    outcome = _extract_failing_tests_pytest(stdout)
    assert outcome == ["tests/a.py::test_x", "tests/b.py::test_y"]


# ---------------------------------------------------------------------------
# _extract_failing_tests_unittest
# ---------------------------------------------------------------------------


def test_extract_failing_tests_unittest_failure() -> None:
    """A 'FAIL: test_name' line gives ['test_name']."""
    assert _extract_failing_tests_unittest("FAIL: test_something", "") == ["test_something"]


def test_extract_failing_tests_unittest_error() -> None:
    """An 'ERROR: test_name' line gives ['test_name']."""
    assert _extract_failing_tests_unittest("", "ERROR: test_erreur") == ["test_erreur"]


def test_extract_failing_tests_unittest_empty() -> None:
    """Output without FAIL:/ERROR: gives an empty list."""
    assert _extract_failing_tests_unittest("OK\n2 tests", "") == []


# ---------------------------------------------------------------------------
# _read_coverage_json
# ---------------------------------------------------------------------------


def test_read_coverage_json_missing(tmp_path: Path) -> None:
    """A missing file gives None."""
    assert _read_coverage_json(tmp_path / "coverage.json", tmp_path) is None


def test_read_coverage_json_valid(tmp_path: Path) -> None:
    """A valid JSON file returns a dict with absolute paths unchanged."""
    source_path = tmp_path / "coverage.json"
    absolu = str(tmp_path / "foo.py")
    data = {"files": {absolu: {"executed_lines": [1, 2], "missing_lines": [3]}}}
    source_path.write_text(json.dumps(data), encoding="utf-8")
    assert _read_coverage_json(source_path, tmp_path) == data


def test_read_coverage_json_corrupt(tmp_path: Path) -> None:
    """An invalid JSON file gives None without propagating an exception."""
    source_path = tmp_path / "coverage.json"
    source_path.write_text("{ invalid json }", encoding="utf-8")
    assert _read_coverage_json(source_path, tmp_path) is None


def test_read_coverage_json_reanchors_relative_paths(tmp_path: Path) -> None:
    """coverage.py returns paths relative to the analyzed root, so re-anchor them absolutely.

    Without re-anchoring, the mapper would resolve 'pkg/mod.py' against the caller's
    cwd and find no file.
    """
    source_path = tmp_path / "coverage.json"
    source_path.write_text(json.dumps({"files": {"pkg/mod.py": {"executed_lines": [1]}}}), encoding="utf-8")

    outcome = _read_coverage_json(source_path, tmp_path)

    assert outcome is not None
    assert list(outcome["files"]) == [str((tmp_path / "pkg" / "mod.py").resolve())]


# ---------------------------------------------------------------------------
# execute_tests - monkeypatched subprocess
# ---------------------------------------------------------------------------


def _mock_proc(returncode: int = 0, stdout: str = "1 passed", stderr: str = "") -> MagicMock:
    """Build a mock subprocess.CompletedProcess."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def test_execute_tests_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A successful subprocess (returncode=0) gives success=True."""
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _mock_proc(0, "1 passed", ""))  # noqa: ARG005
    outcome = execute_tests(tmp_path)
    assert outcome.success is True
    assert outcome.returncode == 0
    assert outcome.framework == "pytest"


def test_execute_tests_failure_returncode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A subprocess return code of 1 gives success=False."""
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _mock_proc(1, "FAILED tests/t.py::test_x\n1 failed", ""))  # noqa: ARG005
    outcome = execute_tests(tmp_path)
    assert outcome.success is False
    assert outcome.failing_tests == ["tests/t.py::test_x"]


def test_execute_tests_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """TimeoutExpired gives success=False and returncode=-1."""
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")

    def _raise_timeout(*a: object, **kw: object) -> None:  # noqa: ARG001
        raise subprocess.TimeoutExpired(cmd=[], timeout=30)

    monkeypatch.setattr(subprocess, "run", _raise_timeout)
    outcome = execute_tests(tmp_path, timeout_per_test=30)
    assert outcome.success is False
    assert outcome.returncode == -1
    assert "Timeout" in outcome.stderr


def test_execute_tests_oserror(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """OSError (missing executable) gives success=False and returncode=-1."""
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")

    def _raise_os(*a: object, **kw: object) -> None:  # noqa: ARG001
        msg = "pytest not found"
        raise OSError(msg)

    monkeypatch.setattr(subprocess, "run", _raise_os)
    outcome = execute_tests(tmp_path)
    assert outcome.success is False
    assert outcome.returncode == -1
    assert "pytest not found" in outcome.stderr


def test_execute_tests_missing_coverage_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When coverage.json is not produced, coverage_json=None without an exception."""
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _mock_proc())  # noqa: ARG005
    outcome = execute_tests(tmp_path)
    assert outcome.coverage_json is None


def test_execute_tests_propagates_detected_framework(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The detected framework is propagated in ExecutionResult."""
    (tmp_path / "test_exemple.py").write_text("import unittest\n", encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _mock_proc())  # noqa: ARG005
    outcome = execute_tests(tmp_path)
    assert outcome.framework == "unittest"


# ---------------------------------------------------------------------------
# Couverture du chemin unittest (it.10)
#
# `coverage run` writes only a binary data file; without a second `coverage json`
# pass, unittest coverage was always lost. Without COVERAGE_FILE, the data file
# would be written into the analyzed project.
# ---------------------------------------------------------------------------


class TestUnittestCommands:
    """Construction of unittest commands."""

    def test_command_uses_coverage_run_when_available(self, tmp_path: Path) -> None:
        """When coverage.py is available, the suite runs under 'coverage run'."""
        cmd = _unittest_command(tmp_path, with_coverage=True)

        assert cmd[1:4] == ["-m", "coverage", "run"]
        assert "unittest" in cmd
        assert "discover" in cmd

    def test_command_uses_unittest_without_coverage(self, tmp_path: Path) -> None:
        """Without coverage.py, 'python -m coverage run' would fail, so the suite still runs."""
        cmd = _unittest_command(tmp_path, with_coverage=False)

        assert "coverage" not in cmd
        assert cmd[1:4] == ["-m", "unittest", "discover"]

    def test_coverage_json_command_targets_file(self, tmp_path: Path) -> None:
        """The second pass writes the report to the requested explicit path."""
        coverage_file = tmp_path / "coverage.json"
        cmd = _coverage_json_command(coverage_file)

        assert cmd[1:4] == ["-m", "coverage", "json"]
        assert cmd[-2:] == ["-o", str(coverage_file)]


class TestUnittestCoverageEndToEnd:
    """Real unittest project execution must propagate coverage to the result."""

    SOURCE = "def add(a, b):\n    return a + b\n\n\ndef jamais_appelee():\n    return None\n"
    SUITE = (
        "import unittest\n\nfrom calc import add\n\n\n"
        "class TestAdd(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n"
    )

    def _project(self, tmp_path: Path) -> Path:
        (tmp_path / "calc.py").write_text(self.SOURCE, encoding="utf-8")
        (tmp_path / "test_calc.py").write_text(self.SUITE, encoding="utf-8")
        return tmp_path

    def test_coverage_json_is_produced(self, tmp_path: Path) -> None:
        """Framework 'unittest' produces coverage_json with the measured source file."""
        root = self._project(tmp_path)

        outcome = execute_tests(root)

        assert outcome.framework == "unittest"
        assert outcome.success is True
        assert outcome.coverage_json is not None
        files = outcome.coverage_json["files"]
        assert str((root / "calc.py").resolve()) in files

    def test_report_paths_are_absolute(self, tmp_path: Path) -> None:
        """Report keys are absolute so the mapper can resolve them outside the analyzed cwd."""
        root = self._project(tmp_path)

        outcome = execute_tests(root)

        assert outcome.coverage_json is not None
        assert all(Path(mapping_key).is_absolute() for mapping_key in outcome.coverage_json["files"])

    def test_analyzed_project_is_not_polluted(self, tmp_path: Path) -> None:
        """No .coverage artifact is left in the analyzed project."""
        root = self._project(tmp_path)

        execute_tests(root)

        assert not list(root.glob(".coverage*"))
