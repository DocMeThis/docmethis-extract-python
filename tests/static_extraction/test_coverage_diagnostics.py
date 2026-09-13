# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for targeted coverage diagnostics (phase 6, iteration 1.10).

Root cause: map_coverage emitted a generic warning for every coverage_json=None.
Fix: ExecutionResult.coverage_diagnostic carries the context; map_coverage warns on
timeouts, logs debug for other causes, and test_runner logs an error when pytest-cov is absent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from docmethis_extract_python.static_extraction.dynamic_analysis.coverage import map_coverage
from docmethis_extract_python.static_extraction.dynamic_analysis.test_runner import (
    ExecutionResult,
    _pytest_command,
    execute_tests,
)
from docmethis_extract_python.static_extraction.models import (
    FunctionRecord,
    MethodType,
    ModuleRecord,
    Visibility,
)

if TYPE_CHECKING:
    import pytest


def _result(
    coverage_json: dict | None = None,
    diagnostic: str | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        success=True,
        framework="pytest",
        returncode=0,
        stdout="",
        stderr="",
        coverage_json=coverage_json,
        coverage_diagnostic=diagnostic,
    )


def _fn() -> FunctionRecord:
    return FunctionRecord(
        qualified_name="pkg.mod.foo",
        file_path=Path("/dummy.py"),
        line_start=1,
        line_end=5,
        col_start=0,
        col_end=0,
        method_kind=MethodType.FUNCTION,
        visibility=Visibility.PUBLIC,
        parent_class=None,
        parent_module="pkg.mod",
        existing_docstring=None,
    )


def _module() -> ModuleRecord:
    return ModuleRecord(
        file_path=Path("/dummy.py"),
        module_name="pkg.mod",
        functions=[_fn()],
    )


# ---------------------------------------------------------------------------
# Tests - map_coverage: log level by diagnostic
# ---------------------------------------------------------------------------


class TestMapCoverageDiagnostic:
    """Verify that map_coverage uses the correct log level for each diagnostic."""

    def test_without_json_without_diagnostic_logs_debug_not_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """coverage_json=None and diagnostic=None logs debug only, not warning."""
        with caplog.at_level(logging.DEBUG, logger="docmethis_extract_python.static_extraction.dynamic_analysis.coverage"):
            map_coverage(_result(coverage_json=None, diagnostic=None), [_module()])

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert not warnings
        assert any("no JSON report" in r.message for r in debugs)

    def test_without_json_with_timeout_diagnostic_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """coverage_json=None and diagnostic='timeout (30s)' logs a warning mentioning timeout."""
        with caplog.at_level(logging.DEBUG, logger="docmethis_extract_python.static_extraction.dynamic_analysis.coverage"):
            map_coverage(_result(coverage_json=None, diagnostic="timeout (30s)"), [_module()])

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings
        assert any("timeout" in r.message for r in warnings)

    def test_without_json_with_pytest_cov_diagnostic_logs_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """A pytest-cov diagnostic logs debug because the error was already logged."""
        with caplog.at_level(logging.DEBUG, logger="docmethis_extract_python.static_extraction.dynamic_analysis.coverage"):
            map_coverage(
                _result(coverage_json=None, diagnostic="pytest-cov not installed"),
                [_module()],
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warnings

    def test_function_unchanged_when_json_missing(self) -> None:
        """coverage_json=None leaves every FunctionRecord unchanged."""
        mod = _module()
        fn = mod.functions[0]
        map_coverage(_result(coverage_json=None), [mod])
        assert fn.test_coverage is None


# ---------------------------------------------------------------------------
# Tests - execute_tests: missing pytest-cov logs error and returns an ExecutionResult.
# ---------------------------------------------------------------------------


class TestExecuteTestsWithoutPytestCov:
    """Verify execute_tests behavior when pytest-cov is absent."""

    def test_missing_pytest_cov_logs_error(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Missing pytest-cov emits logger.error."""
        (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")

        with (
            patch("importlib.util.find_spec", return_value=None),
            caplog.at_level(logging.ERROR, logger="docmethis_extract_python.static_extraction.dynamic_analysis.test_runner"),
        ):
            execute_tests(tmp_path)

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert errors
        assert any("pytest-cov" in r.message for r in errors)

    def test_missing_pytest_cov_returns_diagnostic(self, tmp_path: Path) -> None:
        """Missing pytest-cov returns an ExecutionResult with coverage_diagnostic set."""
        (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")

        with patch("importlib.util.find_spec", return_value=None):
            outcome = execute_tests(tmp_path)

        assert outcome.coverage_diagnostic is not None
        assert "pytest-cov" in outcome.coverage_diagnostic
        assert outcome.coverage_json is None

    def test_present_pytest_cov_does_not_short_circuit(self, tmp_path: Path) -> None:
        """When pytest-cov is present, find_spec returns an object and execution is not skipped."""
        (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        fake_spec = object()

        with (
            patch("importlib.util.find_spec", return_value=fake_spec),
            # Simulate a subprocess error instead of running pytest.
            patch(
                "docmethis_extract_python.static_extraction.dynamic_analysis.test_runner.subprocess.run",
                side_effect=OSError("subprocess unavailable"),
            ),
        ):
            outcome = execute_tests(tmp_path)

        # No pytest-cov short circuit occurred; the result came through the OSError handler.
        assert outcome.coverage_diagnostic is None or "pytest-cov" not in (outcome.coverage_diagnostic or "")


# ---------------------------------------------------------------------------
# Tests - missing pytest-cov loses coverage but not the test suite.
# ---------------------------------------------------------------------------


class TestPytestCovAbsentStillExecutes:
    """`success` attests to test results and must never be True without execution.

    Skipping the suite when pytest-cov was missing returned `success=True, returncode=0`,
    claiming success for a suite that never ran. It also discarded dynamic call and failing-test
    analysis, which do not depend on pytest-cov.
    """

    SUITE = "def test_ok():\n    assert True\n\n\ndef test_ko():\n    assert False\n"

    def _project(self, tmp_path: Path) -> Path:
        (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        (tmp_path / "test_suite.py").write_text(self.SUITE, encoding="utf-8")
        return tmp_path

    def test_suite_actually_executes(self, tmp_path: Path) -> None:
        """A failing test must fail the result, proving that pytest ran."""
        root = self._project(tmp_path)

        with patch("importlib.util.find_spec", return_value=None):
            outcome = execute_tests(root)

        assert outcome.success is False
        assert outcome.returncode != 0
        assert any("test_ko" in t for t in outcome.failing_tests)

    def test_diagnostic_explains_missing_coverage(self, tmp_path: Path) -> None:
        """Coverage is lost and the diagnostic explains why."""
        root = self._project(tmp_path)

        with patch("importlib.util.find_spec", return_value=None):
            outcome = execute_tests(root)

        assert outcome.coverage_json is None
        assert outcome.coverage_diagnostic == "pytest-cov not installed"

    def test_command_omits_cov_when_unavailable(self, tmp_path: Path) -> None:
        """The --cov options come from pytest-cov; passing them without it would fail pytest."""
        cmd = _pytest_command(tmp_path, None)

        assert not [arg for arg in cmd if arg.startswith("--cov")]
        assert "docmethis_extract_python.static_extraction.dynamic_analysis.call_collector" in cmd

    def test_command_includes_cov_when_available(self, tmp_path: Path) -> None:
        """The nominal path is unchanged."""
        cmd = _pytest_command(tmp_path, tmp_path / "coverage.json")

        assert [arg for arg in cmd if arg.startswith("--cov")]
