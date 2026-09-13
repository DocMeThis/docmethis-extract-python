# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Isolated test runner for iteration 1.9.

Run the analyzed project's test suite in a separate subprocess with coverage
capture (coverage.py). Never modify the caller process state.

Supported frameworks: pytest (preferred), unittest.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ExecutionResult", "execute_tests"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Result of a test execution with coverage.

    Parameters
    ----------
    success:
        True when all tests passed (return code 0). Never set it to True without
        actually running the suite: this is the only field that attests to test
        results. The explicit exception is ``framework == "none"``, when no tests exist.
    framework :
        "pytest" | "unittest" - detected framework, or "none" when no tests ran.
    returncode :
        Subprocess return code (0 = success, 1 = failure, 2+ = error).
    stdout :
        Captured standard output.
    stderr :
        Captured standard error.
    coverage_json :
        coverage.py JSON report content (None when unavailable).
    failing_tests:
        Identifiers of failed tests (pytest node ids or unittest names).

    """

    success: bool
    framework: str
    returncode: int
    stdout: str
    stderr: str
    coverage_json: dict | None = None
    failing_tests: list[str] = field(default_factory=list)
    raw_calls: list[dict] = field(default_factory=list)  # calls collected by the plugin (it.9 phase 2)
    # Readable reason why coverage_json is None (None = coverage produced or unknown cause).
    coverage_diagnostic: str | None = None


# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------


def _detect_framework(root: Path) -> str:
    """Return "pytest" or "unittest" from configuration files or test imports.

    Priority: config files (pytest.ini, pyproject.toml, setup.cfg), then import
    scanning in test files. Raise ValueError when the framework cannot be determined.
    """
    if (root / "pytest.ini").exists():
        return "pytest"

    pyproject = root / "pyproject.toml"
    if pyproject.exists() and "[tool.pytest" in pyproject.read_text(encoding="utf-8"):
        return "pytest"

    setup_cfg = root / "setup.cfg"
    if setup_cfg.exists() and "[tool:pytest]" in setup_cfg.read_text(encoding="utf-8"):
        return "pytest"

    votes: dict[str, int] = {"pytest": 0, "unittest": 0}
    for source_path in (*root.rglob("test_*.py"), *root.rglob("*_test.py")):
        payload = source_path.read_text(encoding="utf-8", errors="ignore")
        if "import pytest" in payload:
            votes["pytest"] += 1
        if "import unittest" in payload:
            votes["unittest"] += 1

    if votes["pytest"] > votes["unittest"]:
        return "pytest"
    if votes["unittest"] > votes["pytest"]:
        return "unittest"

    msg = f"Test framework not detected in {root} (votes: {votes})."
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Build commands
# ---------------------------------------------------------------------------


def _pytest_command(root: Path, coverage_file: Path | None) -> list[str]:
    """Build the pytest command with call-collection plugin and coverage when available.

    When ``coverage_file`` is ``None``, remove ``--cov`` options from pytest-cov,
    which pytest would reject when not installed. The suite still runs because
    call collection and failed-test detection do not depend on it.
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--tb=no",
        "-q",
        "-p",
        "docmethis_extract_python.static_extraction.dynamic_analysis.call_collector",
    ]

    if coverage_file is not None:
        cmd += [f"--cov={root}", f"--cov-report=json:{coverage_file}"]
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            cmd.append(f"--cov-config={pyproject}")

    cmd.append(str(root))
    return cmd


def _unittest_command(root: Path, *, with_coverage: bool) -> list[str]:
    """Build the unittest discover command, using ``coverage run`` when available.

    When ``with_coverage`` is False, fall back to unittest alone. Without
    coverage.py, ``python -m coverage run`` would fail and the suite would not
    run at all. Lose coverage, not the suite, matching ``_pytest_command`` without pytest-cov.
    """
    discover = [sys.executable, "-m", "unittest", "discover", "-s", str(root)]
    if not with_coverage:
        return discover

    return [sys.executable, "-m", "coverage", "run", "--source", str(root), *discover[1:]]


def _coverage_json_command(coverage_file: Path) -> list[str]:
    """Build the command converting ``coverage run`` data to a JSON report.

    ``coverage run`` writes only a binary data file. Without this second pass,
    no JSON exists and unittest coverage would always be lost.
    """
    return [sys.executable, "-m", "coverage", "json", "-o", str(coverage_file)]


# ---------------------------------------------------------------------------
# Extract failed tests
# ---------------------------------------------------------------------------


def _extract_failing_tests_pytest(stdout: str) -> list[str]:
    """Extract failed-test node ids from pytest output."""
    failing: list[str] = []
    for source_line in stdout.splitlines():
        stripped = source_line.strip()
        if stripped.startswith("FAILED "):
            node_id = stripped[len("FAILED ") :].split(" - ")[0].strip()
            failing.append(node_id)

    return failing


def _extract_failing_tests_unittest(stdout: str, stderr: str) -> list[str]:
    """Extract failed-test names from unittest output."""
    failing: list[str] = []
    for source_line in (stdout + "\n" + stderr).splitlines():
        stripped = source_line.strip()
        if stripped.startswith(("FAIL:", "ERROR:")):
            parts = stripped.split(":", 1)
            if len(parts) == 2:  # noqa: PLR2004
                failing.append(parts[1].strip())

    return failing


# ---------------------------------------------------------------------------
# Read the coverage JSON report
# ---------------------------------------------------------------------------


def _absolutize_paths(coverage_json: dict, root: Path) -> dict:
    """Re-anchor relative ``files`` keys to ``root``.

    coverage.py makes paths relative to the producing process cwd, here the
    analyzed project root. The caller has a different cwd; without re-anchoring,
    ``coverage_mapper`` would find no files and silently lose coverage. Existing
    absolute keys are left unchanged.
    """
    files = coverage_json.get("files")
    if not isinstance(files, dict):
        return coverage_json

    absolute_paths: dict[str, object] = {}
    for mapping_key, data in files.items():
        source_path = Path(mapping_key)
        absolute_paths[str(source_path if source_path.is_absolute() else (root / source_path).resolve())] = data

    coverage_json["files"] = absolute_paths
    return coverage_json


def _read_coverage_json(source_path: Path, root: Path) -> dict | None:
    """Read the coverage JSON at an explicit path and absolutize file paths."""
    if not source_path.exists():
        return None
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Unable to read coverage report: %s", source_path)
        return None

    if not isinstance(payload, dict):
        logger.warning("Malformed coverage report (expected an object): %s", source_path)
        return None

    return _absolutize_paths(payload, root)


def _read_calls_json(source_path: Path) -> list[dict]:
    """Read the appels.json file produced by the call_collector plugin."""
    if not source_path.exists():
        return []
    try:
        data = json.loads(source_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Unable to read calls file: %s", source_path)
        return []

    if isinstance(data, list):
        return data

    logger.warning("Malformed calls file (expected a list): %s", source_path)
    return []


def _generate_unittest_coverage_json(
    coverage_file: Path,
    root: Path,
    env: dict[str, str],
    timeout: int,
) -> str | None:
    """Convert ``coverage run`` data to JSON and return a diagnostic on failure.

    Return None when the report was produced. ``COVERAGE_FILE`` in ``env`` locates
    the data file used for test execution.
    """
    try:
        proc = subprocess.run(  # noqa: PLW1510, S603
            _coverage_json_command(coverage_file),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(root),
            env=env,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Test runner - unable to generate coverage report: %s", exc)
        return "coverage report not generated"

    if proc.returncode != 0:
        logger.warning(
            "Test runner - 'coverage json' failed (code %d): %s",
            proc.returncode,
            proc.stderr.strip() or proc.stdout.strip(),
        )
        return "coverage report not generated"

    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def execute_tests(
    root: Path,
    timeout_per_test: int = 30,
) -> ExecutionResult:
    """Run the project test suite in an isolated subprocess with coverage.

    The caller process is never polluted: internal project exceptions remain in
    the subprocess. Return a failed ExecutionResult when the subprocess cannot
    start (for example, pytest is absent) or exceeds the global timeout.

    Parameters
    ----------
    root:
        Project root to test.
    timeout_per_test:
        Global timeout in seconds (default: 30 s), used for the complete
        subprocess rather than each individual test.

    Returns
    -------
    ExecutionResult
        Complete result with coverage JSON when available.

    """
    root = root.resolve()
    try:
        framework = _detect_framework(root)
    except ValueError:
        logger.info("Test runner - no framework detected in %s; execution skipped.", root)
        return ExecutionResult(success=True, framework="none", returncode=0, stdout="", stderr="")

    # If coverage is absent, lose coverage but not the suite. Short-circuiting
    # would report success without running tests and would also lose collected
    # calls and failed-test analysis, which do not depend on coverage.
    coverage_module = "pytest_cov" if framework == "pytest" else "coverage"
    coverage_available = importlib.util.find_spec(coverage_module) is not None
    coverage_diagnostic = None

    if not coverage_available:
        package = "pytest-cov" if framework == "pytest" else "coverage"
        logger.error(
            "Test runner - %s is not installed: coverage will be unavailable for %s "
            "(the suite will still run). Install with: pip install %s",
            package,
            root,
            package,
        )
        coverage_diagnostic = f"{package} not installed"

    with tempfile.TemporaryDirectory() as tmp_dir:
        coverage_file = Path(tmp_dir) / "coverage.json"
        calls_file = Path(tmp_dir) / "appels.json"

        cmd = (
            _pytest_command(root, coverage_file if coverage_available else None)
            if framework == "pytest"
            else _unittest_command(root, with_coverage=coverage_available)
        )

        logger.debug("Test runner - command: %s", " ".join(cmd))

        # Environment variables for the call_collector plugin. COVERAGE_FILE keeps
        # binary data in the temporary directory; otherwise coverage.py would
        # write it to the subprocess cwd, the analyzed project.
        env = {
            **os.environ,
            "DOCMETHIS_APPELS_OUTPUT": str(calls_file),
            "DOCMETHIS_PROJECT_ROOT": str(root),
            "COVERAGE_FILE": str(Path(tmp_dir) / ".coverage"),
        }

        try:
            proc = subprocess.run(  # noqa: PLW1510, S603
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_per_test,
                cwd=str(root),
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Test runner - timeout (%ds) exceeded for %s", timeout_per_test, root)
            return ExecutionResult(
                success=False,
                framework=framework,
                returncode=-1,
                stdout="",
                stderr=f"Timeout after {timeout_per_test}s.",
                coverage_diagnostic=f"timeout ({timeout_per_test}s)",
            )
        except OSError as exc:
            logger.warning("Test runner - unable to start subprocess: %s", exc)
            return ExecutionResult(
                success=False,
                framework=framework,
                returncode=-1,
                stdout="",
                stderr=str(exc),
                coverage_diagnostic=coverage_diagnostic,
            )

        # pytest-cov writes JSON directly; unittest needs a second 'coverage json'
        # pass over the data file produced by 'coverage run'.
        if coverage_available and framework == "unittest":
            coverage_diagnostic = _generate_unittest_coverage_json(
                coverage_file,
                root,
                env,
                timeout_per_test,
            )

        coverage_json = _read_coverage_json(coverage_file, root) if coverage_available else None

        # Collected calls - the plugin exists only for pytest.
        raw_calls = _read_calls_json(calls_file) if framework == "pytest" else []

        if framework == "pytest":
            failing_tests = _extract_failing_tests_pytest(proc.stdout)
        elif framework == "unittest":
            failing_tests = _extract_failing_tests_unittest(proc.stdout, proc.stderr)
        else:
            msg = f"Extraction for '{framework}' is not implemented."
            raise ValueError(msg)

        return ExecutionResult(
            success=proc.returncode == 0,
            framework=framework,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            coverage_json=coverage_json,
            failing_tests=failing_tests,
            raw_calls=raw_calls,
            coverage_diagnostic=coverage_diagnostic,
        )
