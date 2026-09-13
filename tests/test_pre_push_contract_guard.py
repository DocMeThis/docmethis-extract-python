# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Unit tests for the extract-python pre-push contract guard."""

from __future__ import annotations

import importlib.util
import io
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


_ROOT_DIR = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT_DIR / "scripts" / "pre_push_contract_guard.py"


def _completed_process(*, stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    """Build a minimal subprocess result for test doubles."""
    return subprocess.CompletedProcess(args=["git"], returncode=returncode, stdout=stdout, stderr="")


def _return_origin_main(*_args: object, **_kwargs: object) -> str:
    """Return a stable default remote ref for test doubles."""
    return "origin/main"


def _return_empty_list(*_args: object, **_kwargs: object) -> list[str]:
    """Return no files for test doubles."""
    return []


@pytest.fixture
def guard_module() -> ModuleType:
    """Load the guard script without importing the installed package."""
    spec = importlib.util.spec_from_file_location("pre_push_contract_guard", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upstream_remote_ref_ignores_local_upstream(
    guard_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore a local upstream to avoid false positives during pre-push."""

    def _fake_git(command: list[str]) -> subprocess.CompletedProcess[str]:
        assert command == ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]
        return _completed_process(stdout="local-branch\n")

    monkeypatch.setattr(guard_module, "_git", _fake_git)

    assert guard_module._upstream_remote_ref() is None


def test_fallback_files_uses_merge_base_after_local_upstream(
    guard_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the remote merge-base when the configured upstream is local."""
    commands: list[tuple[str, ...]] = []

    def _fake_git(command: list[str]) -> subprocess.CompletedProcess[str] | None:
        commands.append(tuple(command))
        if command == ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]:
            return _completed_process(stdout="local-branch\n")
        if command == ["git", "merge-base", "origin/main", "HEAD"]:
            return _completed_process(stdout="base123\n")
        if command == ["git", "diff", "--name-only", "base123", "HEAD"]:
            return _completed_process(stdout="tests/test_public_contract.py\n")
        return None

    monkeypatch.setattr(guard_module, "_git", _fake_git)

    assert guard_module._files_fallback(remote_default_ref="origin/main") == ["tests/test_public_contract.py"]
    assert ("git", "diff", "--name-only", "@{u}..HEAD") not in commands


def test_fallback_files_uses_remote_upstream(
    guard_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer an available remote upstream over merge-base fallback."""
    commands: list[tuple[str, ...]] = []

    def _fake_git(command: list[str]) -> subprocess.CompletedProcess[str] | None:
        commands.append(tuple(command))
        if command == ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]:
            return _completed_process(stdout="origin/feature\n")
        if command == ["git", "diff", "--name-only", "origin/feature..HEAD"]:
            return _completed_process(stdout="README.md\n")
        return None

    monkeypatch.setattr(guard_module, "_git", _fake_git)

    assert guard_module._files_fallback(remote_default_ref="origin/main") == ["README.md"]
    assert ("git", "merge-base", "origin/main", "HEAD") not in commands


def test_contract_pattern_matches_exact_path_and_directory_prefix(guard_module: ModuleType) -> None:
    """Match exact files and directory prefixes without matching similar names."""
    patterns = ["pyproject.toml", "tests/static_extraction/"]

    assert guard_module._is_contractual("pyproject.toml", patterns)
    assert guard_module._is_contractual("tests/static_extraction/test_models.py", patterns)
    assert not guard_module._is_contractual("pyproject.toml.bak", patterns)
    assert not guard_module._is_contractual("tests/static_extraction.py", patterns)


def test_main_returns_ok_for_non_contractual_changes(
    guard_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow a push that does not touch a configured contract path."""
    monkeypatch.setattr(guard_module, "_load_patterns", lambda: ["tests/test_public_contract.py"])
    monkeypatch.setattr(guard_module, "_remote_name", lambda: "origin")
    monkeypatch.setattr(guard_module, "_remote_default_ref", _return_origin_main)
    monkeypatch.setattr(guard_module, "_read_refs_from_stdin", _return_empty_list)
    monkeypatch.setattr(guard_module, "_files_fallback", lambda **_kwargs: ["README.md"])
    monkeypatch.setattr(guard_module.sys, "stdin", io.StringIO(""))

    assert guard_module.main() == guard_module.EXIT_OK


def test_main_rejects_contractual_changes(
    guard_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject a push that touches a configured contract path."""
    monkeypatch.setattr(guard_module, "_load_patterns", lambda: ["tests/test_public_contract.py"])
    monkeypatch.setattr(guard_module, "_remote_name", lambda: "origin")
    monkeypatch.setattr(guard_module, "_remote_default_ref", _return_origin_main)
    monkeypatch.setattr(guard_module, "_read_refs_from_stdin", _return_empty_list)
    monkeypatch.setattr(guard_module, "_files_fallback", lambda **_kwargs: ["tests/test_public_contract.py"])
    monkeypatch.setattr(guard_module.sys, "stdin", io.StringIO(""))

    assert guard_module.main() == guard_module.EXIT_CONTRACTUAL
    assert "contractual files changed" in capsys.readouterr().err


def test_main_fails_closed_when_changed_files_are_unknown(
    guard_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return an internal error when the changed-file range cannot be found."""
    monkeypatch.setattr(guard_module, "_load_patterns", lambda: ["pyproject.toml"])
    monkeypatch.setattr(guard_module, "_remote_name", lambda: "origin")
    monkeypatch.setattr(guard_module, "_remote_default_ref", lambda _remote: None)
    monkeypatch.setattr(guard_module, "_read_refs_from_stdin", _return_empty_list)
    monkeypatch.setattr(guard_module, "_files_fallback", lambda **_kwargs: None)
    monkeypatch.setattr(guard_module.sys, "stdin", io.StringIO(""))

    assert guard_module.main() == guard_module.EXIT_INTERNAL_ERROR
    assert "unable to determine changed files" in capsys.readouterr().err
