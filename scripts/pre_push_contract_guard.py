# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Pre-push guard for the extract-python public contract."""

# ruff: noqa: INP001

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TextIO

__all__: list[str] = []

EXIT_OK = 0
EXIT_CONTRACTUAL = 1
EXIT_INTERNAL_ERROR = 2

_DEFAULT_BRANCHES = ("main", "master", "develop")
_NULL_SHA_LENGTH = 40
_PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"
_REMOTE_NAME_ARG_INDEX = 1
_STDIN_FIELDS_COUNT = 4
_SUBPROCESS_TIMEOUT = 30


def _git(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run a git command with a bounded timeout."""
    try:
        return subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        print(
            f"WARNING: git command timed out after {_SUBPROCESS_TIMEOUT}s: {' '.join(command)}",
            file=sys.stderr,
        )
        return None


def _remote_name() -> str:
    """Return the remote name passed by the pre-push hook."""
    if len(sys.argv) > _REMOTE_NAME_ARG_INDEX and sys.argv[_REMOTE_NAME_ARG_INDEX].strip():
        return sys.argv[_REMOTE_NAME_ARG_INDEX].strip()
    return "origin"


def _remote_default_ref(remote_name: str) -> str | None:
    """Return the remote default branch ref, such as ``origin/main``."""
    result_head = _git(["git", "symbolic-ref", "--quiet", "--short", f"refs/remotes/{remote_name}/HEAD"])
    if result_head is not None and result_head.returncode == 0 and result_head.stdout.strip():
        return result_head.stdout.strip()

    for branch in _DEFAULT_BRANCHES:
        fallback_ref = f"{remote_name}/{branch}"
        result_fallback = _git(["git", "rev-parse", "--verify", "--quiet", fallback_ref])
        if result_fallback is not None and result_fallback.returncode == 0:
            return fallback_ref

    return None


def _load_patterns() -> list[str]:
    """Load contract path patterns from ``pyproject.toml``."""
    try:
        with _PYPROJECT_PATH.open("rb") as file_handle:
            data = tomllib.load(file_handle)
    except FileNotFoundError:
        print(f"ERROR: configuration file not found: {_PYPROJECT_PATH}", file=sys.stderr)
        raise SystemExit(EXIT_INTERNAL_ERROR) from None
    except tomllib.TOMLDecodeError as exc:
        print(f"ERROR: cannot parse {_PYPROJECT_PATH}: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_INTERNAL_ERROR) from None

    return data.get("tool", {}).get("contract-guard", {}).get("patterns", [])


def _read_refs_from_stdin(stdin: TextIO) -> list[tuple[str, str]]:
    """Read pushed refs from pre-push stdin."""
    if stdin.isatty():
        return []

    refs: list[tuple[str, str]] = []
    for line in stdin:
        parts = line.split()
        if len(parts) >= _STDIN_FIELDS_COUNT:
            refs.append((parts[1], parts[3]))
    return refs


def _is_null_sha(sha: str) -> bool:
    """Return whether a SHA represents a deleted ref."""
    return len(sha) == _NULL_SHA_LENGTH and all(character == "0" for character in sha)


def _files_for_ref(local_sha: str, remote_sha: str, *, remote_default_ref: str | None) -> list[str] | None:
    """Return files changed for one pushed ref, or None when detection fails."""
    if _is_null_sha(local_sha):
        return []

    if _is_null_sha(remote_sha):
        if remote_default_ref is None:
            return None
        result_base = _git(["git", "merge-base", remote_default_ref, local_sha])
        if result_base is None or result_base.returncode != 0 or not result_base.stdout.strip():
            return None
        base = result_base.stdout.strip()
    else:
        base = remote_sha

    result = _git(["git", "diff", "--name-only", base, local_sha])
    if result is None or result.returncode != 0:
        return None
    return [line for line in result.stdout.splitlines() if line]


def _files_from_refs(refs: list[tuple[str, str]], *, remote_default_ref: str | None) -> list[str] | None:
    """Return the union of files changed by all pushed refs."""
    changed: set[str] = set()
    for local_sha, remote_sha in refs:
        files = _files_for_ref(local_sha, remote_sha, remote_default_ref=remote_default_ref)
        if files is None:
            return None
        changed.update(files)
    return sorted(changed)


def _upstream_remote_ref() -> str | None:
    """Return a remote upstream ref, ignoring local upstreams."""
    result = _git(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if result is None or result.returncode != 0:
        return None

    upstream_ref = result.stdout.strip()
    if not upstream_ref or "/" not in upstream_ref:
        return None
    return upstream_ref


def _files_fallback(*, remote_default_ref: str | None) -> list[str] | None:
    """Return files changed by HEAD against its upstream or remote merge-base."""
    upstream_remote_ref = _upstream_remote_ref()
    if upstream_remote_ref is not None:
        result = _git(["git", "diff", "--name-only", f"{upstream_remote_ref}..HEAD"])
        if result is not None and result.returncode == 0:
            return [line for line in result.stdout.splitlines() if line]

    if remote_default_ref is None:
        return None

    result_base = _git(["git", "merge-base", remote_default_ref, "HEAD"])
    if result_base is None or result_base.returncode != 0 or not result_base.stdout.strip():
        return None

    merge_base = result_base.stdout.strip()
    result_diff = _git(["git", "diff", "--name-only", merge_base, "HEAD"])
    if result_diff is None or result_diff.returncode != 0:
        return None
    return [line for line in result_diff.stdout.splitlines() if line]


def _is_contractual(path: str, patterns: list[str]) -> bool:
    """Return whether a changed path matches an exact or directory pattern."""
    changed_path = Path(path)
    return any(changed_path.is_relative_to(Path(pattern)) for pattern in patterns)


def _print_detection_error(*, remote_name: str, remote_default_ref: str | None) -> None:
    """Print the fail-closed message used when changed files cannot be determined."""
    print(file=sys.stderr)
    print("ERROR: unable to determine changed files for this push.", file=sys.stderr)
    if remote_default_ref is None:
        print(
            f"Check that the default branch of remote '{remote_name}' is resolvable (for example, git fetch --all --prune).",
            file=sys.stderr,
        )
    else:
        print(f"Check that {remote_default_ref} is accessible (git fetch), then retry.", file=sys.stderr)
    print("Use git push --no-verify to confirm manually if needed.", file=sys.stderr)
    print(file=sys.stderr)


def _print_warning(contractual_files: list[str]) -> None:
    """Print the warning for changed contract files."""
    print(file=sys.stderr)
    print("WARNING: contractual files changed in this push:", file=sys.stderr)
    for path in contractual_files:
        print(f"   {path}", file=sys.stderr)
    print(file=sys.stderr)
    print("These files define or verify public extract-python contracts.", file=sys.stderr)
    print("Use git push --no-verify to confirm that the change is intentional.", file=sys.stderr)
    print(file=sys.stderr)


def main() -> int:
    """Run the pre-push contract guard."""
    patterns = _load_patterns()
    if not patterns:
        return EXIT_OK

    remote_name = _remote_name()
    remote_default_ref = _remote_default_ref(remote_name)
    refs = _read_refs_from_stdin(sys.stdin)
    changed_files = (
        _files_from_refs(refs, remote_default_ref=remote_default_ref)
        if refs
        else _files_fallback(remote_default_ref=remote_default_ref)
    )

    if changed_files is None:
        _print_detection_error(remote_name=remote_name, remote_default_ref=remote_default_ref)
        return EXIT_INTERNAL_ERROR

    contractual_files = sorted({path for path in changed_files if _is_contractual(path, patterns)})
    if not contractual_files:
        return EXIT_OK

    _print_warning(contractual_files)
    return EXIT_CONTRACTUAL


if __name__ == "__main__":
    sys.exit(main())
