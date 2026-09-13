# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Write the provenance manifest for this repository's built distributions."""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

__all__ = ["main"]

SCHEMA_VERSION = "r6-owner-artifact-v1"
INDEX_URL = "https://pkg.docmethis.com"


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Create one manifest beside the artifacts in ``dist``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args(argv)

    project = tomllib.loads((Path("pyproject.toml")).read_text(encoding="utf-8"))["project"]
    name = project["name"]
    version = project["version"]
    dist = args.dist.resolve()
    artifacts = sorted(
        path for path in dist.iterdir() if path.is_file() and (path.name.endswith(".whl") or path.name.endswith(".tar.gz"))
    )
    expected_prefix = f"{name.replace('-', '_')}-{version}"
    valid_names = {f"{expected_prefix}.tar.gz"}
    if not artifacts or any(
        not path.name.startswith(f"{expected_prefix}-") and path.name not in valid_names for path in artifacts
    ):
        message = f"unexpected build artifacts in {dist}"
        raise RuntimeError(message)
    git = shutil.which("git")
    if git is None:
        message = "git is required to record source provenance"
        raise RuntimeError(message)
    source_commit = subprocess.run(  # noqa: S603
        [git, "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "repository": args.repository,
        "source_commit": source_commit,
        "index": INDEX_URL,
        "distribution": {"name": name, "version": version},
        "artifacts": [
            {
                "filename": path.name,
                "kind": "wheel" if path.name.endswith(".whl") else "sdist",
                "sha256": _sha256(path),
            }
            for path in artifacts
        ],
    }
    output = dist / "r6-owner-artifact.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(f"Wrote {output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
