# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Extract project metadata from several possible sources."""

from __future__ import annotations

import ast
import configparser
import logging
import re
import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)

# Python files carrying the package version (numpy, dateutil, httpx, ...)
VERSION_FILE_NAMES: tuple[str, ...] = ("version.py", "__version__.py", "_version.py")

# Directories to skip while looking for packages
EXCLUDED_DIRECTORIES: frozenset[str] = frozenset(
    {
        "tests",
        "test",
        "docs",
        "doc",
        "examples",
        "example",
        "build",
        "dist",
        "vendor",
        "third_party",
    }
)


@dataclass(frozen=True)
class ProjectMetadata:
    """Metadata extracted from a Python project.

    Attributes
    ----------
    identifier_name : str
        Project identifier (name).
    version : str
        Project version.

    """

    identifier_name: str
    version: str = ""


def extract_metadata(root: Path) -> ProjectMetadata:
    """Extract the project name and version.

    Both fields are searched independently through a shared cascade:
    - Name: pyproject.toml > setup.cfg > PKG-INFO > setup.py > directory name
    - Version: pyproject.toml (when explicit) > setup.cfg > PKG-INFO > setup.py
      > version file > __init__.py

    This retrieves a version from version.py even when pyproject.toml declares dynamic versioning.

    Parameters
    ----------
    root : Path
        Root directory of the project from which metadata is extracted.

    Returns
    -------
    ProjectMetadata
        A ProjectMetadata object containing the resolved project name and version, with the directory name used as the name and an
        empty string used as the version when no source provides a value.

    """
    identifier_name: str | None = None
    version: str | None = None

    for n, v in _sources(root):
        identifier_name = identifier_name or n
        version = version or v
        if identifier_name and version:
            break

    return ProjectMetadata(identifier_name=identifier_name or root.name, version=version or "")


def _sources(root: Path) -> Iterator[tuple[str | None, str | None]]:
    """Yield (name, version) pairs from each source in priority order.

    Parameters
    ----------
    root : Path
        Root directory of the project used to locate metadata source files such as pyproject.toml, setup.cfg, setup.py, and
        package version files.

    Returns
    -------
    Iterator[tuple[str | None, str | None]]
        An iterator of (name, version) pairs from each source, in priority order. Each pair may contain None for name or version
        if the source does not define it.

    """
    yield _read_pyproject(root / "pyproject.toml")
    yield _read_setup_cfg(root / "setup.cfg")
    yield _read_pkginfo(root)
    yield _read_setup_py(root / "setup.py")
    yield _read_file_version(root)
    yield _read_init_package(root)


def _read_pyproject(source_path: Path) -> tuple[str | None, str | None]:
    """Extract name and version from pyproject.toml.

    Parameters
    ----------
    source_path : Path
        Path to the pyproject.toml file from which to extract project name and version.

    Returns
    -------
    tuple[str | None, str | None]
        Extract name and version from pyproject.toml and return them as a tuple of optional strings. The first element is the
        project name, the second is the version; either is None if absent or not determinable. If the file is missing or cannot be
        parsed, both elements are None.

    """
    if not source_path.is_file():
        return None, None

    try:
        with source_path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError:
        logger.warning("Invalid pyproject.toml: %s", source_path)
        return None, None

    project_record = data.get("project", {})
    identifier_name = project_record.get("name") or None
    # A missing version for dynamic = ["version"] yields None so the cascade continues.
    version = project_record.get("version") or None

    return identifier_name, version


def _read_setup_cfg(source_path: Path) -> tuple[str | None, str | None]:
    """Extract name and version from setup.cfg.

    Parameters
    ----------
    source_path : Path
        Path to the setup.cfg file to read.

    Returns
    -------
    tuple[str | None, str | None]
        Returns a tuple containing the project name and version extracted from the metadata section of setup.cfg. If setup.cfg is
        missing, cannot be parsed, or the metadata section does not define name or version, the corresponding tuple element is
        None.

    """
    if not source_path.is_file():
        return None, None

    config = configparser.ConfigParser()

    try:
        config.read(str(source_path), encoding="utf-8")
    except configparser.Error:
        return None, None

    if not config.has_section("metadata"):
        return None, None

    identifier_name = config.get("metadata", "name", fallback=None) or None
    version = config.get("metadata", "version", fallback=None) or None

    return identifier_name, version


def _read_pkginfo(root: Path) -> tuple[str | None, str | None]:
    """Extract name and version from PKG-INFO or *.egg-info/PKG-INFO.

    Parameters
    ----------
    root : Path
        Root directory from which to locate PKG-INFO or *.egg-info/PKG-INFO files.

    Returns
    -------
    tuple[str | None, str | None]
        A tuple of two optional strings: the package name and version extracted from the PKG-INFO file. Either element is None if
        the corresponding field is missing or no valid PKG-INFO file is found.

    """
    candidates = [root / "PKG-INFO", *root.glob("*.egg-info/PKG-INFO")]

    for source_path in candidates:
        if not source_path.is_file():
            continue

        try:
            payload = source_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        identifier_name: str | None = None
        version: str | None = None

        for source_line in payload.splitlines():
            if source_line.startswith("Name:"):
                identifier_name = source_line[5:].strip() or None

            elif source_line.startswith("Version:"):
                version = source_line[8:].strip() or None

        if identifier_name or version:
            return identifier_name, version

    return None, None


def _read_setup_py(source_path: Path) -> tuple[str | None, str | None]:
    """Extract name and version from setup.py with a regex heuristic.

    Parameters
    ----------
    source_path : Path
        Path to the setup.py file from which to extract the name and version.

    Returns
    -------
    tuple[str | None, str | None]
        A tuple containing the extracted project name and version from setup.py. Each element is None if the corresponding value
        cannot be found, or if the file is missing or cannot be read as UTF-8.

    """
    if not source_path.is_file():
        return None, None

    try:
        payload = source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None, None

    name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', payload)
    version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', payload)

    identifier_name = name_match.group(1) if name_match else None
    version = version_match.group(1) if version_match else None

    return identifier_name, version


def _read_file_version(root: Path) -> tuple[str | None, str | None]:
    """Find version.py / __version__.py / _version.py in packages at the root.

    Parameters
    ----------
    root : Path
        Root directory under which packages are searched for version files.

    Returns
    -------
    tuple[str | None, str | None]
        Returns a tuple of two optional strings; the first element is always None, and the second element is the extracted version
        string when a version file is found, otherwise None.

    """
    for package_dir in _find_packages(root):
        for file_name in VERSION_FILE_NAMES:
            source_path = package_dir / file_name
            if not source_path.is_file():
                continue

            version = _extract_version_ast(source_path)
            if version:
                return None, version

    return None, None


def _read_init_package(root: Path) -> tuple[str | None, str | None]:
    """Find __version__ in package __init__.py files at the root.

    Parameters
    ----------
    root : Path
        The root directory to search for packages.

    Returns
    -------
    tuple[str | None, str | None]
        Returns a tuple of two optional strings. The first element is always None; the second element contains the version string
        found in a package __init__.py file, or None if no version is found.

    """
    for package_dir in _find_packages(root):
        source_path = package_dir / "__init__.py"
        if not source_path.is_file():
            continue

        version = _extract_version_ast(source_path)
        if version:
            return None, version

    return None, None


def _find_packages(root: Path) -> list[Path]:
    """Return direct Python package directories below the root.

    Parameters
    ----------
    root : Path
        The directory to search for direct Python package directories.

    Returns
    -------
    list[Path]
        A list of Path objects representing the direct Python package directories found below the root.

    """
    return [
        d
        for d in sorted(root.iterdir())
        if d.is_dir() and d.name not in EXCLUDED_DIRECTORIES and not d.name.startswith(".") and (d / "__init__.py").exists()
    ]


def _extract_version_ast(source_path: Path) -> str | None:
    """Extract the value of __version__, version, or VERSION from a Python file via AST.

    Parameters
    ----------
    source_path : Path
        Path to the Python source file from which the version assignment is extracted.

    Returns
    -------
    str | None
        The extracted version string if a matching assignment is found; otherwise None.

    """
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None

    for syntax_node in ast.walk(tree):
        if not isinstance(syntax_node, ast.Assign):
            continue

        for target_symbol in syntax_node.targets:
            if (
                isinstance(target_symbol, ast.Name)
                and target_symbol.id in ("__version__", "version", "VERSION")
                and isinstance(syntax_node.value, ast.Constant)
                and isinstance(syntax_node.value.value, str)
            ):
                return syntax_node.value.value

    return None
