# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Traverse a project and recursively discover Python files.

Handle exclusion patterns (DEC-002), test files, .pyi stubs, and module-name
resolution.

Iteration 8 (phase 4) - import classification:
``classify_imports`` enriches each ``ImportRecord`` with:
- ``classification``: "stdlib" | "third_party" | "local"
- ``is_unused``: True when the imported name never appears in the module.
"""

from __future__ import annotations

import ast
import fnmatch
import logging
import re
import sys
import tomllib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from docmethis_extract_python.static_extraction.configuration import ConfigurationDocmethis
    from docmethis_extract_python.static_extraction.models import ImportRecord, ProjectRecord

from docmethis_extract_python.static_extraction.ast_parser import parse_file
from docmethis_extract_python.static_extraction.models import Visibility

logger = logging.getLogger(__name__)

EXTENSIONS_PYTHON: frozenset[str] = frozenset({".py", ".pyw", ".pyi"})

TEST_FILE_NAMES: frozenset[str] = frozenset({"conftest.py"})

TEST_FILE_PREFIXES: tuple[str, ...] = ("test_", "tests_")

TEST_FILE_SUFFIXES: tuple[str, ...] = ("_test.py", "_tests.py")

TEST_DIRECTORIES: tuple[str, ...] = ("tests", "test")


def discover_files(root: Path, configuration: ConfigurationDocmethis) -> list[Path]:
    """Recursively discover Python files below *root*.

    Apply default and configured exclusion patterns. Return sorted absolute paths.

    Parameters
    ----------
    root : Path
        The root directory from which the recursive discovery of Python files begins.
    configuration : ConfigurationDocmethis
        The configuration object that supplies the exclusion patterns to apply during discovery.

    Returns
    -------
    list[Path]
        A list of absolute Path objects for the discovered Python files, sorted in ascending order.

    """
    patterns = configuration.all_exclusion_patterns
    files: list[Path] = []

    for source_path in root.rglob("*"):
        if not source_path.is_file():
            continue

        if source_path.suffix not in EXTENSIONS_PYTHON:
            continue

        if _is_excluded(source_path, root, patterns):
            continue

        files.append(source_path.resolve())

    files.sort()
    return files


def _is_excluded(source_path: Path, root: Path, patterns: tuple[str, ...]) -> bool:
    """Check whether *path* matches an exclusion pattern.

    Patterns ending in ``/`` are tested against each relative-path component. Other patterns are tested against the file name
    only.

    Parameters
    ----------
    source_path : Path
        Path to the file or directory to test against the exclusion patterns.
    root : Path
        The base directory used to compute the relative path of source_path; exclusion patterns are matched against components of
        this relative path.
    patterns : tuple[str, ...]
        Tuple of exclusion patterns. Patterns ending in '/' are matched against each component of the relative path; all other
        patterns are matched against the file name only.

    Returns
    -------
    bool
        True if the source path matches any exclusion pattern; False otherwise.

    """
    relative_path = source_path.relative_to(root)
    parts = relative_path.parts
    file_name = source_path.name

    for pattern in patterns:
        if pattern.endswith("/"):
            # Directory pattern: check whether any path component matches.
            pattern_sans_slash = pattern.rstrip("/")
            for part in parts[:-1]:  # Exclude the file name.
                if fnmatch.fnmatch(part, pattern_sans_slash):
                    return True

        elif fnmatch.fnmatch(file_name, pattern):
            return True

    return False


def is_test_file(source_path: Path, root: Path) -> bool:
    """Determine whether a file is a test file.

    A file is considered a test when:
    - Its name starts with ``test_`` or ends with ``_test.py``/``_tests.py``.
    - It is ``conftest.py``.
    - It is inside a ``tests/`` or ``test/`` directory.

    Parameters
    ----------
    source_path : Path
        Path to the file to inspect for test-file characteristics.
    root : Path
        The root directory used to resolve source_path to a relative path, so the function can determine whether the file resides
        inside a tests/ or test/ directory.

    Returns
    -------
    bool
        True if the file is considered a test file; False otherwise.

    """
    identifier_name = source_path.name

    if identifier_name in TEST_FILE_NAMES:
        return True

    if identifier_name.startswith(TEST_FILE_PREFIXES) or identifier_name.endswith(TEST_FILE_SUFFIXES):
        return True

    relative_path = source_path.relative_to(root)
    return any(part in TEST_DIRECTORIES for part in relative_path.parts[:-1])


def is_stub_file(source_path: Path) -> bool:
    """Determine whether a file is a type stub (.pyi).

    Parameters
    ----------
    source_path : Path
        Path to the file to check for whether it is a type stub.

    Returns
    -------
    bool
        True if the file is a type stub (has a .pyi suffix), False otherwise.

    """
    return source_path.suffix == ".pyi"


def resolve_module_name(source_path: Path, root: Path) -> str:
    """Resolve the dotted module name from a file path.

    Parameters
    ----------
    source_path : Path
        Path to the source file to resolve into a dotted module name.
    root : Path
        Root directory of the project, used to compute the module-relative path.

    Returns
    -------
    str
        The dotted module name derived from the path relative to the project root.

    Examples
    --------
    ``portion/interval.py`` -> ``"portion.interval"``
    ``portion/__init__.py`` -> ``"portion"``

    """
    relative_path = source_path.relative_to(root)
    parts = list(relative_path.with_suffix("").parts)

    # __init__ corresponds to the parent package, not an __init__ module.
    if parts and parts[-1] == "__init__":
        parts.pop()

    return ".".join(parts) if parts else source_path.stem


# ---------------------------------------------------------------------------
# Classification des imports (it.8 phase 4)
# ---------------------------------------------------------------------------

# Standard-library module names (Python 3.10+).
_STDLIB_NAMES: frozenset[str] = frozenset(sys.stdlib_module_names)

# Regex to extract the raw package name from a PEP 508 dependency.
# Examples: "httpx>=0.28,<1.0" -> "httpx"; "httpx[http2]>=0" -> "httpx".
_DEP_NAME_RE = re.compile(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)")


def _load_third_party_dependencies(project_root: Path) -> frozenset[str]:
    """Load third-party package names declared in pyproject.toml.

    Read ``[project].dependencies`` and ``[dependency-groups].*``. Normalize names to lowercase with hyphens replaced by
    underscores (PEP 503/625). Return an empty frozenset when pyproject.toml is missing or unreadable.

    Parameters
    ----------
    project_root : Path
        Root directory of the project whose pyproject.toml is scanned for third-party dependencies.

    Returns
    -------
    frozenset[str]
        A frozenset of normalized third-party dependency names declared in pyproject.toml, with each name lowercased and hyphens
        replaced by underscores. Returns an empty frozenset if pyproject.toml is missing, unreadable, or contains no matching
        dependencies.

    """
    pyproject = project_root / "pyproject.toml"
    if not pyproject.is_file():
        logger.debug("pyproject.toml absent in %s - third-party classification disabled.", project_root)
        return frozenset()
    try:
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
    except Exception:  # noqa: BLE001
        logger.warning("Unable to read pyproject.toml in %s - third-party classification disabled.", project_root)
        return frozenset()

    raw_dependencies: list[str] = list(data.get("project", {}).get("dependencies", []))
    for group_deps in data.get("dependency-groups", {}).values():
        for dep in group_deps:
            if isinstance(dep, str):
                raw_dependencies.append(dep)
            elif isinstance(dep, dict) and "include-group" not in dep:
                # PEP 735 dict-form sans include-group
                raw_dependencies.append(str(dep))

    names: set[str] = set()
    for raw_dependency in raw_dependencies:
        m = _DEP_NAME_RE.match(raw_dependency.strip())
        if m:
            names.add(m.group(1).lower().replace("-", "_"))
    return frozenset(names)


def _collect_used_names(source_path: Path) -> frozenset[str]:
    """Return the identifiers (``ast.Name.id``) used in the file.

    Walk the complete AST iteratively. Import aliases are not ``ast.Name`` nodes (they are stored as strings in
    ``alias.name``/``alias.asname``), so names introduced by imports do not skew the result.

    Parameters
    ----------
    source_path : Path
        Path to the Python source file whose AST will be walked to collect used identifier names.

    Returns
    -------
    frozenset[str]
        A frozenset of strings containing the names of all identifiers used in the file. If the file cannot be parsed, an empty
        frozenset is returned.

    """
    tree = parse_file(source_path)
    if tree is None:
        return frozenset()
    return frozenset(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))


def _imported_local_name(imp: ImportRecord) -> str:
    """Return the name introduced into the local scope by an ImportRecord.

    - ``import X [as Z]`` -> ``Z`` when aliased, otherwise the first segment of ``X``
    - ``from X import Y [as Z]`` -> ``Z`` when aliased, otherwise ``Y``

    Precondition: star imports (``imp.name == "*"``) are filtered upstream
    and must not be passed to this function.

    Parameters
    ----------
    imp : ImportRecord
        The ImportRecord to inspect. Must not be a star import (name == '*'), as those are filtered upstream.

    Returns
    -------
    str
        The name introduced into the local scope by an ImportRecord: the alias if present, otherwise the imported name, or the
        first segment of the module path for a plain module import.

    """
    return imp.alias or imp.name or imp.module.split(".")[0]


def _classify_import(imp: ImportRecord, third_party: frozenset[str]) -> str:
    """Return an ImportRecord classification: "stdlib" | "third_party" | "local".

    Parameters
    ----------
    imp : ImportRecord
        The ImportRecord to classify based on its module path.
    third_party : frozenset[str]
        A frozenset of normalized third-party package names used to determine whether an import's root module belongs to a
        third-party dependency.

    Returns
    -------
    str
        Return an ImportRecord classification: 'stdlib', 'third_party', or 'local'.

    """
    # Relative imports are always local.
    if imp.module.startswith("."):
        return "local"

    root = imp.module.split(".")[0]

    if root in _STDLIB_NAMES:
        return "stdlib"

    # Normalize the name for comparison with third-party dependencies (PEP 503).
    if root.lower().replace("-", "_") in third_party:
        return "third_party"

    return "local"


def classify_imports(project_record: ProjectRecord) -> None:
    """Add classification and ``is_unused`` to every project ``ImportRecord``.

    Classification :
    - ``"stdlib"``       : root module in ``sys.stdlib_module_names``
    - ``"third_party"``: root module listed in ``[project].dependencies`` in pyproject.toml
    - ``"local"``        : sinon (import relatif ou module interne au projet)

    ``is_unused`` :
    - ``True`` when the name introduced into local scope appears nowhere else in the module.
    - Star imports (``name == "*"``) always have ``is_unused = False`` because
      their symbols cannot be resolved statically.
    - ``type_checking_only`` imports (``if TYPE_CHECKING``) never have ``is_unused``
      because they are used only by the type checker (outside runtime).

    Limites connues :
    - Usage detection relies on an identifier appearing as ``ast.Name`` in the AST.
      Uses in string annotations (``"MyType"``), ``__all__``, or dynamic ``eval()``
      are not detected, so false positives are possible.
    - Package hierarchy is not considered: ``import pkg.sub`` introduces ``pkg``
      into scope; when only ``pkg.sub`` is referenced through ``Attribute``,
      detection sees the ``Name(id="pkg")`` in the attribute chain.

    Parameters
    ----------
    project_record : ProjectRecord
        The project record containing the modules and project root used to classify imports; its import records are updated in
        place.

    """
    third_party = _load_third_party_dependencies(project_record.project_root)

    for module in project_record.modules:
        used_names = _collect_used_names(module.file_path)

        for imp in module.imports:
            imp.classification = _classify_import(imp, third_party)

            # Star imports: impossible to analyze statically, never unused.
            if imp.name == "*":
                imp.is_unused = False
                continue

            # TYPE_CHECKING imports: used only by the type checker, never unused.
            if imp.type_checking_only:
                imp.is_unused = False
                continue

            local_name = _imported_local_name(imp)
            imp.is_unused = local_name not in used_names


_VISIBILITY_ORDER: dict[Visibility, int] = {
    Visibility.PUBLIC: 0,
    Visibility.PROTECTED: 1,
    Visibility.PRIVATE: 2,
}


def effective_visibility(module_visibility: Visibility, symbol_visibility: Visibility) -> Visibility:
    """Return the combined visibility of a symbol in its context.

    The most restrictive visibility wins (maximum in the order public < protected < private): a public member of a private
    container is private. The function is associative, so module -> class -> member nesting is safe.

    Parameters
    ----------
    module_visibility : Visibility
        The visibility of the module or enclosing context that contains the symbol.
    symbol_visibility : Visibility
        The visibility declared on the symbol itself.

    Returns
    -------
    Visibility
        The combined visibility of the symbol, determined by the most restrictive of the two input visibilities.

    """
    return max((module_visibility, symbol_visibility), key=_VISIBILITY_ORDER.__getitem__)
