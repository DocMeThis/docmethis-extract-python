# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Incremental analysis cache - storage and invalidation by SHA-256 hash.

The cache is a ``.docmethis_cache.json`` file at the project root (DEC-003).
Each module is indexed by its normalized absolute POSIX path.

The file hash alone is insufficient: a ``ModuleRecord`` also depends on the model
schema and extraction settings. The file therefore stores a global fingerprint,
and any mismatch invalidates the whole cache (see ``cache_fingerprint``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from docmethis_extract_python.static_extraction.models import (
    ModuleRecord,
    _convert_for_json,
    deserialize_module_record,
)

if TYPE_CHECKING:
    from pathlib import Path

    from docmethis_extract_python.static_extraction.configuration import ConfigurationDocmethis

logger = logging.getLogger(__name__)

CACHE_FILE_NAME = ".docmethis_cache.json"

# Serialized schema version. Increment when a field is added, removed, or
# redefined in ModuleRecord/ClassRecord/FunctionRecord. Otherwise an entry written
# by an older version reloads with defaults for new fields (for example,
# has_overloads=False in a pre-it.10 cache) without reporting an error.
CACHE_SCHEMA_VERSION = 5

# ConfigurationDocmethis fields that change a ModuleRecord for identical source.
# They are part of the fingerprint, so changing them must invalidate the cache.
# `exclude_patterns` is absent: it filters analyzed files without changing a file
# record. `full_reanalysis` already bypasses the cache.
# Consistency is checked by test_cache.py::test_cache_fingerprint_covers_extraction_fields.
EXTRACTION_CONFIG_FIELDS: tuple[str, ...] = ("mixin_max_methods",)


def compute_file_hash(source_path: Path) -> str:
    """Return the SHA-256 hash of a file's contents.

    Parameters
    ----------
    source_path : Path
        Path to the file whose SHA-256 hash is to be computed.

    Returns
    -------
    str
        The SHA-256 hash of the file's contents as a hexadecimal string.

    """
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


def cache_fingerprint(configuration: ConfigurationDocmethis) -> dict[str, Any]:
    """Return the fingerprint controlling cache reuse, excluding file contents.

    Combine the serialized schema version and extraction settings. Runs with different fingerprints cannot share entries, even for
    identical source files.

    Parameters
    ----------
    configuration : ConfigurationDocmethis
        Configuration object whose extraction settings are used to build the cache fingerprint.

    Returns
    -------
    dict[str, Any]
        A dictionary that maps the key 'version_schema' to the current cache schema version and the key 'config' to a dictionary
        of extraction configuration field names and their values from the supplied configuration.

    """
    return {
        "version_schema": CACHE_SCHEMA_VERSION,
        "config": {identifier_name: getattr(configuration, identifier_name) for identifier_name in EXTRACTION_CONFIG_FIELDS},
    }


def load_cache(root_project: Path, configuration: ConfigurationDocmethis) -> dict[str, dict[str, Any]]:
    """Load the cache from ``.docmethis_cache.json``.

    Return a dict indexed by POSIX path. Return an empty dict when the file is missing, invalid, or has a different fingerprint
    from the current run; entries are then not reusable and the project is fully reanalyzed.

    Parameters
    ----------
    root_project : Path
        Root project directory used to resolve the cache file location.
    configuration : ConfigurationDocmethis
        Configuration used to compute the cache fingerprint; the cache is reused only when its stored fingerprint matches this
        configuration.

    Returns
    -------
    dict[str, dict[str, Any]]
        A dictionary mapping POSIX path strings to cache entry dictionaries. It is empty when the cache file is missing, invalid,
        corrupted, or has a fingerprint that does not match the current run.

    """
    path_cache = root_project / CACHE_FILE_NAME

    if not path_cache.is_file():
        logger.debug("No cache found in %s - full analysis.", root_project)
        return {}

    try:
        payload = path_cache.read_text(encoding="utf-8")
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Invalid cache in %s - ignored.", path_cache)
        return {}

    if not isinstance(data, dict):
        logger.warning("Corrupted cache in %s - ignored.", path_cache)
        return {}

    # A cache written before fingerprints were introduced has no key. Treat it as
    # a fingerprint mismatch and reanalyze it.
    expected_fingerprint = cache_fingerprint(configuration)
    if data.get("empreinte") != expected_fingerprint:
        logger.info(
            "Cache for %s has a different fingerprint (%r != %r) - full reanalysis.",
            root_project,
            data.get("empreinte"),
            expected_fingerprint,
        )
        return {}

    modules = data.get("modules", {})
    if not isinstance(modules, dict):
        logger.warning("Corrupted cache in %s - ignored.", path_cache)
        return {}

    logger.info("Cache loaded: %d entry/entries.", len(modules))
    return modules


def save_cache(root_project: Path, modules: list[ModuleRecord], configuration: ConfigurationDocmethis) -> None:
    """Save a list of ModuleRecords to ``.docmethis_cache.json``.

    Parameters
    ----------
    root_project : Path
        Root project directory where the cache file will be saved.
    modules : list[ModuleRecord]
        List of ModuleRecord objects to be saved to the cache file.
    configuration : ConfigurationDocmethis
        Configuration settings used to compute the cache fingerprint for the saved cache file.

    """
    entries: dict[str, Any] = {}
    for module in modules:
        mapping_key = module.file_path.as_posix()
        entries[mapping_key] = asdict(module)

    cache = {"empreinte": cache_fingerprint(configuration), "modules": entries}

    path_cache = root_project / CACHE_FILE_NAME
    path_cache.write_text(
        json.dumps(cache, default=_convert_for_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Cache saved: %d module(s).", len(modules))


def is_cache_valid(source_path: Path, cache: dict[str, dict[str, Any]]) -> bool:
    """Check whether the cache is valid for a given file.

    Return ``True`` when the file hash matches the cached hash.

    Parameters
    ----------
    source_path : Path
        Path to the source file whose cache validity is being checked.
    cache : dict[str, dict[str, Any]]
        A dictionary mapping source file paths to cached metadata, including a 'file_hash' entry used to validate the cache.

    Returns
    -------
    bool
        True if the cache contains an entry for the source path and its stored file hash matches the current file hash; False
        otherwise.

    """
    mapping_key = source_path.as_posix()
    if mapping_key not in cache:
        return False

    current_hash = compute_file_hash(source_path)
    return cache[mapping_key].get("file_hash", "") == current_hash


def restore_module_from_cache(data_cache: dict[str, Any]) -> ModuleRecord:
    """Rebuild a ModuleRecord from cached data.

    Parameters
    ----------
    data_cache : dict[str, Any]
        Cached data representing a serialized ModuleRecord, used to rebuild the record.

    Returns
    -------
    ModuleRecord
        A ModuleRecord reconstructed from the provided cache data, with the from_cache flag set to True.

    """
    record = deserialize_module_record(data_cache)
    record.from_cache = True
    return record
