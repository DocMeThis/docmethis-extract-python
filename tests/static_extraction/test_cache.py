# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for cache storage and invalidation."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

from docmethis_extract_python.static_extraction.cache import (
    CACHE_FILE_NAME,
    EXTRACTION_CONFIG_FIELDS,
    cache_fingerprint,
    compute_file_hash,
    is_cache_valid,
    load_cache,
    restore_module_from_cache,
    save_cache,
)
from docmethis_extract_python.static_extraction.configuration import ConfigurationDocmethis
from docmethis_extract_python.static_extraction.models import ModuleRecord

if TYPE_CHECKING:
    from pathlib import Path

CONFIG = ConfigurationDocmethis()


def _make_module_record(tmp_path: Path, identifier_name: str = "mod", payload: str = "x = 1\n") -> ModuleRecord:
    """Create a test ModuleRecord with a real file."""
    source_file = tmp_path / f"{identifier_name}.py"
    source_file.write_text(payload, encoding="utf-8")
    return ModuleRecord(
        file_path=source_file.resolve(),
        module_name=identifier_name,
        file_hash=compute_file_hash(source_file.resolve()),
    )


class TestComputeHash:
    """Tests for SHA-256 hash computation."""

    def test_hash_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("hello", encoding="utf-8")
        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_changes_with_content(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("hello", encoding="utf-8")
        h1 = compute_file_hash(f)
        f.write_text("world", encoding="utf-8")
        h2 = compute_file_hash(f)
        assert h1 != h2


class TestLoadAndSave:
    """Tests for saving and loading the cache."""

    def test_missing_cache(self, tmp_path: Path) -> None:
        cache = load_cache(tmp_path, CONFIG)
        assert cache == {}

    def test_save_and_load(self, tmp_path: Path) -> None:
        module = _make_module_record(tmp_path)
        save_cache(tmp_path, [module], CONFIG)
        cache = load_cache(tmp_path, CONFIG)
        assert len(cache) == 1

    def test_invalid_cache(self, tmp_path: Path) -> None:
        (tmp_path / ".docmethis_cache.json").write_text("not json", encoding="utf-8")
        cache = load_cache(tmp_path, CONFIG)
        assert cache == {}

    def test_roundtrip_module_record(self, tmp_path: Path) -> None:
        module = _make_module_record(tmp_path, "test_mod", "def foo(): pass\n")
        save_cache(tmp_path, [module], CONFIG)
        cache = load_cache(tmp_path, CONFIG)
        mapping_key = module.file_path.as_posix()
        restored = restore_module_from_cache(cache[mapping_key])
        assert restored.module_name == "test_mod"
        assert restored.file_path == module.file_path
        assert restored.from_cache is True


class TestCacheValidity:
    """Tests for cache validation."""

    def test_valid_cache(self, tmp_path: Path) -> None:
        module = _make_module_record(tmp_path)
        save_cache(tmp_path, [module], CONFIG)
        cache = load_cache(tmp_path, CONFIG)
        assert is_cache_valid(module.file_path, cache) is True

    def test_invalid_after_modification(self, tmp_path: Path) -> None:
        module = _make_module_record(tmp_path)
        save_cache(tmp_path, [module], CONFIG)
        cache = load_cache(tmp_path, CONFIG)
        # Modify the source file.
        module.file_path.write_text("y = 2\n", encoding="utf-8")
        assert is_cache_valid(module.file_path, cache) is False

    def test_file_missing_from_cache(self, tmp_path: Path) -> None:
        f = tmp_path / "unknown.py"
        f.write_text("x = 1", encoding="utf-8")
        assert is_cache_valid(f, {}) is False


class TestCacheFingerprint:
    """A file hash is insufficient; schema and extraction settings must invalidate."""

    def test_configuration_change_invalidates_cache(self, tmp_path: Path) -> None:
        """A different mixin threshold invalidates entries with unchanged sources."""
        module = _make_module_record(tmp_path)
        save_cache(tmp_path, [module], ConfigurationDocmethis(mixin_max_methods=5))

        assert load_cache(tmp_path, ConfigurationDocmethis(mixin_max_methods=5)) != {}
        assert load_cache(tmp_path, ConfigurationDocmethis(mixin_max_methods=10)) == {}

    def test_non_extraction_configuration_does_not_invalidate(self, tmp_path: Path) -> None:
        """exclude_patterns filters analyzed files but changes no record."""
        module = _make_module_record(tmp_path)
        save_cache(tmp_path, [module], CONFIG)

        other_configuration = ConfigurationDocmethis(extra_exclude_patterns=("vendor/",))
        assert load_cache(tmp_path, other_configuration) != {}

    def test_different_schema_version_invalidates_cache(self, tmp_path: Path) -> None:
        """A cache written under another schema is not loaded with defaults."""
        module = _make_module_record(tmp_path)
        save_cache(tmp_path, [module], CONFIG)

        source_path = tmp_path / CACHE_FILE_NAME
        data = json.loads(source_path.read_text(encoding="utf-8"))
        data["empreinte"]["version_schema"] = 999
        source_path.write_text(json.dumps(data), encoding="utf-8")

        assert load_cache(tmp_path, CONFIG) == {}

    def test_legacy_cache_without_fingerprint_is_invalid(self, tmp_path: Path) -> None:
        """Legacy flat format triggers reanalysis instead of loading missing fields."""
        module = _make_module_record(tmp_path)
        save_cache(tmp_path, [module], CONFIG)

        source_path = tmp_path / CACHE_FILE_NAME
        data = json.loads(source_path.read_text(encoding="utf-8"))
        source_path.write_text(json.dumps(data["modules"]), encoding="utf-8")

        assert load_cache(tmp_path, CONFIG) == {}

    def test_fingerprint_covers_extraction_fields(self) -> None:
        """Guardrail: every new configuration field must be classified explicitly.

        This test fails when a field is added to ConfigurationDocmethis. That is
        intentional: it forces a decision about whether the field affects a ModuleRecord
        (-> EXTRACTION_CONFIG_FIELDS, otherwise a stale cache could be reused silently).
        """
        non_extraction = {"exclude_patterns", "extra_exclude_patterns", "full_reanalysis"}
        fields = {field.name for field in dataclasses.fields(ConfigurationDocmethis)}

        assert fields == non_extraction | set(EXTRACTION_CONFIG_FIELDS)

    def test_fingerprint_is_json_serializable(self) -> None:
        """The fingerprint is compared after a JSON round trip and must survive it."""
        fingerprint = cache_fingerprint(CONFIG)
        assert json.loads(json.dumps(fingerprint)) == fingerprint
