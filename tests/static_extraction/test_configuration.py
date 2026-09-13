# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for the configuration module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docmethis_extract_python.static_extraction.configuration import (
    DEFAULT_EXCLUSIONS,
    ConfigurationDocmethis,
    load_configuration,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestConfigurationDocmethis:
    """Tests for the ConfigurationDocmethis dataclass."""

    def test_default_values(self) -> None:
        config = ConfigurationDocmethis()
        assert config.exclude_patterns == DEFAULT_EXCLUSIONS
        assert config.extra_exclude_patterns == ()
        assert config.full_reanalysis is False

    def test_all_exclusion_patterns_without_extra(self) -> None:
        config = ConfigurationDocmethis()
        assert config.all_exclusion_patterns == DEFAULT_EXCLUSIONS

    def test_all_exclusion_patterns_with_extra(self) -> None:
        config = ConfigurationDocmethis(extra_exclude_patterns=("migrations/", "legacy/"))
        outcome = config.all_exclusion_patterns
        assert outcome == (*DEFAULT_EXCLUSIONS, "migrations/", "legacy/")

    def test_frozen(self) -> None:
        config = ConfigurationDocmethis()
        try:
            config.full_reanalysis = True  # type: ignore[misc]
        except AttributeError:
            pass
        else:
            msg = "ConfigurationDocmethis should be frozen"
            raise AssertionError(msg)


class TestLoadConfiguration:
    """Tests for loading configuration from pyproject.toml."""

    def test_without_pyproject(self, tmp_path: Path) -> None:
        config = load_configuration(tmp_path)
        assert config == ConfigurationDocmethis()

    def test_pyproject_without_docmethis_section(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"\n', encoding="utf-8")
        config = load_configuration(tmp_path)
        assert config == ConfigurationDocmethis()

    def test_empty_docmethis_section(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.docmethis]\n", encoding="utf-8")
        config = load_configuration(tmp_path)
        assert config.extra_exclude_patterns == ()

    def test_pyproject_with_exclude_patterns(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[tool.docmethis]\nexclude_patterns = ["migrations/", "legacy/"]\n',
            encoding="utf-8",
        )
        config = load_configuration(tmp_path)
        assert config.extra_exclude_patterns == ("migrations/", "legacy/")
        # Defaults are always present.
        assert config.exclude_patterns == DEFAULT_EXCLUSIONS

    def test_invalid_pyproject(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("this is not valid TOML [[[", encoding="utf-8")
        config = load_configuration(tmp_path)
        assert config == ConfigurationDocmethis()

    def test_default_exclusions_include_pycache(self) -> None:
        assert "__pycache__/" in DEFAULT_EXCLUSIONS
        assert ".venv/" in DEFAULT_EXCLUSIONS
        assert ".git/" in DEFAULT_EXCLUSIONS
