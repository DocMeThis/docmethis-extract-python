# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Load configuration from [tool.docmethis] in pyproject.toml."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING

from docmethis_extract_python.static_extraction.ast_inheritance import DEFAULT_MIXIN_MAX_METHODS

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _load_default_exclusions() -> tuple[str, ...]:
    """Load default exclusion patterns from the packaged exclusions.toml file.

    Returns
    -------
    tuple[str, ...]
        Loads the default exclusion patterns from the bundled TOML configuration file and returns them as a tuple of strings.

    """
    resource = files("docmethis_extract_python").joinpath("config", "exclusions.toml")
    return tuple(tomllib.loads(resource.read_text(encoding="utf-8"))["patterns"])


# Default exclusions (DEC-002)
DEFAULT_EXCLUSIONS: tuple[str, ...] = _load_default_exclusions()


@dataclass(frozen=True)
class ConfigurationDocmethis:
    """Configuration loaded from [tool.docmethis] in pyproject.toml.

    Default exclusions are always applied.
    Users can add patterns through ``exclude_patterns``.

    Attributes
    ----------
    exclude_patterns : tuple[str, ...]
        Default exclusion patterns, always applied.
    extra_exclude_patterns : tuple[str, ...]
        Additional exclusion patterns provided by the user.
    full_reanalysis : bool
        Whether a full reanalysis is forced.
    mixin_max_methods : int
        Method threshold beyond which a *Mixin class is excluded by the heuristic.

    """

    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUSIONS
    extra_exclude_patterns: tuple[str, ...] = ()
    full_reanalysis: bool = False
    # Method threshold beyond which a *Mixin class is excluded by the heuristic.
    mixin_max_methods: int = DEFAULT_MIXIN_MAX_METHODS

    @property
    def all_exclusion_patterns(self) -> tuple[str, ...]:
        """Return the combination of default and additional exclusions.

        Returns
        -------
        tuple[str, ...]
            Returns a tuple of strings containing the combined exclusion patterns, formed by appending the additional exclusion
            patterns to the default exclusion patterns.

        """
        return self.exclude_patterns + self.extra_exclude_patterns


def load_configuration(root_project: Path) -> ConfigurationDocmethis:
    """Load configuration from pyproject.toml at the project root.

    Return the default configuration when the file or ``[tool.docmethis]`` section is missing.

    Parameters
    ----------
    root_project : Path
        The root directory of the project from which the pyproject.toml configuration is loaded.

    Returns
    -------
    ConfigurationDocmethis
        A ConfigurationDocmethis instance populated from the [tool.docmethis] section of pyproject.toml, or the default
        configuration if the file is missing, invalid, or the section is absent.

    """
    path_pyproject = root_project / "pyproject.toml"

    if not path_pyproject.is_file():
        logger.debug("No pyproject.toml found in %s - using default configuration.", root_project)
        return ConfigurationDocmethis()

    try:
        with path_pyproject.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError:
        logger.warning("Invalid pyproject.toml in %s - using default configuration.", root_project)
        return ConfigurationDocmethis()

    section = data.get("tool", {}).get("docmethis", {})

    return ConfigurationDocmethis(
        extra_exclude_patterns=tuple(section.get("exclude_patterns", ())),
        mixin_max_methods=_read_mixin_max_methods(section, path_pyproject),
    )


def _read_mixin_max_methods(section: dict, path_pyproject: Path) -> int:
    """Read ``mixin_max_methods`` and return the default when the value is unusable.

    A direct `int(...)` call would expose a raw ``TypeError``/``ValueError`` at the CLI for a simple typo in ``pyproject.toml``; a
    value <= 0 would silently disable mixin detection because no class has a negative number of methods.

    Parameters
    ----------
    section : dict
        Configuration section from which mixin_max_methods is read.
    path_pyproject : Path
        Path to the pyproject.toml file, used in warning messages when the configured value is invalid.

    Returns
    -------
    int
        Returns the configured mixin_max_methods value when it is a positive integer; otherwise returns the default threshold.

    """
    raw = section.get("mixin_max_methods", DEFAULT_MIXIN_MAX_METHODS)

    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw

    logger.warning(
        "Invalid mixin_max_methods (%r) in %s - applying default threshold %d.",
        raw,
        path_pyproject,
        DEFAULT_MIXIN_MAX_METHODS,
    )
    return DEFAULT_MIXIN_MAX_METHODS
