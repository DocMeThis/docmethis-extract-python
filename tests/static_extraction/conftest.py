# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Shared fixtures for static_extraction tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from docmethis_extract_python.static_extraction.models import ProjectRecord

from docmethis_extract_python.static_extraction.__main__ import analyze_project

PORTION_PROJECT_PATH = Path(__file__).resolve().parent.parent / "mock" / "portion"


@pytest.fixture
def root_portion() -> Path:
    """Return the absolute path to the mock ``portion`` project."""
    return PORTION_PROJECT_PATH


@pytest.fixture(scope="session")
def result_portion() -> ProjectRecord:
    """Analyze ``portion`` once per test session."""
    return analyze_project(PORTION_PROJECT_PATH)
