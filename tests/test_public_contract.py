# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for the candidate distribution boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import docmethis_extract_python
from docmethis_extract_python import api
from docmethis_extract_python.static_extraction.configuration import DEFAULT_EXCLUSIONS

PACKAGE_ROOT = Path(docmethis_extract_python.__file__).resolve().parent
EXPECTED_EXPORTS = {
    "ClassRecord",
    "Confidence",
    "FunctionRecord",
    "MethodType",
    "ModuleRecord",
    "ParameterType",
    "ProjectRecord",
    "Provenance",
    "Visibility",
}


def test_root_exports_are_frozen() -> None:
    """The candidate exposes only the nine R2 model exports at its root."""
    assert set(docmethis_extract_python.__all__) == EXPECTED_EXPORTS
    assert all(hasattr(docmethis_extract_python, name) for name in EXPECTED_EXPORTS)


def test_default_exclusions_are_package_data() -> None:
    """Default exclusions load without a monorepo-root configuration file."""
    assert "__pycache__/" in DEFAULT_EXCLUSIONS
    assert ".git/" in DEFAULT_EXCLUSIONS


def test_inter_package_api_is_available() -> None:
    """The downstream-facing API remains available under the target namespace."""
    assert callable(api.analyze_project)
    assert callable(api.apply_project_passes)
    assert callable(api.build_callgraph)
    assert callable(api.extract_module_record)
    assert callable(api.implicit_receiver_exclusion_reason)
    assert (
        api.implicit_receiver_exclusion_reason(
            parameter_name="self",
            method_kind=api.MethodType.INSTANCE_METHOD,
        )
        == "implicit_instance_receiver"
    )
    assert callable(api.iter_functions)
    assert callable(api.serialize_project)
    assert callable(api.deserialize_project_record)
    assert api.utc_now_iso().endswith("Z")


def test_source_has_no_historical_private_imports() -> None:
    """The candidate source does not import the monorepo or historical namespace."""
    forbidden_prefixes = ("docmethis.", "docmethis_gateway", "gateway")

    for source_path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported_names = [node.module or ""]
            else:
                continue

            assert not any(name.startswith(forbidden_prefixes) for name in imported_names), source_path
