# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Independent exception analysis depth, catch boundaries, and persisted distances."""

import json
from pathlib import Path

import pytest

from docmethis_extract_python.api import (
    Confidence,
    ConfigurationDocmethis,
    analyze_project,
    apply_project_passes,
    deserialize_project_record,
    iter_functions,
    load_configuration,
    serialize_project,
)


@pytest.mark.parametrize("depth", [0, 1, 2, 4])
def test_analysis_depth_is_exact_and_roundtrips(tmp_path: Path, depth: int) -> None:
    """Analysis retains exact call distances and does not propagate through handlers."""
    (tmp_path / "module.py").write_text(
        "def a():\n    return b()\n"
        "def b():\n    return c()\n"
        "def c():\n    return d()\n"
        "def d():\n    return leaf()\n"
        "def leaf():\n    raise ValueError('bad')\n"
        "def caught():\n    try:\n        leaf()\n    except ValueError:\n        return None\n",
        encoding="utf-8",
    )
    config = ConfigurationDocmethis(exception_analysis_depth=depth, exception_documentation_depth=min(1, depth))
    project = analyze_project(tmp_path, config, skip_dynamic=True, write_cache=False)
    restored = deserialize_project_record(json.loads(serialize_project(project)))
    functions = {function.qualified_name.rsplit(".", 1)[-1]: function for function in iter_functions(restored.modules)}
    for name, distance in (("a", 4), ("b", 3), ("c", 2), ("d", 1), ("leaf", 0)):
        exceptions = functions[name].exceptions or []
        assert [exception.propagation_depth for exception in exceptions] == ([distance] if distance <= depth else [])
    assert not functions["caught"].exceptions
    assert functions["caught"].exceptions_confidence is Confidence.EXPLICIT
    apply_project_passes(restored, exception_analysis_depth=0)
    assert not functions["a"].exceptions
    assert functions["leaf"].exceptions[0].propagation_depth == 0


def test_shortest_path_and_cycles_do_not_duplicate_facts(tmp_path: Path) -> None:
    """The same fact reached through several paths keeps the shortest distance."""
    (tmp_path / "module.py").write_text(
        "def a():\n    b()\n    leaf()\ndef b():\n    a()\n    leaf()\ndef leaf():\n    raise ValueError('bad')\n",
        encoding="utf-8",
    )
    project = analyze_project(tmp_path, ConfigurationDocmethis(exception_analysis_depth=4), skip_dynamic=True, write_cache=False)
    for function in iter_functions(project.modules):
        assert len(function.exceptions) == 1
        assert function.exceptions[0].propagation_depth == (0 if function.qualified_name.endswith(".leaf") else 1)


@pytest.mark.parametrize(("analysis", "documentation"), [(0, 0), (4, 0), (2, 1), (4, 4)])
def test_depth_configuration_is_independent(tmp_path: Path, analysis: int, documentation: int) -> None:
    """A deeper AST does not expand the documentary scope."""
    (tmp_path / "pyproject.toml").write_text(
        f"[tool.docmethis]\nexception_analysis_depth = {analysis}\nexception_documentation_depth = {documentation}\n",
        encoding="utf-8",
    )
    config = load_configuration(tmp_path)
    assert config.exception_analysis_depth == analysis
    assert config.exception_documentation_depth == documentation


@pytest.mark.parametrize("values", [(True, 0), (2, False), (-1, 0), (2, "1"), (1, 2)])
def test_invalid_depth_configuration_is_rejected(values: tuple[object, object]) -> None:
    """Reject booleans, coercion, negative depths, and documentation beyond analysis."""
    with pytest.raises(ValueError, match=r"exception_.*depth"):
        ConfigurationDocmethis(exception_analysis_depth=values[0], exception_documentation_depth=values[1])


def test_scope_changes_reuse_raw_cache_but_recompute_enriched_facts(tmp_path: Path) -> None:
    """Changing scope never reloads previously enriched exceptions as local facts."""
    (tmp_path / "module.py").write_text("def a():\n    leaf()\ndef leaf():\n    raise ValueError('bad')\n", encoding="utf-8")
    analyze_project(tmp_path, ConfigurationDocmethis(exception_analysis_depth=4), skip_dynamic=True)
    project = analyze_project(
        tmp_path,
        ConfigurationDocmethis(exception_analysis_depth=0, exception_documentation_depth=0),
        skip_dynamic=True,
        write_cache=False,
    )
    assert project.quality_report.files_from_cache == 1
    function = next(function for function in iter_functions(project.modules) if function.qualified_name.endswith(".a"))
    assert not function.exceptions


@pytest.mark.parametrize("depth", [None, True, -1])
def test_serialized_exception_distances_must_be_explicit_valid_integers(tmp_path: Path, depth: object) -> None:
    """A malformed persisted distance cannot silently become a direct-call fact."""
    (tmp_path / "module.py").write_text("def leaf():\n    raise ValueError('bad')\n", encoding="utf-8")
    project = analyze_project(tmp_path, skip_dynamic=True, write_cache=False)
    data = json.loads(serialize_project(project))
    data["modules"][0]["functions"][0]["exceptions"][0]["propagation_depth"] = depth
    with pytest.raises(ValueError, match="propagation_depth"):
        deserialize_project_record(data)
