# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Integration tests: end-to-end analysis of the portion project."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from docmethis_extract_python.static_extraction import __main__ as module_main
from docmethis_extract_python.static_extraction.__main__ import analyze_project
from docmethis_extract_python.static_extraction.cache import CACHE_FILE_NAME
from docmethis_extract_python.static_extraction.configuration import ConfigurationDocmethis
from docmethis_extract_python.static_extraction.models import (
    Confidence,
    ProjectRecord,
    deserialize_project_record,
    serialize_project,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestAnalyzePortionProject:
    """Integration tests for the mock portion project."""

    def test_analyze_complete(self, result_portion: ProjectRecord) -> None:
        assert result_portion.project_name == "portion"
        assert len(result_portion.modules) > 5
        total_fn = sum(len(m.functions) for m in result_portion.modules)
        total_cls = sum(len(m.classes) for m in result_portion.modules)
        total_methods = sum(len(c.methods) for m in result_portion.modules for c in m.classes)
        assert total_fn + total_methods > 10
        assert total_cls > 0

    def test_no_errors(self, result_portion: ProjectRecord) -> None:
        report = result_portion.quality_report
        assert report is not None
        assert report.files_with_errors == 0

    def test_signature_is_populated(self, result_portion: ProjectRecord) -> None:
        for module in result_portion.modules:
            for fn in module.functions:
                assert fn.signature is not None
                assert fn.signature_confidence is Confidence.EXPLICIT
                assert fn.complexity is not None
                assert fn.complexity.loc >= 1
            for class_ in module.classes:
                for meth in class_.methods:
                    assert meth.signature is not None
                    assert meth.complexity is not None

    def test_complete_json_serialization(self, result_portion: ProjectRecord) -> None:
        json_str = serialize_project(result_portion)
        data = json.loads(json_str)
        assert "modules" in data
        assert len(data["modules"]) > 0


class TestEffectiveCache:
    """Test cache effectiveness."""

    def test_second_execution_uses_cache(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "cache-test"\n', encoding="utf-8")
        (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("class Bar:\n    def method(self):\n        pass\n", encoding="utf-8")
        analyze_project(tmp_path)
        result_two = analyze_project(tmp_path)
        cached_count = sum(1 for module in result_two.modules if module.from_cache)
        assert cached_count == len(result_two.modules)

    def test_full_reanalysis_ignore_cache(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "cache-test"\n', encoding="utf-8")
        (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("class Bar:\n    def method(self):\n        pass\n", encoding="utf-8")
        analyze_project(tmp_path)
        config = ConfigurationDocmethis(full_reanalysis=True)
        result_two = analyze_project(tmp_path, config)
        cached_count = sum(1 for module in result_two.modules if module.from_cache)
        assert cached_count == 0


class TestCliFullReanalysis:
    """Verify that the CLI preserves user configuration."""

    PYPROJECT = '[project]\nname = "cfg"\n\n[tool.docmethis]\nmixin_max_methods = 42\nexclude_patterns = ["vendor/"]\n'

    def _config_after_main(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *argv: str) -> ConfigurationDocmethis:
        """Run main() with the given arguments and return the passed configuration."""
        (tmp_path / "pyproject.toml").write_text(self.PYPROJECT, encoding="utf-8")
        captures: list[ConfigurationDocmethis] = []

        def _faux_analyze(root: Path, configuration: ConfigurationDocmethis | None = None, **_: object) -> ProjectRecord:
            assert configuration is not None
            captures.append(configuration)
            return ProjectRecord(project_root=root, project_name="cfg")

        monkeypatch.setattr(module_main, "analyze_project", _faux_analyze)
        monkeypatch.setattr(sys, "argv", ["docmethis", str(tmp_path), *argv])
        module_main.main()

        assert len(captures) == 1
        return captures[0]

    def test_full_reanalysis_preserves_user_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--full-reanalysis changes only full_reanalysis, not the rest of [tool.docmethis]."""
        config = self._config_after_main(tmp_path, monkeypatch, "--full-reanalysis")

        assert config.full_reanalysis is True
        assert config.mixin_max_methods == 42
        assert "vendor/" in config.extra_exclude_patterns

    def test_config_unchanged_without_full_reanalysis(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the flag, the loaded configuration is passed unchanged."""
        config = self._config_after_main(tmp_path, monkeypatch)

        assert config.full_reanalysis is False
        assert config.mixin_max_methods == 42


class TestMinimalProject:
    """Test projects created from scratch."""

    def test_project_empty(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "vide"\nversion = "0.1.0"\n',
            encoding="utf-8",
        )
        outcome = analyze_project(tmp_path)
        assert outcome.project_name == "vide"
        assert outcome.modules == []

    def test_file_syntax_error(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "test"\n',
            encoding="utf-8",
        )
        (tmp_path / "ok.py").write_text("def foo(): pass\n", encoding="utf-8")
        (tmp_path / "bad.py").write_text("def foo(:\n", encoding="utf-8")
        outcome = analyze_project(tmp_path)
        assert len(outcome.modules) == 1
        assert outcome.modules[0].module_name == "ok"
        report = outcome.quality_report
        assert report is not None
        assert report.files_with_errors == 1

    def test_modification_is_detected(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "test"\n',
            encoding="utf-8",
        )
        source_file = tmp_path / "mod.py"
        source_file.write_text("x = 1\n", encoding="utf-8")

        # First execution.
        analyze_project(tmp_path)

        # Modify the file.
        source_file.write_text("x = 2\ny = 3\n", encoding="utf-8")

        # Second execution.
        result_two = analyze_project(tmp_path)
        assert len(result_two.modules) == 1
        assert result_two.modules[0].from_cache is False

    def test_cache_file_is_created(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "test"\n',
            encoding="utf-8",
        )
        (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
        analyze_project(tmp_path)
        assert (tmp_path / CACHE_FILE_NAME).is_file()

    def test_copy_project_modification_reanalyzes_only_one_file(self, tmp_path: Path) -> None:
        """A synthetic project modification causes only the changed file to be reanalyzed."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
        (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("def bar():\n    return 2\n", encoding="utf-8")

        r1 = analyze_project(tmp_path)
        module_count = len(r1.modules)

        (tmp_path / "a.py").write_text("def foo():\n    return 42\n", encoding="utf-8")

        r2 = analyze_project(tmp_path)
        cached_count = sum(1 for module in r2.modules if module.from_cache)
        assert cached_count == module_count - 1


def _make_simple_project(tmp_path: Path, payload: str) -> Path:
    """Create a minimal project with one Python file."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\nversion = "0.1.0"\n', encoding="utf-8")
    (tmp_path / "mod.py").write_text(payload, encoding="utf-8")
    return tmp_path


class TestIteration3IOEffects:
    """Integration tests: I/O effects are populated for all functions."""

    def test_io_effects_populated_for_all_functions(self, tmp_path: Path) -> None:
        payload = "def foo():\n    return 1\n\ndef bar(x):\n    return x\n"
        _make_simple_project(tmp_path, payload)
        outcome = analyze_project(tmp_path)
        for module in outcome.modules:
            for fn in module.functions:
                assert fn.io_effects is not None
                assert fn.io_effects_confidence is Confidence.EXPLICIT

    def test_io_effects_detect_print(self, tmp_path: Path) -> None:
        payload = "def foo():\n    print('hello')\n"
        _make_simple_project(tmp_path, payload)
        outcome = analyze_project(tmp_path)
        fn = outcome.modules[0].functions[0]
        assert fn.io_effects is not None
        assert fn.io_effects.has_output
        assert any(d.category == "has_output" for d in fn.io_effects.details)

    def test_portion_io_effects_are_populated(self, result_portion: ProjectRecord) -> None:
        for module in result_portion.modules:
            for fn in module.functions:
                assert fn.io_effects is not None, f"io_effects missing from {fn.qualified_name}"
            for class_ in module.classes:
                for meth in class_.methods:
                    assert meth.io_effects is not None, f"io_effects missing from {meth.qualified_name}"


class TestIteration3IsAsync:
    """Integration tests: asynchronous function detection."""

    def test_function_async_is_async_true(self, tmp_path: Path) -> None:
        payload = "async def fetch():\n    pass\n\ndef sync():\n    pass\n"
        _make_simple_project(tmp_path, payload)
        outcome = analyze_project(tmp_path)
        fns = {fn.qualified_name: fn for fn in outcome.modules[0].functions}
        assert fns["mod.fetch"].is_async is True
        assert fns["mod.sync"].is_async is False


class TestIteration3Percentiles:
    """Integration tests: percentiles are calculated after analysis."""

    def test_percentiles_calculated_after_analysis(self, tmp_path: Path) -> None:
        payload = "def f1():\n    pass\n\ndef f2(x):\n    if x:\n        return x\n    return 0\n"
        _make_simple_project(tmp_path, payload)
        outcome = analyze_project(tmp_path)
        for module in outcome.modules:
            for fn in module.functions:
                assert fn.complexity is not None
                assert fn.complexity.cyclomatic_percentile is not None
                assert fn.complexity.loc_percentile is not None
                assert fn.complexity.max_depth_percentile is not None
                assert 0.0 <= fn.complexity.cyclomatic_percentile <= 1.0

    def test_portion_percentiles_non_none(self, result_portion: ProjectRecord) -> None:
        for module in result_portion.modules:
            for fn in module.functions:
                assert fn.complexity is not None
                assert fn.complexity.cyclomatic_percentile is not None


class TestIteration3CacheRoundtrip:
    """Integration tests: cache round trips for complexity, io_effects, and is_async."""

    def test_complexity_survives_cache_roundtrip(self, tmp_path: Path) -> None:
        payload = "def f(x):\n    if x:\n        return x\n    return 0\n"
        _make_simple_project(tmp_path, payload)

        r1 = analyze_project(tmp_path)
        fn1 = r1.modules[0].functions[0]
        assert fn1.complexity is not None

        r2 = analyze_project(tmp_path)
        assert r2.modules[0].from_cache is True
        fn2 = r2.modules[0].functions[0]
        assert fn2.complexity is not None
        assert fn2.complexity.cyclomatic == fn1.complexity.cyclomatic
        assert fn2.complexity.loc == fn1.complexity.loc
        assert fn2.complexity.max_depth == fn1.complexity.max_depth
        assert fn2.io_effects is not None
        assert fn2.is_async == fn1.is_async

    def test_json_serialization_roundtrip_it3(self, tmp_path: Path) -> None:
        payload = "async def fetch():\n    print('x')\n    with open('f', 'w') as fh:\n        fh.write_text('data')\n"
        _make_simple_project(tmp_path, payload)
        outcome = analyze_project(tmp_path)

        json_str = serialize_project(outcome)
        data = json.loads(json_str)
        restored = deserialize_project_record(data)

        fn_orig = outcome.modules[0].functions[0]
        fn_restored = restored.modules[0].functions[0]

        assert fn_restored.is_async is True
        assert fn_restored.complexity is not None
        assert fn_restored.complexity.cyclomatic == fn_orig.complexity.cyclomatic
        assert fn_restored.io_effects is not None
        assert fn_restored.io_effects.has_output == fn_orig.io_effects.has_output
        assert fn_restored.io_effects.writes_files == fn_orig.io_effects.writes_files
