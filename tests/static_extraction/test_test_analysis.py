# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for static test-to-production-function linking (iteration 6)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docmethis_extract_python.static_extraction.models import ProjectRecord

from docmethis_extract_python.static_extraction.dynamic_analysis.test_analysis import (
    _build_alias_maps,
)
from docmethis_extract_python.static_extraction.models import (
    Confidence,
    ImportRecord,
    iter_functions,
)

# ---------------------------------------------------------------------------
# Unit tests - _build_alias_maps
# ---------------------------------------------------------------------------


def test_alias_maps_import_direct() -> None:
    """An ``import X as Y`` creates an entry in ``alias_modules``."""
    imports = [ImportRecord(module="portion", name=None, alias="P")]
    directs, modules = _build_alias_maps(imports)
    assert modules["P"] == "portion"
    assert not directs


def test_alias_maps_from_import() -> None:
    """A ``from X import Y`` creates an entry in ``alias_directs``."""
    imports = [ImportRecord(module="portion", name="closed", alias=None)]
    directs, modules = _build_alias_maps(imports)
    assert directs["closed"] == "portion.closed"
    assert not modules


def test_alias_maps_from_import_with_alias() -> None:
    """A ``from X import Y as Z`` creates ``alias_directs[Z] = 'X.Y'``."""
    imports = [ImportRecord(module="portion", name="closed", alias="closed_alias")]
    directs, _ = _build_alias_maps(imports)
    assert directs["closed_alias"] == "portion.closed"


def test_alias_maps_excludes_type_checking() -> None:
    """TYPE_CHECKING imports are not included in aliases."""
    imports = [ImportRecord(module="portion", name="Interval", alias=None, type_checking_only=True)]
    directs, modules = _build_alias_maps(imports)
    assert not directs
    assert not modules


def test_alias_maps_import_without_alias() -> None:
    """An ``import portion`` creates ``alias_modules['portion'] = 'portion'``."""
    imports = [ImportRecord(module="portion", name=None, alias=None)]
    _, modules = _build_alias_maps(imports)
    assert modules["portion"] == "portion"


# ---------------------------------------------------------------------------
# Integration tests with the root_portion fixture
# ---------------------------------------------------------------------------


def test_integration_tested_functions(result_portion: ProjectRecord) -> None:
    """Public production functions have at least one TestReference after analysis."""
    modules_prod = [m for m in result_portion.modules if not m.is_test and not m.is_stub]
    tested_functions = [
        fn for fn in iter_functions(modules_prod) if fn.test_coverage is not None and fn.test_coverage.is_directly_tested
    ]
    assert len(tested_functions) >= 1, "No production function was identified as tested"


def test_integration_tested_confidence(result_portion: ProjectRecord) -> None:
    """Tested functions have test_coverage_confidence == INFERRED_HIGH."""
    modules_prod = [m for m in result_portion.modules if not m.is_test and not m.is_stub]
    for fn in iter_functions(modules_prod):
        if fn.test_coverage is not None and fn.test_coverage.is_directly_tested:
            assert fn.test_coverage_confidence == Confidence.INFERRED_HIGH


def test_integration_untested_confidence(result_portion: ProjectRecord) -> None:
    """Untested functions have test_coverage_confidence == ABSENT."""
    modules_prod = [m for m in result_portion.modules if not m.is_test and not m.is_stub]
    for fn in iter_functions(modules_prod):
        if fn.test_coverage is not None and not fn.test_coverage.is_directly_tested:
            assert fn.test_coverage_confidence == Confidence.ABSENT


def test_integration_references_are_non_empty(result_portion: ProjectRecord) -> None:
    """Each TestReference for a tested function has valid fields."""
    modules_prod = [m for m in result_portion.modules if not m.is_test and not m.is_stub]
    for fn in iter_functions(modules_prod):
        if fn.test_coverage is None:
            continue
        for ref in fn.test_coverage.test_references:
            assert ref.test_qualified_name
            assert ref.test_module
            assert ref.test_file.exists()
            assert ref.test_line > 0
            assert ref.call_line > 0
            assert ref.framework in {"pytest", "unittest"}


def test_integration_pytest_framework(result_portion: ProjectRecord) -> None:
    """Portion tests without TestCase are detected as pytest tests."""
    modules_prod = [m for m in result_portion.modules if not m.is_test and not m.is_stub]
    refs = [
        ref for fn in iter_functions(modules_prod) if fn.test_coverage is not None for ref in fn.test_coverage.test_references
    ]
    assert any(r.framework == "pytest" for r in refs), "No pytest test detected"


def test_integration_test_count_is_consistent(result_portion: ProjectRecord) -> None:
    """test_count matches the number of TestReference entries."""
    modules_prod = [m for m in result_portion.modules if not m.is_test and not m.is_stub]
    for fn in iter_functions(modules_prod):
        if fn.test_coverage is not None and fn.test_coverage.is_directly_tested:
            assert fn.test_coverage.test_count == len(fn.test_coverage.test_references)


def test_integration_deduplicates_references(result_portion: ProjectRecord) -> None:
    """TestReference entries have no duplicate (test_qualified_name, test_module) pairs."""
    modules_prod = [m for m in result_portion.modules if not m.is_test and not m.is_stub]
    for fn in iter_functions(modules_prod):
        if fn.test_coverage is None:
            continue
        seen: set[tuple[str, str]] = set()
        for ref in fn.test_coverage.test_references:
            mapping_key = (ref.test_qualified_name, ref.test_module)
            assert mapping_key not in seen, f"Duplicate: {mapping_key}"
            seen.add(mapping_key)
