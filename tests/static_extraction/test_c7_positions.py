# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""C7 tests for module and class positions."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

from docmethis_extract_python.static_extraction.ast_analysis import extract_module_record
from docmethis_extract_python.static_extraction.models import Visibility, deserialize_module_record

if TYPE_CHECKING:
    from pathlib import Path


def test_module_and_class_positions_for_check_c7(tmp_path: Path) -> None:
    """Module 1 exposes the lines needed by the module and class checks."""
    code = dedent(
        '''
        """Module summary."""

        class Service:
            """Class summary."""

            def run(self) -> None:
                return None
        '''
    ).lstrip()
    source_file = tmp_path / "module.py"
    source_file.write_text(code, encoding="utf-8")

    record = extract_module_record(source_file, "module", is_stub=False, is_test=False, file_hash="abc")

    assert record is not None
    assert record.line_start == 1
    assert record.line_end == len(code.splitlines())
    assert record.docstring_line_start == 1
    assert record.visibility is Visibility.PUBLIC

    class_ = record.classes[0]
    assert class_.line_start == 3
    assert class_.line_end == 7
    assert class_.docstring_line_start == 4


def test_cache_deserialization_accepts_records_without_c7_positions(tmp_path: Path) -> None:
    """Legacy caches without module/docstring positions remain readable."""
    source_file = tmp_path / "module.py"
    data = {
        "file_path": str(source_file),
        "module_name": "module",
        "functions": [],
        "classes": [
            {
                "qualified_name": "module.Service",
                "file_path": str(source_file),
                "line_start": 1,
                "line_end": 2,
                "col_start": 0,
                "col_end": 0,
                "visibility": "public",
                "parent_module": "module",
                "existing_docstring": None,
            }
        ],
    }

    record = deserialize_module_record(data)

    assert record.line_start == 1
    assert record.line_end == 0
    assert record.visibility is Visibility.PUBLIC
    assert record.docstring_line_start is None
    assert record.classes[0].docstring_line_start is None
