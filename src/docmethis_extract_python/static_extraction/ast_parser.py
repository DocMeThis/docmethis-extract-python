# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Read and parse Python files into AST trees."""

from __future__ import annotations

import ast
import logging
import tokenize
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def parse_file(source_path: Path) -> ast.Module | None:
    """Return a Python file's AST, or ``None`` when it cannot be parsed (DEC-T03).

    Parameters
    ----------
    source_path : Path
        Path to the Python file to parse.

    Returns
    -------
    ast.Module | None
        The parsed AST module for the given source file, or None if the file cannot be parsed.

    """
    outcome = parse_file_with_lines(source_path)
    return outcome[0] if outcome is not None else None


def parse_file_with_lines(source_path: Path) -> tuple[ast.Module, list[str]] | None:
    """Return ``(AST tree, source lines)`` or ``None`` when parsing fails (DEC-T03).

    Parameters
    ----------
    source_path : Path
        The path to the Python source file to read and parse.

    Returns
    -------
    tuple[ast.Module, list[str]] | None
        A tuple containing the parsed AST module and the source code as a list of lines, or None if the file cannot be read or
        parsed.

    """
    try:
        with source_path.open("rb") as f:
            encoding = tokenize.detect_encoding(f.readline)[0]
        source = source_path.read_text(encoding=encoding)
    except (SyntaxError, UnicodeDecodeError, OSError):
        logger.warning("Unable to read %s - unsupported encoding.", source_path)
        return None

    try:
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError:
        logger.warning("Syntax error in %s - file ignored.", source_path)
        return None

    return tree, source.splitlines()
