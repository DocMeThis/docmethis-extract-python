# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Extract source text for FunctionRecord, ClassRecord, and ModuleRecord.

Three pure functions receive an AST node and the file's source lines, then return
a string ready for storage.

Architectural decisions:
- DEC-017: replace the complete method body with ``...`` (including the docstring)
- DEC-018: include closures as-is in FunctionRecord.source_code
- DEC-019: do not dedent; preserve source indentation exactly
- DEC-020: preserve multi-line signatures; handle one-line cases separately
"""

from __future__ import annotations

import ast

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def extract_source_function(
    node: _FunctionNode,
    lines: list[str],
) -> str:
    """Extract raw function source (lines line_start through line_end inclusive).

    Return the lines joined by newline without transformation or dedenting.

    Parameters
    ----------
    node : _FunctionNode
        AST node representing the function to extract, providing 1-based lineno and end_lineno attributes that delimit the source
        lines.
    lines : list[str]
        A list of strings representing the source code lines from which the function source is extracted, using the node's start
        and end line numbers to select the relevant slice.

    Returns
    -------
    str
        A string containing the raw source code of the function, formed by joining the selected lines with newline characters; the
        lines are returned unchanged, without transformation or dedenting.

    """
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def build_class_skeleton(
    node: ast.ClassDef,
    lines: list[str],
) -> str:
    """Return a class skeleton: declaration, attributes, and method stubs.

    Direct method bodies (``FunctionDef`` / ``AsyncFunctionDef``) are replaced by ``...``. Non-function nodes (attributes,
    annotations, ``__slots__``) are kept line by line.

    Parameters
    ----------
    node : ast.ClassDef
        The abstract syntax tree node of type ast.ClassDef representing the class from which the skeleton is built.
    lines : list[str]
        Source code lines of the file being analyzed, used to extract the original text for the class skeleton.

    Returns
    -------
    str
        A string containing the class skeleton, including the class declaration, attributes, and method stubs, with direct method
        bodies replaced by '...' and non-function nodes preserved line by line.

    """
    return "\n".join(_build_body_skeleton(node.body, lines, _effective_start(node), node.end_lineno))


def build_module_skeleton(
    function_records: list[_FunctionNode],
    classes: list[ast.ClassDef],
    lines: list[str],
) -> str:
    """Return a module skeleton: imports, constants, and function/class stubs.

    Top-level function bodies are replaced by ``...``. Classes are reduced to their skeleton through ``build_class_skeleton``.
    Imports, constants, and other lines are kept intact.

    Parameters
    ----------
    function_records : list[_FunctionNode]
        A list of function nodes representing the top-level functions to include in the module skeleton.
    classes : list[ast.ClassDef]
        A list of AST ClassDef nodes representing the top-level classes to include in the module skeleton.
    lines : list[str]
        The source lines of the module, used to copy imports, constants, and other intact content while building the module
        skeleton.

    Returns
    -------
    str
        A string containing the generated module skeleton, with imports, constants, and function/class stubs preserved, top-level
        function bodies replaced by ellipses, and classes reduced to their skeletons.

    """
    # Sort top-level nodes by appearance, including decorators.
    all_nodes: list[_FunctionNode | ast.ClassDef] = [*function_records, *classes]
    all_nodes.sort(key=_effective_start)

    output: list[str] = []
    i = 1  # 1-indexed, next line to process

    for node in all_nodes:
        node_start = _effective_start(node)

        # Copy lines preceding this node (imports, constants, etc.).
        output.extend(lines[i - 1 : node_start - 1])
        i = node_start

        if isinstance(node, ast.ClassDef):
            squelette = build_class_skeleton(node, lines)
            output.extend(squelette.split("\n"))
            i = node.end_lineno + 1
        else:
            i = _strip_function_body(node, lines, i, output)

    # Copy lines remaining after the last node.
    output.extend(lines[i - 1 :])

    return "\n".join(output)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _effective_start(node: _FunctionNode | ast.ClassDef) -> int:
    """Return the node's first line, including decorators (1-indexed).

    Parameters
    ----------
    node : _FunctionNode | ast.ClassDef
        The AST node (function or class definition) whose effective start line, including decorators, is to be computed.

    Returns
    -------
    int
        The node's first line number, including decorators, as a 1-indexed integer.

    """
    if node.decorator_list:
        return node.decorator_list[0].lineno
    return node.lineno


def _strip_function_body(
    node: _FunctionNode,
    lines: list[str],
    i: int,
    output: list[str],
) -> int:
    """Add stripped function lines to ``output``.

    Keep decorators and the signature, replacing the body with ``...``. Return the next line index to process (1-indexed).

    Parameters
    ----------
    node : _FunctionNode
        AST node representing the function definition whose body is to be stripped and replaced with an ellipsis.
    lines : list[str]
        The source code lines of the file being processed, used to extract and copy the decorator and signature lines.
    i : int
        The current line index (1-indexed) into lines from which to start processing.
    output : list[str]
        A list of strings to which the stripped function lines are appended; it is modified in place.

    Returns
    -------
    int
        The next line index to process (1-indexed).

    """
    if node.lineno == node.body[0].lineno:
        # One-line case (for example, ``def f(x): return x``): copy any
        # decorators and replace the def+body line.
        output.extend(lines[i - 1 : node.lineno - 1])
        indent = " " * node.col_offset
        kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        output.append(f"{indent}{kw} {node.name}(...): ...")
        i = node.end_lineno + 1
    else:
        # Multi-line case: copy decorators and signature, then replace the body.
        output.extend(lines[i - 1 : node.body[0].lineno - 1])
        indent = " " * node.body[0].col_offset
        output.append(f"{indent}...")
        i = node.end_lineno + 1
    return i


def _build_body_skeleton(
    body: list[ast.stmt],
    lines: list[str],
    start: int,
    end: int,
) -> list[str]:
    """Return class-body lines with method bodies stripped.

    Non-function nodes are copied line by line.

    Parameters
    ----------
    body : list[ast.stmt]
        Body of the class as a list of AST statement nodes; used to identify method definitions and to determine which lines to
        copy or strip.
    lines : list[str]
        Source code lines of the class body, used to copy non-method portions while stripping method bodies.
    start : int
        The 1-indexed line number in the source lines where the class body begins; processing of body lines starts from this
        position.
    end : int
        The 1-indexed end line of the class body (inclusive).

    Returns
    -------
    list[str]
        Returns the class-body lines with method bodies stripped, preserving non-function nodes line by line.

    """
    output: list[str] = []
    i = start  # 1-indexed

    methods = [n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for child in methods:
        fn_start = _effective_start(child)

        # Copy lines up to this method (class attributes, etc.).
        output.extend(lines[i - 1 : fn_start - 1])
        i = fn_start

        i = _strip_function_body(child, lines, i, output)

    # Copy remaining body lines (attributes after the last method, etc.).
    output.extend(lines[i - 1 : end])

    return output
