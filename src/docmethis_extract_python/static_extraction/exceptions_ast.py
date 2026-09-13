# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Statically extract raised exceptions and except blocks (iteration 7, Module 1).

Walk a function body to extract:
- explicit ``raise`` statements (type, message, reraise, chained, raised condition)
- ``except`` blocks (caught types, reraised, swallowed)

Fundamental rule (DEC-007): emit only explicit ``raise`` statements.
Do not speculate about possible exceptions.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from docmethis_extract_python.static_extraction.models import ExceptBlock, ExceptionRecord, RaisedCondition

__all__ = ["extract_exceptions"]


# ---------------------------------------------------------------------------
# Helpers - extract an exception's type and message
# ---------------------------------------------------------------------------


def _exception_type(exc: ast.expr | None) -> str:
    """Return the textual name of a raised exception, or '<unknown>' when unknown.

    Parameters
    ----------
    exc : ast.expr | None
        The AST expression node representing the raised exception, or None if the exception type is unknown.

    Returns
    -------
    str
        The textual name of the raised exception, or '<unknown>' when the exception expression is None or cannot be determined.

    """
    if exc is None:
        return "<unknown>"
    if isinstance(exc, ast.Name):
        return exc.id
    if isinstance(exc, ast.Attribute):
        return ast.unparse(exc)
    if isinstance(exc, ast.Call):
        return _exception_type(exc.func)
    return ast.unparse(exc)


def _exception_message(exc: ast.expr | None) -> str | None:
    """Return a complete literal exception message, otherwise None.

    Parameters
    ----------
    exc : ast.expr | None
        The AST expression node representing the exception call to inspect; may be None.

    Returns
    -------
    str | None
        The exception message as a string when the exception expression is a call with exactly one constant string argument and no
        keywords; otherwise, None.

    """
    if not isinstance(exc, ast.Call) or len(exc.args) != 1 or exc.keywords:
        return None
    first_arg = exc.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        return first_arg.value
    return None


# ---------------------------------------------------------------------------
# Traversal context for resolving a reraise type
# ---------------------------------------------------------------------------


def _enclosing_handler_types(handler: ast.ExceptHandler) -> list[str]:
    """Return the types caught by an except handler.

    Parameters
    ----------
    handler : ast.ExceptHandler
        The ast.ExceptHandler node from which to extract the caught exception types.

    Returns
    -------
    list[str]
        A list of exception type names caught by the handler.

    """
    if handler.type is None:
        return []
    if isinstance(handler.type, ast.Tuple):
        return [_exception_type(elt) for elt in handler.type.elts]
    return [_exception_type(handler.type)]


_CONDITION_BARRIER_TYPES: tuple[type[ast.AST], ...] = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.TryStar,
    ast.With,
    ast.AsyncWith,
    ast.Match,
)


def _contains_unambiguous_raise(statements: list[ast.stmt]) -> bool:
    """Return whether statements contain a direct raise or a nested if-raise.

    Parameters
    ----------
    statements : list[ast.stmt]
        Statements in one branch of an ``if``.

    Returns
    -------
    bool
        True when the branch contains a direct ``raise`` or another ``if``
        whose branch contains one. Loops, exception handlers, and other
        control-flow constructs are deliberately not inspected.

    """
    for statement in statements:
        if isinstance(statement, ast.Raise):
            return True
        if isinstance(statement, ast.If) and (
            _contains_unambiguous_raise(statement.body) or _contains_unambiguous_raise(statement.orelse)
        ):
            return True
    return False


def _combine_condition(
    parent: RaisedCondition | None,
    expression: str,
    line: int,
) -> RaisedCondition:
    """Build a condition for one deterministic branch, preserving its parent guard."""
    if parent is None:
        return RaisedCondition(expression=expression, line=line)
    return RaisedCondition(expression=f"({parent.expression}) and ({expression})", line=line)


def _branch_condition(
    parent: RaisedCondition | None,
    statements: list[ast.stmt],
    expression: str,
    line: int,
    *,
    conditions_allowed: bool,
) -> RaisedCondition | None:
    """Return a branch condition only when that branch contains an explicit raise path."""
    if not conditions_allowed or not _contains_unambiguous_raise(statements):
        return parent
    return _combine_condition(parent, expression, line)


def _exception_record(
    raise_node: ast.Raise,
    handler: ast.ExceptHandler | None,
    condition: RaisedCondition | None,
) -> ExceptionRecord:
    """Build one local exception record from a raise statement."""
    if raise_node.exc is None:
        types = _enclosing_handler_types(handler) if handler is not None else []
        exc_type = types[0] if len(types) == 1 else "<unknown>"
        return ExceptionRecord(
            exception_type=exc_type,
            message=None,
            raise_line=raise_node.lineno,
            is_reraise=True,
            chained_from=False,
            condition=condition,
        )
    return ExceptionRecord(
        exception_type=_exception_type(raise_node.exc),
        message=_exception_message(raise_node.exc),
        raise_line=raise_node.lineno,
        is_reraise=False,
        chained_from=raise_node.cause is not None,
        condition=condition,
    )


# ---------------------------------------------------------------------------
# Main traversal with raised-condition attachment
# ---------------------------------------------------------------------------


def _walk_with_conditions(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ExceptionRecord]:
    """Yield ExceptionRecords and attach conditions to deterministic if branches.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The AST node representing the function or async function definition whose body is traversed to collect exception records
        and propagate guard conditions.

    Returns
    -------
    Iterator[ExceptionRecord]
        Returns an iterator of ExceptionRecord objects for raise statements encountered while walking the function's AST, with
        condition information attached only to raises on deterministic ``if`` paths.

    """
    stack: list[tuple[ast.AST, ast.ExceptHandler | None, RaisedCondition | None, bool]] = [
        (statement, None, None, True) for statement in reversed(node.body)
    ]
    while stack:
        current, handler, parent_condition, conditions_allowed = stack.pop()

        if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue

        if isinstance(current, ast.Raise):
            yield _exception_record(current, handler, parent_condition)
            continue

        if isinstance(current, ast.If):
            body_condition = _branch_condition(
                parent_condition,
                current.body,
                ast.unparse(current.test),
                current.lineno,
                conditions_allowed=conditions_allowed,
            )
            else_condition = _branch_condition(
                parent_condition,
                current.orelse,
                f"not ({ast.unparse(current.test)})",
                current.lineno,
                conditions_allowed=conditions_allowed,
            )
            stack.extend((child, handler, else_condition, conditions_allowed) for child in reversed(current.orelse))
            stack.extend((child, handler, body_condition, conditions_allowed) for child in reversed(current.body))
            continue

        if isinstance(current, _CONDITION_BARRIER_TYPES):
            stack.extend((child, handler, None, False) for child in reversed(list(ast.iter_child_nodes(current))))
            continue

        if isinstance(current, ast.ExceptHandler):
            stack.extend((child, current, None, False) for child in reversed(current.body))
            continue

        stack.extend(
            (child, handler, parent_condition, conditions_allowed) for child in reversed(list(ast.iter_child_nodes(current)))
        )


# ---------------------------------------------------------------------------
# Extract except blocks
# ---------------------------------------------------------------------------


def _is_trivial(statement: ast.stmt) -> bool:
    """Return True when a statement is pass or ... (an effectively empty body).

    Parameters
    ----------
    statement : ast.stmt
        The AST statement node to check for triviality.

    Returns
    -------
    bool
        True if the statement is a pass statement or an ellipsis expression, indicating an effectively empty body; False
        otherwise.

    """
    return isinstance(statement, ast.Pass) or (
        isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and statement.value.value is ...
    )


def _try_body_range(try_node: ast.Try) -> tuple[int, int]:
    """Return the line range of the enclosing try body (start, end).

    Parameters
    ----------
    try_node : ast.Try
        The AST node representing the try statement whose body's line range is to be computed.

    Returns
    -------
    tuple[int, int]
        A tuple of two integers containing the start and end line numbers of the enclosing try body.

    """
    end = max((stmt.end_lineno or stmt.lineno for stmt in try_node.body), default=try_node.lineno)
    return try_node.lineno, end


def _build_except_block(handler: ast.ExceptHandler, try_range: tuple[int, int]) -> ExceptBlock:
    """Build an ExceptBlock with the enclosing try range.

    Parameters
    ----------
    handler : ast.ExceptHandler
        The AST node representing the except handler block to be analyzed and converted into an ExceptBlock.
    try_range : tuple[int, int]
        The start and end line numbers of the enclosing try statement, used to populate the try_start and try_end fields of the
        resulting ExceptBlock.

    Returns
    -------
    ExceptBlock
        An ExceptBlock containing the caught exception types, whether the handler re-raises or swallows the exception, and the
        handler line and enclosing try range bounds.

    """
    caught_types = _enclosing_handler_types(handler)

    # Detect reraise: a bare `raise` in the handler, without entering nested
    # scopes (functions, classes) or nested handlers (except).
    reraise_stack: list[ast.AST] = list(handler.body)
    reraised = False
    while reraise_stack:
        statement = reraise_stack.pop()
        if isinstance(statement, ast.Raise) and statement.exc is None:
            reraised = True
            break
        if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.ExceptHandler):
            reraise_stack.extend(ast.iter_child_nodes(statement))

    # Swallowed: body reduced to pass / ... without a re-raise.
    swallowed = not reraised and all(_is_trivial(statement) for statement in handler.body)

    return ExceptBlock(
        caught_types=caught_types,
        is_reraised=reraised,
        is_swallowed=swallowed,
        line=handler.lineno,
        try_start=try_range[0],
        try_end=try_range[1],
    )


def _extract_except_blocks(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ExceptBlock]:
    """Extract all except blocks from a function body without entering nested scopes.

    Each handler is attached to the line range of its enclosing ``try`` body (``try_start``/``try_end``) to correlate raises with
    their handlers.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The AST node representing the function or async function whose body is inspected for except blocks.

    Returns
    -------
    list[ExceptBlock]
        A list of ExceptBlock objects representing the except handlers extracted from the function body, excluding handlers inside
        nested function or class scopes, with each block associated to the line range of its enclosing try body.

    """
    blocks: list[ExceptBlock] = []
    stack: list[tuple[ast.AST, tuple[int, int] | None]] = [(stmt, None) for stmt in node.body]
    while stack:
        current, _range = stack.pop()

        if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue

        if isinstance(current, ast.Try):
            inner_range = _try_body_range(current)
            blocks.extend(_build_except_block(handler, inner_range) for handler in current.handlers)
            stack.extend((stmt, inner_range) for stmt in current.body)
            for handler in current.handlers:
                stack.extend((stmt, inner_range) for stmt in handler.body)
            stack.extend((stmt, inner_range) for stmt in current.orelse)
            continue

        stack.extend((child, _range) for child in ast.iter_child_nodes(current))
    return blocks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_exceptions(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[list[ExceptionRecord], list[ExceptBlock]]:
    """Extract raised exceptions and except blocks from a function body.

    Return a tuple ``(exceptions, except_blocks)``:
    - ``exceptions``: explicit raises with type, optional message, and raised condition.
    - ``except_blocks``: handlers with caught types, reraise, and swallow state.

    Emit only explicit ``raise`` statements; do not speculate (DEC-007). Raises in nested functions are ignored.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The AST node representing the function or async function whose body is inspected for raised exceptions and except blocks.

    Returns
    -------
    tuple[list[ExceptionRecord], list[ExceptBlock]]
        A tuple containing two lists: the first holds ExceptionRecord objects for each explicit raise found in the function body,
        and the second holds ExceptBlock objects for each except handler, including reraise and swallow state.

    """
    exceptions = list(_walk_with_conditions(node))
    blocks = _extract_except_blocks(node)
    return exceptions, blocks
