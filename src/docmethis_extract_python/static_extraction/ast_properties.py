# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Detect observable function properties through AST analysis (iteration 7, Module 1).

Detect three categories (DEC-007):
- **preconditions**: guards (if...raise) and asserts in the body preamble.
- **postconditions**: asserts and if...raise in the final five statements.
- **invariants**: asserts outside the preamble/postconditions (including loops).

For ``if expr: raise T``, set ``related_exception = "T"``.
"""

from __future__ import annotations

import ast

from docmethis_extract_python.static_extraction.models import Confidence, ObservableProperty

__all__ = ["extract_observable_properties"]

# Number of fallback statements for guards outside the strict preamble.
_FALLBACK_GUARD_COUNT = 10
# Number of final statements inspected for postconditions.
_N_POSTCONDITIONS = 5


# ---------------------------------------------------------------------------
# Helpers - statement classification
# ---------------------------------------------------------------------------


def _raise_in_if(if_node: ast.If) -> str | None:
    """Return the exception type when an if contains one raise, otherwise None.

    Parameters
    ----------
    if_node : ast.If
        The ast.If node to inspect for a single raise statement.

    Returns
    -------
    str | None
        The exception type raised in the if block as a string, or None if the block does not contain exactly one raise statement.

    """
    useful_body = [statement for statement in if_node.body if not isinstance(statement, ast.Pass)]
    if len(useful_body) == 1 and isinstance(useful_body[0], ast.Raise):
        raise_node = useful_body[0]
        if raise_node.exc is not None:
            exc = raise_node.exc
            node_type = exc.func if isinstance(exc, ast.Call) else exc
            if isinstance(node_type, ast.Name):
                return node_type.id
            if isinstance(node_type, ast.Attribute):
                return ast.unparse(node_type)
    return None


def _is_guard(statement: ast.stmt) -> bool:
    """Return True when a statement is a guard (if...raise or assert).

    Parameters
    ----------
    statement : ast.stmt
        The AST statement node to inspect for guard behavior.

    Returns
    -------
    bool
        True if the statement is a guard (an assert or an if statement that raises); False otherwise.

    """
    if isinstance(statement, ast.Assert):
        return True
    if isinstance(statement, ast.If):
        return _raise_in_if(statement) is not None
    return False


def _referenced_names(statement: ast.stmt) -> set[str]:
    """Return names referenced in a guard condition.

    Parameters
    ----------
    statement : ast.stmt
        The AST statement to inspect for names referenced in its guard condition.

    Returns
    -------
    set[str]
        Returns a set of strings containing the names referenced in the guard condition of the given statement. For assert and if
        statements, this is the set of identifier names appearing in the test expression; for any other statement, an empty set is
        returned.

    """
    if isinstance(statement, ast.Assert):
        return {node.id for node in ast.walk(statement.test) if isinstance(node, ast.Name)}
    if isinstance(statement, ast.If):
        return {node.id for node in ast.walk(statement.test) if isinstance(node, ast.Name)}
    return set()


def _is_pure_assignment(statement: ast.stmt) -> bool:
    """Return True when a statement is a simple or annotated assignment.

    Parameters
    ----------
    statement : ast.stmt
        The AST statement node to check.

    Returns
    -------
    bool
        True if the given statement is an ast.Assign, ast.AnnAssign, or ast.AugAssign node; False otherwise.

    """
    return isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign)


def _is_preamble_statement(statement: ast.stmt) -> bool:
    """Return True when a statement can be part of the preamble (guard or assignment).

    Parameters
    ----------
    statement : ast.stmt
        The AST statement node to inspect for preamble eligibility.

    Returns
    -------
    bool
        Indicate whether the given statement qualifies as a preamble statement, i.e., a guard or a pure assignment.

    """
    return _is_guard(statement) or _is_pure_assignment(statement)


def _body_without_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    """Return the function body without its initial docstring.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The AST node representing the function or async function definition from which to extract the body without the initial
        docstring.

    Returns
    -------
    list[ast.stmt]
        A list of AST statements comprising the function body, excluding the leading docstring expression when one exists.

    """
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return list(body)


# ---------------------------------------------------------------------------
# Traversal - one walker, excluding nested functions/classes
# ---------------------------------------------------------------------------


def _walk_without_nested_scopes(statements: list[ast.stmt]) -> list[ast.stmt]:
    """Yield all statements recursively without entering nested definitions.

    Parameters
    ----------
    statements : list[ast.stmt]
        The initial list of AST statement nodes from which the recursive traversal starts.

    Returns
    -------
    list[ast.stmt]
        Recursively collects all statement nodes from the provided list, descending into child statements while skipping nested
        scopes introduced by function, async function, and class definitions, and returns the accumulated statements as a list.

    """
    result: list[ast.stmt] = []
    stack: list[ast.stmt] = list(statements)
    while stack:
        statement = stack.pop()
        result.append(statement)
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        stack.extend(child for child in ast.iter_child_nodes(statement) if isinstance(child, ast.stmt))
    return result


# ---------------------------------------------------------------------------
# Accumulator - centralized ObservableProperty construction
# ---------------------------------------------------------------------------


class _Accumulator:
    """Collect ObservableProperties from heuristic analysis of a function body.

    Centralize ObservableProperty construction and invariant deduplication by line.
    """

    def __init__(self) -> None:
        """Initialize an accumulator with an empty property list and sets for tracking seen and precondition lines."""
        self._properties: list[ObservableProperty] = []
        self._seen_lines: set[int] = set()  # invariant deduplication only
        self._precondition_lines: set[int] = set()  # exclude pre/post duplicates

    def add_guard(self, category: str, statement: ast.stmt, confidence: Confidence) -> None:
        """Add an assert or if-raise as a property with the given category and confidence.

        Parameters
        ----------
        category : str
            The category label to assign to the added property, for example 'precondition'.
        statement : ast.stmt
            The AST statement node representing the guard to add; expected to be an assert or an if statement whose body raises an
            exception.
        confidence : Confidence
            The confidence level associated with the extracted property.

        """
        if isinstance(statement, ast.Assert):
            self._properties.append(
                ObservableProperty(
                    category=category,
                    raw_expression=ast.unparse(statement.test),
                    line=statement.lineno,
                    confidence=confidence,
                )
            )
            if category == "precondition":
                self._precondition_lines.add(statement.lineno)
        elif isinstance(statement, ast.If):
            exc_type = _raise_in_if(statement)
            if exc_type is not None:
                self._properties.append(
                    ObservableProperty(
                        category=category,
                        raw_expression=ast.unparse(statement.test),
                        line=statement.lineno,
                        confidence=confidence,
                        related_exception=exc_type,
                    )
                )
                if category == "precondition":
                    self._precondition_lines.add(statement.lineno)

    def add_postcondition(self, statement: ast.stmt) -> None:
        """Add a postcondition: assert -> EXPLICIT, if-raise -> INFERRED_LOW.

        Parameters
        ----------
        statement : ast.stmt
            Statement to process as a postcondition; assert statements are treated as explicit postconditions, and if statements
            as inferred-low postconditions.

        """
        if statement.lineno in self._precondition_lines:
            return
        if isinstance(statement, ast.Assert):
            self.add_guard("postcondition", statement, Confidence.EXPLICIT)
        elif isinstance(statement, ast.If):
            self.add_guard("postcondition", statement, Confidence.INFERRED_LOW)

    def add_invariant(self, statement: ast.Assert) -> None:
        """Add an assert as an invariant when its line has not been recorded.

        Parameters
        ----------
        statement : ast.Assert
            The AST Assert node representing the assert statement to add as an invariant.

        """
        if statement.lineno not in self._seen_lines:
            self._seen_lines.add(statement.lineno)
            self.add_guard("invariant", statement, Confidence.EXPLICIT)

    def build(self) -> list[ObservableProperty]:
        """Return accumulated properties.

        Returns
        -------
        list[ObservableProperty]
            The accumulated properties as a list of ObservableProperty instances.

        """
        return list(self._properties)


# ---------------------------------------------------------------------------
# Analysis phases
# ---------------------------------------------------------------------------


def _analyze_preconditions(body: list[ast.stmt], acc: _Accumulator, param_names: frozenset[str]) -> int:
    """Analyze the preamble and fallback guards. Return the preamble end.

    Parameters
    ----------
    body : list[ast.stmt]
        The list of AST statements to analyze for preconditions.
    acc : _Accumulator
        Accumulator to which detected precondition guards are added.
    param_names : frozenset[str]
        Names of the function parameters to consider when identifying fallback guards that reference them.

    Returns
    -------
    int
        An integer representing the index in the body where the preamble ends, equal to the number of consecutive leading preamble
        statements before the first non-preamble statement.

    """
    preamble_end = 0
    for stmt in body:
        if _is_preamble_statement(stmt):
            preamble_end += 1
        else:
            break

    for stmt in body[:preamble_end]:
        acc.add_guard("precondition", stmt, Confidence.EXPLICIT)

    end_fallback = min(preamble_end + _FALLBACK_GUARD_COUNT, len(body))
    for stmt in body[preamble_end:end_fallback]:
        if _is_guard(stmt) and _referenced_names(stmt) & param_names:
            acc.add_guard("precondition", stmt, Confidence.INFERRED_LOW)

    return preamble_end


def _get_start_pc(preamble_end: int, body: list[ast.stmt]) -> int:
    """Returns the larger of preamble_end and the body length minus the number of postconditions.

    Parameters
    ----------
    preamble_end : int
        Returns the starting program counter after the preamble and body. It computes the maximum of the preamble_end offset and
        the length of the body minus the number of postconditions.
    body : list[ast.stmt]
        Returns the starting program counter (PC) as the maximum of the preamble end and the length of the body minus the number
        of postconditions.

    Returns
    -------
    int
        Returns the starting program counter, which is the greater of preamble_end and the length of the body minus the number of
        postconditions.

    """
    return max(preamble_end, len(body) - _N_POSTCONDITIONS)


def _analyze_postconditions(body: list[ast.stmt], preamble_end: int, acc: _Accumulator) -> None:
    """Analyze final statements for postconditions.

    Parameters
    ----------
    body : list[ast.stmt]
        The list of AST statements comprising the function body from which final statements are extracted as postcondition
        candidates.
    preamble_end : int
        Integer index indicating the end of the preamble within the body statement list, used to determine the starting point for
        analyzing postconditions.
    acc : _Accumulator
        An accumulator object that collects and stores postconditions identified from the analyzed statements.

    """
    start_pc = _get_start_pc(preamble_end, body)
    for stmt in body[start_pc:]:
        acc.add_postcondition(stmt)


def _analyze_invariants(body: list[ast.stmt], preamble_end: int, acc: _Accumulator) -> None:
    """Analyze asserts in the middle section and loop bodies.

    Parameters
    ----------
    body : list[ast.stmt]
        The list of AST statement nodes that make up the function body to analyze.
    preamble_end : int
        Index in the body list marking the end of the preamble; analysis of invariant assertions starts from this position onward.
    acc : _Accumulator
        The accumulator that receives the invariants discovered from asserts in the analyzed sections.

    """
    start_pc = _get_start_pc(preamble_end, body)

    # Direct section between the preamble and postconditions.
    for statement in _walk_without_nested_scopes(body[preamble_end:start_pc]):
        if isinstance(statement, ast.Assert):
            acc.add_invariant(statement)

    # for/while bodies in body[preamble_end:] (including postconditions).
    for statement in body[preamble_end:]:
        if isinstance(statement, ast.For | ast.AsyncFor | ast.While):
            for loop_statement in _walk_without_nested_scopes(list(statement.body)):
                if isinstance(loop_statement, ast.Assert):
                    acc.add_invariant(loop_statement)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_observable_properties(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ObservableProperty]:
    """Extract observable properties from a function through an AST heuristic.

    Return ``ObservableProperty`` entries covering:
    - preconditions (preamble: explicit guards and asserts or N=10 fallback)
    - postconditions (five final direct statements)
    - invariants (asserts in the middle body, including loops)

    A function without an assert or raise returns an empty list. Asserts in nested functions are ignored.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        AST node representing the function or async function definition whose observable properties are extracted.

    Returns
    -------
    list[ObservableProperty]
        A list of ObservableProperty entries extracted from the function, covering preconditions, postconditions, and invariants.
        Returns an empty list if the function has no assert or raise; asserts in nested functions are ignored.

    """
    body = _body_without_docstring(node)
    if not body:
        return []

    args = node.args
    param_names = frozenset(
        arg.arg
        for arg in args.posonlyargs
        + args.args
        + args.kwonlyargs
        + ([args.vararg] if args.vararg else [])
        + ([args.kwarg] if args.kwarg else [])
    )

    acc = _Accumulator()
    preamble_end = _analyze_preconditions(body, acc, param_names)
    _analyze_postconditions(body, preamble_end, acc)
    _analyze_invariants(body, preamble_end, acc)
    return acc.build()
