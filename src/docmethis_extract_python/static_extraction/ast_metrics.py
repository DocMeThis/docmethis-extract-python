# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""AST complexity metrics and I/O side-effect detection (iteration 3, Module 1)."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from docmethis_extract_python.static_extraction.models import ProjectRecord

from docmethis_extract_python.static_extraction.models import Complexity, IODetail, IOEffects, iter_functions

__all__ = ["compute_complexity", "compute_project_percentiles", "detect_io"]

# ---------------------------------------------------------------------------
# Constants - detection tables
# ---------------------------------------------------------------------------

_MCCABE_NODES: tuple[type[ast.AST], ...] = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.ExceptHandler,
    ast.match_case,
)

# Exclude bare "http": it is too generic. "http.client" is distinctive enough (stdlib HTTP client).
_PREFIXES_NETWORK: frozenset[str] = frozenset({"requests", "urllib", "socket", "aiohttp", "httpx"})
_NETWORK_TWO_SEGMENT_PREFIXES: frozenset[str] = frozenset({"http.client"})
_READ_ATTRIBUTES: frozenset[str] = frozenset({"read_text", "read_bytes"})
_WRITE_ATTRIBUTES: frozenset[str] = frozenset({"write_text", "write_bytes"})
_MODES_WRITE: frozenset[str] = frozenset({"w", "wb", "a", "ab", "x", "xb", "r+", "w+", "a+"})

# ---------------------------------------------------------------------------
# Traverse a function body
# ---------------------------------------------------------------------------


def _walk_body_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.AST]:
    """Yield all nodes in a function body without entering nested definitions.

    Traverse only the body (node.body); exclude annotations and default values. Yield nested FunctionDef, AsyncFunctionDef, and
    ClassDef nodes without exploring them.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The AST node representing the function or async function whose body is to be traversed.

    Returns
    -------
    Iterator[ast.AST]
        An iterator over the AST nodes in the function body, excluding annotations, default values, and the contents of nested
        function and class definitions.

    """
    stack: list[ast.AST] = list(node.body)
    while stack:
        current = stack.pop()
        yield current
        if not isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            stack.extend(ast.iter_child_nodes(current))


# ---------------------------------------------------------------------------
# LOC
# ---------------------------------------------------------------------------


def _compute_loc(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Return function lines of code, excluding the docstring but including comments.

    Comments are not in the AST and cannot be subtracted.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The AST node representing the function or async function definition whose lines of code are to be computed.

    Returns
    -------
    int
        The number of lines of code in the function, excluding the docstring but including comments.

    """
    total = node.end_lineno - node.lineno + 1  # type: ignore[operator]
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        ds = node.body[0]
        total -= ds.end_lineno - ds.lineno + 1  # type: ignore[operator]
    return total


# ---------------------------------------------------------------------------
# Profondeur d'imbrication
# ---------------------------------------------------------------------------


def _try_subblocks(statement: ast.Try | ast.TryStar) -> list[list[ast.stmt]]:
    """Return Try or TryStar sub-blocks (body, handlers, orelse, finally).

    Parameters
    ----------
    statement : ast.Try | ast.TryStar
        The Try or TryStar AST node whose sub-blocks are to be extracted.

    Returns
    -------
    list[list[ast.stmt]]
        A list of statement-block lists extracted from the Try or TryStar statement: the body, each exception handler body, the
        orelse block if present, and the finally block if present.

    """
    blocks: list[list[ast.stmt]] = [statement.body]
    blocks += [handler.body for handler in statement.handlers]
    if statement.orelse:
        blocks.append(statement.orelse)
    if statement.finalbody:
        blocks.append(statement.finalbody)
    return blocks


def _if_blocks(statement: ast.If) -> list[list[ast.stmt]]:
    """Return all blocks in an if/elif/else chain at the same depth.

    Flatten elif branches: all bodies (if, elif, else) are at the same level,
    so elif does not increment depth.

    Parameters
    ----------
    statement : ast.If
        The AST If node representing the if statement whose body and orelse chain are to be flattened into a list of blocks.

    Returns
    -------
    list[list[ast.stmt]]
        A list of statement-block lists for each branch in the if/elif/else chain, preserving source order and with elif branches
        flattened so all bodies are at the same depth.

    """
    blocks: list[list[ast.stmt]] = [statement.body]
    orelse = statement.orelse
    while orelse:
        if len(orelse) == 1 and isinstance(orelse[0], ast.If):
            blocks.append(orelse[0].body)
            orelse = orelse[0].orelse
        else:
            blocks.append(orelse)
            break
    return blocks


def _subblocks_of(statement: ast.stmt) -> list[list[ast.stmt]]:
    """Return substatement lists for a statement's control blocks.

    Do not enter nested function or class definitions. Flatten if/elif/else chains to avoid over-counting depth.

    Parameters
    ----------
    statement : ast.stmt
        The AST statement whose control-flow substatement blocks are to be extracted.

    Returns
    -------
    list[list[ast.stmt]]
        A list of statement-block lists for the statement's control-flow bodies. Each inner list is a sequence of AST statements
        from one body, such as an if/else branch, loop body or orelse, try handler, or match case. Returns an empty list for
        function, async function, and class definitions, and for statements without control blocks.

    """
    if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return []
    if isinstance(statement, ast.Match):
        return [case.body for case in statement.cases]
    if isinstance(statement, ast.Try | ast.TryStar):
        return _try_subblocks(statement)
    blocks: list[list[ast.stmt]] = []
    if isinstance(statement, ast.If):
        blocks = _if_blocks(statement)
    elif isinstance(statement, ast.For | ast.AsyncFor | ast.While):
        blocks.append(statement.body)  # type: ignore[union-attr]
        if statement.orelse:  # type: ignore[union-attr]
            blocks.append(statement.orelse)  # type: ignore[union-attr]
    elif isinstance(statement, ast.With | ast.AsyncWith):
        blocks.append(statement.body)
    return blocks


def _max_statement_depth(statements: list[ast.stmt], depth: int) -> int:
    """Return maximum control-block nesting depth in a list of statements.

    Counting rules (see ``_subblocks_of``):
    - ``if`` / ``for`` / ``while`` / ``with`` / ``try`` / ``match`` increment depth by one;
    - ``elif`` blocks stay at the parent ``if`` level;
    - terminal ``if`` else and ``for``/``while`` else blocks increment depth;
    - ``try`` increments for its body and each ``except``, ``else``, and ``finally``;
    - nested functions, coroutines, and classes are not explored.
    Depth therefore measures structural control-block depth, not function-call depth or cyclomatic complexity.

    Parameters
    ----------
    statements : list[ast.stmt]
        A list of AST statement nodes whose maximum control-block nesting depth is computed.
    depth : int
        Current control-block nesting depth from which to start measuring; the returned maximum is at least this value.

    Returns
    -------
    int
        Return the maximum control-block nesting depth found in the given list of AST statements, starting from the supplied
        current depth. Recursively visits sub-blocks of each statement and returns the greatest depth reached.

    """
    max_reached = depth
    for statement in statements:
        subblocks = _subblocks_of(statement)
        for block in subblocks:
            max_reached = max(max_reached, _max_statement_depth(block, depth + 1))
    return max_reached


# ---------------------------------------------------------------------------
# Cyclomatic complexity
# ---------------------------------------------------------------------------


def compute_complexity(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Complexity:
    """Return complexity metrics from a function AST node.

    Branch-counting rules:
    - each ``if`` (or ``elif``) counts as 1 (elif belongs to the same ``ast.If`` node);
    - a terminal ``else`` (not followed by an ``elif``) adds 1;
    - each ``match`` case counts as 1.
    The value is the number of explicitly named paths in the conditional structure, not the total number of execution paths (which
    includes implicit paths where no branch is taken).

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The function definition node to analyze. It must be either a regular function or an async function definition.

    Returns
    -------
    Complexity
        Returns a Complexity object containing the computed metrics: loc, cyclomatic complexity, branch count, maximum statement
        nesting depth, and total statement count.

    """
    cyclomatic = 1
    branches = 0
    statements = 0

    for n in _walk_body_function(node):
        if isinstance(n, ast.stmt):
            statements += 1
        if isinstance(n, _MCCABE_NODES):
            cyclomatic += 1
        if isinstance(n, ast.BoolOp):
            cyclomatic += len(n.values) - 1
        if isinstance(n, ast.If):
            branches += 1
            if n.orelse and not isinstance(n.orelse[0], ast.If):
                branches += 1
        if isinstance(n, ast.match_case):
            branches += 1

    max_depth = _max_statement_depth(node.body, 0)
    loc = _compute_loc(node)

    return Complexity(
        loc=loc,
        cyclomatic=cyclomatic,
        branches=branches,
        max_depth=max_depth,
        statements=statements,
    )


# ---------------------------------------------------------------------------
# Detect I/O side effects
# ---------------------------------------------------------------------------


class _Accumulator:
    """Internal I/O detection state that builds IOEffects from found patterns."""

    def __init__(self) -> None:
        """Initialize an accumulator with an empty list of I/O details and an empty set of categories."""
        self._details: list[IODetail] = []
        self._categories: set[str] = set()

    def add(self, pattern: str, line: int, category: str) -> None:
        """Add a detected I/O pattern to the accumulator.

        Parameters
        ----------
        pattern : str
            The I/O pattern string to add to the accumulator.
        line : int
            Line number in the source code where the pattern was detected.
        category : str
            The category to which the detected I/O pattern belongs.

        """
        self._categories.add(category)
        self._details.append(IODetail(pattern=pattern, line=line, category=category))

    def build(self) -> IOEffects:
        """Return IOEffects built from accumulated patterns.

        Returns
        -------
        IOEffects
            Return IOEffects built from accumulated patterns.

        """
        return IOEffects(
            reads_files="reads_files" in self._categories,
            writes_files="writes_files" in self._categories,
            network_access="network_access" in self._categories,
            system_calls="system_calls" in self._categories,
            mutates_global="mutates_global" in self._categories,
            has_output="has_output" in self._categories,
            details=self._details,
        )


def _name_call(node: ast.Call) -> str:
    """Return a call's textual name (for example, 'os.system', 'print', 'p.read_text').

    Limitation: import aliases are not resolved. ``import requests as r; r.get()``
    returns ``"r.get"`` and escapes network detection. Fixing this would require propagating the alias table from
    ``ModuleRecord.imports``, outside iteration 3.

    Parameters
    ----------
    node : ast.Call
        The AST Call node whose textual name is returned.

    Returns
    -------
    str
        The textual name of the call as a string. For a plain name call, this is the identifier; for an attribute call, this is
        the dotted path of the attribute chain (for example, 'os.system' or 'p.read_text'). Returns an empty string when the call
        target is neither a name nor an attribute expression.

    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        segments: list[str] = [func.attr]
        current: ast.expr = func.value
        while isinstance(current, ast.Attribute):
            segments.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            segments.append(current.id)
        return ".".join(reversed(segments))
    return ""


def _first_segment(identifier_name: str) -> str:
    """Return the first segment of a dotted name (for example, 'os.path.join' -> 'os').

    Parameters
    ----------
    identifier_name : str
        The dotted name whose first segment is to be extracted.

    Returns
    -------
    str
        The first segment of the dotted name, i.e., the substring before the first period.

    """
    return identifier_name.split(".", maxsplit=1)[0]


def _first_two_segments(identifier_name: str) -> str:
    """Return the first two segments of a dotted name (for example, 'http.client.HTTPConnection' -> 'http.client').

    Parameters
    ----------
    identifier_name : str
        The dotted name from which to extract the first two segments.

    Returns
    -------
    str
        The first two dot-separated segments of the input identifier name, joined by a dot.

    """
    return ".".join(identifier_name.split(".")[:2])


def _categorize_call_name(identifier_name: str) -> str | None:
    """Return the I/O category of a call name by prefix, or None when uncategorized.

    Parameters
    ----------
    identifier_name : str
        The call name whose I/O category is to be determined by prefix analysis.

    Returns
    -------
    str | None
        Returns the I/O category of a call name by prefix, or None when uncategorized.

    """
    prefix = _first_segment(identifier_name)
    if prefix in _PREFIXES_NETWORK or _first_two_segments(identifier_name) in _NETWORK_TWO_SEGMENT_PREFIXES:
        return "network_access"
    if prefix == "subprocess":
        return "system_calls"
    if prefix == "os":
        suffix = identifier_name[3:] if identifier_name.startswith("os.") else ""
        if suffix == "system" or suffix.startswith(("exec", "spawn")):
            return "system_calls"
        if suffix in {"rename", "remove", "unlink", "rmdir", "makedirs", "mkdir"}:
            return "writes_files"
    if prefix == "shutil":
        return "writes_files"
    return None


def _extract_mode_open(node: ast.Call) -> str | None:
    """Return the mode passed to open() as a string, or None when indeterminate.

    Parameters
    ----------
    node : ast.Call
        The AST Call node representing the open() call whose mode argument is to be extracted.

    Returns
    -------
    str | None
        The mode passed to open() as a string, or None when indeterminate.

    """
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):  # noqa: PLR2004
        return node.args[1].value
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _analyze_open(node: ast.Call, acc: _Accumulator) -> None:
    """Add an I/O pattern for an open() call according to its mode.

    Parameters
    ----------
    node : ast.Call
        The AST Call node representing the open() invocation to be analyzed.
    acc : _Accumulator
        Accumulator to which the detected I/O pattern for the open() call is added.

    """
    mode = _extract_mode_open(node)
    if mode is None:
        acc.add("open(...)", node.lineno, "reads_files")
        acc.add("open(...)", node.lineno, "writes_files")
    elif mode in _MODES_WRITE:
        acc.add(f"open(..., {mode!r})", node.lineno, "writes_files")
    else:
        acc.add(f"open(..., {mode!r})", node.lineno, "reads_files")


def _analyze_call(node: ast.Call, acc: _Accumulator) -> None:
    """Add I/O patterns for an ast.Call node to the accumulator.

    Parameters
    ----------
    node : ast.Call
        The AST Call node representing the function call to analyze.
    acc : _Accumulator
        The accumulator that receives the I/O pattern entries detected from the AST call node.

    """
    identifier_name = _name_call(node)
    if not identifier_name:
        return
    if identifier_name == "open":
        _analyze_open(node, acc)
        return
    if identifier_name == "print":
        acc.add("print(...)", node.lineno, "has_output")
        return
    if identifier_name.startswith("logging."):
        acc.add(f"{identifier_name}(...)", node.lineno, "has_output")
        return
    if isinstance(node.func, ast.Attribute):
        if node.func.attr in _READ_ATTRIBUTES:
            acc.add(f".{node.func.attr}()", node.lineno, "reads_files")
            return
        if node.func.attr in _WRITE_ATTRIBUTES:
            acc.add(f".{node.func.attr}()", node.lineno, "writes_files")
            return
    category = _categorize_call_name(identifier_name)
    if category is not None:
        acc.add(f"{identifier_name}(...)", node.lineno, category)


def _is_global_mutation(node: ast.Assign) -> bool:
    """Return True when an assignment targets os.environ or sys.modules.

    Detect only these two known forms. Implicit mutations of global variables (for example, ``module_level_dict["key"] = v``
    without ``global``) require scope analysis outside iteration 3.

    Parameters
    ----------
    node : ast.Assign
        The AST assignment node to inspect for a global mutation.

    Returns
    -------
    bool
        True when an assignment targets os.environ or sys.modules; otherwise False. Only these two known forms are detected.
        Implicit mutations of global variables, such as module_level_dict['key'] = v without a global declaration, require scope
        analysis outside iteration 3.

    """
    for target in node.targets:
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute):
            attr = target.value
            if isinstance(attr.value, ast.Name) and attr.value.id in {"os", "sys"} and attr.attr in {"environ", "modules"}:
                return True
    return False


def detect_io(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> IOEffects:
    """Return side effects detected in a function's direct body.

    ``mutates_global`` covers ``global x`` declarations and assignments to ``os.environ`` / ``sys.modules``. Implicit module-level
    mutations without a ``global`` declaration are not detected (scope analysis is required).

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The function or async function definition node whose direct body is scanned for I/O side effects.

    Returns
    -------
    IOEffects
        An IOEffects object summarizing the side effects detected in the function's direct body, including global variable
        declarations and mutations of os.environ or sys.modules.

    """
    acc = _Accumulator()
    for n in _walk_body_function(node):
        if isinstance(n, ast.Global):
            for name in n.names:
                acc.add(f"global {name}", n.lineno, "mutates_global")
        elif isinstance(n, ast.Call):
            _analyze_call(n, acc)
        elif isinstance(n, ast.Assign) and _is_global_mutation(n):
            acc.add("mutation os.environ / sys.modules", n.lineno, "mutates_global")
    return acc.build()


# ---------------------------------------------------------------------------
# Project-level percentiles.
# ---------------------------------------------------------------------------


def _percentile(item_value: int, values: list[int]) -> float:
    """Return the percentile rank (0.0-1.0) of a value among all values.

    Defined as (rank - 1) / (n - 1), using the mean rank for ties. Return 0.0 for an empty or one-item list.

    Parameters
    ----------
    item_value : int
        The value whose percentile rank is to be computed.
    values : list[int]
        The list of integer values against which the percentile rank of item_value is calculated.

    Returns
    -------
    float
        The percentile rank of item_value among the provided values, as a float in the range 0.0 to 1.0. It is computed as (rank -
        1) / (n - 1) using the mean rank for ties, and returns 0.0 when values is empty or contains a single item.

    """
    n = len(values)
    if n <= 1:
        return 0.0
    count_below = sum(1 for x in values if x < item_value)
    count_equal = sum(1 for x in values if x == item_value)
    rank = count_below + 1 if count_equal == 0 else count_below + (count_equal + 1) / 2
    return (rank - 1) / (n - 1)


def compute_project_percentiles(project_record: ProjectRecord) -> None:
    """Update complexity percentiles on every project FunctionRecord.

    Mutate Complexity objects in place. Ignore functions whose complexity is None. Percentiles are not cached; this function is
    called after save_cache.

    Parameters
    ----------
    project_record : ProjectRecord
        The project record whose function records are scanned and updated with complexity percentiles in place.

    """
    function_records = [f for f in iter_functions(project_record.modules) if f.complexity is not None]
    if not function_records:
        return

    all_cyclomatic = [f.complexity.cyclomatic for f in function_records]  # type: ignore[union-attr]
    all_loc = [f.complexity.loc for f in function_records]  # type: ignore[union-attr]
    all_depth = [f.complexity.max_depth for f in function_records]  # type: ignore[union-attr]

    for f in function_records:
        c = f.complexity  # type: ignore[union-attr]
        c.cyclomatic_percentile = _percentile(c.cyclomatic, all_cyclomatic)
        c.loc_percentile = _percentile(c.loc, all_loc)
        c.max_depth_percentile = _percentile(c.max_depth, all_depth)
