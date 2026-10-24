# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Conservative control-flow classification for normal function returns."""

from __future__ import annotations

import ast
from enum import Enum, auto

from docmethis_extract_python.static_extraction.models import NormalReturnStatus

__all__ = ["infer_normal_return_status"]


class _Exit(Enum):
    """Structural exit kinds observed while walking one function body.

    Attributes
    ----------
    NORMAL
        A class attribute of the enum class _Exit representing the NORMAL structural exit kind. Its value is produced by auto().
    RETURN
        A class attribute of the enum member _Exit.RETURN that represents the structural exit kind for a return statement observed
        while walking a function body.
    RAISE
        A class attribute of the _Exit enumeration representing the structural exit kind for a raise statement.
    BREAK
        BREAK is a member of the _Exit enumeration, representing the structural exit kind for a break statement observed while
        walking a function body.
    CONTINUE
        CONTINUE is an enumeration member of the _Exit class, defined with an automatically assigned value via auto(). It
        represents a structural exit kind corresponding to a continue statement observed while walking a function body.
    UNKNOWN
        An enumeration member named UNKNOWN, defined with auto(), representing an unrecognized structural exit kind.

    """

    NORMAL = auto()
    RETURN = auto()
    RAISE = auto()
    BREAK = auto()
    CONTINUE = auto()
    UNKNOWN = auto()


def infer_normal_return_status(node: ast.FunctionDef | ast.AsyncFunctionDef) -> NormalReturnStatus:
    """Classify normal completion without inferring opaque call behavior.

    Decorated functions, generators, context managers, and unsupported control statements remain unknown rather than being treated
    as non-returning.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        Accepts a function or async function definition node and returns a NormalReturnStatus classifying normal completion:
        UNKNOWN when the node has decorators or its body contains yield, POSSIBLE when its body can reach normal exit or a return,
        UNKNOWN when its body may reach an unknown exit, break, or continue, and PROVEN_ABSENT otherwise.

    Returns
    -------
    NormalReturnStatus
        Returns UNKNOWN when the function is decorated, contains a yield, or has exits marked UNKNOWN, BREAK, or CONTINUE. Returns
        POSSIBLE when the body has a normal or return exit. Otherwise returns PROVEN_ABSENT.

    """
    if node.decorator_list or _contains_yield(node.body):
        return NormalReturnStatus.UNKNOWN

    exits = _block(node.body)
    if _Exit.NORMAL in exits or _Exit.RETURN in exits:
        return NormalReturnStatus.POSSIBLE
    if _Exit.UNKNOWN in exits or _Exit.BREAK in exits or _Exit.CONTINUE in exits:
        return NormalReturnStatus.UNKNOWN
    return NormalReturnStatus.PROVEN_ABSENT


def _block(statements: list[ast.stmt]) -> set[_Exit]:
    """Return the possible exits from a sequential statement block.

    Parameters
    ----------
    statements : list[ast.stmt]
        Accepts a list of AST statements representing a sequential block. Iterates over the statements while the accumulated exit
        set still contains the normal exit; for each such statement it removes the normal exit and adds the exits produced by
        processing that statement. Stops iterating once the normal exit is no longer present. Returns the resulting set of exits.

    Returns
    -------
    set[_Exit]
        Returns the set of possible exits from the sequential block. Each statement is processed while the special normal-exit
        marker remains in the set; that marker is removed and the statement's exits are added. If processing ends before all
        statements are consumed, the remaining statements are ignored. The returned set consists of the exits accumulated from the
        processed statements.

    """
    exits = {_Exit.NORMAL}
    for statement in statements:
        if _Exit.NORMAL not in exits:
            break
        exits.remove(_Exit.NORMAL)
        exits.update(_statement(statement))
    return exits


def _statement(node: ast.stmt) -> set[_Exit]:  # noqa: C901, PLR0911 - exhaustive AST statement dispatch.
    """Return conservative exits for one statement.

    Parameters
    ----------
    node : ast.stmt
        Accepts a single positional-or-keyword argument named node, annotated as an ast.stmt. Returns a set of _Exit values
        describing conservative exits for that statement, determined by its AST node type: Return yields RETURN; Raise yields
        RAISE; Break yields BREAK; Continue yields CONTINUE; If yields the union of the exits of its body block and, when an else
        clause exists, its orelse block, otherwise NORMAL; While, For/AsyncFor, Try, and Match delegate to their respective
        helpers; TryStar, With, and AsyncWith yield UNKNOWN; Assert yields NORMAL and RAISE; Pass, Expr, Assign, AnnAssign,
        AugAssign, Delete, Import, ImportFrom, Global, Nonlocal, FunctionDef, AsyncFunctionDef, and ClassDef yield NORMAL; any
        other statement yields UNKNOWN.

    Returns
    -------
    set[_Exit]
        Returns a set containing a conservative exit classification for the given statement. Returns {_Exit.RETURN} for a return
        statement, {_Exit.RAISE} for a raise statement, {_Exit.BREAK} for a break statement, and {_Exit.CONTINUE} for a continue
        statement. For an if statement, returns the union of the exits of its body block and either its orelse block exits or
        {_Exit.NORMAL} when there is no orelse. For while, for/async-for, try, and match statements, returns the result of the
        corresponding helper. For try-star, with, or async-with statements, returns {_Exit.UNKNOWN}. For an assert statement,
        returns {_Exit.NORMAL, _Exit.RAISE}. For pass, expression, assignment, annotated assignment, augmented assignment, delete,
        import, import-from, global, nonlocal, function definition, async function definition, or class definition statements,
        returns {_Exit.NORMAL}. Otherwise returns {_Exit.UNKNOWN}.

    """
    if isinstance(node, ast.Return):
        return {_Exit.RETURN}
    if isinstance(node, ast.Raise):
        return {_Exit.RAISE}
    if isinstance(node, ast.Break):
        return {_Exit.BREAK}
    if isinstance(node, ast.Continue):
        return {_Exit.CONTINUE}
    if isinstance(node, ast.If):
        return _block(node.body) | (_block(node.orelse) if node.orelse else {_Exit.NORMAL})
    if isinstance(node, ast.While):
        return _while(node)
    if isinstance(node, ast.For | ast.AsyncFor):
        return _for(node)
    if isinstance(node, ast.Try):
        return _try(node)
    if isinstance(node, ast.Match):
        return _match(node)
    if isinstance(node, ast.TryStar | ast.With | ast.AsyncWith):
        return {_Exit.UNKNOWN}
    if isinstance(node, ast.Assert):
        return {_Exit.NORMAL, _Exit.RAISE}
    if isinstance(
        node,
        ast.Pass
        | ast.Expr
        | ast.Assign
        | ast.AnnAssign
        | ast.AugAssign
        | ast.Delete
        | ast.Import
        | ast.ImportFrom
        | ast.Global
        | ast.Nonlocal
        | ast.FunctionDef
        | ast.AsyncFunctionDef
        | ast.ClassDef,
    ):
        return {_Exit.NORMAL}
    return {_Exit.UNKNOWN}


def _while(node: ast.While) -> set[_Exit]:
    """Handle loops only when a literal test proves they execute at least once.

    Parameters
    ----------
    node : ast.While
        Accepts an ast.While node and returns a set of _Exit outcomes computed from the loop's body, test, and else block; when
        the literal test value is false, returns the else-block exits.

    Returns
    -------
    set[_Exit]
        Returns a set of exit outcomes for a while loop. When the test is a literal false, the result is the exit set of the else
        block. Otherwise it includes body exit outcomes among RETURN, RAISE, UNKNOWN, and BREAK, with BREAK replaced by NORMAL; if
        the test is not a literal true, the else-block exit outcomes are also added.

    """
    body_exits = _block(node.body)
    test_value = _constant_truth(node.test)
    else_exits = _block(node.orelse) if node.orelse else {_Exit.NORMAL}

    if test_value is False:
        return else_exits

    exits = {outcome for outcome in body_exits if outcome in {_Exit.RETURN, _Exit.RAISE, _Exit.UNKNOWN, _Exit.BREAK}}
    if _Exit.BREAK in exits:
        exits.remove(_Exit.BREAK)
        exits.add(_Exit.NORMAL)
    if test_value is not True:
        exits.update(else_exits)
    return exits


def _for(node: ast.For | ast.AsyncFor) -> set[_Exit]:
    """Include the zero-iteration/exhaustion path for an unknown iterable.

    Parameters
    ----------
    node : ast.For | ast.AsyncFor
        Accepts an ast.For or ast.AsyncFor node. Computes exits from the loop body and, if present, the orelse block; when orelse
        is empty, starts with a normal exit. Adds RETURN, RAISE, and UNKNOWN outcomes found in the body, and adds a normal exit if
        the body contains a BREAK. Returns the resulting set of exits.

    Returns
    -------
    set[_Exit]
        Returns a set of _Exit values describing the outcomes of a for or async for statement: the exits of the else block when
        present, otherwise NORMAL; plus RETURN, RAISE, or UNKNOWN outcomes from the body; plus NORMAL when the body contains
        BREAK.

    """
    body_exits = _block(node.body)
    exits = _block(node.orelse) if node.orelse else {_Exit.NORMAL}
    exits.update(outcome for outcome in body_exits if outcome in {_Exit.RETURN, _Exit.RAISE, _Exit.UNKNOWN})
    if _Exit.BREAK in body_exits:
        exits.add(_Exit.NORMAL)
    return exits


def _try(node: ast.Try) -> set[_Exit]:
    """Keep implicit exceptions reachable through handlers and finally blocks.

    Parameters
    ----------
    node : ast.Try
        Accepts an ast.Try node and returns the set of exit paths reachable from it, treating the try body as possibly raising and
        including exits from the orelse block when the body can complete normally, from each handler's body, and from the
        finalbody. When a finally block exists, each original exit is combined with each final exit, keeping the original exit if
        the final exit is normal and otherwise using the final exit.

    Returns
    -------
    set[_Exit]
        Returns a set of exit outcomes for the try statement. It combines the exits of the body with a raise exit, removes normal
        exits, adds the orelse block exits (or normal if the orelse block is absent) when the body can exit normally, and adds the
        exits of each handler body. If a finalbody exists, each existing exit is paired with each final-block exit, keeping the
        original exit when the final exit is normal and using the final exit otherwise.

    """
    # ponytail: over-approximate implicit exceptions; refine their reachability if false negatives matter.
    body_exits = _block(node.body) | {_Exit.RAISE}
    exits = body_exits - {_Exit.NORMAL}
    if _Exit.NORMAL in body_exits:
        exits.update(_block(node.orelse) if node.orelse else {_Exit.NORMAL})
    if node.handlers:
        for handler in node.handlers:
            exits.update(_block(handler.body))

    if node.finalbody:
        final_exits = _block(node.finalbody)
        exits = {original if final_exit is _Exit.NORMAL else final_exit for original in exits for final_exit in final_exits}
    return exits


def _match(node: ast.Match) -> set[_Exit]:
    """Include fallthrough unless an unguarded irrefutable case is present.

    Parameters
    ----------
    node : ast.Match
        Accepts an AST Match node and returns a set of exit outcomes. For each case in the match, adds the exits produced by
        processing the case body. Tracks whether any case is both unguarded and irrefutable. If no such exhaustive case exists,
        adds the normal exit outcome to the result. Returns the set of exits.

    Returns
    -------
    set[_Exit]
        Returns a set of exits collected from every case body in the match statement; if no case is both unguarded and
        irrefutable, the set also includes the normal fallthrough exit.

    """
    exits: set[_Exit] = set()
    exhaustive = False
    for case in node.cases:
        exits.update(_block(case.body))
        exhaustive = exhaustive or (case.guard is None and _pattern_is_irrefutable(case.pattern))
    if not exhaustive:
        exits.add(_Exit.NORMAL)
    return exits


def _constant_truth(node: ast.expr) -> bool | None:
    """Return literal truthiness only for immutable constants.

    Parameters
    ----------
    node : ast.expr
        Returns the boolean truthiness of the constant's value when node is an ast.Constant whose value is a bool, int, float,
        str, bytes, or None; otherwise returns None.

    Returns
    -------
    bool | None
        Returns a boolean indicating the truthiness of the constant's value when the node is an AST Constant whose value is a
        bool, int, float, str, bytes, or None; returns None otherwise.

    """
    if not isinstance(node, ast.Constant) or type(node.value) not in {bool, int, float, str, bytes, type(None)}:
        return None
    return bool(node.value)


class _YieldDetector(ast.NodeVisitor):
    """Find yields in this function without entering nested scopes.

    Attributes
    ----------
    found
        A class-level boolean attribute, initialized to False, on the _YieldDetector class that extends ast.NodeVisitor.
    """

    found: bool = False

    def visit_Yield(self, _node: ast.Yield) -> None:
        """Marks the detector as having found a yield expression when visiting an ast.Yield node.

        Parameters
        ----------
        _node : ast.Yield
            Accepts an ast.Yield node as a positional or keyword argument. On invocation, sets the detector's found attribute to
            True.

        """
        self.found = True

    def visit_YieldFrom(self, _node: ast.YieldFrom) -> None:
        """Sets the found flag to True when a yield-from expression is visited by the AST node visitor.

        Parameters
        ----------
        _node : ast.YieldFrom
            Accepts a positional-or-keyword argument named _node annotated as ast.YieldFrom. The method sets the instance
            attribute found to True.

        """
        self.found = True

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Visits a function definition's decorators, arguments, and return annotation without traversing its body.

        Parameters
        ----------
        node : ast.FunctionDef | ast.AsyncFunctionDef
            Accepts a function definition node (synchronous or asynchronous). Visits each decorator in the node's decorator list,
            then visits the node's arguments, and if the node has a return annotation, visits that annotation as well. Returns no
            value.

        """
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handles async function definitions by delegating to the synchronous FunctionDef visitor.

        Parameters
        ----------
        node : ast.AsyncFunctionDef
            Accepts an AST async function definition node and processes it by delegating to visit_FunctionDef with that node.

        """
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Visits the arguments node of a lambda expression to continue detection.

        Parameters
        ----------
        node : ast.Lambda
            Accepts a positional-or-keyword argument named node, annotated as an ast.Lambda. The method visits the lambda's args
            node.

        """
        self.visit(node.args)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visits the decorators, base classes, and keywords of a class definition node.

        Parameters
        ----------
        node : ast.ClassDef
            Accepts an ast.ClassDef node and visits each of its decorators, bases, and keywords.

        """
        for expression in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expression)


def _contains_yield(statements: list[ast.stmt]) -> bool:
    """Return whether this function body contains a yield in its own scope.

    Parameters
    ----------
    statements : list[ast.stmt]
        Takes a list of AST statement nodes and returns a boolean indicating whether those statements contain a yield in their own
        scope, excluding nested scopes.

    Returns
    -------
    bool
        Returns the value of the detector's found attribute after visiting each supplied statement with a new _YieldDetector
        instance, producing a bool.

    """
    detector = _YieldDetector()
    for statement in statements:
        detector.visit(statement)
    return detector.found


def _pattern_is_irrefutable(pattern: ast.pattern) -> bool:
    """Recognize the simple irrefutable match patterns used to prove exhaustiveness.

    Parameters
    ----------
    pattern : ast.pattern
        Accepts an AST pattern node and determines whether it is an irrefutable match pattern. Returns true for a MatchAs node
        with no sub-pattern, otherwise recurses into a MatchAs sub-pattern; returns true for a MatchStar node; returns true for a
        MatchOr node if any of its child patterns is irrefutable; returns false for any other pattern.

    Returns
    -------
    bool
        Returns True when the pattern is an ast.MatchStar, or an ast.MatchAs whose nested pattern is None or itself irrefutable,
        or an ast.MatchOr with at least one irrefutable child; returns False for any other pattern.

    """
    if isinstance(pattern, ast.MatchAs):
        return pattern.pattern is None or _pattern_is_irrefutable(pattern.pattern)
    if isinstance(pattern, ast.MatchStar):
        return True
    if isinstance(pattern, ast.MatchOr):
        return any(_pattern_is_irrefutable(child) for child in pattern.patterns)
    return False
