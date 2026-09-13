# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Normalize type annotations to PEP 585/604 form.

Transform legacy forms (``Optional[X]``, ``List[X]``, ``Union[X, Y]``) into
Python 3.10+ forms (``X | None``, ``list[X]``, ``X | Y``).
Use only the standard ``ast`` module.
"""

from __future__ import annotations

import ast

from docmethis_extract_python.static_extraction.models import MISSING_VALUE

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

_PEP585_NAMES: dict[str, str] = {
    "List": "list",
    "typing.List": "list",
    "Dict": "dict",
    "typing.Dict": "dict",
    "Tuple": "tuple",
    "typing.Tuple": "tuple",
    "Set": "set",
    "typing.Set": "set",
    "FrozenSet": "frozenset",
    "typing.FrozenSet": "frozenset",
    "Type": "type",
    "typing.Type": "type",
}

_OPTIONAL_NAMES: frozenset[str] = frozenset({"Optional", "typing.Optional"})
_UNION_NAMES: frozenset[str] = frozenset({"Union", "typing.Union"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_type(raw: str) -> str:
    """Return the annotation normalized to PEP 585/604 form.

    Return ``raw`` unchanged when:

    - ``raw == MISSING_VALUE``
    - AST parsing fails (non-standard or invalid annotation)

    Parameters
    ----------
    raw : str
        The type annotation string to normalize.

    Returns
    -------
    str
        Returns the annotation string normalized to PEP 585/604 form. If raw is the missing-value sentinel or cannot be parsed as
        an AST expression, the original raw string is returned unchanged.

    """
    if raw == MISSING_VALUE:
        return MISSING_VALUE
    try:
        tree = ast.parse(raw, mode="eval")
    except SyntaxError:
        return raw
    return ast.unparse(_normalize_node(tree.body))


# ---------------------------------------------------------------------------
# Recursive transformation (private)
# ---------------------------------------------------------------------------


def _extract_name_type(syntax_node: ast.expr) -> str:
    """Extract the qualified name from a type node.

    Examples: ``Name('Optional')`` -> ``'Optional'``,
    ``Attribute(Name('typing'), 'List')`` -> ``'typing.List'``.

    Parameters
    ----------
    syntax_node : ast.expr
        The syntax node representing the type expression from which to extract the qualified name.

    Returns
    -------
    str
        Returns the qualified name extracted from the type node as a string, or an empty string if the node is not a recognized
        name or attribute form.

    """
    if isinstance(syntax_node, ast.Name):
        return syntax_node.id
    if isinstance(syntax_node, ast.Attribute) and isinstance(syntax_node.value, ast.Name):
        return f"{syntax_node.value.id}.{syntax_node.attr}"
    return ""


def _normalize_subscript(identifier_name: str, slice_: ast.expr) -> ast.expr:
    """Apply PEP 585/604 rules to an identified ``Subscript`` node.

    Parameters
    ----------
    identifier_name : str
        The name of the type identifier (e.g., 'Optional', 'Union', 'List', or a custom generic) that determines which PEP 585/604
        normalization rule is applied to the subscript node.
    slice_ : ast.expr
        The AST expression representing the subscript slice of the Subscript node to be normalized.

    Returns
    -------
    ast.expr
        The normalized AST expression for the subscript after applying PEP 585/604 rules, or None if the identifier is not a
        recognized optional, union, or PEP 585 generic.

    """
    if identifier_name in _OPTIONAL_NAMES:
        # Optional[X] -> X | None
        return ast.BinOp(left=_normalize_node(slice_), op=ast.BitOr(), right=ast.Constant(value=None))

    if identifier_name in _UNION_NAMES:
        # Union[X, Y, Z] -> X | Y | Z (left-associative)
        args = [_normalize_node(e) for e in slice_.elts] if isinstance(slice_, ast.Tuple) else [_normalize_node(slice_)]
        outcome: ast.expr = args[0]
        for arg in args[1:]:
            outcome = ast.BinOp(left=outcome, op=ast.BitOr(), right=arg)
        return outcome

    if identifier_name in _PEP585_NAMES:
        # List[X] -> list[X] / Dict[K, V] -> dict[K, V] / ...
        return ast.Subscript(
            value=ast.Name(id=_PEP585_NAMES[identifier_name], ctx=ast.Load()),
            slice=_normalize_node(slice_),
            ctx=ast.Load(),
        )

    # Unknown type (for example, MyGeneric[X]): None tells _normalize_node to
    # recurse into the slice while keeping the value unchanged.
    return None


def _normalize_node(syntax_node: ast.expr) -> ast.expr:
    """Apply PEP 585/604 transformations recursively to an AST node.

    Parameters
    ----------
    syntax_node : ast.expr
        AST expression node to be normalized.

    Returns
    -------
    ast.expr
        Recursively normalizes an AST expression node by applying PEP 585/604 type transformations, returning the transformed AST
        expression node.

    """
    if isinstance(syntax_node, ast.Subscript):
        identifier_name = _extract_name_type(syntax_node.value)
        outcome = _normalize_subscript(identifier_name, syntax_node.slice)
        if outcome is not None:
            return outcome
        return ast.Subscript(value=syntax_node.value, slice=_normalize_node(syntax_node.slice), ctx=syntax_node.ctx)

    if isinstance(syntax_node, ast.Tuple):
        # Slice multi-args (ex : ``Dict[K, V]``) ou ``Tuple[X, Y]`` comme type
        return ast.Tuple(elts=[_normalize_node(e) for e in syntax_node.elts], ctx=syntax_node.ctx)

    if isinstance(syntax_node, ast.List):
        # ``Callable[[A, B], R]``: the parameter list.
        return ast.List(elts=[_normalize_node(e) for e in syntax_node.elts], ctx=syntax_node.ctx)

    if isinstance(syntax_node, ast.BinOp) and isinstance(syntax_node.op, ast.BitOr):
        # ``X | Y`` is already PEP 604: recurse into operands.
        return ast.BinOp(
            left=_normalize_node(syntax_node.left),
            op=ast.BitOr(),
            right=_normalize_node(syntax_node.right),
        )

    # Name, Attribute, Constant, Starred, etc. : retourner sans modification
    return syntax_node
