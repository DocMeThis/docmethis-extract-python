# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Unit tests for type_normalization.py."""

from docmethis_extract_python.static_extraction.models import MISSING_VALUE
from docmethis_extract_python.static_extraction.type_normalization import normalize_type

# ---------------------------------------------------------------------------
# Basic cases - simple types.
# ---------------------------------------------------------------------------


def test_int_unchanged() -> None:
    assert normalize_type("int") == "int"


def test_str_unchanged() -> None:
    assert normalize_type("str") == "str"


def test_none_unchanged() -> None:
    assert normalize_type("None") == "None"


def test_missing_unchanged() -> None:
    assert normalize_type("<absent>") == MISSING_VALUE


def test_custom_type_unchanged() -> None:
    assert normalize_type("MyCustomType") == "MyCustomType"


def test_invalid_syntax_unchanged() -> None:
    assert normalize_type("invalid syntax @@@") == "invalid syntax @@@"


# ---------------------------------------------------------------------------
# Optional
# ---------------------------------------------------------------------------


def test_optional_int() -> None:
    assert normalize_type("Optional[int]") == "int | None"


def test_optional_str() -> None:
    assert normalize_type("Optional[str]") == "str | None"


def test_qualified_optional() -> None:
    assert normalize_type("typing.Optional[str]") == "str | None"


def test_optional_custom() -> None:
    assert normalize_type("Optional[MyClass]") == "MyClass | None"


# ---------------------------------------------------------------------------
# Union
# ---------------------------------------------------------------------------


def test_union_two_types() -> None:
    assert normalize_type("Union[str, int]") == "str | int"


def test_union_three_types() -> None:
    assert normalize_type("Union[str, int, float]") == "str | int | float"


def test_union_with_none() -> None:
    assert normalize_type("Union[int, None]") == "int | None"


def test_qualified_union() -> None:
    assert normalize_type("typing.Union[str, int]") == "str | int"


# ---------------------------------------------------------------------------
# PEP 585 - legacy forms to lowercase.
# ---------------------------------------------------------------------------


def test_list_int() -> None:
    assert normalize_type("List[int]") == "list[int]"


def test_dict_str_int() -> None:
    assert normalize_type("Dict[str, int]") == "dict[str, int]"


def test_tuple_int_str() -> None:
    assert normalize_type("Tuple[int, str]") == "tuple[int, str]"


def test_set_float() -> None:
    assert normalize_type("Set[float]") == "set[float]"


def test_frozenset_str() -> None:
    assert normalize_type("FrozenSet[str]") == "frozenset[str]"


def test_type_myclass() -> None:
    assert normalize_type("Type[MyClass]") == "type[MyClass]"


def test_qualified_list() -> None:
    assert normalize_type("typing.List[int]") == "list[int]"


def test_qualified_dict() -> None:
    assert normalize_type("typing.Dict[str, int]") == "dict[str, int]"


# ---------------------------------------------------------------------------
# Already normalized - passthrough.
# ---------------------------------------------------------------------------


def test_lowercase_list_unchanged() -> None:
    assert normalize_type("list[int]") == "list[int]"


def test_lowercase_dict_unchanged() -> None:
    assert normalize_type("dict[str, int]") == "dict[str, int]"


def test_pep604_unchanged() -> None:
    assert normalize_type("str | None") == "str | None"


def test_pep604_three_types_unchanged() -> None:
    assert normalize_type("str | int | float") == "str | int | float"


# ---------------------------------------------------------------------------
# Nesting.
# ---------------------------------------------------------------------------


def test_optional_list() -> None:
    assert normalize_type("Optional[List[int]]") == "list[int] | None"


def test_list_optional() -> None:
    assert normalize_type("List[Optional[int]]") == "list[int | None]"


def test_dict_str_list_int() -> None:
    assert normalize_type("Dict[str, List[int]]") == "dict[str, list[int]]"


def test_dict_str_list_optional_int() -> None:
    assert normalize_type("Dict[str, List[Optional[int]]]") == "dict[str, list[int | None]]"


def test_optional_dict() -> None:
    assert normalize_type("Optional[Dict[str, int]]") == "dict[str, int] | None"


def test_union_with_list() -> None:
    assert normalize_type("Union[List[int], str]") == "list[int] | str"


# ---------------------------------------------------------------------------
# Callable - passthrough for args (nested List).
# ---------------------------------------------------------------------------


def test_callable_unchanged() -> None:
    assert normalize_type("Callable[[int, str], bool]") == "Callable[[int, str], bool]"


def test_callable_with_optional_return() -> None:
    # Optional[str] return is normalized; List args pass through.
    assert normalize_type("Callable[[int], Optional[str]]") == "Callable[[int], str | None]"
