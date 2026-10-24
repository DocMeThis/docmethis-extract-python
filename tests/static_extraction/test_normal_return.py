# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Conservative AST classification of normal function completion."""

from __future__ import annotations

import ast
import inspect
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from docmethis_extract_python.static_extraction.ast_analysis import extract_module_record
from docmethis_extract_python.static_extraction.models import NormalReturnStatus
from docmethis_extract_python.static_extraction.normal_return import infer_normal_return_status

if TYPE_CHECKING:
    from pathlib import Path


def _status(source: str) -> NormalReturnStatus:
    """Classify the sole top-level function in a source snippet.

    Parameters
    ----------
    source : str
        Accepts a source string, parses it as Python after dedenting, and takes the first top-level statement as a function
        definition. Returns the normal-return status inferred for that function. Asserts that the first statement is a function
        definition.

    Returns
    -------
    NormalReturnStatus
        Returns the NormalReturnStatus inferred from the first top-level function definition parsed from the dedented source
        string.

    """
    module = ast.parse(dedent(source))
    function = module.body[0]
    assert isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
    return infer_normal_return_status(function)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param("def f():\n    raise Error()\n", NormalReturnStatus.PROVEN_ABSENT, id="direct-raise"),
        pytest.param(
            "def f(flag):\n    if flag:\n        raise Error()\n    else:\n        raise OtherError()\n",
            NormalReturnStatus.PROVEN_ABSENT,
            id="all-if-branches-raise",
        ),
        pytest.param(
            "def f(flag):\n    if flag:\n        raise Error()\n",
            NormalReturnStatus.POSSIBLE,
            id="if-falls-through",
        ),
        pytest.param(
            "def f(flag):\n    if flag:\n        return 1\n    raise Error()\n",
            NormalReturnStatus.POSSIBLE,
            id="conditional-return",
        ),
        pytest.param("def f():\n    return None\n", NormalReturnStatus.POSSIBLE, id="explicit-none-return"),
        pytest.param("def f():\n    return\n", NormalReturnStatus.POSSIBLE, id="bare-return"),
        pytest.param(
            "def f():\n    try:\n        raise Error()\n    except Error:\n        raise\n",
            NormalReturnStatus.PROVEN_ABSENT,
            id="handled-reraise",
        ),
        pytest.param(
            "def f():\n    try:\n        raise Error()\n    except Error:\n        pass\n",
            NormalReturnStatus.POSSIBLE,
            id="handled-fallthrough",
        ),
        pytest.param(
            "def f():\n    try:\n        return 1\n    finally:\n        raise Error()\n",
            NormalReturnStatus.PROVEN_ABSENT,
            id="finally-overrides-return",
        ),
        pytest.param(
            "def f():\n    try:\n        pass\n    except Error:\n        raise\n    else:\n        raise Error()\n",
            NormalReturnStatus.PROVEN_ABSENT,
            id="try-else-raise",
        ),
        pytest.param(
            "def f():\n    try:\n        raise ExceptionGroup('g', [ValueError()])\n    except* ValueError:\n        raise\n",
            NormalReturnStatus.UNKNOWN,
            id="try-star-unsupported",
        ),
        pytest.param(
            "def f():\n    while True:\n        raise Error()\n",
            NormalReturnStatus.PROVEN_ABSENT,
            id="true-loop-raise",
        ),
        pytest.param(
            "def f():\n    while True:\n        pass\n",
            NormalReturnStatus.PROVEN_ABSENT,
            id="true-loop-diverges",
        ),
        pytest.param(
            "def f():\n    while True:\n        continue\n",
            NormalReturnStatus.PROVEN_ABSENT,
            id="true-loop-continue",
        ),
        pytest.param(
            "def f():\n    while True:\n        break\n",
            NormalReturnStatus.POSSIBLE,
            id="true-loop-break",
        ),
        pytest.param(
            "def f(flag):\n    while flag:\n        raise Error()\n",
            NormalReturnStatus.POSSIBLE,
            id="while-may-skip",
        ),
        pytest.param(
            "def f(items):\n    for item in items:\n        raise Error()\n",
            NormalReturnStatus.POSSIBLE,
            id="for-may-be-empty",
        ),
        pytest.param(
            "def f(items):\n    for item in items:\n        break\n    else:\n        raise Error()\n",
            NormalReturnStatus.POSSIBLE,
            id="for-break-skips-else",
        ),
        pytest.param(
            "def f(value):\n    match value:\n        case _:\n            raise Error()\n",
            NormalReturnStatus.PROVEN_ABSENT,
            id="match-wildcard",
        ),
        pytest.param(
            "def f(value):\n    match value:\n        case 0:\n            raise Error()\n",
            NormalReturnStatus.POSSIBLE,
            id="match-may-fall-through",
        ),
        pytest.param("def f():\n    return helper()\n", NormalReturnStatus.POSSIBLE, id="opaque-return-call"),
        pytest.param(
            "def f():\n    with manager():\n        raise Error()\n",
            NormalReturnStatus.UNKNOWN,
            id="opaque-context-manager",
        ),
        pytest.param(
            "@wrapper\ndef f():\n    raise Error()\n",
            NormalReturnStatus.UNKNOWN,
            id="opaque-decorator",
        ),
        pytest.param("def f():\n    yield 1\n", NormalReturnStatus.UNKNOWN, id="generator"),
        pytest.param("async def f():\n    raise Error()\n", NormalReturnStatus.PROVEN_ABSENT, id="async-raise"),
        pytest.param(
            "def f():\n    def nested():\n        yield 1\n    raise Error()\n",
            NormalReturnStatus.PROVEN_ABSENT,
            id="nested-generator-is-separate-scope",
        ),
    ],
)
def test_infer_normal_return_status_is_conservative(source: str, expected: NormalReturnStatus) -> None:
    """Only definite normal exits are possible; unsupported constructs stay unknown.

    Parameters
    ----------
    source : str
        Accepts a string argument named source that supplies the code to be analyzed, and a second argument giving the expected
        normal-return status. The function asserts that the status inferred from source is the expected value.
    expected : NormalReturnStatus
        A pytest-style test case that asserts the helper _status, when applied to source, returns exactly the provided expected
        value. It performs no other actions and takes no other inputs besides source and expected.

    """
    assert _status(source) is expected


def test_extract_module_record_stores_normal_return_status(tmp_path: Path) -> None:
    """The AST result is persisted on each extracted function record.

    Parameters
    ----------
    tmp_path : Path
        Provides the directory in which a temporary Python source file is created and written with the test's input code, whose
        path is then passed to the extraction function.

    """
    source = tmp_path / "mod.py"
    source.write_text(
        dedent(
            """\
            def never() -> int:
                raise Error()

            def maybe(flag: bool) -> int:
                if flag:
                    raise Error()
            """
        ),
        encoding="utf-8",
    )

    record = extract_module_record(source, "mod", is_stub=False, is_test=False, file_hash="abc")

    assert record is not None
    statuses = {function.qualified_name: function.normal_return_status for function in record.functions}
    assert statuses == {
        "mod.never": NormalReturnStatus.PROVEN_ABSENT,
        "mod.maybe": NormalReturnStatus.POSSIBLE,
    }


@pytest.mark.parametrize(
    "source",
    [
        "def f():\n    try:\n        while True:\n            1 / 0\n    finally:\n        return 42\n",
        (
            "def f():\n    while True:\n        try:\n            while True:\n                1 / 0\n"
            "        finally:\n            break\n    return 42\n"
        ),
    ],
    ids=["finally-return", "finally-break"],
)
def test_finally_can_return_after_an_implicit_loop_exception(source: str) -> None:
    """An implicit exception can leave a non-terminating loop through finally.

    Parameters
    ----------
    source : str
        Accepts a source string whose code is executed to define a function f, then asserts f() returns 42; also asserts the
        analyzed status of that source is NormalReturnStatus.POSSIBLE.

    """
    assert _status(source) is NormalReturnStatus.POSSIBLE
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - execute only the fixed regression snippets above.
    assert namespace["f"]() == 42


@pytest.mark.parametrize(
    "definition",
    [
        "def inner(value=(yield 1)):\n        pass",
        "async def inner(value=(yield 1)):\n        pass",
        "inner = lambda value=(yield 1): value",
        "@(yield 1)\n    def inner():\n        pass",
        "class Inner((yield 1)):\n        pass",
        "class Inner(metaclass=(yield 1)):\n        pass",
        "@(yield 1)\n    class Inner:\n        pass",
    ],
    ids=[
        "function-default",
        "async-default",
        "lambda-default",
        "function-decorator",
        "class-base",
        "class-keyword",
        "class-decorator",
    ],
)
def test_yields_in_nested_definition_expressions_belong_to_the_outer_function(definition: str) -> None:
    """Skip nested bodies, but inspect defaults, decorators and class arguments.

    Parameters
    ----------
    definition : str
        Accepts a required string argument named definition. The function performs assertions only and returns None.

    """
    source = f"def f() -> int:\n    {definition}\n    raise RuntimeError()\n"
    assert _status(source) is NormalReturnStatus.UNKNOWN
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - execute only the fixed regression snippets above.
    assert inspect.isgeneratorfunction(namespace["f"])
