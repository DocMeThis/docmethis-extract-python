# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Unit tests for AST metrics: complexity, side effects, and percentiles."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from docmethis_extract_python.static_extraction.ast_metrics import (
    _walk_body_function,
    compute_complexity,
    compute_project_percentiles,
    detect_io,
)
from docmethis_extract_python.static_extraction.models import (
    ClassRecord,
    Complexity,
    Confidence,
    FunctionRecord,
    MethodType,
    ModuleRecord,
    ProjectRecord,
    Visibility,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_func(src: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Return the first FunctionDef or AsyncFunctionDef found in src."""
    tree = ast.parse(textwrap.dedent(src))
    ast.fix_missing_locations(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return node
    msg = "No function found in source"
    raise ValueError(msg)


def _make_function_record(name: str = "f", complexity: Complexity | None = None) -> FunctionRecord:
    """Build a minimal FunctionRecord for percentile tests."""
    return FunctionRecord(
        qualified_name=name,
        file_path=Path("test.py"),
        line_start=1,
        line_end=10,
        col_start=0,
        col_end=0,
        method_kind=MethodType.FUNCTION,
        visibility=Visibility.PUBLIC,
        parent_class=None,
        parent_module="test",
        existing_docstring=None,
        complexity=complexity,
        complexity_confidence=Confidence.EXPLICIT if complexity is not None else Confidence.ABSENT,
    )


def _make_complexity(cyclomatic: int = 1, loc: int = 5, max_depth: int = 0) -> Complexity:
    """Build a minimal Complexity for tests."""
    return Complexity(loc=loc, cyclomatic=cyclomatic, branches=0, max_depth=max_depth, statements=1)


def _make_project(function_records: list[FunctionRecord]) -> ProjectRecord:
    """Build a minimal ProjectRecord containing the given functions."""
    module = ModuleRecord(file_path=Path("test.py"), module_name="test", functions=function_records)
    return ProjectRecord(project_root=Path(), project_name="test", modules=[module])


# ---------------------------------------------------------------------------
# TestWalkFunctionBody
# ---------------------------------------------------------------------------


class TestWalkFunctionBody:
    """Tests for _walk_body_function."""

    def test_walk_simple(self) -> None:
        node = _parse_func("def f():\n    pass\n")
        nodes = list(_walk_body_function(node))
        types = [type(n).__name__ for n in nodes]
        assert "Pass" in types

    def test_does_not_descend_into_nested_function(self) -> None:
        node = _parse_func(
            """
            def outer(x):
                if x:
                    pass
                def inner(y):
                    return y
            """
        )
        nodes = list(_walk_body_function(node))
        # inner is yielded as a statement in outer's body.
        inner_nodes = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == "inner"]
        assert len(inner_nodes) == 1
        # But inner's Return is not yielded.
        returns = [n for n in nodes if isinstance(n, ast.Return)]
        assert len(returns) == 0

    def test_does_not_descend_into_nested_class(self) -> None:
        node = _parse_func(
            """
            def outer():
                class Inner:
                    def method(self):
                        return 1
            """
        )
        nodes = list(_walk_body_function(node))
        returns = [n for n in nodes if isinstance(n, ast.Return)]
        assert len(returns) == 0

    def test_excludes_default_parameter_values(self) -> None:
        # open() in default values must not be detected.
        node = _parse_func("def f(x=open('cfg')):\n    pass\n")
        nodes = list(_walk_body_function(node))
        calls_open = [n for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open"]
        assert len(calls_open) == 0


# ---------------------------------------------------------------------------
# TestComputeComplexity
# ---------------------------------------------------------------------------


class TestComputeComplexity:
    """Tests for compute_complexity."""

    def test_function_empty(self) -> None:
        node = _parse_func("def f():\n    pass\n")
        c = compute_complexity(node)
        assert c.cyclomatic == 1
        assert c.branches == 0
        assert c.max_depth == 0
        assert c.statements == 1

    def test_if_without_else(self) -> None:
        node = _parse_func(
            """
            def f(x):
                if x:
                    return 1
                return 0
            """
        )
        c = compute_complexity(node)
        assert c.cyclomatic == 2
        assert c.branches == 1  # Only the if body, no explicit else.
        assert c.max_depth == 1
        assert c.statements == 3  # If, Return(1), Return(0).

    def test_if_elif_else(self) -> None:
        node = _parse_func(
            """
            def f(x):
                if x > 0:
                    pass
                elif x < 0:
                    pass
                else:
                    pass
            """
        )
        c = compute_complexity(node)
        assert c.cyclomatic == 3
        assert c.branches == 3  # corps if + corps elif + else terminal
        assert c.max_depth == 1  # if/elif/else flattened at the same depth.

    def test_boolop_increments_mccabe(self) -> None:
        node = _parse_func(
            """
            def f(a, b, c):
                if a and b and c:
                    pass
            """
        )
        c = compute_complexity(node)
        # Base 1 + If 1 + BoolOp(3 values) + 2 = 4.
        assert c.cyclomatic == 4
        assert c.branches == 1
        assert c.max_depth == 1

    def test_for_with_nested_if(self) -> None:
        node = _parse_func(
            """
            def f(lst):
                for x in lst:
                    if x:
                        pass
            """
        )
        c = compute_complexity(node)
        assert c.cyclomatic == 3  # Base + For + If.
        assert c.max_depth == 2  # for(1) > if(2).

    def test_nested_function_is_not_counted(self) -> None:
        node_outer = _parse_func(
            """
            def outer(x):
                if x:
                    pass
                def inner(a, b, c):
                    if a and b and c and True and False:
                        pass
            """
        )
        c = compute_complexity(node_outer)
        assert c.cyclomatic == 2
        assert c.branches == 1
        assert c.max_depth == 1

    def test_absolute_loc_value(self) -> None:
        # LOC includes the def line and excludes comments (not AST nodes).
        # def f(x):    <- line 1 (included)
        #     return x <- line 2
        node = _parse_func("def f(x):\n    return x\n")
        c = compute_complexity(node)
        assert c.loc == 2  # def + return

    def test_loc_excludes_docstring(self) -> None:
        node = _parse_func(
            """
            def f(x):
                \"\"\"Docstring sur une ligne.\"\"\"
                return x
            """
        )
        c = compute_complexity(node)
        # Three total lines (def, docstring, return) - one docstring line = 2.
        assert c.loc == 2

    def test_async_def(self) -> None:
        node = _parse_func(
            """
            async def f(x):
                if x:
                    pass
            """
        )
        c = compute_complexity(node)
        assert c.cyclomatic == 2

    def test_percentiles_initially_none(self) -> None:
        node = _parse_func("def f():\n    pass\n")
        c = compute_complexity(node)
        assert c.cyclomatic_percentile is None
        assert c.loc_percentile is None
        assert c.max_depth_percentile is None


# ---------------------------------------------------------------------------
# TestDetectIO
# ---------------------------------------------------------------------------


class TestDetectIO:
    """Tests for detect_io."""

    def test_without_side_effects(self) -> None:
        node = _parse_func("def f(x):\n    return x + 1\n")
        io = detect_io(node)
        assert not io.reads_files
        assert not io.writes_files
        assert not io.network_access
        assert not io.system_calls
        assert not io.mutates_global
        assert not io.has_output
        assert io.details == []

    def test_print(self) -> None:
        node = _parse_func("def f():\n    print('hello')\n")
        io = detect_io(node)
        assert io.has_output
        assert not io.reads_files
        assert len(io.details) == 1
        assert io.details[0].category == "has_output"

    def test_open_read(self) -> None:
        node = _parse_func("def f():\n    open('file', 'r')\n")
        io = detect_io(node)
        assert io.reads_files
        assert not io.writes_files

    def test_open_without_mode(self) -> None:
        node = _parse_func("def f():\n    open('file')\n")
        io = detect_io(node)
        assert io.reads_files

    def test_open_write(self) -> None:
        node = _parse_func("def f():\n    open('file', 'w')\n")
        io = detect_io(node)
        assert io.writes_files
        assert not io.reads_files

    def test_open_append(self) -> None:
        node = _parse_func("def f():\n    open('file', 'a')\n")
        io = detect_io(node)
        assert io.writes_files

    def test_open_mode_unknown(self) -> None:
        # Mode in a variable is not statically determinable.
        node = _parse_func("def f(mode):\n    open('file', mode)\n")
        io = detect_io(node)
        assert io.reads_files
        assert io.writes_files

    def test_read_text(self) -> None:
        node = _parse_func("def f(p):\n    p.read_text()\n")
        io = detect_io(node)
        assert io.reads_files

    def test_write_text(self) -> None:
        node = _parse_func("def f(p):\n    p.write_text('data')\n")
        io = detect_io(node)
        assert io.writes_files

    def test_requests(self) -> None:
        node = _parse_func("def f():\n    requests.get('http://x')\n")
        io = detect_io(node)
        assert io.network_access

    def test_subprocess_run(self) -> None:
        node = _parse_func("def f():\n    subprocess.run(['ls'])\n")
        io = detect_io(node)
        assert io.system_calls

    def test_os_system(self) -> None:
        node = _parse_func("def f():\n    os.system('ls')\n")
        io = detect_io(node)
        assert io.system_calls

    def test_global_statement(self) -> None:
        node = _parse_func("def f():\n    global x\n    x = 1\n")
        io = detect_io(node)
        assert io.mutates_global
        detail = next(d for d in io.details if d.category == "mutates_global")
        assert "global" in detail.pattern

    def test_os_environ(self) -> None:
        node = _parse_func("def f():\n    os.environ['KEY'] = 'val'\n")
        io = detect_io(node)
        assert io.mutates_global

    def test_shutil(self) -> None:
        node = _parse_func("def f():\n    shutil.copy('a', 'b')\n")
        io = detect_io(node)
        assert io.writes_files

    def test_logging(self) -> None:
        node = _parse_func("def f():\n    logging.info('msg')\n")
        io = detect_io(node)
        assert io.has_output

    def test_detail_contains_line_number(self) -> None:
        src = "def f():\n    pass\n    print('x')\n"
        node = _parse_func(src)
        io = detect_io(node)
        assert io.details[0].line == 3  # print is on line 3.

    def test_io_in_nested_function_is_not_detected(self) -> None:
        node = _parse_func(
            """
            def outer():
                def inner():
                    print('inner')
                return 1
            """
        )
        io = detect_io(node)
        assert not io.has_output  # print is in inner, not outer.

    def test_with_open(self) -> None:
        node = _parse_func(
            """
            def f():
                with open('file', 'r') as fh:
                    return fh.read()
            """
        )
        io = detect_io(node)
        assert io.reads_files

    def test_unresolved_import_alias_is_false_negative(self) -> None:
        # Known limitation: import aliases are not resolved.
        # "import requests as r; r.get()" returns "r.get", whose prefix "r"
        # is not in _PREFIXES_NETWORK, so network detection silently fails.
        node = _parse_func("def f():\n    r.get('http://x')\n")
        io = detect_io(node)
        assert not io.network_access  # Expected false negative.

    def test_io_in_nested_function_under_if_is_not_detected(self) -> None:
        # Regression: print in a nested def under an if must not leak out.
        node = _parse_func(
            """
            def outer(cond):
                if cond:
                    def inner():
                        print('inner')
                return 1
            """
        )
        io = detect_io(node)
        assert not io.has_output

    def test_global_in_conditional_block(self) -> None:
        # global declared inside an if must be detected.
        node = _parse_func(
            """
            def f(cond):
                if cond:
                    global x
            """
        )
        io = detect_io(node)
        assert io.mutates_global

    def test_http_client_network_access(self) -> None:
        node = _parse_func("def f():\n    http.client.HTTPConnection('host')\n")
        io = detect_io(node)
        assert io.network_access

    def test_try_except_star_mccabe(self) -> None:
        node = _parse_func(
            """
            def f():
                try:
                    pass
                except* TypeError as eg:
                    pass
            """
        )
        c = compute_complexity(node)
        assert c.cyclomatic == 2  # base 1 + ExceptHandler 1

    def test_io_in_for_block_is_detected(self) -> None:
        # Positive case: print in a for body must be detected.
        node = _parse_func(
            """
            def f(lst):
                for x in lst:
                    print(x)
            """
        )
        io = detect_io(node)
        assert io.has_output

    def test_io_in_if_block_is_detected(self) -> None:
        # Positive case: open() in an if body must be detected.
        node = _parse_func(
            """
            def f(cond):
                if cond:
                    open('f.txt', 'w')
            """
        )
        io = detect_io(node)
        assert io.writes_files

    def test_io_in_nested_function_under_for_is_not_detected(self) -> None:
        # Regression: print in a nested def under a for must not leak out.
        node = _parse_func(
            """
            def f(lst):
                for x in lst:
                    def inner():
                        print('fuyard')
            """
        )
        io = detect_io(node)
        assert not io.has_output

    def test_logger_parameter_is_false_negative(self) -> None:
        # Known limitation: logger.info() where logger is a parameter is not detected,
        # because "logger.info" does not start with "logging.".
        node = _parse_func(
            """
            def f(logger):
                logger.info('msg')
            """
        )
        io = detect_io(node)
        assert not io.has_output  # Expected false negative; logger passed as a parameter.


# ---------------------------------------------------------------------------
# TestComputeProjectPercentiles
# ---------------------------------------------------------------------------


class TestComputeProjectPercentiles:
    """Tests for compute_project_percentiles."""

    def test_project_without_functions(self) -> None:
        project_record = _make_project([])
        compute_project_percentiles(project_record)  # Must not raise.

    def test_one_function(self) -> None:
        f = _make_function_record(complexity=_make_complexity(cyclomatic=5, loc=10, max_depth=2))
        project_record = _make_project([f])
        compute_project_percentiles(project_record)
        assert f.complexity is not None
        assert f.complexity.cyclomatic_percentile == pytest.approx(0.0)
        assert f.complexity.loc_percentile == pytest.approx(0.0)
        assert f.complexity.max_depth_percentile == pytest.approx(0.0)

    def test_three_functions_with_distinct_values(self) -> None:
        f1 = _make_function_record("f1", complexity=_make_complexity(cyclomatic=1))
        f2 = _make_function_record("f2", complexity=_make_complexity(cyclomatic=5))
        f3 = _make_function_record("f3", complexity=_make_complexity(cyclomatic=10))
        project_record = _make_project([f1, f2, f3])
        compute_project_percentiles(project_record)
        # f1 (cyclomatic=1): rank 1 -> (1-1)/(3-1) = 0.0.
        assert f1.complexity is not None
        assert f1.complexity.cyclomatic_percentile == pytest.approx(0.0)
        # f2 (cyclomatic=5): rank 2 -> (2-1)/(3-1) = 0.5.
        assert f2.complexity is not None
        assert f2.complexity.cyclomatic_percentile == pytest.approx(0.5)
        # f3 (cyclomatic=10): rank 3 -> (3-1)/(3-1) = 1.0.
        assert f3.complexity is not None
        assert f3.complexity.cyclomatic_percentile == pytest.approx(1.0)

    def test_all_values_identical(self) -> None:
        function_records = [_make_function_record(f"f{i}", complexity=_make_complexity(cyclomatic=3)) for i in range(4)]
        project_record = _make_project(function_records)
        compute_project_percentiles(project_record)
        # Mean rank = (0 + (4+1)/2) = 2.5 -> (2.5-1)/(4-1) = 0.5.
        for f in function_records:
            assert f.complexity is not None
            assert f.complexity.cyclomatic_percentile == pytest.approx(0.5)

    def test_class_methods_are_included(self) -> None:
        m = _make_function_record("cls.method", complexity=_make_complexity(cyclomatic=8))
        cls = ClassRecord(
            qualified_name="Cls",
            file_path=Path("test.py"),
            line_start=1,
            line_end=20,
            col_start=0,
            col_end=0,
            visibility=Visibility.PUBLIC,
            parent_module="test",
            existing_docstring=None,
            methods=[m],
        )
        f = _make_function_record("f", complexity=_make_complexity(cyclomatic=1))
        module = ModuleRecord(file_path=Path("test.py"), module_name="test", functions=[f], classes=[cls])
        project_record = ProjectRecord(project_root=Path(), project_name="test", modules=[module])
        compute_project_percentiles(project_record)
        # f(1): rank 1 -> (1-1)/(2-1) = 0.0; m(8): rank 2 -> (2-1)/(2-1) = 1.0.
        assert f.complexity is not None
        assert f.complexity.cyclomatic_percentile == pytest.approx(0.0)
        assert m.complexity is not None
        assert m.complexity.cyclomatic_percentile == pytest.approx(1.0)

    def test_none_complexity_is_ignored_without_crash(self) -> None:
        function_with_complexity = _make_function_record("f1", complexity=_make_complexity(cyclomatic=5))
        function_without_complexity = _make_function_record("f2", complexity=None)
        project_record = _make_project([function_with_complexity, function_without_complexity])
        compute_project_percentiles(project_record)  # Must not raise.
        assert function_with_complexity.complexity is not None
        assert function_with_complexity.complexity.cyclomatic_percentile == pytest.approx(0.0)
        assert function_without_complexity.complexity is None
