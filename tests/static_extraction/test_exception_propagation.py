# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Unit tests for propagate_exceptions_via_callgraph."""

from __future__ import annotations

from pathlib import Path

from docmethis_extract_python.api import exception_escapes
from docmethis_extract_python.static_extraction.models import (
    CallgraphEdge,
    Confidence,
    ExceptBlock,
    ExceptionRecord,
    FunctionRecord,
    MethodType,
    ModuleRecord,
    ProjectRecord,
    Provenance,
    RaisedCondition,
    Visibility,
)
from docmethis_extract_python.static_extraction.type_resolution import propagate_exceptions_via_callgraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fn(
    qname: str,
    exceptions: list[ExceptionRecord] | None = None,
    except_blocks: list[ExceptBlock] | None = None,
    exceptions_confidence: Confidence = Confidence.ABSENT,
) -> FunctionRecord:
    return FunctionRecord(
        qualified_name=qname,
        file_path=Path("mod.py"),
        line_start=1,
        line_end=5,
        col_start=0,
        col_end=0,
        method_kind=MethodType.FUNCTION,
        visibility=Visibility.PUBLIC,
        parent_class=None,
        parent_module="mod",
        existing_docstring=None,
        exceptions=exceptions,
        exceptions_confidence=exceptions_confidence,
        except_blocks=except_blocks,
    )


def _exc(
    exc_type: str,
    line: int = 1,
    *,
    message: str | None = None,
    is_reraise: bool = False,
    condition: RaisedCondition | None = None,
) -> ExceptionRecord:
    return ExceptionRecord(
        exception_type=exc_type,
        message=message,
        raise_line=line,
        is_reraise=is_reraise,
        chained_from=False,
        condition=condition,
    )


def _edge(caller: str, callee: str, line: int = 1) -> CallgraphEdge:
    return CallgraphEdge(
        source_qname=caller,
        target_qname=callee,
        call_type="direct",
        is_external=False,
        external_lib=None,
        line=line,
    )


def _edge_ext(caller: str, callee: str) -> CallgraphEdge:
    return CallgraphEdge(
        source_qname=caller,
        target_qname=callee,
        call_type="direct",
        is_external=True,
        external_lib=callee.split(".", maxsplit=1)[0],
        line=1,
    )


def _project(*fns: FunctionRecord) -> tuple[ProjectRecord, dict[str, list[CallgraphEdge]]]:
    """Build a minimal ProjectRecord with the given functions and an empty callgraph."""
    module = ModuleRecord(
        file_path=Path("mod.py"),
        module_name="mod",
        functions=list(fns),
    )
    project_record = ProjectRecord(project_root=Path(), project_name="test", modules=[module])
    callgraph: dict[str, list[CallgraphEdge]] = {fn.qualified_name: [] for fn in fns}
    return project_record, callgraph


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPropagateExceptions:
    """Unit tests for exception propagation through the callgraph."""

    def test_direct_propagation_depth_one(self) -> None:
        """G raises ValueError, f calls g without try/except, so f propagates ValueError."""
        fn_g = _fn("mod.g", exceptions=[_exc("ValueError")])
        fn_f = _fn("mod.f")
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]

        propagate_exceptions_via_callgraph(project_record, cg)

        types = [e.exception_type for e in (fn_f.exceptions or [])]
        assert "ValueError" in types

    def test_propagation_provenance_is_callgraph(self) -> None:
        """A propagated exception has provenance=CALLGRAPH_PROPAGATION."""
        fn_g = _fn("mod.g", exceptions=[_exc("ValueError")])
        fn_f = _fn("mod.f")
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]

        propagate_exceptions_via_callgraph(project_record, cg)

        provenances = [e.provenance for e in (fn_f.exceptions or [])]
        assert all(p == Provenance.CALLGRAPH_PROPAGATION for p in provenances)

    def test_local_raise_provenance_is_preserved(self) -> None:
        """A local raise keeps LOCAL_INFERENCE provenance and is not overwritten."""
        fn_g = _fn("mod.g", exceptions=[_exc("ValueError")])
        fn_f = _fn("mod.f", exceptions=[_exc("TypeError")])
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]

        propagate_exceptions_via_callgraph(project_record, cg)

        provenances = {e.exception_type: e.provenance for e in (fn_f.exceptions or [])}
        assert provenances["TypeError"] == Provenance.LOCAL_INFERENCE
        assert provenances["ValueError"] == Provenance.CALLGRAPH_PROPAGATION

    def test_no_propagation_when_caught(self) -> None:
        """G raises ValueError and f catches ValueError, so nothing propagates."""
        fn_g = _fn("mod.g", exceptions=[_exc("ValueError")])
        except_f = ExceptBlock(caught_types=["ValueError"], is_reraised=False, is_swallowed=True, line=3)
        fn_f = _fn("mod.f", except_blocks=[except_f])
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]

        propagate_exceptions_via_callgraph(project_record, cg)

        types = [e.exception_type for e in (fn_f.exceptions or [])]
        assert "ValueError" not in types

    def test_bare_except_blocks_propagation(self) -> None:
        """A bare except blocks all exception propagation."""
        fn_g = _fn("mod.g", exceptions=[_exc("ValueError")])
        bare = ExceptBlock(caught_types=[], is_reraised=False, is_swallowed=True, line=3)
        fn_f = _fn("mod.f", except_blocks=[bare])
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]

        propagate_exceptions_via_callgraph(project_record, cg)

        assert not fn_f.exceptions

    def test_exception_handler_blocks_propagation(self) -> None:
        """G raises ValueError and f catches Exception, so nothing propagates."""
        fn_g = _fn("mod.g", exceptions=[_exc("ValueError")])
        block = ExceptBlock(caught_types=["Exception"], is_reraised=False, is_swallowed=True, line=3)
        fn_f = _fn("mod.f", except_blocks=[block])
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]

        propagate_exceptions_via_callgraph(project_record, cg)

        assert not fn_f.exceptions

    def test_base_exception_handler_blocks_propagation(self) -> None:
        """G raises KeyError and f catches BaseException, so nothing propagates."""
        fn_g = _fn("mod.g", exceptions=[_exc("KeyError")])
        block = ExceptBlock(caught_types=["BaseException"], is_reraised=False, is_swallowed=True, line=3)
        fn_f = _fn("mod.f", except_blocks=[block])
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]

        propagate_exceptions_via_callgraph(project_record, cg)

        assert not fn_f.exceptions

    def test_propagation_depth_two(self) -> None:
        """H raises KeyError, g calls h, and f calls g, so f propagates KeyError at depth 2."""
        fn_h = _fn("mod.h", exceptions=[_exc("KeyError")])
        fn_g = _fn("mod.g")
        fn_f = _fn("mod.f")
        project_record, cg = _project(fn_f, fn_g, fn_h)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]
        cg["mod.g"] = [_edge("mod.g", "mod.h")]

        propagate_exceptions_via_callgraph(project_record, cg)

        types_f = [e.exception_type for e in (fn_f.exceptions or [])]
        assert "KeyError" in types_f

    def test_no_propagation_beyond_depth_two(self) -> None:
        """I raises IOError through h, g, and f, but propagation stops beyond depth 2."""
        fn_i = _fn("mod.i", exceptions=[_exc("IOError")])
        fn_h = _fn("mod.h")
        fn_g = _fn("mod.g")
        fn_f = _fn("mod.f")
        project_record, cg = _project(fn_f, fn_g, fn_h, fn_i)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]
        cg["mod.g"] = [_edge("mod.g", "mod.h")]
        cg["mod.h"] = [_edge("mod.h", "mod.i")]

        propagate_exceptions_via_callgraph(project_record, cg)

        types_f = [e.exception_type for e in (fn_f.exceptions or [])]
        assert "IOError" not in types_f

    def test_no_duplicate_exceptions(self) -> None:
        """Two ValueError raises in G produce one ValueError entry in f."""
        fn_g = _fn("mod.g", exceptions=[_exc("ValueError", line=2), _exc("ValueError", line=5)])
        fn_f = _fn("mod.f")
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]

        propagate_exceptions_via_callgraph(project_record, cg)

        types = [e.exception_type for e in (fn_f.exceptions or [])]
        assert types.count("ValueError") == 1

    def test_external_calls_are_ignored(self) -> None:
        """Edges to external functions do not propagate exceptions."""
        fn_f = _fn("mod.f")
        project_record, cg = _project(fn_f)
        cg["mod.f"] = [_edge_ext("mod.f", "os.getcwd")]

        propagate_exceptions_via_callgraph(project_record, cg)

        assert not fn_f.exceptions

    def test_absent_confidence_becomes_inferred_high(self) -> None:
        """When f has no exceptions (ABSENT), confidence becomes INFERRED_HIGH."""
        fn_g = _fn("mod.g", exceptions=[_exc("TypeError")])
        fn_f = _fn("mod.f", exceptions_confidence=Confidence.ABSENT)
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]

        propagate_exceptions_via_callgraph(project_record, cg)

        assert fn_f.exceptions_confidence == Confidence.INFERRED_HIGH

    def test_explicit_confidence_is_not_downgraded(self) -> None:
        """When f already has EXPLICIT exceptions, confidence is unchanged."""
        exc_propre = _exc("RuntimeError")
        fn_g = _fn("mod.g", exceptions=[_exc("TypeError")])
        fn_f = _fn("mod.f", exceptions=[exc_propre], exceptions_confidence=Confidence.EXPLICIT)
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]

        propagate_exceptions_via_callgraph(project_record, cg)

        assert fn_f.exceptions_confidence == Confidence.EXPLICIT

    def test_callee_catches_before_caller(self) -> None:
        """G catches KeyError from h, so f does not propagate KeyError."""
        fn_h = _fn("mod.h", exceptions=[_exc("KeyError")])
        except_g = ExceptBlock(caught_types=["KeyError"], is_reraised=False, is_swallowed=True, line=3)
        fn_g = _fn("mod.g", except_blocks=[except_g])
        fn_f = _fn("mod.f")
        project_record, cg = _project(fn_f, fn_g, fn_h)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]
        cg["mod.g"] = [_edge("mod.g", "mod.h")]

        propagate_exceptions_via_callgraph(project_record, cg)

        types_f = [e.exception_type for e in (fn_f.exceptions or [])]
        assert "KeyError" not in types_f

    def test_empty_callgraph_does_not_propagate(self) -> None:
        """An empty callgraph produces no propagation."""
        fn_f = _fn("mod.f")
        project_record, cg = _project(fn_f)

        propagate_exceptions_via_callgraph(project_record, cg)

        assert not fn_f.exceptions

    def test_handlers_only_catch_raises_inside_their_try_range(self) -> None:
        """A handler outside the callee raise range does not suppress propagation."""
        fn_g = _fn("mod.g", exceptions=[_exc("ValueError", line=20)])
        block = ExceptBlock(
            caught_types=["ValueError"],
            is_reraised=False,
            is_swallowed=True,
            line=5,
            try_start=1,
            try_end=10,
        )
        fn_g.except_blocks = [block]
        fn_f = _fn("mod.f")
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g", line=1)]

        propagate_exceptions_via_callgraph(project_record, cg)

        assert [exception.exception_type for exception in fn_f.exceptions or []] == ["ValueError"]

    def test_caller_handler_only_catches_call_inside_its_try_range(self) -> None:
        """A caller handler outside the callsite range does not suppress propagation."""
        fn_g = _fn("mod.g", exceptions=[_exc("ValueError")])
        block = ExceptBlock(
            caught_types=["ValueError"],
            is_reraised=False,
            is_swallowed=True,
            line=5,
            try_start=10,
            try_end=20,
        )
        fn_f = _fn("mod.f", except_blocks=[block])
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g", line=1)]

        propagate_exceptions_via_callgraph(project_record, cg)

        assert [exception.exception_type for exception in fn_f.exceptions or []] == ["ValueError"]

    def test_propagation_preserves_conditions_messages_and_reraise(self) -> None:
        """Propagation retains source evidence without making it local to the caller."""
        condition = RaisedCondition(expression="value < 0", line=2)
        fn_g = _fn(
            "mod.g",
            exceptions=[_exc("ValueError", message="bad value", is_reraise=True, condition=condition)],
            except_blocks=[
                ExceptBlock(
                    caught_types=["ValueError"],
                    is_reraised=True,
                    is_swallowed=False,
                    line=4,
                    try_start=1,
                    try_end=5,
                )
            ],
        )
        fn_f = _fn("mod.f")
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g", line=8)]

        propagate_exceptions_via_callgraph(project_record, cg)

        assert fn_f.exceptions is not None
        propagated = fn_f.exceptions[0]
        assert propagated.provenance is Provenance.CALLGRAPH_PROPAGATION
        assert propagated.message == "bad value"
        assert propagated.condition is condition
        assert propagated.is_reraise is True
        assert propagated.raise_line == 8

    def test_local_and_propagated_same_type_are_both_retained(self) -> None:
        """A local raise is not hidden by a propagated raise of the same type."""
        local = _exc("ValueError", condition=RaisedCondition(expression="value < 0", line=2))
        fn_g = _fn("mod.g", exceptions=[_exc("ValueError")])
        fn_f = _fn("mod.f", exceptions=[local])
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]

        propagate_exceptions_via_callgraph(project_record, cg)

        assert fn_f.exceptions is not None
        assert [exception.provenance for exception in fn_f.exceptions] == [
            Provenance.LOCAL_INFERENCE,
            Provenance.CALLGRAPH_PROPAGATION,
        ]
        assert fn_f.exceptions[0].condition is local.condition

    def test_conditional_and_unconditional_occurrences_are_retained(self) -> None:
        """A conditionless propagated record does not hide a known source condition."""
        condition = RaisedCondition(expression="value < 0", line=2)
        fn_g = _fn("mod.g", exceptions=[_exc("ValueError"), _exc("ValueError", condition=condition)])
        fn_f = _fn("mod.f")
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]

        propagate_exceptions_via_callgraph(project_record, cg)

        assert fn_f.exceptions is not None
        conditions = {
            exception.condition.expression if exception.condition is not None else None for exception in fn_f.exceptions
        }
        assert conditions == {None, "value < 0"}

    def test_unknown_exception_type_remains_unknown_when_propagated(self) -> None:
        """Propagation does not turn unknown evidence into a guessed exception type."""
        fn_g = _fn("mod.g", exceptions=[_exc("<unknown>", is_reraise=True)])
        fn_f = _fn("mod.f")
        project_record, cg = _project(fn_f, fn_g)
        cg["mod.f"] = [_edge("mod.f", "mod.g")]

        propagate_exceptions_via_callgraph(project_record, cg)

        assert fn_f.exceptions is not None
        assert fn_f.exceptions[0].exception_type == "<unknown>"
        assert fn_f.exceptions[0].provenance is Provenance.CALLGRAPH_PROPAGATION

    def test_propagation_order_is_stable_across_edge_order(self) -> None:
        """Propagated records follow callsite order rather than input edge order."""
        fn_a = _fn("mod.a", exceptions=[_exc("ValueError")])
        fn_b = _fn("mod.b", exceptions=[_exc("TypeError")])
        fn_f = _fn("mod.f")
        project_record, cg = _project(fn_f, fn_b, fn_a)
        cg["mod.f"] = [_edge("mod.f", "mod.b", line=20), _edge("mod.f", "mod.a", line=10)]

        propagate_exceptions_via_callgraph(project_record, cg)

        assert [exception.exception_type for exception in fn_f.exceptions or []] == ["ValueError", "TypeError"]


class TestExceptionEscapes:
    """Unit tests for the shared exception visibility predicate."""

    def test_local_exception_caught_by_a_swallowing_handler_does_not_escape(self) -> None:
        function = _fn(
            "mod.f",
            except_blocks=[ExceptBlock(caught_types=["ValueError"], is_reraised=False, is_swallowed=True, line=3)],
        )

        assert exception_escapes(function, _exc("ValueError", line=2)) is False

    def test_local_exception_without_a_handler_escapes(self) -> None:
        function = _fn("mod.f")

        assert exception_escapes(function, _exc("ValueError")) is True

    def test_propagated_exception_is_already_exposed(self) -> None:
        function = _fn("mod.f")
        exception = _exc("ValueError")
        exception.provenance = Provenance.CALLGRAPH_PROPAGATION

        assert exception_escapes(function, exception) is True
