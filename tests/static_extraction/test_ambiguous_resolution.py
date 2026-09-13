# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for contextual test-to-function disambiguation (phase 2, iteration 1.10)."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from docmethis_extract_python.static_extraction.dynamic_analysis.test_analysis import (
    _arity_compatible,
    _call_argument_count,
    _count_non_variadic_parameters,
    _count_required_parameters,
    _indexer_function,
    _IndexProduction,
    _TestLinksVisitor,
)
from docmethis_extract_python.static_extraction.models import (
    FunctionRecord,
    MethodType,
    ModuleRecord,
    Parameter,
    ParameterType,
    Signature,
    Visibility,
)

if TYPE_CHECKING:
    import pytest

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_fn(
    qualified_name: str,
    parent_module: str,
    visibility: Visibility = Visibility.PUBLIC,
    signature: Signature | None = None,
) -> FunctionRecord:
    return FunctionRecord(
        qualified_name=qualified_name,
        file_path=Path("/dummy.py"),
        line_start=1,
        line_end=10,
        col_start=0,
        col_end=0,
        method_kind=MethodType.FUNCTION,
        visibility=visibility,
        parent_class=None,
        parent_module=parent_module,
        existing_docstring=None,
        signature=signature,
    )


def _make_sig(*kinds: ParameterType, names: list[str] | None = None, defaults: list[str] | None = None) -> Signature:
    """Build a Signature; ``defaults`` gives each parameter's default value."""
    if names is None:
        names = [f"p{i}" for i in range(len(kinds))]
    if defaults is None:
        defaults = ["<absent>"] * len(kinds)
    params = [
        Parameter(name=n, annotation_raw="<absent>", default_value=d, kind=k)
        for n, k, d in zip(names, kinds, defaults, strict=False)
    ]
    return Signature(parameters=params, return_annotation="<absent>", decorators=[])


def _parse_call(expr: str) -> ast.Call:
    """Parse a call expression and return its ast.Call node."""
    tree = ast.parse(expr, mode="eval")
    assert isinstance(tree.body, ast.Call)
    return tree.body


# ---------------------------------------------------------------------------
# Tests - _call_argument_count
# ---------------------------------------------------------------------------


class TestCallArgumentCount:
    """Tests for _call_argument_count."""

    def test_no_args(self) -> None:
        assert _call_argument_count(_parse_call("f()")) == 0

    def test_positional_args(self) -> None:
        assert _call_argument_count(_parse_call("f(1, 2, 3)")) == 3

    def test_keyword_args(self) -> None:
        assert _call_argument_count(_parse_call("f(x=1, y=2)")) == 2

    def test_mixed_args(self) -> None:
        assert _call_argument_count(_parse_call("f(1, x=2)")) == 2

    def test_double_star_expansion_unknown_arity(self) -> None:
        # **d hides the argument count; treating it as zero would filter on false arity.
        assert _call_argument_count(_parse_call("f(**d)")) is None

    def test_double_star_expansion_with_keyword_unknown_arity(self) -> None:
        assert _call_argument_count(_parse_call("f(x=1, **d)")) is None

    def test_star_expansion_unknown_arity(self) -> None:
        # *args hides the argument count; the Starred node is not one argument.
        assert _call_argument_count(_parse_call("f(*a)")) is None

    def test_star_expansion_after_positional_unknown_arity(self) -> None:
        assert _call_argument_count(_parse_call("f(1, *a)")) is None


# ---------------------------------------------------------------------------
# Tests - _count_non_variadic_parameters
# ---------------------------------------------------------------------------


class TestCountNonVariadicParameters:
    """Tests for _count_non_variadic_parameters."""

    def test_signature_absent(self) -> None:
        fn = _make_fn("foo", "mod")
        assert _count_non_variadic_parameters(fn) is None

    def test_simple_parameters(self) -> None:
        sig = _make_sig(
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.POSITIONAL_OR_KEYWORD,
        )
        fn = _make_fn("foo", "mod", signature=sig)
        assert _count_non_variadic_parameters(fn) == 2

    def test_variadic_parameters_excluded(self) -> None:
        sig = _make_sig(
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.VAR_POSITIONAL,
            ParameterType.VAR_KEYWORD,
        )
        fn = _make_fn("foo", "mod", signature=sig)
        assert _count_non_variadic_parameters(fn) == 1

    def test_self_excluded(self) -> None:
        sig = _make_sig(
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.POSITIONAL_OR_KEYWORD,
            names=["self", "x"],
        )
        fn = _make_fn("Cls.foo", "mod", signature=sig)
        assert _count_non_variadic_parameters(fn) == 1

    def test_cls_excluded(self) -> None:
        sig = _make_sig(
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.POSITIONAL_OR_KEYWORD,
            names=["cls", "x"],
        )
        fn = _make_fn("Cls.foo", "mod", signature=sig)
        assert _count_non_variadic_parameters(fn) == 1

    def test_keyword_only_counted(self) -> None:
        sig = _make_sig(
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.KEYWORD_ONLY,
        )
        fn = _make_fn("foo", "mod", signature=sig)
        assert _count_non_variadic_parameters(fn) == 2


# ---------------------------------------------------------------------------
# Tests - _count_required_parameters
# ---------------------------------------------------------------------------


class TestCountRequiredParameters:
    """Tests for _count_required_parameters."""

    def test_signature_absent(self) -> None:
        assert _count_required_parameters(_make_fn("foo", "mod")) is None

    def test_defaults_excluded(self) -> None:
        """foo(a, b=1, c=2) requires only one argument."""
        sig = _make_sig(
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.POSITIONAL_OR_KEYWORD,
            names=["a", "b", "c"],
            defaults=["<absent>", "1", "2"],
        )
        fn = _make_fn("foo", "mod", signature=sig)
        assert _count_required_parameters(fn) == 1
        assert _count_non_variadic_parameters(fn) == 3

    def test_keyword_only_without_default_is_required(self) -> None:
        sig = _make_sig(
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.KEYWORD_ONLY,
            names=["a", "flag"],
        )
        assert _count_required_parameters(_make_fn("foo", "mod", signature=sig)) == 2


# ---------------------------------------------------------------------------
# Tests - _arity_compatible
# ---------------------------------------------------------------------------


class TestArityCompatible:
    """Tests for _arity_compatible - interval [required, total]."""

    def test_signature_absent_incompatible(self) -> None:
        """Without a signature there is no information, so the function is never selected."""
        fn = _make_fn("foo", "mod")
        assert _arity_compatible(fn, 1) is False

    def test_exact_required_parameters(self) -> None:
        sig = _make_sig(ParameterType.POSITIONAL_OR_KEYWORD, ParameterType.POSITIONAL_OR_KEYWORD)
        fn = _make_fn("foo", "mod", signature=sig)
        assert _arity_compatible(fn, 2) is True
        assert _arity_compatible(fn, 1) is False
        assert _arity_compatible(fn, 3) is False

    def test_parameter_with_default_is_optional(self) -> None:
        """foo(a, b=1) accepts 1 or 2 arguments."""
        sig = _make_sig(
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.POSITIONAL_OR_KEYWORD,
            names=["a", "b"],
            defaults=["<absent>", "1"],
        )
        fn = _make_fn("foo", "mod", signature=sig)
        assert _arity_compatible(fn, 1) is True
        assert _arity_compatible(fn, 2) is True
        assert _arity_compatible(fn, 0) is False
        assert _arity_compatible(fn, 3) is False

    def test_non_literal_default_remains_optional(self) -> None:
        """A SPECIAL_VALUE default (call or expression) remains a default."""
        sig = _make_sig(
            ParameterType.POSITIONAL_OR_KEYWORD,
            names=["a"],
            defaults=["<special>"],
        )
        fn = _make_fn("foo", "mod", signature=sig)
        assert _arity_compatible(fn, 0) is True

    def test_variadic_without_upper_bound(self) -> None:
        """foo(a, *args) accepts any number of arguments >= 1."""
        sig = _make_sig(ParameterType.POSITIONAL_OR_KEYWORD, ParameterType.VAR_POSITIONAL)
        fn = _make_fn("foo", "mod", signature=sig)
        assert _arity_compatible(fn, 1) is True
        assert _arity_compatible(fn, 7) is True
        assert _arity_compatible(fn, 0) is False

    def test_self_ignored(self) -> None:
        """The self parameter does not count toward a method call's arity."""
        sig = _make_sig(
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.POSITIONAL_OR_KEYWORD,
            names=["self", "x"],
        )
        fn = _make_fn("Cls.foo", "mod", signature=sig)
        assert _arity_compatible(fn, 1) is True
        assert _arity_compatible(fn, 2) is False


# ---------------------------------------------------------------------------
# Tests - _resolve_contextually (criteria 1-4)
# ---------------------------------------------------------------------------


class TestResolveContextually:
    """Tests for _IndexProduction._resolve_contextually."""

    def test_criterion_one_parent_module_equals_test_module(self) -> None:
        """Criterion 1: parent_module == test_module selects the sole candidate."""
        fn_a = _make_fn("foo", "pkg.sub")
        fn_b = _make_fn("foo", "pkg.other")
        result = _IndexProduction._resolve_contextually([fn_a, fn_b], "foo", test_module="pkg.sub")
        assert result is fn_a

    def test_criterion_one_parent_module_prefixes_test_module(self) -> None:
        """Criterion 1: parent_module prefixes test_module, so it is selected."""
        fn_a = _make_fn("foo", "pkg")
        fn_b = _make_fn("foo", "other.pkg")
        result = _IndexProduction._resolve_contextually([fn_a, fn_b], "foo", test_module="pkg.tests.test_foo")
        assert result is fn_a

    def test_criterion_two_matches_argument_count(self) -> None:
        """Criterion 2: only the candidate with two non-variadic parameters is selected for two args."""
        sig_two = _make_sig(ParameterType.POSITIONAL_OR_KEYWORD, ParameterType.POSITIONAL_OR_KEYWORD)
        sig_three = _make_sig(
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.POSITIONAL_OR_KEYWORD,
        )
        fn_a = _make_fn("foo", "pkg.a", signature=sig_two)
        fn_b = _make_fn("foo", "pkg.b", signature=sig_three)
        result = _IndexProduction._resolve_contextually([fn_a, fn_b], "foo", test_module="tests.bar", n_args=2)
        assert result is fn_a

    def test_criterion_two_skipped_when_argument_count_is_none(self) -> None:
        """Criterion 2 is skipped when n_args is None (capture unavailable)."""
        sig_two = _make_sig(ParameterType.POSITIONAL_OR_KEYWORD, ParameterType.POSITIONAL_OR_KEYWORD)
        # fn_a: private, with a signature that would match if n_args=2 were supplied.
        fn_a = _make_fn("foo", "pkg.a", visibility=Visibility.PRIVATE, signature=sig_two)
        # fn_b: only public candidate; criterion 3 must select it.
        fn_b = _make_fn("foo", "pkg.b", visibility=Visibility.PUBLIC)
        fn_c = _make_fn("foo", "pkg.c", visibility=Visibility.PRIVATE)
        # Without n_args, criterion 2 is skipped; criterion 3 selects fn_b.
        result = _IndexProduction._resolve_contextually([fn_a, fn_b, fn_c], "foo", test_module="tests.bar", n_args=None)
        assert result is fn_b

    def test_criterion_three_selects_only_public(self) -> None:
        """Criterion 3: only the public candidate is selected."""
        fn_pub = _make_fn("foo", "pkg.a", visibility=Visibility.PUBLIC)
        fn_priv = _make_fn("foo", "pkg.b", visibility=Visibility.PRIVATE)
        fn_prot = _make_fn("foo", "pkg.c", visibility=Visibility.PROTECTED)
        result = _IndexProduction._resolve_contextually([fn_pub, fn_priv, fn_prot], "foo", test_module="tests.x")
        assert result is fn_pub

    def test_criterion_four_selects_longest_common_prefix(self) -> None:
        """Criterion 4: select the candidate with the longest common prefix."""
        near_function = _make_fn("foo", "alpha.beta.gamma")
        far_function = _make_fn("foo", "alpha.delta")
        result = _IndexProduction._resolve_contextually(
            [near_function, far_function], "foo", test_module="alpha.beta.tests.test_foo"
        )
        assert result is near_function

    def test_criterion_two_selects_candidate_with_default(self) -> None:
        """A one-argument call selects foo(a, b=1) and excludes foo(x, y).

        Counting all non-variadic parameters gave two for both candidates: neither
        matched, the criterion filtered nothing, and resolution remained ambiguous.
        """
        default_signature = _make_sig(
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.POSITIONAL_OR_KEYWORD,
            names=["a", "b"],
            defaults=["<absent>", "1"],
        )
        two_required_signature = _make_sig(ParameterType.POSITIONAL_OR_KEYWORD, ParameterType.POSITIONAL_OR_KEYWORD)
        fn_a = _make_fn("foo", "pkg.a", signature=default_signature)
        fn_b = _make_fn("foo", "pkg.b", signature=two_required_signature)

        result = _IndexProduction._resolve_contextually([fn_a, fn_b], "foo", test_module="tests.x", n_args=1)
        assert result is fn_a

    def test_criterion_two_accepts_large_variadic_call(self) -> None:
        """A three-argument call selects foo(*args) and excludes foo(a)."""
        variadic_signature = _make_sig(ParameterType.VAR_POSITIONAL, names=["args"])
        one_parameter_signature = _make_sig(ParameterType.POSITIONAL_OR_KEYWORD, names=["a"])
        fn_a = _make_fn("foo", "pkg.a", signature=variadic_signature)
        fn_b = _make_fn("foo", "pkg.b", signature=one_parameter_signature)

        result = _IndexProduction._resolve_contextually([fn_a, fn_b], "foo", test_module="tests.x", n_args=3)
        assert result is fn_a

    def test_criterion_two_excludes_too_many_arguments(self) -> None:
        """The interval has an upper bound: foo(a, b=1) does not accept three args."""
        default_signature = _make_sig(
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.POSITIONAL_OR_KEYWORD,
            names=["a", "b"],
            defaults=["<absent>", "1"],
        )
        three_parameter_signature = _make_sig(
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.POSITIONAL_OR_KEYWORD,
            ParameterType.POSITIONAL_OR_KEYWORD,
        )
        fn_a = _make_fn("foo", "pkg.a", signature=default_signature)
        fn_b = _make_fn("foo", "pkg.b", signature=three_parameter_signature)

        result = _IndexProduction._resolve_contextually([fn_a, fn_b], "foo", test_module="tests.x", n_args=3)
        assert result is fn_b

    def test_criterion_four_stops_at_divergence(self) -> None:
        """A matching segment after divergence does not count toward the common prefix.

        'a.x.c' shares one segment with 'a.b.c', not two. Without stopping at
        divergence, it would tie 'a.b.z' and create artificial ambiguity.
        """
        wrong_function = _make_fn("foo", "a.x.c")
        right_function = _make_fn("foo", "a.b.z")

        result = _IndexProduction._resolve_contextually([wrong_function, right_function], "foo", test_module="a.b.c")
        assert result is right_function

    def test_still_ambiguous_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """After the four criteria, return None and log a warning if still ambiguous."""
        fn_a = _make_fn("foo", "alpha.beta")
        fn_b = _make_fn("foo", "alpha.beta")
        with caplog.at_level(logging.WARNING):
            result = _IndexProduction._resolve_contextually([fn_a, fn_b], "foo", test_module="tests.x")
        assert result is None
        assert "Ambiguous resolution" in caplog.text

    def test_unreduced_criterion_continues_to_next(self) -> None:
        """If criterion 1 does not reduce the candidates, criterion 2 takes over."""
        one_signature = _make_sig(ParameterType.POSITIONAL_OR_KEYWORD)
        two_signature = _make_sig(ParameterType.POSITIONAL_OR_KEYWORD, ParameterType.POSITIONAL_OR_KEYWORD)
        fn_a = _make_fn("foo", "pkg.a", signature=one_signature)
        fn_b = _make_fn("foo", "pkg.b", signature=two_signature)
        # test_module "tests.x" is prefixed by neither "pkg.a" nor "pkg.b"; skip criterion 1.
        # n_args=1 lets criterion 2 select fn_a.
        result = _IndexProduction._resolve_contextually([fn_a, fn_b], "foo", test_module="tests.x", n_args=1)
        assert result is fn_a


# ---------------------------------------------------------------------------
# Integration tests for _resolve_unique.
# ---------------------------------------------------------------------------


class TestResolveUnique:
    """Tests for _IndexProduction._resolve_unique and contextual criteria."""

    def test_zero_candidates(self) -> None:
        assert _IndexProduction._resolve_unique([], "foo") is None

    def test_one_candidate(self) -> None:
        fn = _make_fn("foo", "pkg")
        assert _IndexProduction._resolve_unique([fn], "foo") is fn

    def test_multiple_candidates_without_test_module_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """Without test_module, multiple candidates produce a warning and return None."""
        fn_a = _make_fn("foo", "pkg.a")
        fn_b = _make_fn("foo", "pkg.b")
        with caplog.at_level(logging.WARNING):
            result = _IndexProduction._resolve_unique([fn_a, fn_b], "foo")
        assert result is None
        assert "Ambiguous resolution" in caplog.text

    def test_multiple_candidates_with_test_module_are_disambiguated(self) -> None:
        """With test_module, contextual disambiguation returns the correct candidate."""
        fn_a = _make_fn("foo", "pkg.sub")
        fn_b = _make_fn("foo", "pkg.other")
        result = _IndexProduction._resolve_unique([fn_a, fn_b], "foo", test_module="pkg.sub.tests")
        assert result is fn_a


# ---------------------------------------------------------------------------
# Tests - find_by_unique_simple_name with disambiguation
# ---------------------------------------------------------------------------


class TestProductionIndexDisambiguation:
    """Tests for find_by_unique_simple_name with contextual disambiguation."""

    def _build_index(self, function_records: list[FunctionRecord]) -> _IndexProduction:
        index = _IndexProduction()
        for fn in function_records:
            _indexer_function(fn, fn.parent_module, index)
        return index

    def test_unique_simple_name_is_disambiguated_by_module(self) -> None:
        """find_by_unique_simple_name resolves ambiguity with test_module."""
        fn_a = _make_fn("foo", "pkg.moduleA")
        fn_b = _make_fn("foo", "pkg.moduleB")
        index = self._build_index([fn_a, fn_b])
        result = index.find_by_unique_simple_name("foo", test_module="pkg.moduleA.tests")
        assert result is fn_a

    def test_ambiguous_simple_name_without_test_module_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """Without test_module, an ambiguous name returns None with a warning."""
        fn_a = _make_fn("foo", "pkg.a")
        fn_b = _make_fn("foo", "pkg.b")
        index = self._build_index([fn_a, fn_b])
        with caplog.at_level(logging.WARNING):
            result = index.find_by_unique_simple_name("foo")
        assert result is None
        assert "Ambiguous resolution" in caplog.text


# ---------------------------------------------------------------------------
# Tests - unpacked calls, from visitor to produced links
# ---------------------------------------------------------------------------


class TestUnpackedCalls:
    """An `f(*args)` / `f(**kwargs)` call must not use invented arity for disambiguation.

    Two same-named candidates have different arities and cannot be separated by criteria 1, 3, and 4.
    Only criterion 2 can decide. Counting the `Starred` node as an argument selected the one-parameter
    candidate and created an invented link instead of preserving the ambiguity.
    """

    def _index(self) -> _IndexProduction:
        one_parameter_signature = _make_sig(ParameterType.POSITIONAL_OR_KEYWORD)
        three_parameter_signature = _make_sig(*[ParameterType.POSITIONAL_OR_KEYWORD] * 3)
        index = _IndexProduction()
        for fn in (
            _make_fn("process", "pkg.a", signature=one_parameter_signature),
            _make_fn("process", "pkg.b", signature=three_parameter_signature),
        ):
            _indexer_function(fn, fn.parent_module, index)
        return index

    def _link_keys(self, body: str) -> list[str]:
        module = ModuleRecord(file_path=Path("/tests/test_x.py"), module_name="tests.test_x", is_test=True)
        visitor = _TestLinksVisitor(module, self._index())
        visitor.visit(ast.parse(f"def test_x():\n    {body}\n"))
        return [mapping_key for mapping_key, _ in visitor.links]

    def test_explicit_call_is_selected_by_arity(self) -> None:
        """A known arity always separates the candidates on the nominal path."""
        assert self._link_keys("process(1)") == ["pkg.a.process"]

    def test_star_args_produce_no_link(self) -> None:
        """`*args`: unknown arity skips criterion 2 and preserves ambiguity instead of a false link."""
        assert self._link_keys("process(*args)") == []

    def test_double_star_kwargs_produce_no_link(self) -> None:
        """`**kwargs` behaves likewise; literal arity zero would incorrectly exclude both candidates."""
        assert self._link_keys("process(**kwargs)") == []
