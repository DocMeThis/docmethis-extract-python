# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Unit tests for ast_inheritance: C3 MRO, protocols, and attributes."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from docmethis_extract_python.static_extraction.ast_inheritance import (
    analyze_inheritance_project,
    classify_methods,
    detect_protocols,
    extract_bases_raw,
    extract_class_attributes,
    extract_instance_attributes,
    is_abstract_class,
    is_mixin,
    resolve_mro,
)
from docmethis_extract_python.static_extraction.models import (
    ClassHierarchy,
    ClassRecord,
    Confidence,
    FunctionRecord,
    MethodOrigin,
    MethodType,
    ModuleRecord,
    ProjectRecord,
    Visibility,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_class(src: str) -> ast.ClassDef:
    tree = ast.parse(textwrap.dedent(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return node
    msg = "No class found"
    raise ValueError(msg)


def _make_class_record(
    qualified_name: str,
    parent_module: str = "pkg.mod",
    method_names: list[str] | None = None,
    hierarchy: ClassHierarchy | None = None,
) -> ClassRecord:
    methods = [
        FunctionRecord(
            qualified_name=f"{qualified_name}.{m}",
            file_path=Path("x.py"),
            line_start=1,
            line_end=2,
            col_start=0,
            col_end=10,
            method_kind=MethodType.INSTANCE_METHOD,
            visibility=Visibility.PUBLIC,
            parent_class=qualified_name,
            parent_module=parent_module,
            existing_docstring=None,
        )
        for m in method_names or []
    ]
    return ClassRecord(
        qualified_name=qualified_name,
        file_path=__import__("pathlib").Path("x.py"),
        line_start=1,
        line_end=10,
        col_start=0,
        col_end=0,
        visibility=Visibility.PUBLIC,
        parent_module=parent_module,
        existing_docstring=None,
        methods=methods,
        hierarchy=hierarchy,
    )


# ---------------------------------------------------------------------------
# extract_bases_raw
# ---------------------------------------------------------------------------


def test_bases_raw_none() -> None:
    node = _parse_class("class Foo: pass")
    assert extract_bases_raw(node) == []


def test_bases_raw_simple() -> None:
    node = _parse_class("class Foo(Bar, Baz): pass")
    assert extract_bases_raw(node) == ["Bar", "Baz"]


def test_bases_raw_qualified() -> None:
    node = _parse_class("class Foo(pkg.Base): pass")
    assert extract_bases_raw(node) == ["pkg.Base"]


def test_bases_raw_excludes_metaclass() -> None:
    # metaclass= est dans node.keywords, pas node.bases
    node = _parse_class("class Foo(metaclass=ABCMeta): pass")
    assert extract_bases_raw(node) == []


# ---------------------------------------------------------------------------
# is_abstract_class
# ---------------------------------------------------------------------------


def test_abstract_via_abc() -> None:
    node = _parse_class("class Foo(ABC): pass")
    assert is_abstract_class(node) is True


def test_abstract_via_qualified_abc() -> None:
    node = _parse_class("class Foo(abc.ABC): pass")
    assert is_abstract_class(node) is True


def test_abstract_via_metaclass() -> None:
    node = _parse_class("class Foo(metaclass=ABCMeta): pass")
    assert is_abstract_class(node) is True


def test_abstract_via_abstractmethod() -> None:
    src = """
    class Foo:
        @abstractmethod
        def bar(self): ...
    """
    node = _parse_class(src)
    assert is_abstract_class(node) is True


def test_non_abstract() -> None:
    node = _parse_class("class Foo(Bar): pass")
    assert is_abstract_class(node) is False


# ---------------------------------------------------------------------------
# is_mixin
# ---------------------------------------------------------------------------


def test_is_mixin_by_name_without_init() -> None:
    src = """
    class LogMixin:
        def log(self): pass
    """
    node = _parse_class(src)
    assert is_mixin(node) is True


def test_is_mixin_with_simple_init() -> None:
    """An __init__ containing only pass is simple, so the mixin is kept (phase 7a it.10)."""
    src = """
    class LogMixin:
        def __init__(self): pass
    """
    node = _parse_class(src)
    assert is_mixin(node) is True


def test_complex_init_is_not_a_mixin() -> None:
    """An __init__ with an external call is excluded."""
    src = """
    class LogMixin:
        def __init__(self): setup()
    """
    node = _parse_class(src)
    assert is_mixin(node) is False


def test_non_mixin_name() -> None:
    node = _parse_class("class Foo: pass")
    assert is_mixin(node) is False


# ---------------------------------------------------------------------------
# extract_class_attributes
# ---------------------------------------------------------------------------


def test_class_attributes_annotated_assignment() -> None:
    src = """
    class Foo:
        x: int
        y: str = "hello"
    """
    node = _parse_class(src)
    assert extract_class_attributes(node) == ["x", "y"]


def test_class_attributes_assignment() -> None:
    src = """
    class Foo:
        MAX = 10
        DEFAULT = None
    """
    node = _parse_class(src)
    assert extract_class_attributes(node) == ["MAX", "DEFAULT"]


def test_class_attributes_exclude_methods() -> None:
    src = """
    class Foo:
        x: int
        def method(self): pass
    """
    node = _parse_class(src)
    assert extract_class_attributes(node) == ["x"]


# ---------------------------------------------------------------------------
# extract_instance_attributes
# ---------------------------------------------------------------------------


def test_attributes_instance_init() -> None:
    src = """
    class Foo:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    """
    node = _parse_class(src)
    # L'ordre peut varier selon le walk (LIFO) — on compare les ensembles.
    assert set(extract_instance_attributes(node)) == {"x", "y"}


def test_attributes_instance_multi_methods() -> None:
    src = """
    class Foo:
        def __init__(self):
            self.x = 1
        def setup(self):
            self.y = 2
    """
    node = _parse_class(src)
    assert "x" in extract_instance_attributes(node)
    assert "y" in extract_instance_attributes(node)


def test_instance_attributes_have_no_duplicates() -> None:
    src = """
    class Foo:
        def __init__(self):
            self.x = 1
        def reset(self):
            self.x = 0
    """
    node = _parse_class(src)
    result = extract_instance_attributes(node)
    assert result.count("x") == 1


def test_instance_attributes_ignore_nested_functions() -> None:
    src = """
    class Foo:
        def __init__(self):
            def inner():
                self.hidden = True
            self.x = 1
    """
    node = _parse_class(src)
    result = extract_instance_attributes(node)
    assert "x" in result
    assert "hidden" not in result


def test_attributes_instance_annassign() -> None:
    src = """
    class Foo:
        def __init__(self):
            self.x: int = 0
    """
    node = _parse_class(src)
    assert "x" in extract_instance_attributes(node)


# ---------------------------------------------------------------------------
# detect_protocols
# ---------------------------------------------------------------------------


def test_detect_iterator() -> None:
    cls = _make_class_record("pkg.mod.Iter", method_names=["__iter__", "__next__"])
    protocols = detect_protocols(cls, {})
    names = [p.name for p in protocols]
    assert "Iterator" in names


def test_detect_iterable_only() -> None:
    cls = _make_class_record("pkg.mod.Iter", method_names=["__iter__"])
    protocols = detect_protocols(cls, {})
    names = [p.name for p in protocols]
    assert "Iterable" in names
    assert "Iterator" not in names


def test_detect_context_manager() -> None:
    cls = _make_class_record("pkg.mod.CM", method_names=["__enter__", "__exit__"])
    protocols = detect_protocols(cls, {})
    assert any(p.name == "ContextManager" for p in protocols)


def test_detect_async_context_manager() -> None:
    cls = _make_class_record("pkg.mod.ACM", method_names=["__aenter__", "__aexit__"])
    protocols = detect_protocols(cls, {})
    assert any(p.name == "AsyncContextManager" for p in protocols)


def test_detect_callable() -> None:
    cls = _make_class_record("pkg.mod.F", method_names=["__call__"])
    assert any(p.name == "Callable" for p in detect_protocols(cls, {}))


def test_detect_none_protocol() -> None:
    cls = _make_class_record("pkg.mod.Plain", method_names=["run", "stop"])
    assert detect_protocols(cls, {}) == []


def test_protocol_dunder_methods_are_correct() -> None:
    cls = _make_class_record("pkg.mod.Iter", method_names=["__iter__", "__next__", "other"])
    protocols = detect_protocols(cls, {})
    it = next(p for p in protocols if p.name == "Iterator")
    assert set(it.dunder_methods) == {"__iter__", "__next__"}


def test_protocol_detected_via_inheritance() -> None:
    # Parent provides __next__, child provides __iter__: Iterator via MRO.
    parent = _make_class_record(
        "pkg.mod.Base",
        method_names=["__next__"],
        hierarchy=ClassHierarchy(direct_parents=[], mro_list=[], is_abstract=False),
    )
    child_class = _make_class_record(
        "pkg.mod.Child",
        method_names=["__iter__"],
        hierarchy=ClassHierarchy(
            direct_parents=["pkg.mod.Base"],
            mro_list=["pkg.mod.Child", "pkg.mod.Base", "object"],
            is_abstract=False,
        ),
    )
    index = {"pkg.mod.Base": parent, "pkg.mod.Child": child_class}
    protocols = detect_protocols(child_class, index)
    assert any(p.name == "Iterator" for p in protocols)


def test_protocol_not_detected_without_index() -> None:
    # Without an index, inherited dunders are not visible.
    child_class = _make_class_record(
        "pkg.mod.Child",
        method_names=["__iter__"],
        hierarchy=ClassHierarchy(
            direct_parents=["pkg.mod.Base"],
            mro_list=["pkg.mod.Child", "pkg.mod.Base", "object"],
            is_abstract=False,
        ),
    )
    protocols = detect_protocols(child_class, {})
    assert not any(p.name == "Iterator" for p in protocols)


# ---------------------------------------------------------------------------
# resolve_mro
# ---------------------------------------------------------------------------


def _build_index(*classes: ClassRecord) -> dict[str, ClassRecord]:
    return {class_.qualified_name: class_ for class_ in classes}


def test_mro_class_without_parents() -> None:
    cls = _make_class_record("pkg.mod.Foo", hierarchy=ClassHierarchy(direct_parents=[], mro_list=[], is_abstract=False))
    index = _build_index(cls)
    mro = resolve_mro(cls, index)
    assert mro is not None
    assert mro == ["pkg.mod.Foo", "object"]


def test_mro_simple_inheritance() -> None:
    base = _make_class_record("pkg.mod.Base", hierarchy=ClassHierarchy(direct_parents=[], mro_list=[], is_abstract=False))
    child = _make_class_record(
        "pkg.mod.Child", hierarchy=ClassHierarchy(direct_parents=["pkg.mod.Base"], mro_list=[], is_abstract=False)
    )
    index = _build_index(base, child)
    mro = resolve_mro(child, index)
    assert mro is not None
    assert mro == ["pkg.mod.Child", "pkg.mod.Base", "object"]


def test_mro_multiple_inheritance() -> None:
    # Diamond : D(B, C), B(A), C(A)
    a = _make_class_record("m.A", hierarchy=ClassHierarchy(direct_parents=[], mro_list=[], is_abstract=False))
    b = _make_class_record("m.B", hierarchy=ClassHierarchy(direct_parents=["m.A"], mro_list=[], is_abstract=False))
    c = _make_class_record("m.C", hierarchy=ClassHierarchy(direct_parents=["m.A"], mro_list=[], is_abstract=False))
    d = _make_class_record("m.D", hierarchy=ClassHierarchy(direct_parents=["m.B", "m.C"], mro_list=[], is_abstract=False))
    index = _build_index(a, b, c, d)
    mro = resolve_mro(d, index)
    assert mro is not None
    # C3 for the diamond: D, B, C, A, object.
    assert mro == ["m.D", "m.B", "m.C", "m.A", "object"]


def test_mro_external_class() -> None:
    cls = _make_class_record(
        "pkg.mod.Foo", hierarchy=ClassHierarchy(direct_parents=["external.Base"], mro_list=[], is_abstract=False)
    )
    index = _build_index(cls)
    mro = resolve_mro(cls, index)
    assert mro is not None
    assert "pkg.mod.Foo" in mro
    assert "external.Base" in mro


def test_mro_starts_with_class() -> None:
    cls = _make_class_record("pkg.mod.Foo", hierarchy=ClassHierarchy(direct_parents=[], mro_list=[], is_abstract=False))
    mro = resolve_mro(cls, _build_index(cls))
    assert mro is not None
    assert mro[0] == "pkg.mod.Foo"


def test_mro_ends_with_object() -> None:
    cls = _make_class_record("pkg.mod.Foo", hierarchy=ClassHierarchy(direct_parents=[], mro_list=[], is_abstract=False))
    mro = resolve_mro(cls, _build_index(cls))
    assert mro is not None
    assert mro[-1] == "object"


def test_mro_cycle_returns_none() -> None:
    """An inheritance cycle returns None (inconsistent MRO)."""
    a = _make_class_record("m.A", hierarchy=ClassHierarchy(direct_parents=["m.B"], mro_list=[], is_abstract=False))
    b = _make_class_record("m.B", hierarchy=ClassHierarchy(direct_parents=["m.A"], mro_list=[], is_abstract=False))
    index = _build_index(a, b)
    assert resolve_mro(a, index) is None


# ---------------------------------------------------------------------------
# analyze_inheritance_project - integration
# ---------------------------------------------------------------------------


def _build_project(*modules: ModuleRecord) -> ProjectRecord:
    return ProjectRecord(
        project_root=Path(),
        project_name="test",
        modules=list(modules),
    )


def _build_module(module_name: str, *classes: ClassRecord) -> ModuleRecord:
    return ModuleRecord(
        file_path=Path(f"{module_name.replace('.', '/')}.py"),
        module_name=module_name,
        classes=list(classes),
    )


def test_analyze_inheritance_project_populates_mro() -> None:
    base = _make_class_record("pkg.mod.Base", parent_module="pkg.mod")
    child = _make_class_record("pkg.mod.Child", parent_module="pkg.mod")
    # Set the hierarchy with unresolved raw parents.
    base.hierarchy = ClassHierarchy(direct_parents=[], mro_list=[], is_abstract=False)
    child.hierarchy = ClassHierarchy(direct_parents=["pkg.mod.Base"], mro_list=[], is_abstract=False)

    project_record = _build_project(_build_module("pkg.mod", base, child))
    analyze_inheritance_project(project_record)

    assert child.hierarchy.mro_list[0] == "pkg.mod.Child"
    assert "pkg.mod.Base" in child.hierarchy.mro_list
    assert child.hierarchy.mro_list[-1] == "object"


def test_analyze_inheritance_project_detects_protocols() -> None:
    class_ = _make_class_record("pkg.mod.MyIter", parent_module="pkg.mod", method_names=["__iter__", "__next__"])
    class_.hierarchy = ClassHierarchy(direct_parents=[], mro_list=[], is_abstract=False)

    project_record = _build_project(_build_module("pkg.mod", class_))
    analyze_inheritance_project(project_record)

    assert any(p.name == "Iterator" for p in class_.protocols)


def test_analyze_inheritance_project_confidence_explicit() -> None:
    class_ = _make_class_record("pkg.mod.Foo", parent_module="pkg.mod")
    class_.hierarchy = ClassHierarchy(direct_parents=[], mro_list=[], is_abstract=False)

    project_record = _build_project(_build_module("pkg.mod", class_))
    analyze_inheritance_project(project_record)

    assert class_.hierarchy_confidence is Confidence.EXPLICIT


def test_analyze_inheritance_project_confidence_inferred_low_on_cycle() -> None:
    """A class with an inheritance cycle has an empty mro_list: INFERRED_LOW."""
    a = _make_class_record("m.A", parent_module="m")
    b = _make_class_record("m.B", parent_module="m")
    a.hierarchy = ClassHierarchy(direct_parents=["m.B"], mro_list=[], is_abstract=False)
    b.hierarchy = ClassHierarchy(direct_parents=["m.A"], mro_list=[], is_abstract=False)

    project_record = _build_project(_build_module("m", a, b))
    analyze_inheritance_project(project_record)

    assert a.hierarchy_confidence is Confidence.INFERRED_LOW


def test_analyze_inheritance_project_resolves_local_name() -> None:
    """The raw name 'Base' is resolved to 'pkg.mod.Base' through local classes."""
    base = _make_class_record("pkg.mod.Base", parent_module="pkg.mod")
    base.hierarchy = ClassHierarchy(direct_parents=[], mro_list=[], is_abstract=False)

    child = _make_class_record("pkg.mod.Child", parent_module="pkg.mod")
    child.hierarchy = ClassHierarchy(direct_parents=["Base"], mro_list=[], is_abstract=False)

    project_record = _build_project(_build_module("pkg.mod", base, child))
    analyze_inheritance_project(project_record)

    assert "pkg.mod.Base" in child.hierarchy.direct_parents


# ---------------------------------------------------------------------------
# classify_methods tests
# ---------------------------------------------------------------------------


def test_classifier_methods_new_without_parents() -> None:
    """A method on a class without parents is NEW."""
    class_ = _make_class_record("m.Foo", method_names=["run"])
    class_.hierarchy = ClassHierarchy(direct_parents=[], mro_list=["m.Foo", "object"], is_abstract=False)
    index: dict[str, ClassRecord] = {"m.Foo": class_}

    classify_methods(class_, index)

    method = class_.methods[0]
    assert method.method_origin is MethodOrigin.NEW


def test_classifier_methods_overridden() -> None:
    """A method present on a parent is OVERRIDDEN."""
    base = _make_class_record("m.Base", method_names=["run"])
    base.hierarchy = ClassHierarchy(direct_parents=[], mro_list=["m.Base", "object"], is_abstract=False)

    child = _make_class_record("m.Child", method_names=["run"])
    child.hierarchy = ClassHierarchy(direct_parents=["m.Base"], mro_list=["m.Child", "m.Base", "object"], is_abstract=False)
    index = {"m.Base": base, "m.Child": child}

    classify_methods(child, index)

    assert child.methods[0].method_origin is MethodOrigin.OVERRIDDEN


def test_classifier_methods_new_method_absent_from_parent() -> None:
    """A method absent from all parents is NEW."""
    base = _make_class_record("m.Base", method_names=["setup"])
    base.hierarchy = ClassHierarchy(direct_parents=[], mro_list=["m.Base", "object"], is_abstract=False)

    child = _make_class_record("m.Child", method_names=["run"])
    child.hierarchy = ClassHierarchy(direct_parents=["m.Base"], mro_list=["m.Child", "m.Base", "object"], is_abstract=False)
    index = {"m.Base": base, "m.Child": child}

    classify_methods(child, index)

    assert child.methods[0].method_origin is MethodOrigin.NEW


def test_classifier_methods_mixed() -> None:
    """NEW and OVERRIDDEN methods can coexist in one class."""
    base = _make_class_record("m.Base", method_names=["setup"])
    base.hierarchy = ClassHierarchy(direct_parents=[], mro_list=["m.Base", "object"], is_abstract=False)

    child = _make_class_record("m.Child", method_names=["setup", "run"])
    child.hierarchy = ClassHierarchy(direct_parents=["m.Base"], mro_list=["m.Child", "m.Base", "object"], is_abstract=False)
    index = {"m.Base": base, "m.Child": child}

    classify_methods(child, index)

    origins = {m.qualified_name.split(".")[-1]: m.method_origin for m in child.methods}
    assert origins["setup"] is MethodOrigin.OVERRIDDEN
    assert origins["run"] is MethodOrigin.NEW


def test_classifier_methods_without_hierarchy_does_not_fail() -> None:
    """A class without a hierarchy is unchanged (method_origin remains None)."""
    class_ = _make_class_record("m.Foo", method_names=["run"])
    index: dict[str, ClassRecord] = {"m.Foo": class_}

    classify_methods(class_, index)

    assert class_.methods[0].method_origin is None


def test_classifier_methods_unknown_parent_is_ignored() -> None:
    """A parent absent from the index is ignored without error; the method is NEW."""
    child = _make_class_record("m.Child", method_names=["run"])
    child.hierarchy = ClassHierarchy(
        direct_parents=["m.ExternalBase"], mro_list=["m.Child", "m.ExternalBase", "object"], is_abstract=False
    )
    index: dict[str, ClassRecord] = {"m.Child": child}

    classify_methods(child, index)

    assert child.methods[0].method_origin is MethodOrigin.NEW


def test_analyze_inheritance_project_classifies_methods() -> None:
    """analyze_inheritance_project calls classify_methods and populates method_origin."""
    base = _make_class_record("pkg.mod.Base", method_names=["run"], parent_module="pkg.mod")
    base.hierarchy = ClassHierarchy(direct_parents=[], mro_list=[], is_abstract=False)

    child = _make_class_record("pkg.mod.Child", method_names=["run", "extra"], parent_module="pkg.mod")
    child.hierarchy = ClassHierarchy(direct_parents=["pkg.mod.Base"], mro_list=[], is_abstract=False)

    project_record = _build_project(_build_module("pkg.mod", base, child))
    analyze_inheritance_project(project_record)

    origins = {m.qualified_name.split(".")[-1]: m.method_origin for m in child.methods}
    assert origins["run"] is MethodOrigin.OVERRIDDEN
    assert origins["extra"] is MethodOrigin.NEW
