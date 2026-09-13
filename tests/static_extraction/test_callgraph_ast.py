# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Unit tests for callgraph_ast: local pass, global pass, and integration."""

from __future__ import annotations

import textwrap
from pathlib import Path

from docmethis_extract_python.static_extraction.callgraph_ast import (
    _CALLGRAPH_MAX_DEPTH,
    _analyze_callgraph,
    _local_pass,
    _module_mapping,
    _resolve_name,
    build_callgraph,
)
from docmethis_extract_python.static_extraction.models import (
    CallgraphEdge,
    ClassHierarchy,
    ClassRecord,
    Confidence,
    FunctionRecord,
    ImportRecord,
    MethodType,
    ModuleRecord,
    ProjectRecord,
    Visibility,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fn(
    qname: str,
    file_path: Path = Path("pkg/mod.py"),
    line_start: int = 1,
    line_end: int = 5,
    parent_class: str | None = None,
    parent_module: str = "pkg.mod",
    method_kind: MethodType | None = None,
) -> FunctionRecord:
    kind = method_kind or (MethodType.INSTANCE_METHOD if parent_class else MethodType.FUNCTION)
    return FunctionRecord(
        qualified_name=qname,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        col_start=0,
        col_end=0,
        method_kind=kind,
        visibility=Visibility.PUBLIC,
        parent_class=parent_class,
        parent_module=parent_module,
        existing_docstring=None,
    )


def _cls(
    qname: str,
    methods: list[FunctionRecord] | None = None,
    hierarchy: ClassHierarchy | None = None,
    file_path: Path = Path("pkg/mod.py"),
) -> ClassRecord:
    return ClassRecord(
        qualified_name=qname,
        file_path=file_path,
        line_start=1,
        line_end=20,
        col_start=0,
        col_end=0,
        visibility=Visibility.PUBLIC,
        parent_module="pkg.mod",
        existing_docstring=None,
        methods=methods or [],
        hierarchy=hierarchy,
    )


def _module(
    module_name: str,
    functions: list[FunctionRecord] | None = None,
    classes: list[ClassRecord] | None = None,
    imports: list[ImportRecord] | None = None,
    file_path: Path = Path("pkg/mod.py"),
    *,
    is_stub: bool = False,
) -> ModuleRecord:
    return ModuleRecord(
        file_path=file_path,
        module_name=module_name,
        functions=functions or [],
        classes=classes or [],
        imports=imports or [],
        is_stub=is_stub,
    )


def _edge(
    source_qname: str,
    target_qname: str,
    *,
    call_type: str = "direct",
    is_external: bool = False,
    external_lib: str | None = None,
    line: int = 1,
) -> CallgraphEdge:
    return CallgraphEdge(
        source_qname=source_qname,
        target_qname=target_qname,
        call_type=call_type,
        is_external=is_external,
        external_lib=external_lib,
        line=line,
    )


# ---------------------------------------------------------------------------
# Tests _module_mapping
# ---------------------------------------------------------------------------


class TestMappingModule:
    """Unit tests for _module_mapping."""

    def test_local_functions(self) -> None:
        fn_a = _fn("pkg.mod.foo")
        module = _module("pkg.mod", functions=[fn_a])
        m = _module_mapping(module)
        assert m.get("foo") == "pkg.mod.foo"
        assert m.get("pkg.mod.foo") == "pkg.mod.foo"

    def test_import_from(self) -> None:
        imp = ImportRecord(module="os.path", name="join", alias=None)
        module = _module("pkg.mod", imports=[imp])
        m = _module_mapping(module)
        assert m.get("join") == "os.path.join"

    def test_import_as(self) -> None:
        imp = ImportRecord(module="os.path", name="join", alias="pjoin")
        module = _module("pkg.mod", imports=[imp])
        m = _module_mapping(module)
        assert m.get("pjoin") == "os.path.join"

    def test_import_module(self) -> None:
        imp = ImportRecord(module="os", name=None, alias=None)
        module = _module("pkg.mod", imports=[imp])
        m = _module_mapping(module)
        assert m.get("os") == "os"

    def test_local_class(self) -> None:
        cls = _cls("pkg.mod.Foo")
        module = _module("pkg.mod", classes=[cls])
        m = _module_mapping(module)
        assert m.get("Foo") == "pkg.mod.Foo"

    def test_submodule_import(self) -> None:
        """import pkg.submod -> key 'pkg', value 'pkg' (first segment)."""  # noqa: D403
        imp = ImportRecord(module="pkg.submod", name=None, alias=None)
        module = _module("pkg.mod", imports=[imp])
        m = _module_mapping(module)
        assert m.get("pkg") == "pkg"

    def test_submodule_import_alias(self) -> None:
        """import pkg.submod as sub -> key 'sub', value 'sub'."""  # noqa: D403
        imp = ImportRecord(module="pkg.submod", name=None, alias="sub")
        module = _module("pkg.mod", imports=[imp])
        m = _module_mapping(module)
        assert m.get("sub") == "sub"

    def test_method_via_class_dot(self) -> None:
        m_fn = _fn("pkg.mod.Foo.bar", parent_class="pkg.mod.Foo")
        cls = _cls("pkg.mod.Foo", methods=[m_fn])
        module = _module("pkg.mod", classes=[cls])
        m = _module_mapping(module)
        assert m.get("Foo.bar") == "pkg.mod.Foo.bar"


# ---------------------------------------------------------------------------
# Tests _resolve_name
# ---------------------------------------------------------------------------


class TestResolveName:
    """Unit tests for _resolve_name."""

    def setup_method(self) -> None:
        self.fn_a = _fn("pkg.mod.foo")
        self.fn_b = _fn("pkg.mod.bar")
        self.index_fn = {
            "pkg.mod.foo": self.fn_a,
            "pkg.mod.bar": self.fn_b,
        }
        self.mapping = {
            "foo": "pkg.mod.foo",
            "bar": "pkg.mod.bar",
            "os": "os",
        }

    def test_direct_local_match(self) -> None:
        qname, is_ext, ext_lib = _resolve_name("foo", self.mapping, self.index_fn)
        assert qname == "pkg.mod.foo"
        assert not is_ext
        assert ext_lib is None

    def test_direct_external_match(self) -> None:
        qname, is_ext, ext_lib = _resolve_name("os", self.mapping, self.index_fn)
        assert qname == "os"
        assert is_ext
        assert ext_lib == "os"

    def test_first_segment(self) -> None:
        qname, is_ext, ext_lib = _resolve_name("os.path", self.mapping, self.index_fn)
        assert qname == "os.path"
        assert is_ext
        assert ext_lib == "os"

    def test_external_fallback(self) -> None:
        _qname, is_ext, ext_lib = _resolve_name("unknown.func", self.mapping, self.index_fn)
        assert is_ext
        assert ext_lib == "unknown"

    def test_external_library_first_segment(self) -> None:
        _, is_ext, ext_lib = _resolve_name("numpy.array", self.mapping, self.index_fn)
        assert is_ext
        assert ext_lib == "numpy"


# ---------------------------------------------------------------------------
# Tests _analyze_callgraph - recursion
# ---------------------------------------------------------------------------


class TestAnalyzeCallgraphRecursion:
    """Tests for recursion detection through _analyze_callgraph."""

    def setup_method(self) -> None:
        self.index_fn = {
            "pkg.mod.f": _fn("pkg.mod.f"),
            "pkg.mod.g": _fn("pkg.mod.g"),
            "pkg.mod.h": _fn("pkg.mod.h"),
        }

    def test_direct_recursion(self) -> None:
        callgraph = {"pkg.mod.f": [_edge("pkg.mod.f", "pkg.mod.f")]}
        analysis = _analyze_callgraph(callgraph, self.index_fn, _CALLGRAPH_MAX_DEPTH)
        assert analysis.nodes["pkg.mod.f"].is_recursive is True

    def test_no_recursion(self) -> None:
        callgraph = {
            "pkg.mod.f": [_edge("pkg.mod.f", "pkg.mod.g")],
            "pkg.mod.g": [],
        }
        analysis = _analyze_callgraph(callgraph, self.index_fn, _CALLGRAPH_MAX_DEPTH)
        assert analysis.nodes["pkg.mod.f"].is_recursive is False
        assert analysis.nodes["pkg.mod.g"].is_recursive is False

    def test_indirect_recursion(self) -> None:
        # f -> g -> f: indirect cycle.
        callgraph = {
            "pkg.mod.f": [_edge("pkg.mod.f", "pkg.mod.g")],
            "pkg.mod.g": [_edge("pkg.mod.g", "pkg.mod.f")],
        }
        analysis = _analyze_callgraph(callgraph, self.index_fn, _CALLGRAPH_MAX_DEPTH)
        assert analysis.nodes["pkg.mod.f"].is_recursive is True
        assert analysis.nodes["pkg.mod.g"].is_recursive is True

    def test_long_indirect_recursion(self) -> None:
        # f -> g -> h -> f.
        callgraph = {
            "pkg.mod.f": [_edge("pkg.mod.f", "pkg.mod.g")],
            "pkg.mod.g": [_edge("pkg.mod.g", "pkg.mod.h")],
            "pkg.mod.h": [_edge("pkg.mod.h", "pkg.mod.f")],
        }
        analysis = _analyze_callgraph(callgraph, self.index_fn, _CALLGRAPH_MAX_DEPTH)
        assert analysis.nodes["pkg.mod.f"].is_recursive is True
        assert analysis.nodes["pkg.mod.g"].is_recursive is True
        assert analysis.nodes["pkg.mod.h"].is_recursive is True

    def test_recursion_does_not_mark_nodes_outside_cycle(self) -> None:
        # h2 -> f -> g -> f (f <-> g cycle), but h2 is outside the cycle.
        index = {**self.index_fn, "pkg.mod.h2": _fn("pkg.mod.h2")}
        callgraph = {
            "pkg.mod.h2": [_edge("pkg.mod.h2", "pkg.mod.f")],
            "pkg.mod.f": [_edge("pkg.mod.f", "pkg.mod.g")],
            "pkg.mod.g": [_edge("pkg.mod.g", "pkg.mod.f")],
        }
        analysis = _analyze_callgraph(callgraph, index, _CALLGRAPH_MAX_DEPTH)
        assert analysis.nodes["pkg.mod.f"].is_recursive is True
        assert analysis.nodes["pkg.mod.g"].is_recursive is True
        assert analysis.nodes["pkg.mod.h2"].is_recursive is False


# ---------------------------------------------------------------------------
# Tests _analyze_callgraph - depths
# ---------------------------------------------------------------------------


class TestAnalyzeCallgraphDepths:
    """Tests for depth calculation through _analyze_callgraph."""

    def setup_method(self) -> None:
        self.index_fn = {
            "pkg.mod.f": _fn("pkg.mod.f"),
            "pkg.mod.g": _fn("pkg.mod.g"),
            "pkg.mod.h": _fn("pkg.mod.h"),
        }

    def test_leaf(self) -> None:
        callgraph = {"pkg.mod.f": []}
        analysis = _analyze_callgraph(callgraph, self.index_fn, _CALLGRAPH_MAX_DEPTH)
        assert analysis.nodes["pkg.mod.f"].depth == 0

    def test_depth_one(self) -> None:
        callgraph = {
            "pkg.mod.f": [_edge("pkg.mod.f", "pkg.mod.g")],
            "pkg.mod.g": [],
        }
        analysis = _analyze_callgraph(callgraph, self.index_fn, _CALLGRAPH_MAX_DEPTH)
        assert analysis.nodes["pkg.mod.f"].depth == 1
        assert analysis.nodes["pkg.mod.g"].depth == 0

    def test_depth_two(self) -> None:
        callgraph = {
            "pkg.mod.f": [_edge("pkg.mod.f", "pkg.mod.g")],
            "pkg.mod.g": [_edge("pkg.mod.g", "pkg.mod.h")],
            "pkg.mod.h": [],
        }
        analysis = _analyze_callgraph(callgraph, self.index_fn, _CALLGRAPH_MAX_DEPTH)
        assert analysis.nodes["pkg.mod.f"].depth == 2

    def test_cap_max_depth(self) -> None:
        # A chain of _CALLGRAPH_MAX_DEPTH + 2 functions is capped at _CALLGRAPH_MAX_DEPTH.
        fns = {f"pkg.mod.f{i}": _fn(f"pkg.mod.f{i}") for i in range(_CALLGRAPH_MAX_DEPTH + 2)}
        names = list(fns.keys())
        callgraph = {names[i]: [_edge(names[i], names[i + 1])] for i in range(len(names) - 1)}
        callgraph[names[-1]] = []
        analysis = _analyze_callgraph(callgraph, fns, _CALLGRAPH_MAX_DEPTH)
        assert analysis.nodes[names[0]].depth == _CALLGRAPH_MAX_DEPTH

    def test_cycle_does_not_block(self) -> None:
        # Cycle f -> g -> f must not loop forever.
        callgraph = {
            "pkg.mod.f": [_edge("pkg.mod.f", "pkg.mod.g")],
            "pkg.mod.g": [_edge("pkg.mod.g", "pkg.mod.f")],
        }
        analysis = _analyze_callgraph(callgraph, self.index_fn, _CALLGRAPH_MAX_DEPTH)
        assert isinstance(analysis.nodes["pkg.mod.f"].depth, int)
        assert isinstance(analysis.nodes["pkg.mod.g"].depth, int)

    def test_external_calls_ignored(self) -> None:
        callgraph = {
            "pkg.mod.f": [_edge("pkg.mod.f", "os.path.join", is_external=True, external_lib="os")],
        }
        analysis = _analyze_callgraph(callgraph, self.index_fn, _CALLGRAPH_MAX_DEPTH)
        assert analysis.nodes["pkg.mod.f"].depth == 0


# ---------------------------------------------------------------------------
# Local pass tests (temporary files).
# ---------------------------------------------------------------------------


class TestLocalPass:
    """Unit tests for _local_pass."""

    def test_call_direct_local(self, tmp_path: Path) -> None:
        src = textwrap.dedent("""\
            def foo():
                pass

            def bar():
                foo()
        """)
        f = tmp_path / "mod.py"
        f.write_text(src, encoding="utf-8")

        fn_foo = _fn("pkg.mod.foo", file_path=f, line_start=1, line_end=2, parent_module="pkg.mod")
        fn_bar = _fn("pkg.mod.bar", file_path=f, line_start=4, line_end=5, parent_module="pkg.mod")
        module = _module("pkg.mod", functions=[fn_foo, fn_bar], file_path=f)
        project_record = ProjectRecord(project_root=tmp_path, project_name="test", modules=[module])

        index_fn = {fn.qualified_name: fn for fn in [fn_foo, fn_bar]}
        index_cls = {}

        cg = _local_pass(project_record, index_fn, index_cls)

        callees_bar = [e.target_qname for e in cg.get("pkg.mod.bar", [])]
        assert "pkg.mod.foo" in callees_bar

    def test_detects_external_call(self, tmp_path: Path) -> None:
        src = textwrap.dedent("""\
            import os

            def foo():
                os.getcwd()
        """)
        f = tmp_path / "mod.py"
        f.write_text(src, encoding="utf-8")

        fn_foo = _fn("pkg.mod.foo", file_path=f, line_start=3, line_end=4, parent_module="pkg.mod")
        imp = ImportRecord(module="os", name=None, alias=None)
        module = _module("pkg.mod", functions=[fn_foo], imports=[imp], file_path=f)
        project_record = ProjectRecord(project_root=tmp_path, project_name="test", modules=[module])
        index_fn = {"pkg.mod.foo": fn_foo}
        index_cls = {}

        cg = _local_pass(project_record, index_fn, index_cls)

        edges = cg.get("pkg.mod.foo", [])
        ext_edges = [e for e in edges if e.is_external]
        assert any(e.external_lib == "os" for e in ext_edges)

    def test_call_qualifie_call_type(self, tmp_path: Path) -> None:
        src = textwrap.dedent("""\
            import os

            def foo():
                os.getcwd()
        """)
        f = tmp_path / "mod.py"
        f.write_text(src, encoding="utf-8")

        fn_foo = _fn("pkg.mod.foo", file_path=f, line_start=3, line_end=4, parent_module="pkg.mod")
        imp = ImportRecord(module="os", name=None, alias=None)
        module = _module("pkg.mod", functions=[fn_foo], imports=[imp], file_path=f)
        project_record = ProjectRecord(project_root=tmp_path, project_name="test", modules=[module])
        index_fn = {"pkg.mod.foo": fn_foo}
        index_cls = {}

        cg = _local_pass(project_record, index_fn, index_cls)

        edges = cg.get("pkg.mod.foo", [])
        assert any(e.call_type == "qualified_call" for e in edges)

    def test_local_submodule_call_is_not_external(self, tmp_path: Path) -> None:
        """import pkg.submod + pkg.submod.fn() produces a local edge."""  # noqa: D403
        src = textwrap.dedent("""\
            import pkg.submod

            def foo():
                pkg.submod.helper()
        """)
        f = tmp_path / "mod.py"
        f.write_text(src, encoding="utf-8")

        fn_helper = _fn("pkg.submod.helper", file_path=f, line_start=1, line_end=2, parent_module="pkg.submod")
        fn_foo = _fn("pkg.mod.foo", file_path=f, line_start=3, line_end=4, parent_module="pkg.mod")
        imp = ImportRecord(module="pkg.submod", name=None, alias=None)
        module = _module("pkg.mod", functions=[fn_foo], imports=[imp], file_path=f)
        project_record = ProjectRecord(project_root=tmp_path, project_name="test", modules=[module])
        index_fn = {"pkg.mod.foo": fn_foo, "pkg.submod.helper": fn_helper}
        index_cls = {}

        cg = _local_pass(project_record, index_fn, index_cls)

        edges = cg.get("pkg.mod.foo", [])
        assert any(e.target_qname == "pkg.submod.helper" and not e.is_external for e in edges)

    def test_self_method_resolved_via_mro(self, tmp_path: Path) -> None:
        src = textwrap.dedent("""\
            class Foo:
                def helper(self):
                    pass

                def run(self):
                    self.helper()
        """)
        f = tmp_path / "mod.py"
        f.write_text(src, encoding="utf-8")

        fn_helper = _fn(
            "pkg.mod.Foo.helper", file_path=f, line_start=2, line_end=3, parent_class="pkg.mod.Foo", parent_module="pkg.mod"
        )
        fn_run = _fn(
            "pkg.mod.Foo.run", file_path=f, line_start=5, line_end=6, parent_class="pkg.mod.Foo", parent_module="pkg.mod"
        )

        hierarchy = ClassHierarchy(direct_parents=[], mro_list=["pkg.mod.Foo", "object"], is_abstract=False)
        cls_foo = _cls("pkg.mod.Foo", methods=[fn_helper, fn_run], hierarchy=hierarchy, file_path=f)
        module = _module("pkg.mod", classes=[cls_foo], file_path=f)
        project_record = ProjectRecord(project_root=tmp_path, project_name="test", modules=[module])

        index_fn = {fn.qualified_name: fn for fn in [fn_helper, fn_run]}
        index_cls = {"pkg.mod.Foo": cls_foo}

        cg = _local_pass(project_record, index_fn, index_cls)

        callees_run = [e.target_qname for e in cg.get("pkg.mod.Foo.run", [])]
        assert "pkg.mod.Foo.helper" in callees_run

    def test_super_method_resolved(self, tmp_path: Path) -> None:
        src = textwrap.dedent("""\
            class Base:
                def setup(self):
                    pass

            class Child(Base):
                def setup(self):
                    super().setup()
        """)
        f = tmp_path / "mod.py"
        f.write_text(src, encoding="utf-8")

        fn_base_setup = _fn(
            "pkg.mod.Base.setup", file_path=f, line_start=2, line_end=3, parent_class="pkg.mod.Base", parent_module="pkg.mod"
        )
        fn_child_setup = _fn(
            "pkg.mod.Child.setup", file_path=f, line_start=6, line_end=7, parent_class="pkg.mod.Child", parent_module="pkg.mod"
        )

        hier_base = ClassHierarchy(direct_parents=[], mro_list=["pkg.mod.Base", "object"], is_abstract=False)
        hier_child = ClassHierarchy(
            direct_parents=["pkg.mod.Base"], mro_list=["pkg.mod.Child", "pkg.mod.Base", "object"], is_abstract=False
        )

        cls_base = _cls("pkg.mod.Base", methods=[fn_base_setup], hierarchy=hier_base, file_path=f)
        cls_child = _cls("pkg.mod.Child", methods=[fn_child_setup], hierarchy=hier_child, file_path=f)

        module = _module("pkg.mod", classes=[cls_base, cls_child], file_path=f)
        project_record = ProjectRecord(project_root=tmp_path, project_name="test", modules=[module])

        index_fn = {fn.qualified_name: fn for fn in [fn_base_setup, fn_child_setup]}
        index_cls = {"pkg.mod.Base": cls_base, "pkg.mod.Child": cls_child}

        cg = _local_pass(project_record, index_fn, index_cls)

        callees = [e.target_qname for e in cg.get("pkg.mod.Child.setup", [])]
        assert "pkg.mod.Base.setup" in callees

    def test_staticmethod_first_parameter_is_not_self(self, tmp_path: Path) -> None:
        """@staticmethod: the first parameter is not self.

        other.run() must not be resolved through Foo's MRO.
        Treat it as a normal qualified call instead.
        """
        src = textwrap.dedent("""\
            class Foo:
                def run(self):
                    pass

                @staticmethod
                def process(other):
                    other.run()
        """)
        f = tmp_path / "mod.py"
        f.write_text(src, encoding="utf-8")

        fn_run = _fn(
            "pkg.mod.Foo.run", file_path=f, line_start=2, line_end=3, parent_class="pkg.mod.Foo", parent_module="pkg.mod"
        )
        fn_process = _fn(
            "pkg.mod.Foo.process",
            file_path=f,
            line_start=6,
            line_end=7,
            parent_class="pkg.mod.Foo",
            parent_module="pkg.mod",
            method_kind=MethodType.STATIC_METHOD,
        )

        hier = ClassHierarchy(direct_parents=[], mro_list=["pkg.mod.Foo", "object"], is_abstract=False)
        cls_foo = _cls("pkg.mod.Foo", methods=[fn_run, fn_process], hierarchy=hier, file_path=f)
        module = _module("pkg.mod", classes=[cls_foo], file_path=f)
        project_record = ProjectRecord(project_root=tmp_path, project_name="test", modules=[module])
        index_fn = {fn.qualified_name: fn for fn in [fn_run, fn_process]}
        index_cls = {"pkg.mod.Foo": cls_foo}

        cg = _local_pass(project_record, index_fn, index_cls)

        edges = cg.get("pkg.mod.Foo.process", [])
        # No via_self edge to Foo.run should be generated.
        via_self_vers_run = [e for e in edges if e.call_type == "via_self" and e.target_qname == "pkg.mod.Foo.run"]
        assert not via_self_vers_run


# ---------------------------------------------------------------------------
# Tests for a module shipped as .py + .pyi.
# ---------------------------------------------------------------------------


class TestStubAndImplementation:
    """A module with implementation and stub: the stub must not evict the implementation.

    `resolve_module_name()` ignores the extension, so `pkg/mod.py` and `pkg/mod.pyi` share
    the same `module_name` and their functions share `qualified_name`. Indexing ASTs by
    `module_name` paired the `.py` FunctionRecords with the stub AST - divergent lines,
    missing nodes - then replaced real edges with the stub's empty edges.
    """

    IMPL = textwrap.dedent("""\
        def foo():
            pass


        def bar():
            foo()
    """)
    # Lines intentionally differ from the implementation; this difference made the AST
    # node unavailable when both files were conflated.
    STUB = textwrap.dedent("""\
        def foo() -> None: ...
        def bar() -> None: ...
    """)

    def _project(self, tmp_path: Path) -> tuple[ProjectRecord, FunctionRecord]:
        impl = tmp_path / "mod.py"
        impl.write_text(self.IMPL, encoding="utf-8")
        stub = tmp_path / "mod.pyi"
        stub.write_text(self.STUB, encoding="utf-8")

        fn_foo = _fn("pkg.mod.foo", file_path=impl, line_start=1, line_end=2)
        fn_bar = _fn("pkg.mod.bar", file_path=impl, line_start=5, line_end=6)
        module_impl = _module("pkg.mod", functions=[fn_foo, fn_bar], file_path=impl)

        stub_foo = _fn("pkg.mod.foo", file_path=stub, line_start=1, line_end=1)
        stub_bar = _fn("pkg.mod.bar", file_path=stub, line_start=2, line_end=2)
        module_stub = _module("pkg.mod", functions=[stub_foo, stub_bar], file_path=stub, is_stub=True)

        # Discovery order sorts "mod.py" before "mod.pyi", so the stub arrived last.
        project_record = ProjectRecord(project_root=tmp_path, project_name="test", modules=[module_impl, module_stub])
        return project_record, fn_bar

    def test_implementation_edges_are_preserved(self, tmp_path: Path) -> None:
        """The `.py` bar -> foo edge survives the stub."""
        project_record, _ = self._project(tmp_path)
        index_fn = {fn.qualified_name: fn for module in project_record.modules for fn in module.functions}

        cg = _local_pass(project_record, index_fn, {})

        assert [e.target_qname for e in cg.get("pkg.mod.bar", [])] == ["pkg.mod.foo"]

    def test_implementation_records_are_enriched(self, tmp_path: Path) -> None:
        """End to end, the `.py` FunctionRecord receives the edges."""
        project_record, fn_bar = self._project(tmp_path)

        build_callgraph(project_record)

        assert [e.target_qname for e in fn_bar.callgraph] == ["pkg.mod.foo"]
        assert fn_bar.callgraph_confidence == Confidence.EXPLICIT

    def test_stub_without_implementation_is_still_analyzed(self, tmp_path: Path) -> None:
        """A standalone stub is the only module source and is not discarded."""
        stub = tmp_path / "mod.pyi"
        stub.write_text(self.STUB, encoding="utf-8")
        stub_bar = _fn("pkg.mod.bar", file_path=stub, line_start=2, line_end=2)
        module_stub = _module("pkg.mod", functions=[stub_bar], file_path=stub, is_stub=True)
        project_record = ProjectRecord(project_root=tmp_path, project_name="test", modules=[module_stub])

        cg = _local_pass(project_record, {"pkg.mod.bar": stub_bar}, {})

        assert cg["pkg.mod.bar"] == []


# ---------------------------------------------------------------------------
# build_callgraph integration tests.
# ---------------------------------------------------------------------------


class TestBuildCallgraph:
    """Integration tests for build_callgraph."""

    def test_edges_are_written_to_function_records(self, tmp_path: Path) -> None:
        src = textwrap.dedent("""\
            def foo():
                pass

            def bar():
                foo()
        """)
        f = tmp_path / "mod.py"
        f.write_text(src, encoding="utf-8")

        fn_foo = _fn("pkg.mod.foo", file_path=f, line_start=1, line_end=2, parent_module="pkg.mod")
        fn_bar = _fn("pkg.mod.bar", file_path=f, line_start=4, line_end=5, parent_module="pkg.mod")
        module = _module("pkg.mod", functions=[fn_foo, fn_bar], file_path=f)
        project_record = ProjectRecord(project_root=tmp_path, project_name="test", modules=[module])

        build_callgraph(project_record)

        assert fn_bar.callgraph is not None
        callees_bar = [e.target_qname for e in fn_bar.callgraph]
        assert "pkg.mod.foo" in callees_bar

    def test_direct_recursion_is_marked(self, tmp_path: Path) -> None:
        src = textwrap.dedent("""\
            def fib(n):
                if n <= 1:
                    return n
                return fib(n - 1) + fib(n - 2)
        """)
        f = tmp_path / "mod.py"
        f.write_text(src, encoding="utf-8")

        fn_fib = _fn("pkg.mod.fib", file_path=f, line_start=1, line_end=4, parent_module="pkg.mod")
        module = _module("pkg.mod", functions=[fn_fib], file_path=f)
        project_record = ProjectRecord(project_root=tmp_path, project_name="test", modules=[module])

        build_callgraph(project_record)

        assert fn_fib.is_recursive is True

    def test_callgraph_depth_is_calculated(self, tmp_path: Path) -> None:
        src = textwrap.dedent("""\
            def leaf():
                pass

            def mid():
                leaf()

            def top():
                mid()
        """)
        f = tmp_path / "mod.py"
        f.write_text(src, encoding="utf-8")

        fn_leaf = _fn("pkg.mod.leaf", file_path=f, line_start=1, line_end=2, parent_module="pkg.mod")
        fn_mid = _fn("pkg.mod.mid", file_path=f, line_start=4, line_end=5, parent_module="pkg.mod")
        fn_top = _fn("pkg.mod.top", file_path=f, line_start=7, line_end=8, parent_module="pkg.mod")
        module = _module("pkg.mod", functions=[fn_leaf, fn_mid, fn_top], file_path=f)
        project_record = ProjectRecord(project_root=tmp_path, project_name="test", modules=[module])

        build_callgraph(project_record)

        assert fn_leaf.callgraph_depth == 0
        assert fn_mid.callgraph_depth == 1
        assert fn_top.callgraph_depth == 2

    def test_external_edges_are_marked_external(self, tmp_path: Path) -> None:
        src = textwrap.dedent("""\
            import os

            def foo():
                os.getcwd()
                os.listdir('.')
        """)
        f = tmp_path / "mod.py"
        f.write_text(src, encoding="utf-8")

        fn_foo = _fn("pkg.mod.foo", file_path=f, line_start=3, line_end=5, parent_module="pkg.mod")
        imp = ImportRecord(module="os", name=None, alias=None)
        module = _module("pkg.mod", functions=[fn_foo], imports=[imp], file_path=f)
        project_record = ProjectRecord(project_root=tmp_path, project_name="test", modules=[module])

        build_callgraph(project_record)

        assert fn_foo.callgraph is not None
        ext_edges = [e for e in fn_foo.callgraph if e.is_external]
        assert len(ext_edges) >= 1
        assert all(e.external_lib == "os" for e in ext_edges)

    def test_confidence_is_explicit_when_edges_exist(self, tmp_path: Path) -> None:
        src = textwrap.dedent("""\
            def foo():
                pass

            def bar():
                foo()
        """)
        f = tmp_path / "mod.py"
        f.write_text(src, encoding="utf-8")

        fn_foo = _fn("pkg.mod.foo", file_path=f, line_start=1, line_end=2, parent_module="pkg.mod")
        fn_bar = _fn("pkg.mod.bar", file_path=f, line_start=4, line_end=5, parent_module="pkg.mod")
        module = _module("pkg.mod", functions=[fn_foo, fn_bar], file_path=f)
        project_record = ProjectRecord(project_root=tmp_path, project_name="test", modules=[module])

        build_callgraph(project_record)

        assert fn_bar.callgraph_confidence == Confidence.EXPLICIT
        assert fn_foo.callgraph_confidence == Confidence.INFERRED_LOW

    def test_depth_exceeded_is_marked(self, tmp_path: Path) -> None:
        # A chain of DEPTH+2 functions should mark depth_exceeded on the first edge.
        # DEPTH+2 functions are needed for f1 to reach MAX depth and mark f0 -> f1.
        n = _CALLGRAPH_MAX_DEPTH + 2
        lines = []
        for i in range(n):
            lines.append(f"def f{i}():")
            if i < n - 1:
                lines.append(f"    f{i + 1}()")
            else:
                lines.append("    pass")
        src = "\n".join(lines) + "\n"
        f = tmp_path / "mod.py"
        f.write_text(src, encoding="utf-8")

        # Chaque fonction prend 2 lignes
        fns = [
            _fn(f"pkg.mod.f{i}", file_path=f, line_start=i * 2 + 1, line_end=i * 2 + 2, parent_module="pkg.mod") for i in range(n)
        ]

        module = _module("pkg.mod", functions=fns, file_path=f)
        project_record = ProjectRecord(project_root=tmp_path, project_name="test", modules=[module])
        build_callgraph(project_record)

        # The first function should have at least one depth_exceeded edge.
        exceeded = [e for e in (fns[0].callgraph or []) if e.depth_exceeded]
        assert len(exceeded) >= 1
