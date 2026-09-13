# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Data models for static extraction - the contract between Modules 1 and 2.

Defines shared enums (Confidence, Provenance) and dataclasses
(FunctionRecord, ClassRecord, ModuleRecord, ProjectRecord) enriched by each
iteration.
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Sentinels - missing or non-trivial values in string fields
# ---------------------------------------------------------------------------

MISSING_VALUE: str = "<absent>"  # missing annotation, return, or default
SPECIAL_VALUE: str = "<special>"  # non-trivial default value (call, expression)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def name_simple(qualified_name: str) -> str:
    """Return the last part of a qualified name (for example, 'a.b.foo' -> 'foo').

    Parameters
    ----------
    qualified_name : str
        The qualified name from which to extract the last part as the simple name.

    Returns
    -------
    str
        The last component of the qualified name, obtained by splitting on dots and returning the final element.

    """
    return qualified_name.rsplit(".", maxsplit=1)[-1]


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------


class Confidence(enum.Enum):
    """Confidence level of extracted data.

    Attributes
    ----------
    EXPLICIT : Confidence
        Value explicitly present in the source, such as a type annotation.
    INFERRED_HIGH : Confidence
        Inferred with high confidence.
    INFERRED_LOW : Confidence
        Inferred with low confidence.
    ABSENT : Confidence
        Value absent from the source.

    """

    EXPLICIT = "explicit"
    INFERRED_HIGH = "inferred_high"
    INFERRED_LOW = "inferred_low"
    ABSENT = "absent"


class Provenance(enum.Enum):
    """Source of extracted data.

    Attributes
    ----------
    ANNOTATION : Provenance
        Extracted from a type annotation.
    STUB : Provenance
        Extracted from a type stub.
    LOCAL_INFERENCE : Provenance
        Inferred locally from the function body.
    TEST : Provenance
        Collected from test execution.
    TRACE : Provenance
        Collected from runtime tracing.
    FUZZING : Provenance
        Collected from fuzzing.
    CALLGRAPH_PROPAGATION : Provenance
        Propagated through the callgraph.

    """

    ANNOTATION = "annotation"
    STUB = "stub"
    LOCAL_INFERENCE = "local_inference"
    TEST = "test"
    TRACE = "trace"
    FUZZING = "fuzzing"
    CALLGRAPH_PROPAGATION = "callgraph_propagation"  # it.8: exception propagated through callgraph


class MethodType(enum.Enum):
    """Function or method type.

    Attributes
    ----------
    FUNCTION : MethodType
        Standalone function.
    INSTANCE_METHOD : MethodType
        Instance method taking an implicit ``self``.
    CLASS_METHOD : MethodType
        Class method taking an implicit ``cls``.
    STATIC_METHOD : MethodType
        Static method without an implicit receiver.
    PROPERTY : MethodType
        Property accessor.

    """

    FUNCTION = "function"
    INSTANCE_METHOD = "instance_method"
    CLASS_METHOD = "classmethod"
    STATIC_METHOD = "staticmethod"
    PROPERTY = "property"


class PropertyAccessor(enum.Enum):
    """Accessor role for a property function."""

    GETTER = "getter"
    SETTER = "setter"


def implicit_receiver_exclusion_reason(*, parameter_name: str, method_kind: MethodType) -> str | None:
    """Return the documentation exclusion reason for an implicit receiver.

    Parameters
    ----------
    parameter_name : str
        The name of the implicit receiver parameter to check, such as 'self' or 'cls'.
    method_kind : MethodType
        The kind of method being evaluated, used to determine whether a parameter should be treated as an implicit receiver based
        on its name.

    Returns
    -------
    str | None
        The exclusion reason for an implicit receiver, or None if the parameter is not an implicit receiver for the given method
        kind.

    """
    if method_kind is MethodType.INSTANCE_METHOD and parameter_name == "self":
        return "implicit_instance_receiver"

    if method_kind is MethodType.PROPERTY and parameter_name == "self":
        return "implicit_property_receiver"

    if method_kind is MethodType.CLASS_METHOD and parameter_name == "cls":
        return "implicit_class_receiver"

    return None


class Visibility(enum.Enum):
    """Visibility of a Python symbol.

    Attributes
    ----------
    PUBLIC : Visibility
        Public symbol.
    PROTECTED : Visibility
        Protected symbol (single leading underscore).
    PRIVATE : Visibility
        Private symbol (double leading underscore).

    """

    PUBLIC = "public"
    PROTECTED = "protected"
    PRIVATE = "private"


class ParameterType(enum.Enum):
    """Parameter kind in a function signature.

    Attributes
    ----------
    POSITIONAL_ONLY : ParameterType
        Positional-only parameter.
    POSITIONAL_OR_KEYWORD : ParameterType
        Parameter accepted positionally or by keyword.
    KEYWORD_ONLY : ParameterType
        Keyword-only parameter.
    VAR_POSITIONAL : ParameterType
        Variadic positional parameter (*args).
    VAR_KEYWORD : ParameterType
        Variadic keyword parameter (**kwargs).

    """

    POSITIONAL_ONLY = "positional_only"
    POSITIONAL_OR_KEYWORD = "positional_or_keyword"
    KEYWORD_ONLY = "keyword_only"
    VAR_POSITIONAL = "var_positional"
    VAR_KEYWORD = "var_keyword"


class MethodOrigin(enum.Enum):
    """Method origin relative to the class hierarchy (iteration 8).

    - NEW: absent from all MRO parents
    - OVERRIDDEN: redefines a method inherited from a parent
    - INHERITED: provided by a parent and not redefined in this class

    Attributes
    ----------
    NEW : MethodOrigin
        Absent from all MRO parents.
    OVERRIDDEN : MethodOrigin
        Redefines a method inherited from a parent.
    INHERITED : MethodOrigin
        Provided by a parent and not redefined in this class.

    """

    NEW = "new"
    OVERRIDDEN = "overridden"
    INHERITED = "inherited"


# ---------------------------------------------------------------------------
# Dataclasses - domain types for progressive fields
# ---------------------------------------------------------------------------


@dataclass
class Parameter:
    """Function or method parameter.

    Attributes
    ----------
    name : str
        Name
    annotation_raw : str
        MISSING_VALUE when unannotated
    default_value : str
        MISSING_VALUE, SPECIAL_VALUE, or a literal (for example, "42", "None")
    kind : ParameterType
        Kind
    line_start : int | None
        Line start

    """

    name: str
    annotation_raw: str  # MISSING_VALUE when unannotated
    default_value: str  # MISSING_VALUE, SPECIAL_VALUE, or a literal (for example, "42", "None")
    kind: ParameterType
    line_start: int | None = None

    def __post_init__(self) -> None:
        """Normalize the `kind` field to the ParameterType enum."""
        if isinstance(self.kind, str):
            self.kind = ParameterType(self.kind)


@dataclass
class Decorator:
    """Decorator applied to a function or method.

    Attributes
    ----------
    full_decorator : str
        Ex: "functools.lru_cache(maxsize=128)"
    is_known : bool
        True for property/staticmethod/classmethod/abstractmethod

    """

    full_decorator: str  # ex: "functools.lru_cache(maxsize=128)"
    is_known: bool  # True for property/staticmethod/classmethod/abstractmethod


@dataclass
class Signature:
    """Signature extracted from a function or method (iteration 1).

    Attributes
    ----------
    parameters : list[Parameter]
        List of parameter records.
    return_annotation : str
        MISSING_VALUE when unannotated
    decorators : list[Decorator]
        Decorators

    """

    parameters: list[Parameter]
    return_annotation: str  # MISSING_VALUE when unannotated
    decorators: list[Decorator]


@dataclass
class Complexity:
    """Function complexity metrics (iteration 3).

    Attributes
    ----------
    loc : int
        Lines of code excluding the docstring
    cyclomatic : int
        McCabe cyclomatic complexity (base = 1)
    branches : int
        Explicit branches: +1 per if/elif, terminal else, and match_case
    max_depth : int
        Maximum nesting depth
    statements : int
        Statement nodes in the direct body
    cyclomatic_percentile : float | None
        Cyclomatic percentile
    loc_percentile : float | None
        Loc percentile
    max_depth_percentile : float | None
        Max depth percentile

    """

    loc: int  # Lines of code excluding the docstring
    cyclomatic: int  # McCabe cyclomatic complexity (base = 1)
    branches: int  # Explicit branches: +1 per if/elif, terminal else, and match_case
    max_depth: int  # Maximum nesting depth
    statements: int  # Statement nodes in the direct body

    # Project-level percentiles (None before project percentile calculation)
    cyclomatic_percentile: float | None = None
    loc_percentile: float | None = None
    max_depth_percentile: float | None = None


@dataclass
class TypeInfo:
    """Resolved type for a function parameter or return value (iteration 2).

    Attributes
    ----------
    type_str : str
        Normalized PEP 585/604 form - never "absent"
    confidence : Confidence
        EXPLICIT | INFERRED_HIGH | INFERRED_LOW
    provenance : Provenance
        ANNOTATION | STUB | LOCAL_INFERENCE
    raw : str | None
        Original source annotation (when provenance=ANNOTATION)

    """

    type_str: str  # Normalized PEP 585/604 form - never "absent"
    confidence: Confidence  # EXPLICIT | INFERRED_HIGH | INFERRED_LOW
    provenance: Provenance  # ANNOTATION | STUB | LOCAL_INFERENCE
    raw: str | None = None  # Original source annotation (when provenance=ANNOTATION)


@dataclass
class ResolvedTypes:
    """Resolved types for all function parameters and the return value (iteration 2).

    Attributes
    ----------
    parameters : dict[str, TypeInfo]
        Parameter name -> TypeInfo (absent when unresolved)
    return_type : TypeInfo | None
        None when the return type is unresolved

    """

    parameters: dict[str, TypeInfo]  # parameter name -> TypeInfo (absent when unresolved)
    return_type: TypeInfo | None = None  # None when the return type is unresolved


@dataclass
class IODetail:
    """Concrete occurrence of an I/O pattern detected in a function body.

    Attributes
    ----------
    pattern : str
        Text description, for example, "open(..., 'w')"
    line : int
        Source-file line number
    category : str
        "reads_files" | "writes_files" | "network_access" | "system_calls" | "mutates_global" | "has_output"

    """

    pattern: str  # Text description, for example, "open(..., 'w')"
    line: int  # Source-file line number
    category: str  # "reads_files" | "writes_files" | "network_access" | "system_calls" | "mutates_global" | "has_output"


@dataclass
class IOEffects:
    """Side effects and I/O operations detected in a function (iteration 3).

    Attributes
    ----------
    reads_files : bool
        Reads files
    writes_files : bool
        Writes files
    network_access : bool
        Network access
    system_calls : bool
        System calls
    mutates_global : bool
        Mutates global
    has_output : bool
        Has output
    details : list[IODetail]
        Details

    """

    reads_files: bool = False
    writes_files: bool = False
    network_access: bool = False
    system_calls: bool = False
    mutates_global: bool = False
    has_output: bool = False
    details: list[IODetail] = field(default_factory=list)


@dataclass
class RaisedCondition:
    """Syntactic condition under which a raise is reached (iteration 7).

    Attributes
    ----------
    expression : str
        Ast.unparse of the enclosing if test or deterministic combination of nested tests
    line : int
        Closest enclosing if statement line

    """

    expression: str  # ast.unparse of the enclosing if test
    line: int  # if statement line


@dataclass
class ExceptionRecord:
    """Exception explicitly raised in a function (iteration 7).

    Attributes
    ----------
    exception_type : str
        Exception name, for example, "ValueError", "MyError"
    message : str | None
        Message string when inferred, otherwise None
    raise_line : int
        Raise statement line
    is_reraise : bool
        True for a bare `raise` in an except
    chained_from : bool
        True for `raise X from e`
    condition : RaisedCondition | None
        Present when the raise is on a deterministic if branch
    provenance : Provenance
        It.8: CALLGRAPH_PROPAGATION when propagated

    """

    exception_type: str  # exception name, for example, "ValueError", "MyError"
    message: str | None  # message string when inferred, otherwise None
    raise_line: int  # raise statement line
    is_reraise: bool = False  # True for a bare `raise` in an except
    chained_from: bool = False  # True for `raise X from e`
    condition: RaisedCondition | None = None  # present on a deterministic if branch
    provenance: Provenance = Provenance.LOCAL_INFERENCE  # it.8: CALLGRAPH_PROPAGATION when propagated


@dataclass
class ExceptBlock:
    """Except block extracted from a function body (iteration 7).

    ``try_start``/``try_end``: line range of the enclosing ``try`` body
    (added in iteration C10 to correlate a raise with its handlers).

    Attributes
    ----------
    caught_types : list[str]
        Caught types, for example, ["ValueError", "TypeError"]; [] for bare except
    is_reraised : bool
        True when the handler contains a bare `raise`
    is_swallowed : bool
        True when the handler is pass/... without a re-raise
    line : int
        Except line
    try_start : int
        First line of the enclosing try body (0 when unknown)
    try_end : int
        Last line of the enclosing try body (0 when unknown)

    """

    caught_types: list[str]  # caught types, for example, ["ValueError", "TypeError"]; [] for bare except
    is_reraised: bool  # True when the handler contains a bare `raise`
    is_swallowed: bool  # True when the handler is pass/... without a re-raise
    line: int  # except line
    try_start: int = 0  # first line of the enclosing try body (0 when unknown)
    try_end: int = 0  # last line of the enclosing try body (0 when unknown)


@dataclass
class ObservableProperty:
    """Observable property detected by heuristic analysis (iteration 7).

    Categories: "precondition" | "postcondition" | "invariant".

    Attributes
    ----------
    category : str
        Category
    raw_expression : str
        Ast.unparse of the expression or condition
    line : int
        Line
    confidence : Confidence
        Confidence
    related_exception : str | None
        Exception type raised in the same if, when present

    """

    category: str
    raw_expression: str  # ast.unparse of the expression or condition
    line: int
    confidence: Confidence
    related_exception: str | None = None  # exception type raised in the same if, when present


@dataclass
class TestReference:
    """Static reference from a test function to a target function (iteration 6).

    Attributes
    ----------
    test_qualified_name : str
        Qualified name in the test module, for example, "TestHelpers.test_bounds"
    test_module : str
        Test module, for example, "tests.test_interval"
    test_file : Path
        Absolute test-file path
    test_line : int
        Test-function definition line
    framework : str
        "pytest" | "unittest" | "unknown"
    call_line : int
        Call line in the test file

    """

    test_qualified_name: str  # qualified name in the test module, for example, "TestHelpers.test_bounds"
    test_module: str  # test module, for example, "tests.test_interval"
    test_file: Path  # absolute test-file path
    test_line: int  # test-function definition line
    framework: str  # "pytest" | "unittest" | "unknown"
    call_line: int  # call line in the test file


@dataclass
class TestCall:
    """Record of a direct call from a test to a production function (it.9 phase 2).

    Collected dynamically by the pytest plugin (call_collector.py).
    Feeds UsageExample snippet generation in phase 3.

    Attributes
    ----------
    function_name : str
        Qualified_name of the called function, for example, "pkg.mod.my_function"
    test_source : str
        Pytest node_id, for example, "tests/test_foo.py::test_bar"
    args : dict[str, Any]
        Serialized arguments (value or "<non_serializable>")
    result : Any | None
        Serialized return value, None for an exception or None return
    exception : str | None
        Exception type when the function raised

    """

    function_name: str  # qualified_name of the called function, for example, "pkg.mod.my_function"
    test_source: str  # pytest node_id, for example, "tests/test_foo.py::test_bar"
    args: dict[str, Any]  # serialized arguments (value or "<non_serializable>")
    result: Any | None  # serialized return value, None for an exception or None return
    exception: str | None  # exception type when the function raised


@dataclass
class TestCoverage:
    """Test coverage for a function (iterations 6 and 9).

    Static fields (test_references, is_directly_tested, test_count) are populated by
    pure AST analysis (it.6). Dynamic fields (covered_lines, uncovered_lines, ratio,
    has_any_test, has_failing_test) are populated by real execution through coverage.py
    (it.9). Direct calls (calls) are collected by the pytest plugin (it.9 phase 2).

    Attributes
    ----------
    test_references : list[TestReference]
        Test references
    is_directly_tested : bool
        True when at least one test references it directly
    test_count : int
        Number of distinct test functions
    covered_lines : list[int]
        Lines executed by tests
    uncovered_lines : list[int]
        Lines never reached
    ratio : float
        Covered / (covered + uncovered), 0.0 when no test
    has_any_test : bool
        True when at least one test covers a function line
    has_failing_test : bool
        True when at least one covering test fails
    calls : list[TestCall]
        Direct calls collected by the plugin

    """

    # --- Static fields (it.6) ---
    test_references: list[TestReference] = field(default_factory=list)
    is_directly_tested: bool = False  # True when at least one test references it directly
    test_count: int = 0  # number of distinct test functions

    # --- Dynamic coverage fields (it.9 phase 1) ---
    covered_lines: list[int] = field(default_factory=list)  # lines executed by tests
    uncovered_lines: list[int] = field(default_factory=list)  # lines never reached
    ratio: float = 0.0  # covered / (covered + uncovered), 0.0 when no test
    has_any_test: bool = False  # True when at least one test covers a function line
    has_failing_test: bool = False  # True when at least one covering test fails

    # --- Dynamic call fields (it.9 phase 2) ---
    calls: list[TestCall] = field(default_factory=list)  # direct calls collected by the plugin


@dataclass
class UsageExample:
    """Function usage example snippet (it.9 phase 3).

    Generated from dynamically collected TestCalls. Feeds the "Examples" section
    of NumPy docstrings produced by Module 2.

    Attributes
    ----------
    function : str
        Function qualified_name, for example, "pkg.mod.my_function"
    test_source : str
        Pytest node_id, for example, "tests/test_foo.py::test_bar"
    setup : str
        Minimal import to reproduce the example, for example, "from pkg.mod import my_function"
    call : str
        Call expression, for example, "my_function(1, 2)"
    result : str
        Return value as a string, for example, "3"
    provenance : str
        "test" for examples originating from tests
    confidence : str
        "high" for examples from passing tests
    format : str
        "doctest" | "raw"
    contradiction_suspected : bool
        True when an identical call has another result

    """

    function: str  # function qualified_name, for example, "pkg.mod.my_function"
    test_source: str  # pytest node_id, for example, "tests/test_foo.py::test_bar"
    setup: str  # minimal import to reproduce the example, for example, "from pkg.mod import my_function"
    call: str  # call expression, for example, "my_function(1, 2)"
    result: str  # return value as a string, for example, "3"
    provenance: str  # "test" for examples originating from tests
    confidence: str  # "high" for examples from passing tests
    format: str  # "doctest" | "raw"
    contradiction_suspected: bool = False  # True when an identical call has another result


@dataclass
class ImportRecord:
    """Direct import extracted from a module's top level.

    Attributes
    ----------
    module : str
        For example, "os.path", ".utils" (relative)
    name : str | None
        None for `import X`; imported name for `from X import name`
    alias : str | None
        None without an `as` alias
    conditional : bool
        True inside try/except ImportError
    type_checking_only : bool
        True inside if TYPE_CHECKING
    classification : Literal['stdlib', 'third_party', 'local'] | None
        None when not calculated
    is_unused : bool
        True when the imported name never appears in the module

    """

    module: str  # for example, "os.path", ".utils" (relative)
    name: str | None  # None for `import X`; imported name for `from X import name`
    alias: str | None  # None without an `as` alias
    conditional: bool = False  # True inside try/except ImportError
    type_checking_only: bool = False  # True inside if TYPE_CHECKING
    classification: Literal["stdlib", "third_party", "local"] | None = None  # None when not calculated
    is_unused: bool = False  # True when the imported name never appears in the module


# ---------------------------------------------------------------------------
# Dataclasses it.8 - inheritance and callgraph
# ---------------------------------------------------------------------------


@dataclass
class ClassHierarchy:
    """Class inheritance and MRO information (iteration 8).

    Attributes
    ----------
    direct_parents : list[str]
        Qualified names of direct parents in declaration order
    mro_list : list[str]
        C3 linearization: starts with the class and ends with "object"
    is_abstract : bool
        True for ABC or a class with an unimplemented abstractmethod
    is_mixin : bool
        Heuristic: name ends in Mixin, no __init__, few methods

    """

    direct_parents: list[str]  # qualified names of direct parents in declaration order
    mro_list: list[str]  # C3 linearization: starts with the class and ends with "object"
    is_abstract: bool  # True for ABC or a class with an unimplemented abstractmethod
    is_mixin: bool = False  # heuristic: name ends in Mixin, no __init__, few methods


@dataclass
class ImplicitProtocol:
    """Structural protocol implicitly implemented by a class (iteration 8).

    Attributes
    ----------
    name : str
        Ex: "Iterator", "ContextManager", "AsyncContextManager"
    dunder_methods : list[str]
        Dunder methods satisfying this protocol

    """

    name: str  # ex: "Iterator", "ContextManager", "AsyncContextManager"
    dunder_methods: list[str]  # dunder methods satisfying this protocol


@dataclass
class CallgraphEdge:
    """Directed edge in a function callgraph (iteration 8).

    Attributes
    ----------
    source_qname : str
        Caller qualified_name
    target_qname : str
        Callee qualified_name (or best-effort resolved name)
    call_type : Literal['direct', 'via_self', 'via_super', 'callback', 'qualified_call']
        Call type
    is_external : bool
        True when the callee belongs to an external (non-local) library
    external_lib : str | None
        First qualified-name segment when external (for example, "os", "numpy")
    line : int
        Call line in the source file
    depth_exceeded : bool
        True when truncated because maximum depth (5) was exceeded

    """

    source_qname: str  # caller qualified_name
    target_qname: str  # callee qualified_name (or best-effort resolved name)
    call_type: Literal["direct", "via_self", "via_super", "callback", "qualified_call"]
    is_external: bool  # True when the callee belongs to an external (non-local) library
    external_lib: str | None  # first qualified-name segment when external (for example, "os", "numpy")
    line: int  # call line in the source file
    depth_exceeded: bool = False  # True when truncated because maximum depth (5) was exceeded


# ---------------------------------------------------------------------------
# Dataclasses - progressively enriched records
# ---------------------------------------------------------------------------


@dataclass
class FunctionRecord:
    """Complete function or method record, enriched by each iteration.

    Identity fields are populated at creation. Progressive fields (signature,
    types, complexity, and so on) remain None with companion confidence ABSENT
    until their corresponding iteration.

    Attributes
    ----------
    qualified_name : str
        Qualified name
    file_path : Path
        File path
    line_start : int
        Line start
    line_end : int
        Line end
    col_start : int
        Col start
    col_end : int
        Col end
    method_kind : MethodType
        Method kind
    visibility : Visibility
        Visibility
    parent_class : str | None
        Parent class
    parent_module : str
        Parent module
    existing_docstring : str | None
        Existing docstring
    docstring_line_start : int | None
        Docstring line start
    is_async : bool
        Is async
    property_accessor : PropertyAccessor | None
        Property accessor role, when the function is a getter or setter.
    signature : Signature | None
        Signature
    signature_confidence : Confidence
        Confidence of the ``signature`` field.
    types : ResolvedTypes | None
        Types
    types_confidence : Confidence
        Confidence of the ``types`` field.
    complexity : Complexity | None
        Complexity
    complexity_confidence : Confidence
        Confidence of the ``complexity`` field.
    io_effects : IOEffects | None
        Io effects
    io_effects_confidence : Confidence
        Confidence of the ``io_effects`` field.
    source_code : str | None
        Source code
    exceptions : list[ExceptionRecord] | None
        Exceptions
    exceptions_confidence : Confidence
        Confidence of the ``exceptions`` field.
    except_blocks : list[ExceptBlock] | None
        Except blocks
    observable_properties : list[ObservableProperty] | None
        Observable properties
    observable_properties_confidence : Confidence
        Confidence of the ``observable_properties`` field.
    callgraph : list[CallgraphEdge] | None
        Callgraph
    callgraph_confidence : Confidence
        Confidence of the ``callgraph`` field.
    is_recursive : bool
        Is recursive
    callgraph_depth : int
        Maximum reachable depth; -1 when not calculated
    method_origin : MethodOrigin | None
        Method origin
    test_coverage : TestCoverage | None
        Test coverage
    test_coverage_confidence : Confidence
        Confidence of the ``test_coverage`` field.
    examples : list[UsageExample] | None
        Usage examples generated from test calls.
    examples_confidence : Confidence
        Confidence of the ``examples`` field.
    has_overloads : bool
        Has overloads
    overload_signatures : list[Signature]
        Overload signatures
    traces : Any
        Traces
    traces_confidence : Confidence
        Confidence of the ``traces`` field.
    profiling : Any
        Profiling
    profiling_confidence : Confidence
        Confidence of the ``profiling`` field.
    from_cache : bool
        From cache
    file_hash : str
        File hash
    analysis_timestamp : str
        Analysis timestamp

    """

    # --- Identity (always populated) ---
    qualified_name: str
    file_path: Path
    line_start: int
    line_end: int
    col_start: int
    col_end: int
    method_kind: MethodType
    visibility: Visibility
    parent_class: str | None
    parent_module: str
    existing_docstring: str | None
    docstring_line_start: int | None = None
    is_async: bool = False
    property_accessor: PropertyAccessor | None = None

    # --- Progressive fields (populated by later iterations) ---
    # Each field has a companion _confidence field.
    # Any will be replaced by the concrete type in the relevant iteration.

    signature: Signature | None = None
    signature_confidence: Confidence = field(default=Confidence.ABSENT)

    types: ResolvedTypes | None = None
    types_confidence: Confidence = field(default=Confidence.ABSENT)

    complexity: Complexity | None = None
    complexity_confidence: Confidence = field(default=Confidence.ABSENT)

    io_effects: IOEffects | None = None
    io_effects_confidence: Confidence = field(default=Confidence.ABSENT)

    source_code: str | None = None

    exceptions: list[ExceptionRecord] | None = None
    exceptions_confidence: Confidence = field(default=Confidence.ABSENT)

    except_blocks: list[ExceptBlock] | None = None

    observable_properties: list[ObservableProperty] | None = None
    observable_properties_confidence: Confidence = field(default=Confidence.ABSENT)

    # it.8: callgraph and recursion
    callgraph: list[CallgraphEdge] | None = None
    callgraph_confidence: Confidence = field(default=Confidence.ABSENT)
    is_recursive: bool = False
    callgraph_depth: int = -1  # maximum reachable depth; -1 when not calculated

    # it.8: origin in the class hierarchy (None for functions outside a class)
    method_origin: MethodOrigin | None = None

    test_coverage: TestCoverage | None = None
    test_coverage_confidence: Confidence = field(default=Confidence.ABSENT)

    examples: list[UsageExample] | None = None
    examples_confidence: Confidence = field(default=Confidence.ABSENT)

    # it.10: @overload definitions
    has_overloads: bool = False
    overload_signatures: list[Signature] = field(default_factory=list)

    traces: Any = None
    traces_confidence: Confidence = field(default=Confidence.ABSENT)

    profiling: Any = None
    profiling_confidence: Confidence = field(default=Confidence.ABSENT)

    # --- Cache ---
    from_cache: bool = False
    file_hash: str = ""
    analysis_timestamp: str = ""


@dataclass
class ClassRecord:
    """Complete Python class record.

    Attributes
    ----------
    qualified_name : str
        Qualified name
    file_path : Path
        File path
    line_start : int
        Line start
    line_end : int
        Line end
    col_start : int
        Col start
    col_end : int
        Col end
    visibility : Visibility
        Visibility
    parent_module : str
        Parent module
    existing_docstring : str | None
        Existing docstring
    docstring_line_start : int | None
        Docstring line start
    methods : list[FunctionRecord]
        Methods defined directly in this class.
    source_code : str | None
        Source code
    hierarchy : ClassHierarchy | None
        Hierarchy
    hierarchy_confidence : Confidence
        Confidence of the ``hierarchy`` field.
    protocols : list[ImplicitProtocol]
        Protocols
    class_attributes : list[str]
        Defined directly in the class body
    instance_attributes : list[str]
        Detected through self.x = ...
    inherited_methods : list[FunctionRecord]
        Inherited methods
    from_cache : bool
        From cache
    file_hash : str
        File hash
    analysis_timestamp : str
        Analysis timestamp

    """

    # --- Identity ---
    qualified_name: str
    file_path: Path
    line_start: int
    line_end: int
    col_start: int
    col_end: int
    visibility: Visibility
    parent_module: str
    existing_docstring: str | None
    docstring_line_start: int | None = None

    # --- Contents ---
    methods: list[FunctionRecord] = field(default_factory=list)

    source_code: str | None = None

    # --- Progressive fields (it.8) ---
    hierarchy: ClassHierarchy | None = None
    hierarchy_confidence: Confidence = field(default=Confidence.ABSENT)
    protocols: list[ImplicitProtocol] = field(default_factory=list)
    class_attributes: list[str] = field(default_factory=list)  # defined directly in the class body
    instance_attributes: list[str] = field(default_factory=list)  # detected through self.x = ...
    # Methods inherited from a parent and not redefined here (it.8).
    # Stored separately so `methods` contains only methods defined on this class.
    # Each entry is a copy of the parent's FunctionRecord with method_origin = INHERITED.
    inherited_methods: list[FunctionRecord] = field(default_factory=list)

    # --- Cache ---
    from_cache: bool = False
    file_hash: str = ""
    analysis_timestamp: str = ""


@dataclass
class ModuleRecord:
    """Complete Python module record (.py/.pyw/.pyi).

    Attributes
    ----------
    file_path : Path
        File path
    module_name : str
        Dotted path: "portion.interval"
    line_start : int
        Line start
    line_end : int
        Line end
    visibility : Visibility
        Visibility
    docstring_line_start : int | None
        Docstring line start
    functions : list[FunctionRecord]
        Functions
    classes : list[ClassRecord]
        Classes
    is_stub : bool
        Is stub
    is_test : bool
        Is test
    existing_docstring : str | None
        Existing docstring
    imports : list[ImportRecord]
        Imports
    future_annotations : bool
        True si `from __future__ import annotations`
    source_code : str | None
        Source code
    from_cache : bool
        From cache
    file_hash : str
        File hash
    analysis_timestamp : str
        Analysis timestamp

    """

    file_path: Path
    module_name: str  # Dotted path: "portion.interval"
    line_start: int = 1
    line_end: int = 0
    visibility: Visibility = Visibility.PUBLIC
    docstring_line_start: int | None = None

    functions: list[FunctionRecord] = field(default_factory=list)
    classes: list[ClassRecord] = field(default_factory=list)

    is_stub: bool = False
    is_test: bool = False
    existing_docstring: str | None = None

    imports: list[ImportRecord] = field(default_factory=list)
    future_annotations: bool = False  # True si `from __future__ import annotations`

    source_code: str | None = None

    # --- Cache ---
    from_cache: bool = False
    file_hash: str = ""
    analysis_timestamp: str = ""


@dataclass
class ProjectRecord:
    """Root record for an analyzed Python project.

    Attributes
    ----------
    project_root : Path
        Project root
    project_name : str
        Project name
    project_version : str
        Project version
    modules : list[ModuleRecord]
        Modules
    quality_report : Any
        Quality report
    analysis_timestamp : str
        Analysis timestamp
    docmethis_config : dict[str, Any]
        Docmethis config

    """

    project_root: Path
    project_name: str
    project_version: str = ""

    modules: list[ModuleRecord] = field(default_factory=list)
    quality_report: Any = None
    analysis_timestamp: str = ""
    docmethis_config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Serialization / deserialization
# ---------------------------------------------------------------------------

VERSION_SCHEMA_PROJECT_RECORD = 2


def _convert_for_json(obj: object) -> str | int | float | bool | None:
    """Custom converter for non-standard types passed to ``json.dumps``.

    Parameters
    ----------
    obj : object
        The object to be converted to a JSON-serializable value. It is expected to be a Path or Enum instance, but any object can
        be passed; if the type is not supported, a TypeError is raised.

    Returns
    -------
    str | int | float | bool | None
        Converts a non-standard object into a JSON-serializable primitive. Path objects are converted to their POSIX string form,
        Enum members to their value, and unsupported types raise a TypeError.

    Raises
    ------
    TypeError
        Explicitly raised.

    """
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, enum.Enum):
        return obj.value
    msg = f"Type is not serializable: {type(obj)}"
    raise TypeError(msg)


def serialize_project(record: ProjectRecord) -> str:
    """Serialize a complete ProjectRecord to JSON.

    Parameters
    ----------
    record : ProjectRecord
        The ProjectRecord instance to serialize.

    Returns
    -------
    str
        The complete ProjectRecord serialized as a JSON string.

    """
    data = {"schema_version": VERSION_SCHEMA_PROJECT_RECORD, **asdict(record)}
    return json.dumps(data, default=_convert_for_json, ensure_ascii=False, indent=2)


def _deserialize_signature(data: dict[str, Any]) -> Signature:
    """Rebuild a Signature from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the serialized signature data, as produced by dataclasses.asdict(), from which the Signature
        object is reconstructed.

    Returns
    -------
    Signature
        The Signature instance reconstructed from the provided dictionary.

    """
    data["parameters"] = [Parameter(**p) for p in data.get("parameters", [])]
    data["decorators"] = [Decorator(**d) for d in data.get("decorators", [])]
    return Signature(**data)


def _deserialize_type_info(data: dict[str, Any] | None) -> TypeInfo | None:
    """Rebuild a TypeInfo from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any] | None
        The dictionary containing serialized TypeInfo fields, as produced by dataclasses.asdict(), or None to indicate that no
        TypeInfo should be rebuilt.

    Returns
    -------
    TypeInfo | None
        A TypeInfo instance reconstructed from the provided dictionary, or None if data is None.

    """
    if data is None:
        return None
    return TypeInfo(
        type_str=data["type_str"],
        confidence=Confidence(data["confidence"]),
        provenance=Provenance(data["provenance"]),
        raw=data.get("raw"),
    )


def _deserialize_resolved_types(data: dict[str, Any]) -> ResolvedTypes:
    """Rebuild ResolvedTypes from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any]
        The serialized representation of a ResolvedTypes instance, typically generated by dataclasses.asdict(), containing the
        parameters mapping and return type information.

    Returns
    -------
    ResolvedTypes
        A ResolvedTypes instance rebuilt from the input dict, containing deserialized parameters and return type.

    """
    parameters = {identifier_name: _deserialize_type_info(info) for identifier_name, info in data.get("parameters", {}).items()}
    return ResolvedTypes(parameters=parameters, return_type=_deserialize_type_info(data.get("return_type")))


def _deserialize_complexity(data: dict[str, Any]) -> Complexity:
    """Rebuild Complexity from a dict produced by dataclasses.asdict().

    Missing percentiles are returned as None for compatibility with caches from before iteration 3.

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the serialized Complexity fields, typically produced by dataclasses.asdict(), with keys matching
        the dataclass attribute names and values used to reconstruct the object.

    Returns
    -------
    Complexity
        The reconstructed Complexity instance, with percentile fields set to None when missing from the input dictionary.

    """
    return Complexity(
        loc=data["loc"],
        cyclomatic=data.get("cyclomatic", 1),
        branches=data.get("branches", 0),
        max_depth=data.get("max_depth", 0),
        statements=data.get("statements", 0),
        cyclomatic_percentile=data.get("cyclomatic_percentile"),
        loc_percentile=data.get("loc_percentile"),
        max_depth_percentile=data.get("max_depth_percentile"),
    )


def _deserialize_io_detail(data: dict[str, Any]) -> IODetail:
    """Rebuild IODetail from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the serialized IODetail fields, including pattern, line, and category, as produced by
        dataclasses.asdict().

    Returns
    -------
    IODetail
        An IODetail instance reconstructed from the input dictionary.

    """
    return IODetail(
        pattern=data["pattern"],
        line=data["line"],
        category=data["category"],
    )


def _deserialize_io_effects(data: dict[str, Any]) -> IOEffects:
    """Rebuild IOEffects from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the serialized IOEffects fields, typically produced by dataclasses.asdict(), from which the
        IOEffects instance is rebuilt.

    Returns
    -------
    IOEffects
        An IOEffects instance reconstructed from the provided dictionary, with fields populated from the corresponding keys and
        defaults for missing optional fields.

    """
    return IOEffects(
        reads_files=data.get("reads_files", False),
        writes_files=data.get("writes_files", False),
        network_access=data.get("network_access", False),
        system_calls=data.get("system_calls", False),
        mutates_global=data.get("mutates_global", False),
        has_output=data.get("has_output", False),
        details=[_deserialize_io_detail(d) for d in data.get("details", [])],
    )


def _deserialize_raised_condition(data: dict[str, Any] | None) -> RaisedCondition | None:
    """Rebuild RaisedCondition from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any] | None
        A dictionary mapping field names to values for a RaisedCondition, as generated by dataclasses.asdict(), or None if there
        is no condition to rebuild.

    Returns
    -------
    RaisedCondition | None
        Returns a RaisedCondition rebuilt from the provided data, or None if data is None.

    """
    if data is None:
        return None
    return RaisedCondition(expression=data["expression"], line=data["line"])


def _deserialize_exception_record(data: dict[str, Any]) -> ExceptionRecord:
    """Rebuild ExceptionRecord from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the serialized ExceptionRecord fields, typically produced by dataclasses.asdict(), from which the
        ExceptionRecord is reconstructed.

    Returns
    -------
    ExceptionRecord
        Deserialize a dictionary produced by dataclasses.asdict() into an ExceptionRecord instance, restoring all fields including
        the nested raised condition and provenance.

    """
    return ExceptionRecord(
        exception_type=data["exception_type"],
        message=data.get("message"),
        raise_line=data["raise_line"],
        is_reraise=data.get("is_reraise", False),
        chained_from=data.get("chained_from", False),
        condition=_deserialize_raised_condition(data.get("condition")),
        provenance=Provenance(data.get("provenance", Provenance.LOCAL_INFERENCE.value)),
    )


def _deserialize_except_block(data: dict[str, Any]) -> ExceptBlock:
    """Rebuild ExceptBlock from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the serialized ExceptBlock data, typically produced by dataclasses.asdict(), with keys for
        caught_types, is_reraised, is_swallowed, line, and optionally try_start and try_end.

    Returns
    -------
    ExceptBlock
        The reconstructed ExceptBlock instance.

    """
    return ExceptBlock(
        caught_types=data["caught_types"],
        is_reraised=data["is_reraised"],
        is_swallowed=data["is_swallowed"],
        line=data["line"],
        try_start=data.get("try_start", 0),
        try_end=data.get("try_end", 0),
    )


def _deserialize_observable_property(data: dict[str, Any]) -> ObservableProperty:
    """Rebuild ObservableProperty from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the serialized fields of an ObservableProperty, including keys 'category', 'raw_expression',
        'line', 'confidence', and optionally 'related_exception', as produced by dataclasses.asdict().

    Returns
    -------
    ObservableProperty
        An ObservableProperty instance reconstructed from the provided dictionary.

    """
    return ObservableProperty(
        category=data["category"],
        raw_expression=data["raw_expression"],
        line=data["line"],
        confidence=Confidence(data["confidence"]),
        related_exception=data.get("related_exception"),
    )


def _deserialize_class_hierarchy(data: dict[str, Any] | None) -> ClassHierarchy | None:
    """Rebuild ClassHierarchy from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any] | None
        The serialized ClassHierarchy data as a dictionary, or None if no hierarchy should be built.

    Returns
    -------
    ClassHierarchy | None
        The deserialized ClassHierarchy instance, or None when the input data is None.

    """
    if data is None:
        return None
    return ClassHierarchy(
        direct_parents=data.get("direct_parents", []),
        mro_list=data.get("mro_list", data.get("mro", [])),  # compatibility with pre-it.8 caches
        is_abstract=data.get("is_abstract", False),
        is_mixin=data.get("is_mixin", False),
    )


def _deserialize_protocol(data: dict[str, Any]) -> ImplicitProtocol:
    """Rebuild ImplicitProtocol from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the serialized protocol data, typically from dataclasses.asdict(), with the required key 'name'
        and the optional key 'dunder_methods'.

    Returns
    -------
    ImplicitProtocol
        An ImplicitProtocol instance reconstructed from the input dictionary, with the name and dunder_methods attributes
        populated from the corresponding keys.

    """
    return ImplicitProtocol(
        name=data["name"],
        dunder_methods=data.get("dunder_methods", []),
    )


def _deserialize_callgraph_edge(data: dict[str, Any]) -> CallgraphEdge:
    """Rebuild CallgraphEdge from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the serialized CallgraphEdge fields, as produced by dataclasses.asdict(), from which the
        CallgraphEdge is reconstructed.

    Returns
    -------
    CallgraphEdge
        Returns a CallgraphEdge reconstructed from the provided dictionary.

    """
    return CallgraphEdge(
        source_qname=data["source_qname"],
        target_qname=data["target_qname"],
        call_type=data["call_type"],
        is_external=data["is_external"],
        external_lib=data.get("external_lib"),
        line=data["line"],
        depth_exceeded=data.get("depth_exceeded", False),
    )


def _deserialize_progressive_fields(data: dict[str, Any]) -> None:
    """Deserialize optional progressive FunctionRecord fields in place.

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary of FunctionRecord data whose optional progressive fields are deserialized in place, replacing each present
        field's value with its deserialized representation.

    """
    if data.get("signature") is not None:
        data["signature"] = _deserialize_signature(data["signature"])
    if data.get("complexity") is not None:
        data["complexity"] = _deserialize_complexity(data["complexity"])
    if data.get("io_effects") is not None:
        data["io_effects"] = _deserialize_io_effects(data["io_effects"])
    if data.get("types") is not None:
        data["types"] = _deserialize_resolved_types(data["types"])
    if data.get("exceptions") is not None:
        data["exceptions"] = [_deserialize_exception_record(e) for e in data["exceptions"]]
    if data.get("except_blocks") is not None:
        data["except_blocks"] = [_deserialize_except_block(b) for b in data["except_blocks"]]
    if data.get("observable_properties") is not None:
        data["observable_properties"] = [_deserialize_observable_property(p) for p in data["observable_properties"]]
    if data.get("callgraph") is not None:
        data["callgraph"] = [_deserialize_callgraph_edge(e) for e in data["callgraph"]]
    if data.get("method_origin") is not None:
        data["method_origin"] = MethodOrigin(data["method_origin"])


def _deserialize_test_reference(data: dict[str, Any]) -> TestReference:
    """Rebuild TestReference from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any]
        Mapping of field names to values for a TestReference instance, as produced by dataclasses.asdict().

    Returns
    -------
    TestReference
        A TestReference instance reconstructed from the provided dictionary.

    """
    return TestReference(
        test_qualified_name=data["test_qualified_name"],
        test_module=data["test_module"],
        test_file=Path(data["test_file"]),
        test_line=data["test_line"],
        framework=data["framework"],
        call_line=data["call_line"],
    )


def _deserialize_test_call(data: dict[str, Any]) -> TestCall:
    """Rebuild TestCall from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the serialized fields of a TestCall instance, including function_name, test_source, args, result,
        and exception.

    Returns
    -------
    TestCall
        A TestCall instance reconstructed from the provided data dictionary.

    """
    return TestCall(
        function_name=data["function_name"],
        test_source=data["test_source"],
        args=data.get("args", {}),
        result=data.get("result"),
        exception=data.get("exception"),
    )


def _deserialize_test_coverage(data: dict[str, Any]) -> TestCoverage:
    """Rebuild TestCoverage from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the serialized fields of a TestCoverage instance, typically produced by dataclasses.asdict(), used
        to reconstruct the object.

    Returns
    -------
    TestCoverage
        Returns a TestCoverage instance reconstructed from the input dictionary, with fields populated from the corresponding
        keys, default values used for missing keys, and nested test references and calls deserialized recursively.

    """
    return TestCoverage(
        test_references=[_deserialize_test_reference(r) for r in data.get("test_references", [])],
        is_directly_tested=data.get("is_directly_tested", False),
        test_count=data.get("test_count", 0),
        covered_lines=data.get("covered_lines", []),
        uncovered_lines=data.get("uncovered_lines", []),
        ratio=data.get("ratio", 0.0),
        has_any_test=data.get("has_any_test", False),
        has_failing_test=data.get("has_failing_test", False),
        calls=[_deserialize_test_call(a) for a in data.get("calls", [])],
    )


def _deserialize_usage_example(data: dict[str, Any]) -> UsageExample:
    """Rebuild UsageExample from a dict produced by dataclasses.asdict().

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the serialized UsageExample fields, typically produced by dataclasses.asdict(), from which the
        UsageExample instance is rebuilt.

    Returns
    -------
    UsageExample
        A UsageExample instance reconstructed from the provided dictionary, with missing optional fields defaulting to their
        standard values.

    """
    return UsageExample(
        function=data["function"],
        test_source=data["test_source"],
        setup=data["setup"],
        call=data["call"],
        result=data["result"],
        provenance=data.get("provenance", "test"),
        confidence=data.get("confidence", "high"),
        format=data.get("format", "doctest"),
        contradiction_suspected=data.get("contradiction_suspected", False),
    )


def deserialize_function_record(data: dict[str, Any]) -> FunctionRecord:
    """Rebuild a FunctionRecord from JSON data.

    Keys are not validated because the data comes from dataclasses.asdict() (internal cache). A KeyError indicates a corrupted
    cache -> --full-reanalysis.

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the JSON-serialized fields of a FunctionRecord. This dictionary is mutated in place: file paths
        are converted to Path objects, string enum values are converted to their corresponding enum types, confidence values are
        converted to Confidence instances, and nested structures are deserialized before being used to construct the
        FunctionRecord.

    Returns
    -------
    FunctionRecord
        A FunctionRecord reconstructed from the input data.

    """
    data["file_path"] = Path(data["file_path"])
    data["method_kind"] = MethodType(data["method_kind"])
    data["visibility"] = Visibility(data["visibility"])
    if data.get("property_accessor") is not None:
        data["property_accessor"] = PropertyAccessor(data["property_accessor"])
    for mapping_key in [c for c in data if c.endswith("_confidence")]:
        data[mapping_key] = Confidence(data[mapping_key])
    _deserialize_progressive_fields(data)
    if data.get("test_coverage") is not None:
        data["test_coverage"] = _deserialize_test_coverage(data["test_coverage"])
    if data.get("examples") is not None:
        data["examples"] = [_deserialize_usage_example(e) for e in data["examples"]]
    if data.get("overload_signatures"):
        data["overload_signatures"] = [_deserialize_signature(s) for s in data["overload_signatures"]]
    return FunctionRecord(**data)


def deserialize_class_record(data: dict[str, Any]) -> ClassRecord:
    """Rebuild a ClassRecord from JSON data.

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the serialized ClassRecord fields to be deserialized.

    Returns
    -------
    ClassRecord
        Rebuilds a ClassRecord from a JSON-like dictionary, converting nested fields and enum values into their proper types.

    """
    data["file_path"] = Path(data["file_path"])
    data["visibility"] = Visibility(data["visibility"])
    data["methods"] = [deserialize_function_record(m) for m in data.get("methods", [])]
    data["inherited_methods"] = [deserialize_function_record(m) for m in data.get("inherited_methods", [])]
    for mapping_key in [c for c in data if c.endswith("_confidence")]:
        data[mapping_key] = Confidence(data[mapping_key])
    if data.get("hierarchy") is not None:
        data["hierarchy"] = _deserialize_class_hierarchy(data["hierarchy"])
    data["protocols"] = [_deserialize_protocol(p) for p in data.get("protocols", [])]
    return ClassRecord(**data)


def deserialize_module_record(data: dict[str, Any]) -> ModuleRecord:
    """Rebuild a ModuleRecord from JSON data.

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the serialized ModuleRecord data, including fields such as file_path, visibility, functions,
        classes, and imports.

    Returns
    -------
    ModuleRecord
        Rebuilds a ModuleRecord from a JSON-like dictionary, converting nested data such as file paths, visibility, functions,
        classes, and imports into their proper representations, and returns the reconstructed ModuleRecord.

    """
    data["file_path"] = Path(data["file_path"])
    if data.get("visibility") is not None:
        data["visibility"] = Visibility(data["visibility"])
    data["functions"] = [deserialize_function_record(f) for f in data.get("functions", [])]
    data["classes"] = [deserialize_class_record(c) for c in data.get("classes", [])]
    data["imports"] = [
        ImportRecord(
            module=imp["module"],
            name=imp.get("name"),
            alias=imp.get("alias"),
            conditional=imp.get("conditional", False),
            type_checking_only=imp.get("type_checking_only", False),
            classification=imp.get("classification"),
            is_unused=imp.get("is_unused", False),
        )
        for imp in data.get("imports", [])
    ]
    return ModuleRecord(**data)


def deserialize_project_record(data: dict[str, Any]) -> ProjectRecord:
    """Rebuild a ProjectRecord from JSON data.

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary containing the serialized ProjectRecord data, including schema_version, project_root, and modules entries.

    Returns
    -------
    ProjectRecord
        The deserialized ProjectRecord reconstructed from the JSON data.

    Raises
    ------
    ValueError
        Explicitly raised.

    """
    version = data.pop("schema_version", None)
    if version != VERSION_SCHEMA_PROJECT_RECORD:
        msg = f"Incompatible ProjectRecord schema: {version!r} (expected {VERSION_SCHEMA_PROJECT_RECORD})."
        raise ValueError(msg)
    data["project_root"] = Path(data["project_root"])
    data["modules"] = [deserialize_module_record(m) for m in data.get("modules", [])]
    return ProjectRecord(**data)


def iter_functions(modules: list[ModuleRecord]) -> Iterator[FunctionRecord]:
    """Yield all FunctionRecords (module functions and class methods).

    Parameters
    ----------
    modules : list[ModuleRecord]
        A list of ModuleRecord objects whose functions and class methods should be yielded as FunctionRecords.

    Returns
    -------
    Iterator[FunctionRecord]
        An iterator over all FunctionRecords contained in the given modules, including module-level functions and methods of all
        classes.

    """
    for module in modules:
        yield from module.functions
        for class_ in module.classes:
            yield from class_.methods
