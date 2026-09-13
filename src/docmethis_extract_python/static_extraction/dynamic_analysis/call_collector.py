# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Pytest plugin for dynamic test-to-production call collection (it.9 phase 2).

Enabled only when DOCMETHIS_APPELS_OUTPUT is defined.
DOCMETHIS_PROJECT_ROOT identifies the analyzed project root.
Use --no-instrument (or DOCMETHIS_NO_INSTRUMENT=1) to disable it when it interferes.

Mechanism (DEC-011): pytest_runtest_call hook plus wrapping of production functions
in test-module namespaces (patch-on-import). Record only direct calls from test
functions (the frame immediately above the wrapper).
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from docmethis_extract_python.static_extraction.models import name_simple

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

__all__ = ["CallCollector"]

logger = logging.getLogger(__name__)

# Sentinel distinguishing a None return from an exception raised before return.
_SENTINEL: object = object()


# ---------------------------------------------------------------------------
# Internal subprocess data model
# ---------------------------------------------------------------------------


@dataclass
class _RawTestCall:
    """Internal TestCall form for the plugin subprocess (avoids importing models).

    Attributes
    ----------
    function_name : str
        Qualified name of the function called by the test.
    test_source : str
        Name of the test that made the call.
    args : dict[str, Any]
        Serialized arguments passed to the function.
    result : Any | None
        Serialized result of the call, if any.
    exception : str | None
        Exception message raised by the call, if any.

    """

    function_name: str
    test_source: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any | None = None
    exception: str | None = None


# ---------------------------------------------------------------------------
# Argument serialization
# ---------------------------------------------------------------------------


def _serialize_value(item_value: object) -> str | object:
    """Try to serialize a value to JSON; return repr() or '<non_serializable>' otherwise.

    Parameters
    ----------
    item_value : object
        The value to attempt JSON serialization.

    Returns
    -------
    str | object
        Returns the original value if it is JSON-serializable; otherwise, returns its repr() string, or the literal
        '<non_serializable>' if repr() also fails.

    """
    try:
        json.dumps(item_value)
    except (TypeError, ValueError):
        pass
    else:
        return item_value

    try:
        return repr(item_value)
    except Exception:  # noqa: BLE001
        return "<non_serializable>"


def _build_args_dict(fn: Callable[..., object], args: tuple[object, ...], kwargs: dict[str, object]) -> dict[str, object]:
    """Map positional arguments to signature parameter names.

    Parameters
    ----------
    fn : Callable[..., object]
        The callable whose signature is inspected to map positional arguments to parameter names.
    args : tuple[object, ...]
        A tuple of positional arguments to be mapped to the target function's parameter names.
    kwargs : dict[str, object]
        Keyword arguments to include in the resulting argument dictionary; each key is used as the parameter name and its value is
        serialized.

    Returns
    -------
    dict[str, object]
        Returns a dictionary mapping argument names to serialized values. Positional arguments are keyed by the corresponding
        signature parameter names, or by arg0, arg1, ... when the signature cannot be determined; keyword arguments are keyed by
        their original names.

    """
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        result: dict[str, Any] = {}
        for i, item_value in enumerate(args):
            identifier_name = params[i] if i < len(params) else f"arg{i}"
            result[identifier_name] = _serialize_value(item_value)
        for identifier_name, item_value in kwargs.items():
            result[identifier_name] = _serialize_value(item_value)
    except (ValueError, TypeError):
        return {f"arg{i}": _serialize_value(v) for i, v in enumerate(args)}
    else:
        return result


# ---------------------------------------------------------------------------
# Test-module detection
# ---------------------------------------------------------------------------


def _is_test_module(module_name: str, file_path: str) -> bool:
    """Return True when a module appears to be a test file or conftest.

    Parameters
    ----------
    module_name : str
        The fully qualified name of the module to check, used to determine whether it is a test module.
    file_path : str
        Path to the module file, used to determine whether it is a test module based on its file stem.

    Returns
    -------
    bool
        True if the module appears to be a test file or conftest; False otherwise.

    """
    stem = Path(file_path).stem
    if stem.startswith("test") or stem.endswith("_test") or stem == "conftest":
        return True
    return any(part in {"tests", "test"} or part.startswith("test_") for part in module_name.split("."))


# ---------------------------------------------------------------------------
# Plugin pytest
# ---------------------------------------------------------------------------


class CallCollector:
    """Pytest plugin that collects direct calls from tests to production functions.

    Instantiate in pytest_configure() when DOCMETHIS_APPELS_OUTPUT is defined.

    Parameters
    ----------
    root : Path
        Root directory of the project to analyze.
    output_file : Path
        The path to the output file where the collected call data should be written.

    Attributes
    ----------
    current_test : str | None
        Pytest node id of the currently running test, or None.

    """

    def __init__(self, root: Path, output_file: Path) -> None:
        """Initialize the collector with the project root and output-file path.

        Parameters
        ----------
        root : Path
            Root directory of the project to analyze.
        output_file : Path
            The path to the output file where the collected call data should be written.

        """
        self._root = root.resolve()
        self._output_file = output_file
        self.current_test: str | None = None
        self._calls: list[_RawTestCall] = []
        # (owner object, attribute name, original value) for final restoration.
        self._patches: list[tuple[Any, str, Any]] = []

    # ---------------------------------------------------------------------------
    # Pytest lifecycle.
    # ---------------------------------------------------------------------------

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """After collection, patch production functions in test namespaces.

        Parameters
        ----------
        session : pytest.Session
            The pytest session object for the current test run, used to access collected test items and namespaces.

        """
        self._enable_patches(session)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item: pytest.Item) -> Generator[None, None, None]:
        """Record the current test before execution and reset it afterward.

        Parameters
        ----------
        item : pytest.Item
            The pytest test item that is about to be executed.

        Returns
        -------
        Generator[None, None, None]
            A generator that yields a single None value while the current test is recorded, then resets it to None after
            execution.

        """
        self.current_test = item.nodeid
        yield
        self.current_test = None

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG002
        """Restore original functions and write results to the output file.

        Parameters
        ----------
        session : pytest.Session
            The pytest session object that is being finalized.
        exitstatus : int
            The exit status code of the pytest session.

        """
        self._disable_patches()
        self._write_results()

    # ---------------------------------------------------------------------------
    # Patching
    # ---------------------------------------------------------------------------

    def _enable_patches(self, session: pytest.Session) -> None:
        """Find and patch production functions in test-module namespaces.

        Parameters
        ----------
        session : pytest.Session
            The active pytest session whose collected test items are used to locate test modules for patching.

        """
        seen: set[tuple[int, str]] = set()  # (id(owner), attr_name), avoiding duplicates.

        for item in session.items:
            test_module = getattr(item, "module", None)
            if test_module is None:
                continue

            for attr_name, obj in list(vars(test_module).items()):
                if not self._is_patchable(obj):
                    continue

                mapping_key = (id(test_module), attr_name)
                if mapping_key in seen:
                    continue
                seen.add(mapping_key)

                src_module = getattr(obj, "__module__", None) or ""
                qname = f"{src_module}.{getattr(obj, '__name__', attr_name)}"
                wrapper = self._create_wrapper(obj, qname)
                setattr(test_module, attr_name, wrapper)
                self._patches.append((test_module, attr_name, obj))

    def _is_patchable(self, obj: object) -> bool:
        """Return True when obj is an unmocked patchable production function.

        Parameters
        ----------
        obj : object
            The object to inspect to determine whether it is an unmocked patchable production function.

        Returns
        -------
        bool
            True if obj is an unmocked patchable production function; False otherwise.

        """
        # Preliminary checks.
        if not callable(obj) or isinstance(obj, (type, MagicMock)):
            return False

        src_module = getattr(obj, "__module__", None)
        if not src_module:
            return False

        mod = sys.modules.get(src_module)
        if mod is None:
            return False

        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            return False

        # Path checks.
        try:
            resolved_path = str(Path(mod_file).resolve())
            root_str = str(self._root)
            is_in_root = resolved_path.startswith(root_str)
        except (OSError, ValueError):
            return False

        return is_in_root and not _is_test_module(src_module, mod_file)

    def _create_wrapper(self, original: Callable[..., object], qname: str) -> Callable[..., object]:
        """Return a wrapper that collects direct calls from test functions.

        Parameters
        ----------
        original : Callable[..., object]
            The callable to be wrapped; direct calls to it are collected by the returned wrapper.
        qname : str
            The qualified name of the original callable being wrapped, used to identify the function in collected call records.

        Returns
        -------
        Callable[..., object]
            A wrapper callable that invokes the original callable, records direct calls made from test functions, and preserves
            the original function's metadata.

        """
        collector = self

        def wrapper(*args: object, **kwargs: object) -> object:
            # Skip when the target became a mock (for example, unittest.mock.patch in a test).
            if isinstance(original, MagicMock):
                return original(*args, **kwargs)

            # Direct call: the frame immediately above the wrapper is a test function.
            is_direct = False
            try:
                caller_name = sys._getframe(1).f_code.co_name  # noqa: SLF001
                is_direct = caller_name.startswith("test")
            except (ValueError, AttributeError):
                pass

            outcome: object = _SENTINEL
            exception_type: str | None = None

            try:
                outcome = original(*args, **kwargs)
            except Exception as exc:
                exception_type = type(exc).__name__
                raise
            finally:
                if is_direct and collector.current_test is not None:
                    invocation = _RawTestCall(
                        function_name=qname,
                        test_source=collector.current_test,
                        args=_build_args_dict(original, args, kwargs),
                        result=_serialize_value(outcome) if outcome is not _SENTINEL else None,
                        exception=exception_type,
                    )
                    collector._calls.append(invocation)

            return outcome

        wrapper.__name__ = getattr(original, "__name__", name_simple(qname))  # type: ignore[attr-defined]
        wrapper.__doc__ = getattr(original, "__doc__", None)
        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        return wrapper

    def _disable_patches(self) -> None:
        """Restores all patched attributes to their original values and clears the internal patch list."""
        for owner, attr_name, original in reversed(self._patches):
            setattr(owner, attr_name, original)
        self._patches.clear()

    def _write_results(self) -> None:
        """Writes the collected call invocations to the output file as JSON, logging a warning if an OS error occurs."""
        try:
            data = [asdict(invocation) for invocation in self._calls]
            self._output_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("call_collector - unable to write results: %s", exc)


# ---------------------------------------------------------------------------
# Registration hooks (the -p entry point)
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --no-instrument to disable instrumentation.

    Parameters
    ----------
    parser : pytest.Parser
        The pytest argument parser used to register the --no-instrument command-line option.

    """
    # The option may already be registered when the plugin is loaded twice.
    with contextlib.suppress(Exception):
        parser.addoption(
            "--no-instrument",
            action="store_true",
            default=False,
            help="Disable docmethis instrumentation (test-call collection).",
        )


def pytest_configure(config: pytest.Config) -> None:
    """Register CallCollector when DOCMETHIS_APPELS_OUTPUT is defined.

    Parameters
    ----------
    config : pytest.Config
        The pytest configuration object, providing access to the plugin manager and command-line options.

    """
    output_file_str = os.environ.get("DOCMETHIS_APPELS_OUTPUT")
    if not output_file_str:
        return

    no_instrument = os.environ.get("DOCMETHIS_NO_INSTRUMENT") or _get_option(config, "--no-instrument")
    if no_instrument:
        return

    root_str = os.environ.get("DOCMETHIS_PROJECT_ROOT", ".")
    plugin = CallCollector(root=Path(root_str), output_file=Path(output_file_str))
    try:
        config.pluginmanager.register(plugin, "docmethis_call_collector")
    except Exception as exc:  # noqa: BLE001
        logger.warning("call_collector - unable to register plugin: %s", exc)


def _get_option(config: pytest.Config, name: str) -> bool:
    """Read a pytest option without raising when it is not registered.

    Parameters
    ----------
    config : pytest.Config
        The pytest configuration object used to query command-line options.
    name : str
        The name of the pytest option to read.

    Returns
    -------
    bool
        The boolean value of the requested pytest option, or False if the option is not registered or an exception occurs while
        reading it.

    """
    try:
        return config.getoption(name, default=False)
    except Exception:  # noqa: BLE001
        return False
