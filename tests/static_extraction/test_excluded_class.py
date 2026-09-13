# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Tests for the relaxed mixin heuristic (phase 7, iteration 1.10).

Root cause: is_mixin() excluded every *Mixin class with an __init__, even a trivial one.
Fix 7a: a simple __init__ (super().__init__ + self.attr = ...) keeps the mixin.
Fix 7b: the mixin method threshold is configurable through ConfigurationDocmethis; use logger.debug.
"""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING

from docmethis_extract_python.static_extraction.ast_inheritance import DEFAULT_MIXIN_MAX_METHODS, is_mixin
from docmethis_extract_python.static_extraction.configuration import ConfigurationDocmethis

if TYPE_CHECKING:
    import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_class(src: str) -> ast.ClassDef:
    tree = ast.parse(src)
    return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))


# ---------------------------------------------------------------------------
# Phase 7a - simple __init__ keeps the mixin.
# ---------------------------------------------------------------------------


class TestInitSimple:
    """Tests for phase 7a: a simple __init__ keeps the mixin."""

    def test_no_init_is_mixin(self) -> None:
        node = _parse_class("class LogMixin:\n    def log(self): pass")
        assert is_mixin(node)

    def test_super_only_init_is_mixin(self) -> None:
        src = """
class LogMixin:
    def __init__(self):
        super().__init__()
    def log(self): pass
"""
        assert is_mixin(_parse_class(src))

    def test_super_init_with_assignments_is_mixin(self) -> None:
        src = """
class LogMixin:
    def __init__(self):
        super().__init__()
        self.level = "INFO"
        self.prefix = ""
    def log(self): pass
"""
        assert is_mixin(_parse_class(src))

    def test_init_with_docstring_is_mixin(self) -> None:
        src = """
class LogMixin:
    def __init__(self):
        \"\"\"Docstring.\"\"\"
        super().__init__()
    def log(self): pass
"""
        assert is_mixin(_parse_class(src))

    def test_init_with_pass_is_mixin(self) -> None:
        src = """
class LogMixin:
    def __init__(self):
        pass
    def log(self): pass
"""
        assert is_mixin(_parse_class(src))

    def test_complex_init_with_condition_is_excluded(self) -> None:
        src = """
class LogMixin:
    def __init__(self, level=None):
        if level:
            self.level = level
    def log(self): pass
"""
        assert not is_mixin(_parse_class(src))

    def test_complex_init_with_loop_is_excluded(self) -> None:
        src = """
class LogMixin:
    def __init__(self):
        for _ in range(3):
            pass
    def log(self): pass
"""
        assert not is_mixin(_parse_class(src))

    def test_external_init_call_is_excluded(self) -> None:
        src = """
class LogMixin:
    def __init__(self):
        setup_logging()
    def log(self): pass
"""
        assert not is_mixin(_parse_class(src))

    def test_super_init_with_args_is_mixin(self) -> None:
        src = """
class LogMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def log(self): pass
"""
        assert is_mixin(_parse_class(src))

    def test_init_complexe_log_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """Complex __init__ uses logger.debug rather than warning."""
        src = """
class LogMixin:
    def __init__(self):
        if True:
            self.x = 1
    def log(self): pass
"""
        with caplog.at_level(logging.DEBUG, logger="docmethis_extract_python.static_extraction.ast_inheritance"):
            is_mixin(_parse_class(src))

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert not warnings
        assert any("complex __init__" in r.message for r in debugs)


# ---------------------------------------------------------------------------
# Phase 7b - configurable threshold, logger.debug instead of warning.
# ---------------------------------------------------------------------------


class TestConfigurableThreshold:
    """Tests for phase 7b: configurable method threshold and logger.debug."""

    def test_threshold_default_5(self) -> None:
        assert DEFAULT_MIXIN_MAX_METHODS == 5
        # Single source: configuration does not redefine its own default.
        assert ConfigurationDocmethis().mixin_max_methods == DEFAULT_MIXIN_MAX_METHODS

    def test_class_below_threshold_is_mixin(self) -> None:
        methods = "\n".join(f"    def m{i}(self): pass" for i in range(5))
        src = f"class LogMixin:\n{methods}"
        assert is_mixin(_parse_class(src))

    def test_class_above_threshold_is_excluded(self) -> None:
        methods = "\n".join(f"    def m{i}(self): pass" for i in range(6))
        src = f"class LogMixin:\n{methods}"
        assert not is_mixin(_parse_class(src))

    def test_custom_larger_threshold(self) -> None:
        methods = "\n".join(f"    def m{i}(self): pass" for i in range(8))
        src = f"class LogMixin:\n{methods}"
        assert is_mixin(_parse_class(src), mixin_max_methods=10)

    def test_custom_smaller_threshold(self) -> None:
        methods = "\n".join(f"    def m{i}(self): pass" for i in range(3))
        src = f"class LogMixin:\n{methods}"
        assert not is_mixin(_parse_class(src), mixin_max_methods=2)

    def test_threshold_exceeded_logs_debug_not_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Exceeding the threshold logs debug only, not warning."""
        methods = "\n".join(f"    def m{i}(self): pass" for i in range(6))
        src = f"class LogMixin:\n{methods}"
        with caplog.at_level(logging.DEBUG, logger="docmethis_extract_python.static_extraction.ast_inheritance"):
            is_mixin(_parse_class(src))

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert not warnings
        assert any("threshold" in r.message for r in debugs)
