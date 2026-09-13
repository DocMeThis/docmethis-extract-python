# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Shared timestamp utilities."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now_iso() -> str:
    """Return the current UTC time in ISO 8601 format with second precision and a ``Z`` suffix.

    Returns
    -------
    str
        A string containing the current UTC time formatted as ISO 8601 with seconds precision and a trailing Z.

    """
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
