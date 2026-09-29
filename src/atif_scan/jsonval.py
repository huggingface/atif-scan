"""Untrusted JSON values, narrowed to what a caller expects (stdlib only).

Trajectories, Harbor run files and Hub listings are parsed with `json.loads`, so every
value is `object` until checked. These helpers are the one place that narrows them:
a value of the wrong shape becomes `None` (or an empty container), which callers treat
as unknown, never as zero, false or clean. Booleans are never counts or numbers.
"""

from __future__ import annotations

import json
import math
from typing import Any, TypeGuard, cast

# A parsed JSON object whose values haven't been checked yet.
JsonObject = dict[str, object]
# One of atif-scan's own report documents (allowlisted fields it built itself).
Doc = dict[str, Any]


def is_object(value: object) -> TypeGuard[JsonObject]:
    return isinstance(value, dict)


def as_object(value: object) -> JsonObject:
    """The value if it's a JSON object, else an empty one."""
    # json.loads only makes str keys; the values stay unchecked (`object`).
    return cast("JsonObject", value) if isinstance(value, dict) else {}


def as_list(value: object) -> list[object]:
    """The value if it's a JSON array, else an empty one."""
    return cast("list[object]", value) if isinstance(value, list) else []


def as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def count(value: object) -> int | None:
    """A recorded count: a non-negative int (not a bool or float), else None."""
    return value if type(value) is int and value >= 0 else None


def number(value: object, low: float | None = None) -> float | None:
    """A finite int/float (not bool), optionally at least `low`; anything else is None."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    if not math.isfinite(value) or (low is not None and value < low):
        return None
    return float(value)


def load_object(data: bytes | None, limit: int) -> JsonObject:
    """A JSON object from at most `limit` bytes, else an empty one (absent, too large,
    not UTF-8, not JSON, or not an object)."""
    if not data or len(data) > limit:
        return {}
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError):
        return {}
    return as_object(value)
