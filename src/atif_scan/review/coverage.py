"""How much of a trace an answering model read, from its own saved trajectory.

`atif-scan hunt` saves each answering run's ATIF trajectory beside its answer
(`<question>.review.atif.json`). Its read-only trace-tool calls say which steps of the
reviewed trace it read: `read_steps(first, last)` ranges and `read_step_segment` steps,
and whether it opened `trace_outline`. Searches return windows, not steps, and aren't
counted as read. The figures are counts only; nothing of either trace is exported.

A universal answer ("clean", "absent") after reading a small part of the trace is a
guess about the rest. `LOW_COVERAGE` marks it, so reports can count such answers apart;
the answer itself is never changed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ..data.jsonval import as_list, as_object, as_str, count

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

    from ..data.jsonval import Doc

MAX_BYTES = 32 * 1024 * 1024  # answering trajectories observed: ~100-170 KB
LOW_COVERAGE = 0.5  # share of the reviewed trace's steps below which "nothing" is a guess
# Answers that claim nothing was found anywhere: the ones coverage matters for.
UNIVERSAL = frozenset({"clean", "absent", "no_visible_downstream_indicator", "no_evidence"})


def _arguments(call: object) -> dict[str, object]:
    raw = as_object(call).get("arguments")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, RecursionError):
            return {}
    return dict(as_object(raw))


def _tool(call: object) -> str:
    """The tool name without the MCP server prefix (`uv__read_steps` -> `read_steps`)."""
    return (as_str(as_object(call).get("function_name")) or "").rsplit("__", 1)[-1]


def _steps_read(name: str, args: dict[str, object]) -> range:
    if name == "read_steps":
        first, last = count(args.get("first")), count(args.get("last"))
        if first is not None and last is not None and first <= last:
            return range(first, last + 1)
    elif name == "read_step_segment" and (step := count(args.get("step_number"))) is not None:
        return range(step, step + 1)
    return range(0)


def review_coverage(review: Path, step_numbers: Collection[int]) -> Doc | None:
    """{steps_read, share, outline, tool_calls} for one answering run over a trace with
    these step numbers; None when its trajectory is missing, too large or not JSON."""
    try:
        with review.open("rb") as handle:
            data = handle.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            return None
        doc = json.loads(data)
    except (OSError, ValueError, RecursionError):
        return None
    calls = [
        c
        for s in as_list(as_object(doc).get("steps"))
        for c in as_list(as_object(s).get("tool_calls"))
    ]
    known = set(step_numbers)
    read: set[int] = set()
    for call in calls:
        read.update(n for n in _steps_read(_tool(call), _arguments(call)) if n in known)
    return {
        "steps_read": len(read),
        "share": round(len(read) / len(known), 2) if known else None,
        "outline": any(_tool(c) == "trace_outline" for c in calls),
        "tool_calls": len(calls),
    }


def thin(row: Doc) -> bool:
    """A universal answer given after reading less than LOW_COVERAGE of the trace."""
    coverage = as_object(row.get("coverage"))
    share = coverage.get("share")
    return (
        row.get("answer") in UNIVERSAL and isinstance(share, (int, float)) and share < LOW_COVERAGE
    )
