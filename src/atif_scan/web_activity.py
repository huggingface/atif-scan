"""Counts of explicitly recorded web actions; no network access or text export.

Provider backend requests are not actions. Only recognized call structures are read,
never arbitrary nested objects. Frozen argument mappings and tuples are supported.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .jsonval import as_str

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .jsonval import Doc
    from .model import ToolCall, Trace


@dataclass(frozen=True)
class WebActivity:
    searches: int = 0
    opens: int = 0
    finds: int = 0
    known_queries: int = 0
    unknown_query_actions: int = 0
    unknown_actions: int = 0

    def document(self) -> Doc:
        """Numeric allowlist only: never query strings, references, URLs or entities."""
        return {
            "searches": self.searches,
            "opens": self.opens,
            "finds": self.finds,
            "known_queries": self.known_queries,
            "unknown_query_actions": self.unknown_query_actions,
            "unknown_actions": self.unknown_actions,
        }


def _mapping(value: object) -> Mapping[str, object]:
    # Frozen JSON has str keys, but callers may supply other mappings.
    if isinstance(value, Mapping) and all(isinstance(k, str) for k in value):
        return {k: v for k, v in value.items() if isinstance(k, str)}
    return {}


def _sequence(value: object) -> tuple[object, ...] | None:
    return tuple(value) if isinstance(value, list | tuple) else None


def _query(value: object) -> str | None:
    text = as_str(value)
    return text if text and text.strip() else None


def _queries(args: Mapping[str, object]) -> tuple[int, int]:
    primary = _query(args.get("query", args.get("q", args.get("input"))))
    batch = _sequence(args.get("queries"))
    if batch is None:
        return (int(primary is not None), int(primary is None or "queries" in args))
    valid = [_query(v) for v in batch]
    known = sum(v is not None for v in valid)
    # Only this documented alias is deduplicated. Repeated recorded queries elsewhere
    # are separate actions, not a set of unique strings.
    known += int(primary is not None and (not valid or primary != valid[0]))
    malformed_primary = any(k in args for k in ("query", "q")) and primary is None
    unknown = malformed_primary or not (known or batch) or any(v is None for v in valid)
    return known, int(unknown)


def _search(args: Mapping[str, object]) -> WebActivity:
    known, unknown = _queries(args)
    return WebActivity(searches=1, known_queries=known, unknown_query_actions=unknown)


def _sum(rows: Iterable[WebActivity]) -> WebActivity:
    totals = [0] * 6
    for row in rows:
        values = (
            row.searches,
            row.opens,
            row.finds,
            row.known_queries,
            row.unknown_query_actions,
            row.unknown_actions,
        )
        for index, value in enumerate(values):
            totals[index] += value
    return WebActivity(*totals)


def _batch(key: str, value: object) -> WebActivity:
    entries = _sequence(value)
    if not entries:
        return _search({}) if key == "search_query" else WebActivity(unknown_actions=1)
    if key == "search_query":
        return _sum(_search(_mapping(entry)) for entry in entries)
    if key == "find":
        return WebActivity(finds=len(entries))
    return WebActivity(opens=len(entries))


def call_activity(call: ToolCall) -> WebActivity:
    """Read one recognized action envelope, with native normalized-tool fallback."""
    args = _mapping(call.arguments)
    if call.name == "web_search_call":
        action = as_str(args.get("action_type"))
        payload = args
        if action is None:
            payload = _mapping(args.get("action"))
            action = as_str(payload.get("type"))
        return _hosted(action, payload)
    if call.name in ("web__run", "web.run"):
        keys = tuple(k for k in ("search_query", "open", "click", "find") if k in args)
        return _sum(_batch(k, args[k]) for k in keys) if keys else WebActivity(unknown_actions=1)
    if call.tool == "web_search":
        return _search(args)
    return WebActivity(opens=1) if call.tool == "web_fetch" else WebActivity()


def _hosted(action: str | None, args: Mapping[str, object]) -> WebActivity:
    if action == "search":
        return _search(args)
    if action == "open_page":
        return WebActivity(opens=1)
    if action == "find_in_page":
        return WebActivity(finds=1)
    return WebActivity(unknown_actions=1)


def web_activity(trace: Trace | None) -> WebActivity | None:
    """None means no trace data, not zero web activity.

    Code-mode calls are counted unless their parent is itself a recognized web call:
    those derived aliases must not count the same recorded action twice.
    """
    if trace is None:
        return None
    rows: list[WebActivity] = []
    for step in trace.steps:
        if not step.authored:
            continue
        parents = {
            c.id
            for c in step.calls
            if c.result_id is None
            and c.id
            and (
                c.tool in ("web_search", "web_fetch")
                or c.name in ("web_search_call", "web__run", "web.run")
            )
        }
        rows.extend(call_activity(c) for c in step.calls if c.result_id not in parents)
    return _sum(rows)
