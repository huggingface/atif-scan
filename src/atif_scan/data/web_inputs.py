"""Static web-input evidence, separate from returned content and token accounting.

Codex code-mode arguments deliberately remain None: their source is already counted
in the outer program. This small typed summary preserves literal input evidence
without counting that source twice. Reference IDs are targets, not inferred URLs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .model import Channel

if TYPE_CHECKING:
    from .model import ToolCall

REFERENCE = re.compile(r"turn\d+(?:search|view)\d+\Z")


@dataclass(frozen=True)
class WebInput:
    recorded: bool
    source_known: bool


def _target(value: object) -> WebInput:
    if not isinstance(value, str) or not value.strip():
        return WebInput(False, False)
    if value.startswith(("https://", "http://")):
        return WebInput(True, True)
    # Historical exports can omit the reference-producing result, or reuse IDs
    # across compactions. Literal IDs prove an input, never a unique source link.
    return WebInput(bool(REFERENCE.fullmatch(value)), False)


def _entry(action: str, value: object) -> WebInput:
    if not isinstance(value, Mapping):
        return WebInput(False, False)
    if action == "search_query":
        query = value.get("q")
        known = isinstance(query, str) and bool(query.strip())
        return WebInput(known, known)
    return _target(value.get("ref_id", value.get("reference_id", value.get("url"))))


def parse_web_input(name: str, args: object) -> WebInput | None:
    """Read only recognized action envelopes; arbitrary nested prose is not input."""
    if name not in ("web__run", "web.run"):
        return None
    if not isinstance(args, Mapping):
        return WebInput(False, False)
    rows = []
    for action in ("search_query", "open", "click", "find"):
        if action not in args:
            continue
        batch = args[action]
        if not isinstance(batch, list | tuple) or not batch:
            rows.append(WebInput(False, False))
        else:
            rows.extend(_entry(action, entry) for entry in batch)
    return WebInput(
        bool(rows) and all(row.recorded for row in rows),
        bool(rows) and all(row.source_known for row in rows),
    )


def web_input(call: ToolCall) -> WebInput:
    parsed = call.web_input or parse_web_input(call.name, call.arguments)
    if parsed is not None:
        return parsed
    wanted = Channel.QUERY if call.tool == "web_search" else Channel.URL
    fields = [c for ch, c in call.fields if ch == wanted]
    known = bool(fields) and all(c.understood and bool(c.text.strip()) for c in fields)
    return WebInput(known, known)
