"""One-time OpenHands export compatibility exception for private review.

Some ALE OpenHands ATIF exports retain action summaries but omit provider reasoning.
This fixed-format companion reader recovers uniquely linked reasoning without
changing ATIF or scanner results. It is not a native-session adapter framework:
do not extend it to other harnesses or session formats. Prefer fixing the upstream
ATIF exporter or explicit conversion to standard evidence for future format gaps.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..data import account_ids, credentials
from ..data.jsonval import Doc, as_object, as_str, compact_json, load_object
from .cite import mask, trace_secrets

if TYPE_CHECKING:
    from pathlib import Path

    from ..data.model import Trace

MAX_BYTES = 16 * 1024 * 1024
MAX_EVENTS = 20_000
PAGE_CHARS = 6000


@dataclass(frozen=True)
class ReasoningRecord:
    line: int
    step: int
    index: int
    text: str


@dataclass(frozen=True)
class NativeReasoning:
    status: str
    digest: str
    records: tuple[ReasoningRecord, ...] = ()
    unlinked: int = 0
    malformed: int = 0

    def counts(self) -> Doc:
        return {
            "status": self.status,
            "format": "openhands_jsonl",
            "linked": len(self.records),
            "unlinked": self.unlinked,
            "malformed": self.malformed,
            "scanned": False,
        }

    def outline(self) -> Doc:
        return {
            **self.counts(),
            "records": [
                {"record": i, "native_line": r.line, "atif_step": r.step, "characters": len(r.text)}
                for i, r in enumerate(self.records)
            ],
        }

    def page(self, record: int, offset: int = 0, limit: int = 3000) -> Doc:
        if type(record) is not int or not 0 <= record < len(self.records):
            raise ValueError("no such native reasoning record")
        text = self.records[record].text  # whole-field masked before any slicing
        if type(offset) is not int or not 0 <= offset <= len(text):
            raise ValueError("invalid offset")
        if type(limit) is not int or not 1 <= limit <= PAGE_CHARS:
            raise ValueError("invalid limit")
        end = min(offset + limit, len(text))
        return {
            "record": record,
            "native_line": self.records[record].line,
            "atif_step": self.records[record].step,
            "offset": offset,
            "next_offset": end if end < len(text) else None,
            "text": text[offset:end],
        }


def _read(local: Path | None) -> tuple[str, bytes]:
    if local is None:
        return "not_checked", b""
    path = local.parent / "origin_log" / "openhands_cli" / "transcript.jsonl"
    try:
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("unsafe companion")
        if not path.exists():
            return "not_found", b""
        return _read_file(path)
    except (OSError, UnicodeError, ValueError):
        return "unreadable", b""


def _read_file(path: Path) -> tuple[str, bytes]:
    if not path.is_file():
        raise ValueError("not a regular file")
    with path.open("rb") as handle:
        data = handle.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return "oversized", b""
    data.decode("utf-8")
    return "available", data


def _records(trace: Trace, data: bytes) -> tuple[tuple[ReasoningRecord, ...], int, int]:
    events = [load_object(line, MAX_BYTES) for line in data.splitlines()[:MAX_EVENTS]]
    ids = Counter(as_str(e.get("tool_call_id")) for e in events if e.get("kind") == "ActionEvent")
    calls = [
        (c, s.index, trace.step_numbers[s.index])
        for s in trace.steps
        if s.authored
        for c in s.calls
    ]
    call_ids = Counter(c.id for c, _, _ in calls)
    by_id = {c.id: (c, index, step) for c, index, step in calls if call_ids[c.id] == 1}
    raw = data.decode("utf-8")
    known = trace_secrets(trace) | credentials.values((raw,)) | account_ids.values((raw,))
    records = []
    unlinked = 0
    for line, event in enumerate(events, 1):
        text = as_str(event.get("reasoning_content"))
        if not text or event.get("kind") != "ActionEvent" or event.get("source") != "agent":
            continue
        key = as_str(event.get("tool_call_id"))
        match = by_id.get(key or "")
        if (
            not key
            or not match
            or ids[key] != 1
            or not _same_action(event, match[0].name, match[0].arguments)
        ):
            unlinked += 1
            continue
        _, index, step = match
        records.append(ReasoningRecord(line, step, index, mask(text, known)))
    malformed = sum(not e for e in events) + int(len(data.splitlines()) > MAX_EVENTS)
    return tuple(records), unlinked, malformed


def _same_action(event: dict[str, object], name: str, arguments: object) -> bool:
    action = as_object(event.get("action"))
    encoded = compact_json(action)
    return (
        event.get("tool_name") == name
        and bool(action)
        and encoded is not None
        and encoded == compact_json(arguments)
    )


def discover_native_reasoning(local: Path | None, trace: Trace) -> NativeReasoning:
    """Fixed sibling only. Bind content/status; unique IDs AND exact tool/actions link it."""
    status, data = _read(local)
    digest = hashlib.sha256(
        b"openhands-reasoning/v1\0" + status.encode() + b"\0" + data
    ).hexdigest()
    if status != "available":
        return NativeReasoning(status, digest)
    records, unlinked, malformed = _records(trace, data)
    return NativeReasoning(status, digest, records, unlinked, malformed)
