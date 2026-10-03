"""Answers: replies read back from a question bundle, validated against their question and trace
digest, and tallied. Answers annotate; they never change findings, severities or DQ
candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..data.jsonval import as_list, as_object, as_str, count, is_object
from ..evidence.history import discover_history

if TYPE_CHECKING:
    from pathlib import Path

    from ..data.jsonval import Doc, JsonObject
    from ..data.model import Trace
from .catalogue import ANSWER_STEPS, BY_ID, CONFIDENCE, Question
from .prompts import trace_digest

MAX_READ = 64 * 1024  # bytes of a metadata or answer file; larger files are ignored


MAX_OBJECTS = 64  # `{` positions tried when a reply wraps its JSON object in prose


def _unfence(text: str) -> str:
    """Strip a surrounding Markdown code fence (```json … ```)."""
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.startswith("json"):
            text = text[4:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _embedded_object(text: str) -> JsonObject | None:
    """The first JSON object with an `answer` key inside prose ("Sure! {...}")."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    for _ in range(MAX_OBJECTS):
        if start == -1:
            return None
        try:
            value, end = decoder.raw_decode(text, start)
        except (ValueError, RecursionError):
            start = text.find("{", start + 1)
            continue
        if is_object(value) and "answer" in value:
            return value
        start = text.find("{", end)  # skip the object's own nested braces
    return None


def parse_answer(text: str, meta: object) -> Doc | None:
    """A validated answer, or None. Only enum fields and step numbers are kept: the free
    `reason` may quote the trace, so it never enters reports. Answers are checked against
    the question's own enum, not the list recorded in `meta` (metadata on disk may be
    stale or edited)."""
    fields = as_object(meta)
    question = BY_ID.get(as_str(fields.get("question")) or "")
    version = fields.get("version")
    if question is None or not isinstance(version, str):
        return None
    value = _reply(text)
    if value is None or not _valid_reply(question, value):
        return None
    steps = as_list(value.get("steps"))  # a list, or falsy (no steps)
    return {
        "question": question.id,
        "version": version,
        "answer": value["answer"],
        "confidence": value.get("confidence"),
        "steps": sorted({n for s in steps if (n := count(s)) is not None})[:ANSWER_STEPS],
        **({"mechanism": value.get("mechanism")} if question.mechanisms else {}),
    }


def _reply(text: str) -> JsonObject | None:
    """The reply's JSON object: the whole (unfenced) text, else one embedded in prose."""
    try:
        value = json.loads(_unfence(text))
    except (ValueError, RecursionError):
        value = _embedded_object(text)
    return value if is_object(value) else None


def _valid_reply(question: Question, value: JsonObject) -> bool:
    steps = value.get("steps") or []
    return (
        value.get("answer") in question.answers
        and value.get("confidence") in CONFIDENCE
        and isinstance(steps, list)
        and (not question.mechanisms or value.get("mechanism") in question.mechanisms)
    )


def _read(path: Path) -> str:
    """At most MAX_READ bytes of UTF-8 text; a larger file is rejected (ValueError)."""
    with path.open("rb") as f:
        data = f.read(MAX_READ + 1)
    if len(data) > MAX_READ:
        raise ValueError("file_too_large")
    return data.decode("utf-8")


@dataclass
class Answers:
    """Answers found under a questions directory, keyed by input label."""

    # (metadata, answer, status); `load` keeps only metadata naming a known question with
    # string input_id and version.
    by_input: dict[str, list[tuple[Doc, Doc | None, str]]] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path) -> Answers:
        found = cls()
        for meta_path in sorted(root.glob("*/*.json")):
            if meta_path.name.endswith(".answer.json"):
                continue
            try:
                meta = json.loads(_read(meta_path))
            except (OSError, ValueError, RecursionError):
                continue
            if not (
                isinstance(meta, dict)
                and isinstance(meta.get("question"), str)
                and meta["question"] in BY_ID
                and isinstance(meta.get("input_id"), str)
                and isinstance(meta.get("version"), str)
            ):
                continue  # malformed metadata: not ours to report on
            reply = meta_path.with_name(meta_path.stem + ".answer.json")
            answer, status = None, "unanswered"
            if reply.exists():
                try:
                    answer = parse_answer(_read(reply), meta)
                except (OSError, ValueError):
                    answer = None
                status = "answered" if answer else "invalid"
            found.by_input.setdefault(meta["input_id"], []).append((meta, answer, status))
        return found

    def annotate(self, label: str, trace: Trace | None, local: Path | None = None) -> list[Doc]:
        """Report rows for one input: stale answers (other trace/version) are marked."""
        digest = trace_digest(trace) if trace is not None else None
        archive = discover_history(local) if trace is not None and trace.compacted else None
        archive_digest = archive.digest() if archive is not None else None
        rows = []
        for meta, answer, status in self.by_input.get(label, []):
            row = _row(meta, answer, status, digest)
            changed = archive is not None and (
                meta.get("archive_digest") != archive_digest
                if "archive_digest" in meta
                else bool(archive.files)
            )
            if changed and status == "answered":
                row = {"question": meta["question"], "version": meta["version"], "status": "stale"}
            rows.append(row)
        return rows


def _row(meta: Doc, answer: Doc | None, status: str, digest: str | None) -> Doc:
    stale = status == "answered" and (
        meta.get("digest") != digest or BY_ID[meta["question"]].version != meta["version"]
    )
    row = {
        "question": meta["question"],
        "version": meta["version"],
        "status": "stale" if stale else status,
    }
    if answer and not stale:
        row.update(answer=answer["answer"], confidence=answer["confidence"], steps=answer["steps"])
        if "mechanism" in answer:
            row["mechanism"] = answer["mechanism"]
    return row


def tally(items: list[Doc]) -> dict[str, dict[str, int]]:
    """{question: {answer or status: traces}} across report items."""
    out: dict[str, dict[str, int]] = {}
    for item in items:
        for row in item.get("answers") or []:
            key = row.get("answer") or row["status"]
            bucket = out.setdefault(row["question"], {})
            bucket[key] = bucket.get(key, 0) + 1
    return out


__all__ = ["Answers", "parse_answer", "tally"]
