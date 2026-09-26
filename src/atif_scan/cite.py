"""Opt-in citations (`--cite`): the trace text behind a finding, with before/after context.

This is the only output path that contains trace text, so it is off by default and
never used by the report allowlist. Excerpts are bounded and masked for common
credential shapes, but masking is best-effort: treat cited output like the trace itself
and keep it out of Git.
"""

from __future__ import annotations

import re

from .checks import Severity
from .engine import Assessment
from .model import Channel, Content, Locator, Step, Trace

WINDOW = 160  # characters of context on each side of a matched span
CONTEXT = 240  # characters of before/after context
PER_FINDING = 3  # evidence items cited per finding

SECRETS = [
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.S
        ),
        "[private key]",
    ),
    (re.compile(r"(?i)\b(authorization|proxy-authorization)\s*[:=]\s*[^\n\"']+"), r"\1: ***"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer ***"),
    (
        re.compile(
            r"(?i)\b([A-Z0-9_]*(?:api[_-]?key|token|secret|passw(?:or)?d|credential)[A-Z0-9_]*)"
            r"(\s*[:=]\s*|['\"]\s*:\s*['\"])([^\s'\"&]{4,})"
        ),
        r"\1\2***",
    ),
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"), "***"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "***"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "***"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "***"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), "***"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "***"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "***"),
    (re.compile(r"(://[^/\s:@]+):[^/\s@]+@"), r"\1:***@"),
    (re.compile(r"(?i)([?&](?:token|key|sig|signature|access_token|api_key)=)[^&\s]+"), r"\1***"),
]


def mask(text: str) -> str:
    for pattern, replacement in SECRETS:
        text = pattern.sub(replacement, text)
    return text


# Mask the whole text *before* cutting windows, so a secret can't straddle a cut.
def _head(text: str, limit: int = CONTEXT) -> str:
    masked = mask(text)
    return masked[:limit] + ("…" if len(masked) > limit else "")


def _tail(text: str, limit: int = CONTEXT) -> str:
    masked = mask(text)
    return ("…" if len(masked) > limit else "") + masked[-limit:]


MARK = "\x00"


def _window(text: str, start: int, end: int) -> tuple[str, str, str] | None:
    """Mask with the match delimited by markers, then cut; None if masking ate a marker."""
    marked = mask(text[:start] + MARK + text[start:end] + MARK + text[end:])
    parts = marked.split(MARK)
    if len(parts) != 3:
        return None  # a secret overlapped the match; show the masked head instead
    before, match, after = parts
    return (
        ("…" if len(before) > WINDOW else "") + before[-WINDOW:],
        match,
        after[:WINDOW] + ("…" if len(after) > WINDOW else ""),
    )


def _surface(step: Step, at: Locator) -> tuple[Content | None, str | None]:
    """The cited surface text and, for tool arguments, the recorded tool name."""
    if at.channel == Channel.MESSAGE:
        return step.message, None
    if at.channel == Channel.REASONING:
        return step.reasoning, None
    if at.channel == Channel.OBSERVATION and at.observation is not None:
        return step.observations[at.observation].content, None
    if at.call is not None:
        call = step.calls[at.call]
        fields = [c for ch, c in call.fields if ch == at.channel]
        if at.field is not None:
            return call.fields[at.field][1], call.name
        return (Content("\n".join(c.text for c in fields)) if fields else None), call.name
    return None, None


def _metadata(step: Step) -> str:
    parts = []
    if step.step_id_recorded:
        parts.append(f"step_id={step.step_id}")
    if step.timestamp_recorded:
        parts.append(f"timestamp={step.timestamp.isoformat() if step.timestamp else 'invalid'}")
    return " ".join(parts) or "no step metadata recorded"


def cite(trace: Trace, at: Locator) -> dict:
    """One citation: `text` split around the match, plus `before`/`after` context."""
    step = trace.steps[at.step]
    result: dict = {"step": at.step, "channel": at.channel.value, "source": step.source}
    if at.channel == Channel.METADATA:
        result.update(before="", match=_metadata(step), after="")
        return result
    content, tool = _surface(step, at)
    if tool is not None:
        result["tool"] = tool
    text = content.text if content is not None else ""
    window = (
        _window(text.replace(MARK, " "), *at.span) if at.span and at.span[1] <= len(text) else None
    )
    if window is not None:
        result.update(before=window[0], match=window[1], after=window[2])
    else:
        result.update(before="", match=_head(text, 2 * WINDOW), after="")
    # Context: why the agent did it (same step's reasoning/message) and what came back.
    if at.call is not None:
        intent = step.reasoning.text or step.message.text
        call = step.calls[at.call]
        output = next(
            (o.content.text for o in step.observations if o.source_call_id == call.id), None
        )
        result["context_before"] = _tail(intent) if intent else ""
        result["context_after"] = _head(output) if output else ""
    elif at.channel == Channel.OBSERVATION and at.observation is not None:
        source = step.observations[at.observation].source_call_id
        call = next((c for c in step.calls if c.id == source), None)
        result["context_before"] = (
            _head("\n".join(c.text for _, c in call.fields)) if call is not None else ""
        )
        result["context_after"] = ""
    else:
        first = step.calls[0] if step.calls else None
        result["context_before"] = ""
        result["context_after"] = (
            _head("\n".join(c.text for _, c in first.fields)) if first is not None else ""
        )
    return result


def citations(
    trace: Trace, assessments: tuple[Assessment, ...], minimum: Severity
) -> dict[str, list[dict]]:
    """Citations for unexcused findings at/above `minimum`, keyed by check ID."""
    out: dict[str, list[dict]] = {}
    for a in assessments:
        if a.counts and a.spec.severity >= minimum and a.result.evidence:
            out[a.spec.id] = [cite(trace, at) for at in a.result.evidence[:PER_FINDING]]
    return out
