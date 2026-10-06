"""Opt-in citations (`--cite`): the trace text behind a finding, with before/after context.

This is the only output path that contains trace text, so it is off by default and
never used by the report allowlist. Excerpts are bounded and masked for common
credential shapes, but masking is best-effort: treat cited output like the trace itself
and keep it out of Git.
"""

from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING

from ..checks import check_selected
from ..data import account_ids, credentials
from ..data.model import Channel, Content, Locator, Step, ToolCall, Trace

if TYPE_CHECKING:
    from ..checks import Severity
    from ..data.jsonval import Doc
    from ..engine import Assessment

WINDOW = 160  # characters of context on each side of a matched span
CONTEXT = 240  # characters of before/after context
PER_FINDING = 3  # evidence items cited per finding

# Token shapes are `credentials.TOKEN_SHAPES`; these are the masks beyond it and beyond
# `credentials.mask`'s secret-named values: header values, bearer tokens, unquoted
# secret-named values, URL userinfo and query-string keys. The sk-/pk-/rk- shape stays
# here too because credentials requires a digit in it (fewer false findings); citations
# mask it either way.
HEADERS = [
    (re.compile(r"(?i)\b(authorization|proxy-authorization)\s*[:=]\s*[^\n\"']+"), r"\1: ***"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer ***"),
]
SHAPES = [
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"), "***"),
    (re.compile(r"(://[^/\s:@]+):[^/\s@]+@"), r"\1:***@"),
    (re.compile(r"(?i)([?&](?:token|key|sig|signature|access_token|api_key)=)[^&\s]+"), r"\1***"),
]
SECRETS = HEADERS + SHAPES
# NAME: value / NAME=value / "name": "value" where NAME contains a secret word. The name is
# matched as a whole identifier and tested separately (one regex with the word inside the
# identifier was quadratic on long identifiers); the value is only checked for length in
# the lookahead so a rejected name doesn't swallow a later `token=...` in its value.
SECRET_WORD = re.compile(r"(?i)api[_-]?key|token|secret|passw(?:or)?d|credential")
KEYED = re.compile(r"(?<![\w-])([\w-]+)(\s*[:=]\s*|['\"]\s*:\s*['\"])(?=[^\s'\"&]{4})")
KEYED_VALUE = re.compile(r"[^\s'\"&]+")


def _keyed(text: str) -> list[tuple[int, int, int]]:
    """(match start, value start, value end) of secret-named values."""
    out: list[tuple[int, int, int]] = []
    pos = 0
    for m in KEYED.finditer(text):
        if m.start() < pos or not SECRET_WORD.search(m[1]):
            continue
        value = KEYED_VALUE.match(text, m.end())
        if value is None:  # can't happen: KEYED's lookahead saw 4+ value characters here
            raise AssertionError("KEYED matched without a value")
        end = value.end()
        out.append((m.start(), m.end(), end))
        pos = end
    return out


def mask(text: str, known: frozenset[str] = frozenset()) -> str:
    """Mask credential shapes, secret-named values and any `known` secret value (the same
    secret seen elsewhere in the trace, e.g. printed bare on its own line).

    Order matters: private-key blocks first (a header mask stops at the line end), then
    the broad header/assignment masks (a token glued to another would otherwise be cut by
    its shape and the rest left unmatched), then token shapes, then `credentials.mask`,
    whose named-value test sees tokens already masked (a glued token isn't "code").
    OpenAI account identifiers go first, while their keys are still intact."""
    text = _mask_spans(text, account_ids.spans(text))
    text = credentials.PRIVATE_KEY.sub("[private key]", text)
    for pattern, replacement in HEADERS:
        text = pattern.sub(replacement, text)
    text = _mask_keyed(text)
    text = credentials.TOKEN_SHAPES.sub("***", text)
    for pattern, replacement in SHAPES:
        text = pattern.sub(replacement, text)
    return credentials.mask(text, known)


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    out: list[str] = []
    last = 0
    for start, end in sorted(spans):
        if start >= last:
            out += [text[last:start], "***"]
            last = end
    out.append(text[last:])
    return "".join(out)


def _mask_keyed(text: str) -> str:
    out: list[str] = []
    last = 0
    for _, value, end in _keyed(text):
        out += [text[last:value], "***"]
        last = end
    out.append(text[last:])
    return "".join(out)


def trace_secrets(trace: Trace) -> frozenset[str]:
    """Every credential value and OpenAI account identifier anywhere in the trace, so
    each occurrence gets masked (an account UUID repeated bare is no longer key-bound)."""
    texts = []
    for step in trace.steps:
        texts += [step.message.text or "", step.reasoning.text or ""]
        texts += [c.text or "" for call in step.calls for _, c in call.fields]
        texts += [o.content.text or "" for o in step.observations]
    return credentials.values(texts) | account_ids.values(texts)


def _secret_spans(text: str, known: frozenset[str]) -> list[tuple[int, int]]:
    spans = [m.span() for pattern, _ in SECRETS for m in pattern.finditer(text)]
    spans += [(start, end) for start, _, end in _keyed(text)]
    spans += [f.value for f in credentials.find(text)]
    spans += account_ids.spans(text)
    for value in known:
        start = text.find(value)
        while start != -1:
            spans.append((start, start + len(value)))
            start = text.find(value, start + 1)
    return spans


@functools.lru_cache(maxsize=256)
def _masked(text: str, known: frozenset[str]) -> str:
    """`mask`, memoised: one result or intent is often cited by several locators."""
    return mask(text, known)


# Mask the whole text *before* cutting windows, so a secret can't straddle a cut.
def _head(text: str, limit: int = CONTEXT, known: frozenset[str] = frozenset()) -> str:
    masked = _masked(text, known)
    return masked[:limit] + ("…" if len(masked) > limit else "")


def _tail(text: str, limit: int = CONTEXT, known: frozenset[str] = frozenset()) -> str:
    masked = _masked(text, known)
    return ("…" if len(masked) > limit else "") + masked[-limit:]


MARK = "\x00"
MARKED_PARTS = 3  # before, match, after: text split at the two markers


def _window(
    text: str, start: int, end: int, known: frozenset[str] = frozenset()
) -> tuple[str, str, str] | None:
    """Mask with the match delimited by markers, then cut. None if a match boundary falls
    inside a secret (a marker would split it so no pattern matches either half) or if
    masking ate a marker."""
    for s, e in _secret_spans(text, known):
        if s < start < e or s < end < e:
            return None
    marked = mask(text[:start] + MARK + text[start:end] + MARK + text[end:], known)
    parts = marked.split(MARK)
    if len(parts) != MARKED_PARTS:
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
    call = step.calls[at.call] if at.call is not None else None
    return (_call_surface(call, at), call.name) if call is not None else (None, None)


def _call_surface(call: ToolCall, at: Locator) -> Content | None:
    """One recorded argument field, or every field on the cited channel joined."""
    if at.field is not None:
        return call.fields[at.field][1]
    fields = [c for ch, c in call.fields if ch == at.channel]
    return Content("\n".join(c.text for c in fields)) if fields else None


def _metadata(step: Step) -> str:
    parts = []
    if step.step_id_recorded:
        parts.append(f"step_id={step.step_id}")
    if step.timestamp_recorded:
        parts.append(f"timestamp={step.timestamp.isoformat() if step.timestamp else 'invalid'}")
    return " ".join(parts) or "no step metadata recorded"


def cite(trace: Trace, at: Locator, known: frozenset[str] | None = None) -> Doc:
    """One citation: `text` split around the match, plus `before`/`after` context."""
    if known is None:
        known = trace_secrets(trace)
    step = trace.steps[at.step]
    result: Doc = {
        "step": at.step,
        "step_id": trace.step_numbers[at.step],
        "channel": at.channel.value,
        "source": step.source,
    }
    if at.observation is not None and step.observations[at.observation].pairing_reconstructed:
        result["pairing_reconstructed"] = True
    if at.channel == Channel.METADATA:
        result.update(before="", match=_metadata(step), after="")
        return result
    content, tool = _surface(step, at)
    if tool is not None:
        result["tool"] = tool
    before, match, after = _excerpt(content.text if content is not None else "", at, known)
    result.update(before=before, match=match, after=after)
    # Keys already set (pairing_reconstructed) keep their place in the citation.
    result.update(_context(step, at, known))
    return result


def _excerpt(text: str, at: Locator, known: frozenset[str]) -> tuple[str, str, str]:
    """The match with masked context on each side, else the masked head of the text."""
    window = (
        _window(text.replace(MARK, " "), *at.span, known)
        if at.span and at.span[1] <= len(text)
        else None
    )
    return window if window is not None else ("", _head(text, 2 * WINDOW, known), "")


def _fields_head(call: ToolCall | None, known: frozenset[str]) -> str:
    return _head("\n".join(c.text for _, c in call.fields), known=known) if call else ""


def _context(step: Step, at: Locator, known: frozenset[str]) -> Doc:
    """Why the agent did it (same step's reasoning/message) and what came back."""
    if at.call is not None:
        return _call_context(step, step.calls[at.call], known)
    if at.channel == Channel.OBSERVATION and at.observation is not None:
        source = step.observations[at.observation].source_call_id
        call = next((c for c in step.calls if c.id == source), None)
        return {"context_before": _fields_head(call, known), "context_after": ""}
    first = step.calls[0] if step.calls else None
    return {"context_before": "", "context_after": _fields_head(first, known)}


def _call_context(step: Step, call: ToolCall, known: frozenset[str]) -> Doc:
    intent = step.reasoning.text or step.message.text
    observation = next((o for o in step.observations if o.source_call_id == call.result_key), None)
    output = observation.content.text if observation is not None else None
    out: Doc = {}
    if observation is not None and observation.pairing_reconstructed:
        out["pairing_reconstructed"] = True
    out["context_before"] = _tail(intent, known=known) if intent else ""
    out["context_after"] = _head(output, known=known) if output else ""
    return out


def citations(
    trace: Trace,
    assessments: tuple[Assessment, ...],
    minimum: Severity,
    checks: tuple[str, ...] = (),
) -> dict[str, list[Doc]]:
    """Citations for unexcused findings at/above `minimum`, keyed by check ID; with
    `checks` (ID globs), only for the checks they select."""
    out: dict[str, list[Doc]] = {}
    known = trace_secrets(trace)
    for a in assessments:
        if not (a.counts and a.spec.severity >= minimum and a.result.evidence):
            continue
        if not checks or check_selected(a.spec.id, checks):
            out[a.spec.id] = [cite(trace, at, known) for at in a.result.evidence[:PER_FINDING]]
    return out
