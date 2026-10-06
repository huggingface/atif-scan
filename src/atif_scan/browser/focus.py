"""Conservative raw-to-masked focus mapping; never return transformed trace text."""

from __future__ import annotations

from ..data.credentials import PRIVATE_KEY
from ..data.jsonval import Doc, count
from ..evidence.cite import _secret_spans, mask

# Newlines keep markers out of single-line header/assignment replacements.
MARK = "\n\x00\n"
SPAN_SIZE = 2
MARKED_PARTS = 3


def _span(value: object, length: int) -> tuple[int, int] | None:
    if not isinstance(value, (tuple, list)) or len(value) != SPAN_SIZE:
        return None
    start, end = count(value[0]), count(value[1])
    if start is None or end is None or not start < end <= length:
        return None
    return start, end


def _expand(text: str, span: tuple[int, int], known: frozenset[str]) -> tuple[int, int]:
    """Merge overlapping secrets first so expansion covers their transitive union."""
    ranges = _secret_spans(text, known)
    ranges.extend(match.span() for match in PRIVATE_KEY.finditer(text))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start < merged[-1][1]:
            merged[-1] = merged[-1][0], max(end, merged[-1][1])
        else:
            merged.append((start, end))
    start, end = span
    for left, right in merged:
        if left < start < right:
            start = left
        if left < end < right:
            end = right
    return start, end


def masked_focus(text: str, span: object, known: frozenset[str] = frozenset()) -> Doc:
    """Return only proven global Unicode offsets, or an explicit field fallback."""
    fallback: Doc = {"focus_status": "field", "highlight_start": None, "highlight_end": None}
    bounds = _span(span, len(text))
    if bounds is None or "\x00" in text:
        return fallback
    baseline = mask(text, known)
    start, end = _expand(text, bounds, known)
    marked = mask(text[:start] + MARK + text[start:end] + MARK + text[end:], known)
    parts = marked.split(MARK)
    if len(parts) != MARKED_PARTS or "".join(parts) != baseline or not parts[1]:
        return fallback
    return {
        "focus_status": "exact" if (start, end) == bounds else "masked_region",
        "highlight_start": len(parts[0]),
        "highlight_end": len(parts[0]) + len(parts[1]),
    }
