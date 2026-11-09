"""Source-line path receipts in plain/known command-output envelopes.

No source text is returned, and no path is opened. The consumer must bind these
facts to an actual filesystem-search call; a formatted line alone proves nothing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .jsonval import as_object, as_str

MAX_RESULT = 512 * 1024
SOURCE_LINE = re.compile(
    r"^(?P<path>(?:/|~/)[^\r\n:]+(?:\.py|\.pyi|\.pyx|\.pxd)):[0-9]+:[^\r\n]*\S", re.M
)


@dataclass(frozen=True)
class SourceLines:
    paths: tuple[str, ...] = ()
    complete: bool = True


def source_lines(text: str) -> SourceLines:
    if len(text) > MAX_RESULT:
        return SourceLines(complete=False)
    parts = [text]
    complete = True
    if text.lstrip().startswith("{"):
        try:
            fields = as_object(json.loads(text))
        except (ValueError, RecursionError):
            return SourceLines(complete=False)
        parts = [
            value
            for key in ("output", "stdout", "output_head", "output_tail")
            if (value := as_str(fields.get(key))) is not None
        ]
        if not parts:
            return SourceLines(complete=False)
        complete = not any(key in fields for key in ("output_head", "output_tail"))
        complete &= not bool(fields.get("truncated") or fields.get("elided_chars"))
    return SourceLines(
        tuple(
            dict.fromkeys(match["path"] for part in parts for match in SOURCE_LINE.finditer(part))
        ),
        complete,
    )
