"""Terminal colour for the brief: section labels, marks and severities (rich, terminal only)."""

from __future__ import annotations

import re
from typing import IO, TYPE_CHECKING

from ..document import STYLE

if TYPE_CHECKING:
    from rich.text import Text

from .words import LABEL, MARK_HANG

LABELS = "WEB|RUN|SCORE|REPLACED|REVIEW|FINDINGS|AWARENESS|EVIDENCE|TOKENS|COST|SETTINGS|MORE"


SEVERITY_STYLE = {**STYLE, "none": "dim", "unavailable": "yellow"}


# A · context row is dim. wrap() continues that row MARK_HANG spaces past the label
# column, under the text after the mark, so the row pattern misses the rest.
_CONTEXT_LINE = rf"^ {{{LABEL}}}· .*$"
_CONTEXT_LINE_RE = re.compile(_CONTEXT_LINE, re.M)
_MARK_WRAP = re.compile(rf"^ {{{LABEL + MARK_HANG}}}\S")


PATTERNS = [
    (r"^atif-scan .*$", "bold"),
    (r"^Findings set review priority.*$", "dim"),
    (rf"^(?:{LABELS})\b", "bold cyan"),
    (r"✓", "bold green"),
    (r"⚠", "bold yellow"),
    (r"→", "bold magenta"),
    (_CONTEXT_LINE, "dim"),
    (r"(?<=^SCORE {6})\d+\.\d+%(?: ± \d+\.\d+)?", "bold"),
    (r"\best\. [^;·(]*?\d[\d.,–%$<]*", "magenta"),
    (r"^ {11}priority +trials +rewarded +finding$", "dim underline"),
    (r"^ {11}trials +rewarded +the agent$", "dim underline"),
    (r"^ {33,}[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+$", "dim"),
    (r"^--summary .*$", "dim"),
]


COMPILED = [(re.compile(pattern, re.M), style) for pattern, style in PATTERNS] + [
    # Severity words where they are severities: table rows and "critical 38".
    (re.compile(rf"(?<=^ {{11}}){word}\b|\b{word}(?= \d)", re.M), style)
    for word, style in SEVERITY_STYLE.items()
]


def _wraps_context(line: str, hanging: bool) -> bool:
    """A wrap() continuation of a · row, not a new mark or a table row."""
    return hanging and _MARK_WRAP.match(line) is not None


def colourise(text: str) -> Text:
    """The brief as a rich Text with styles applied by pattern.

    A wrapped continuation of a · context line stays dim. The line pattern only
    matches the row that starts with the mark.
    """
    from rich.text import Text  # noqa: PLC0415 - rich is optional, imported only to colour

    out = Text()
    hanging = False
    for line in text.splitlines(keepends=True):
        continuing = _wraps_context(line, hanging)
        styled = Text(line)
        if continuing:
            styled.stylize("dim")
        for pattern, style in COMPILED:
            styled.highlight_regex(pattern, style)
        out.append_text(styled)
        hanging = continuing or _CONTEXT_LINE_RE.match(line) is not None
    return out


def print_brief(text: str, file: IO[str] | None = None) -> None:
    """Coloured on a terminal (rich), plain otherwise."""
    try:
        from rich.console import Console  # noqa: PLC0415 - rich is optional
    except ImportError:
        print(text, end="", file=file)
        return
    console = Console(file=file, highlight=False, soft_wrap=True)
    if not console.is_terminal:
        print(text, end="", file=file)
        return
    console.print(colourise(text), end="")
