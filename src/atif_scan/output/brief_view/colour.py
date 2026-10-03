"""Terminal colour for the brief: section labels, marks and severities (rich, terminal only)."""

from __future__ import annotations

import re
from typing import IO, TYPE_CHECKING

from ..document import STYLE

if TYPE_CHECKING:
    from rich.text import Text


LABELS = "WEB|RUN|SCORE|REVIEW|FINDINGS|AWARENESS|EVIDENCE|TOKENS|COST|SETTINGS|MORE"


SEVERITY_STYLE = {**STYLE, "none": "dim", "unavailable": "yellow"}


PATTERNS = [
    (r"^atif-scan .*$", "bold"),
    (r"^Findings set review priority.*$", "dim"),
    (rf"^(?:{LABELS})\b", "bold cyan"),
    (r"✓", "bold green"),
    (r"⚠", "bold yellow"),
    (r"→", "bold magenta"),
    (r"^ {11}· .*$", "dim"),
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


def colourise(text: str) -> Text:
    """The brief as a rich Text with styles applied by pattern."""
    from rich.text import Text  # noqa: PLC0415 - rich is optional, imported only to colour

    out = Text()
    for line in text.splitlines(keepends=True):
        styled = Text(line)
        for pattern, style in COMPILED:
            styled.highlight_regex(pattern, style)
        out.append_text(styled)
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
