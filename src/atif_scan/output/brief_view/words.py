"""Shared vocabulary of the run brief: marks, widths, words and numbers (plurals, percentages,
money), and the label/wrap layout every section uses."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ...data.jsonval import Doc


OK, WARN, INFO = "✓", "⚠", "·"


MARKS = (OK, WARN, INFO, "→")


WIDTH = 100


LABEL = 11  # the label column, including its trailing space


PAD = " " * LABEL


NBSP = "\u00a0"  # glues a separator to the word before it while wrapping


# How many medium+ checks FINDINGS lists before pointing at --summary.
CHECKS_SHOWN = 8


# Harbor error types named on the EVIDENCE line; the rest are counted.
ERROR_KINDS_SHOWN = 3


# Below this many traces a p5–p95 band is just the extremes: show the range instead.
MIN_TRACES_FOR_PERCENTILES = 20


# A missing-calls estimate at or above this share of the run's calls is run-level news.
MISSING_CALLS_NOTABLE_PCT = 1


# Below half a cent a positive amount reads "<$0.01"; within a cent two costs are equal.
HALF_CENT = 0.005


CENT = 0.01


PRIORITIES = ("critical", "high", "medium", "low", "info", "none", "unavailable")


RUN_KINDS = {"harbor_hub": "Harbor job", "harbor_leaderboard_row": "leaderboard row"}


REASONING = {
    "full": "full text",
    "summarised": "summarised",
    "recorded": "text without a token count",
    "withheld": "withheld (tokens only)",
    "none": "not exposed",
}


RATIO_BASIS = {
    "answer_only": "reasoning excluded (its tokens are reported separately)",
    "all_text": "reasoning included (summaries read lower)",
    "visible_only": "no usable reasoning split for this comparison; lower bound cannot be checked",
}


# Why a rewarded trial can't be cleared (report.uncleared_reasons), in plain words.
NOT_CLEARED = {
    "tool input not readable": "tool input unreadable",
}


Lines = list[str]


# The DQ threshold in words: "a rewarded trial with <level> finding".
LEVEL = {"critical": "a critical", "high": "a high or critical", "medium": "a medium or higher"}


def plural(n: int, noun: str, many: str | None = None) -> str:
    """`1 trial`, `2 trials`, `1,204 findings`; `many` for irregular plurals."""
    return f"{n:,} {noun if n == 1 else many or noun + 's'}"


def has(n: int) -> str:
    """The verb for a count as subject: `1 trial has`, `2 trials have`."""
    return "has" if n == 1 else "have"


def was(n: int) -> str:
    return "was" if n == 1 else "were"


def pct(n: float, total: float) -> str:
    return f"{100 * n / total:.1f}%" if total else "—"


def _tally(value: object) -> dict[str, int]:
    """A brief {code: count} tally (built by brief.py), typed for this view."""
    return {str(k): int(v) for k, v in value.items()} if isinstance(value, dict) else {}


def usd(value: float) -> str:
    """Dollars to the cent; a positive amount under half a cent reads `<$0.01`."""
    return "<$0.01" if 0 < value < HALF_CENT else f"${value:,.2f}"


def rate(value: float) -> str:
    """A $/M-token price: cents when it has them (`$0.15`, `$15.00`), else as written
    (`$0.003`)."""
    return f"${value:,.2f}" if round(value, 2) == value else f"${value:g}"


def counts(pairs: Iterable[tuple[str, int]]) -> str:
    """`AgentTimeoutError 12 · NonZeroAgentExitCodeError 5`."""
    return " · ".join(f"{k}{NBSP}{v:,}" for k, v in pairs)  # a count stays with its label


def spread(r: Doc) -> str:
    """A distribution's middle: p5–p95 over enough traces, else the range."""
    if r["traces"] == 1:
        return ""
    if r["traces"] < MIN_TRACES_FOR_PERCENTILES:
        return f" (range {r['min']:.2f}–{r['max']:.2f})"
    return f" (p5–p95 {r['p5']:.2f}–{r['p95']:.2f})"


def wrap(label: str, body: Lines) -> Lines:
    """A section: the label on its first line, every line wrapped to WIDTH. A body line
    that starts with a mark wraps under the text after the mark; a line starting with
    spaces is pre-formatted (a table row) and kept as it is."""
    out: Lines = []
    for j, text in enumerate(body):
        head = f"{label if j == 0 else '':<{LABEL}}"
        if text.startswith(" "):
            out.append((head + text[1:]).replace(NBSP, " ").rstrip())
            continue
        indent = PAD + ("  " if text[:1] in MARKS and text[1:2] == " " else "")
        # A " · " separator never starts a wrapped line (it would read as a mark).
        glued = text[:2] + text[2:].replace(" · ", NBSP + "· ")
        out += [
            line.replace(NBSP, " ")
            for line in textwrap.wrap(
                glued,
                WIDTH,
                initial_indent=head,
                subsequent_indent=indent,
                break_long_words=False,
                break_on_hyphens=False,
            )
        ] or [head.rstrip()]
    return out


def _title(b: Doc, check: str) -> str:
    return str(b.get("titles", {}).get(check) or check)


def _title_inline(b: Doc, check: str) -> str:
    """A title inside a sentence: lower-cased first letter, unless it starts an acronym
    ("CIFAR labels…", "SSH server…")."""
    title = _title(b, check)
    if title == check or title[1:2].isupper():
        return title
    return title[:1].lower() + title[1:]


def _present(b: Doc) -> int:
    return int(b["overview"]["trials"]["present"])


def _rewards_known(b: Doc) -> bool:
    return bool(b["overview"]["accuracy"])


def _rewarded(b: Doc) -> int:
    """Rewarded trials, from the recorded accuracy over scored trials."""
    ov = b["overview"]
    if not ov["accuracy"]:
        return 0
    scored = _present(b) - int(ov["trials"]["reward_unknown"])
    return round(float(ov["accuracy"][0]) * scored / 100)
