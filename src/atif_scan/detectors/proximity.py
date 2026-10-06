"""Word proximity in prose: lexicon phrases in order, a few words apart, in one sentence.

A regex says "these words, in this order"; it can't say "a task word, then within four
words a provenance link, then within three an evaluation noun" without listing every
filler. `Chain` does, over a lowercase word stream that keeps each word's offsets (for
citations). Words are ASCII letters/digits joined by `.`, `_` or an apostrophe, so
`eval.py` and `don't` are one word, and `benchmark-style` or `harbor/agentic` are two.
Sentences end at `.`/`!`/`?` before a space, or a line break.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

WORD = re.compile(r"[a-z0-9]+(?:[._'][a-z0-9]+)*")
SENTENCE_END = re.compile(r"[.!?]+(?=\s)|\n")

Phrase = tuple[str, ...]
Span = tuple[int, int]


@dataclass(frozen=True)
class Words:
    """A text's lowercase words, with offsets into the text and sentence numbers."""

    text: tuple[str, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    sentence: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.text)


def words(text: str) -> Words:
    low = text.replace("’", "'").lower()
    if len(low) != len(text):  # a few non-ASCII letters lengthen when lowercased
        low = "".join(c.lower() if c.isascii() else " " for c in text.replace("’", "'"))
    boundaries = [m.end() for m in SENTENCE_END.finditer(text)]
    found = list(WORD.finditer(low))
    return Words(
        tuple(m.group() for m in found),
        tuple(m.start() for m in found),
        tuple(m.end() for m in found),
        tuple(bisect_right(boundaries, m.start()) for m in found),
    )


@dataclass(frozen=True)
class Slot:
    """Alternative phrases for one position of a chain, indexed by their first word."""

    by_first: dict[str, tuple[Phrase, ...]] = field(repr=False)

    def at(self, ws: Words, i: int) -> int | None:
        """Length of a phrase of this slot starting at word i, within one sentence."""
        for phrase in self.by_first.get(ws.text[i], ()):
            n = len(phrase)
            if (
                i + n <= len(ws)
                and ws.sentence[i + n - 1] == ws.sentence[i]
                and ws.text[i : i + n] == phrase
            ):
                return n
        return None


def slot(*items: str) -> Slot:
    """A slot from space-separated phrases: slot("from", "part of")."""
    by_first: dict[str, tuple[Phrase, ...]] = {}
    for item in items:
        phrase = tuple(item.split())
        by_first[phrase[0]] = (*by_first.get(phrase[0], ()), phrase)
    return Slot(by_first)


@dataclass(frozen=True)
class Chain:
    """Slots in order; `gaps[k]` words at most between slot k and slot k+1 (0: adjacent).
    The whole chain lies in one sentence."""

    slots: tuple[Slot, ...]
    gaps: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.slots or len(self.gaps) != len(self.slots) - 1:
            raise ValueError("chain_needs_one_gap_between_each_pair_of_slots")

    def finditer(self, ws: Words) -> Iterator[Span]:
        """Each match's character span, from its first word to its last, by start."""
        for i in range(len(ws)):
            last = self._from(ws, i, 0)
            if last is not None:
                yield ws.starts[i], ws.ends[last]

    def search(self, ws: Words) -> Span | None:
        return next(self.finditer(ws), None)

    def _from(self, ws: Words, i: int, k: int) -> int | None:
        """Index of the last word when slots k… match starting exactly at word i."""
        n = self.slots[k].at(ws, i)
        if n is None:
            return None
        last = i + n - 1
        return last if k + 1 == len(self.slots) else self._next(ws, i, last, k)

    def _next(self, ws: Words, i: int, last: int, k: int) -> int | None:
        """Index of the chain's last word when slot k+1 starts at most `gaps[k]` words
        after word `last`, in word i's sentence."""
        for j in range(last + 1, min(last + 2 + self.gaps[k], len(ws))):
            if ws.sentence[j] != ws.sentence[i]:
                break
            end = self._from(ws, j, k + 1)
            if end is not None:
                return end
        return None


def near(ws: Words, span: Span, cues: Iterable[str], within: int) -> bool:
    """Whether a cue is among the `within` words either side of `span`, in any sentence
    (a cue may sit in the sentence before: "…harnesses do? This task looks like an eval")."""
    cue = frozenset(cues)
    first = bisect_right(ws.ends, span[0])  # first word not ending before the span
    after = bisect_left(ws.starts, span[1])  # first word starting after it
    return any(w in cue for w in ws.text[max(first - within, 0) : first]) or any(
        w in cue for w in ws.text[after : after + within]
    )
