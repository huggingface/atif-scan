"""Recall signals: the agent writes benchmark-specific knowledge that nothing in the trace
gave it ("unprimed"). Trained-on-test evidence is judged here by *priming*, not by content:
a token is unprimed when no earlier prompt, copied context or tool result contained it.

Priming is a plain case-insensitive substring test over all earlier context, so tokens
glued into hex dumps or listings still count as seen. Steps before a compaction summary
are gone, so nothing written after one can be called unprimed: scanning stops there and a
negative is `unknown`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from ..checks import CheckSpec, Context, Detection, Status, Unread
from ..data.model import Channel, Locator, Surface, Trace
from ..data.web_inputs import web_input
from ..data.web_results import web_results_complete

if TYPE_CHECKING:
    import re

PROSE_AND_INPUT = frozenset(
    {Channel.MESSAGE, Channel.REASONING, Channel.QUERY, Channel.COMMAND, Channel.ARGUMENTS}
)


def walk(trace: Trace) -> Iterator[tuple[bool, Surface]]:
    """(authored, surface) in trace order: an agent step's own text and call arguments
    come before that step's tool results; prompts and copied steps are context."""
    for step in trace.steps:
        if step.authored:
            for surface in step.authored_surfaces():
                yield True, surface
        else:
            yield False, Surface(Locator(step.index, Channel.MESSAGE), step.message)
        for surface in step.observation_surfaces():
            yield False, surface


Candidates = Callable[[Surface, Context], Iterable[tuple[str, tuple[int, int]]]]


def unrecorded_web_result(trace: Trace) -> Locator | None:
    """The first web search/fetch whose result wasn't recorded (Codex hosted web search).
    What the agent writes after it may be primed by that result: TB2.1 protein-assembly
    searched the instruction's own sentence and then cited tbench.ai."""
    for step, call in trace.agent_calls():
        if call.tool not in ("web_search", "web_fetch"):
            continue
        if not web_results_complete(step, call) or not web_input(call).source_known:
            return Locator(step.index, Channel.METADATA, call=call.index)
    return None


def _horizon(trace: Trace) -> tuple[int, Unread] | None:
    """The first step whose earlier context is incomplete: a compaction summary (earlier
    context is gone), or the step after a web result the trace doesn't hold."""
    limits = [(i, Unread("compacted", Locator(i, Channel.MESSAGE))) for i in trace.compacted]
    blind = unrecorded_web_result(trace)
    if blind is not None:
        limits.append((blind.step + 1, Unread("web_result_not_recorded", blind)))
    if not limits:
        return None
    return min(limits, key=lambda limit: limit[0])


@dataclass
class _Walk:
    """Tokens found while all earlier context was read, and after the first unread part.

    Unread context (an image, an unparsed result, compacted or unrecorded history) can
    only prime tokens, never add them: tokens written after it are `unconfirmed`, and a
    negative still holds when even those aren't enough. Unread *authored* text could hide
    a token, so it leaves the walk incomplete."""

    found: dict[str, Locator] = field(default_factory=dict)
    unconfirmed: dict[str, Locator] = field(default_factory=dict)
    unread: list[Unread] = field(default_factory=list)
    complete: bool = True

    def blind(self, unread: Unread) -> None:
        if not self.unread:
            self.unread.append(unread)

    @property
    def target(self) -> dict[str, Locator]:
        return self.unconfirmed if self.unread else self.found


@dataclass(frozen=True)
class UnprimedDetector:
    """Match authored surfaces (on `channels`) holding a candidate token that no earlier
    context contained. `candidates` yields (token, span); `enough` decides from the set of
    distinct unprimed tokens whether the trace matches (default: any).

    When some earlier context couldn't be read, tokens written after it may have been
    primed there: the result is unknown, with those tokens' first positions as evidence
    and the unread place in `Detection.unread`."""

    spec: CheckSpec
    channels: frozenset[Channel]
    candidates: Candidates = field(repr=False)
    enough: Callable[[set[str], Context], bool] = field(
        default=lambda found, context: bool(found), repr=False
    )
    stop: Callable[[Surface], bool] | None = field(default=None, repr=False)
    # Applied to tokens and context before the substring test (spelling variants).
    fold: Callable[[str], str] = field(default=str.lower, repr=False)

    def evaluate(self, trace: Trace, context: Context) -> Detection:
        walk_ = self._scan(trace, context)
        # Withheld reasoning can't prime anything (it's the agent's own); the recorded
        # text is what's judged, and `Trace.reasoning_exposure` says what it covers.
        complete = trace.agent_steps > 0 and walk_.complete
        unread = tuple(walk_.unread)
        if self.enough(set(walk_.found), context):
            # Found before any unread context: nothing later can prime them.
            return Detection(Status.MATCH, tuple(walk_.found.values()), complete)
        possible = {**walk_.found, **walk_.unconfirmed}
        if unread and self.enough(set(possible), context):
            return Detection(Status.UNKNOWN, tuple(possible.values()), False, unread)
        if not complete:
            return Detection(Status.UNKNOWN, (), False, unread)
        return Detection(Status.NO_MATCH)

    def _scan(self, trace: Trace, context: Context) -> _Walk:
        """Collect unprimed tokens, splitting those written after the first unread
        context from those that every earlier context was read for."""
        state = _Walk()
        seen: list[str] = []
        # Tokens already found in `seen`: it only grows, so they stay primed.
        primed: set[str] = set()
        if trace.head_missing:
            state.blind(Unread("prompt_not_recorded"))
        horizon = _horizon(trace)
        for authored, surface in walk(trace):
            if horizon is not None and surface.at.step >= horizon[0]:
                state.blind(horizon[1])
            if not authored:
                self._read_context(surface, seen, state)
                continue
            if self.stop is not None and self.stop(surface):
                break  # e.g. a benchmark lookup: what follows may be primed by it
            if self._judged(surface):
                state.complete = state.complete and surface.content.understood
                self._collect(surface, context, seen, primed, state)
        return state

    def _read_context(self, surface: Surface, seen: list[str], state: _Walk) -> None:
        """Add a prompt/result's readable text to `seen`; an image in it, or a part that
        couldn't be parsed, may hold more: what's written later is then unconfirmed."""
        if surface.content.media:
            state.blind(Unread("media", surface.at))
        elif not surface.content.understood:
            state.blind(Unread("unreadable", surface.at))
        if surface.content.text:
            seen.append(self.fold(surface.content.text))

    def _judged(self, surface: Surface) -> bool:
        channel = surface.at.channel
        return channel in self.channels and not (
            channel == Channel.PAYLOAD and surface.tool == "shell"
        )

    def _collect(
        self,
        surface: Surface,
        context: Context,
        seen: list[str],
        primed: set[str],
        state: _Walk,
    ) -> None:
        for token, span in self.candidates(surface, context):
            key = self.fold(token)
            if key in state.found or key in state.unconfirmed or key in primed:
                continue
            if any(key in s for s in seen):
                primed.add(key)
                continue
            state.target[key] = replace(surface.at, span=span)


def regex_candidates(pattern: re.Pattern[str]) -> Candidates:
    def candidates(surface: Surface, context: Context) -> Iterator[tuple[str, tuple[int, int]]]:
        for m in pattern.finditer(surface.content.text):
            yield m.group(0), m.span()

    return candidates
