"""Small immutable ATIF projection. Raw text is memory-only and hidden from repr."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from datetime import datetime

    from .web_inputs import WebInput


class Channel(StrEnum):
    MESSAGE = "message"
    REASONING = "reasoning"
    COMMAND = "command"
    PATH = "path"
    QUERY = "query"
    URL = "url"
    OBSERVATION = "observation"
    # Tool-argument strings no rule classified (any tool). Still scanned by text checks.
    ARGUMENTS = "arguments"
    # Argument data that is not an action target: file contents, edits, prose, prompts,
    # search patterns. Available to plugins; built-ins don't scan it.
    PAYLOAD = "payload"
    # Recorded step metadata such as step_id and timestamp; used by integrity checks.
    METADATA = "metadata"


# Channels whose meaning depends on recognizing the tool that produced them.
TOOL_INPUT_CHANNELS = frozenset({Channel.COMMAND, Channel.PATH, Channel.QUERY, Channel.URL})
# Presence only: ATIF reasoning_content does not identify full text versus summaries.
REASONING_EXPOSURE = ("recorded", "withheld", "none")


SPAN_LENGTH = 2  # (start, end)


@dataclass(frozen=True)
class Locator:
    step: int
    channel: Channel
    call: int | None = None
    observation: int | None = None
    # Which classified argument of the call (index into ToolCall.fields).
    field: int | None = None
    # Character offsets of the match within the surface text, when a detector knows them.
    span: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("invalid locator index")
        for value in (self.call, self.observation, self.field):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("invalid locator index")
        if not isinstance(self.channel, Channel):
            raise ValueError("invalid locator channel")
        if self.span is not None and not (
            isinstance(self.span, tuple)
            and len(self.span) == SPAN_LENGTH
            and all(type(v) is int for v in self.span)
            and 0 <= self.span[0] <= self.span[1]
        ):
            raise ValueError("invalid locator span")


@dataclass(frozen=True)
class Content:
    text: str = field(default="", repr=False)
    understood: bool = True
    # An image/audio/video block (or a harness placeholder for one, e.g. "[Image 1]") was
    # part of it: text the agent saw there isn't in `text`.
    media: bool = False
    # sha256 of each inline media payload in it, when every media part has one (see
    # `data.media`); empty when any part is only a placeholder, file reference or
    # unreadable payload. Identities only: the payload itself is never kept.
    media_ids: tuple[str, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class Surface:
    at: Locator
    content: Content = field(repr=False)
    tool: str | None = None  # normalized category
    tool_name: str | None = field(default=None, repr=False)  # as recorded


@dataclass(frozen=True)
class ToolCall:
    index: int
    # Normalized category: shell, read, write, search_files, web_fetch, web_search,
    # inert (orchestration with no evidence value) or other (unrecognized).
    tool: str
    fields: tuple[tuple[Channel, Content], ...] = field(repr=False)
    id: str = field(repr=False)
    name: str = field(repr=False)
    arguments: Mapping[str, object] | None = field(repr=False)
    # The call whose observation holds this call's result: itself, or for a call read
    # from inside a tool program (Codex code mode), the program call.
    result_id: str | None = field(default=None, repr=False)
    web_input: WebInput | None = field(default=None, repr=False)

    # Validated provider status; absent/unrecognized metadata remains unknown.
    status: Literal["incomplete", "completed", "in_progress", "failed"] | None = None
    # Authored payload only; None means unavailable (also used for derived calls).
    raw_argument_chars: int | None = None
    raw_argument_basis: Literal["recorded_json", "raw_text", "compact_json"] | None = None

    @property
    def result_key(self) -> str:
        return self.result_id if self.result_id is not None else self.id


@dataclass(frozen=True)
class Observation:
    source_call_id: str | None = field(repr=False)
    content: Content = field(repr=False)
    # The loader inferred this link, not an exported call ID.
    pairing_reconstructed: bool = False
    # Neither valid positional pairing nor a unique recorded remainder was available.
    # Checks needing the producing call must treat this observation as unresolved.
    pairing_unresolved: bool = False
    # Step-local recorded call index; never a synthesized provider identifier.
    source_call_index: int | None = None
    pairing_method: Literal["position", "unique_remainder"] | None = None
    # One terminal capture for all of the step's typed input (a Terminus 2 batch): it
    # records the batch's output, not which call printed what, so it stays unresolved.
    shared_terminal: bool = False


@dataclass(frozen=True)
class Step:
    index: int
    source: str
    copied: bool
    message: Content = field(repr=False)
    reasoning: Content = field(repr=False)
    calls: tuple[ToolCall, ...] = ()
    observations: tuple[Observation, ...] = field(default=(), repr=False)
    # Recorded metadata. `*_recorded` distinguishes "absent" from "present but invalid".
    step_id: int | None = None
    step_id_recorded: bool = False
    timestamp: datetime | None = None
    timestamp_recorded: bool = False
    # A non-agent step carrying agent-only fields (tool_calls, reasoning, metrics, model).
    agent_only_fields: bool = False
    # metrics.completion_tokens when recorded as a non-negative integer.
    completion_tokens: int | None = None
    # The step's own `model_name` (label-safe), which can differ from the root agent
    # block's: a harness falling back to another model mid-run keeps the configured
    # model in the header (Terminus 2 on TB2.1: 79 of 445 Fable 5 trials ran Opus 4.8).
    model_name: str | None = None
    # Validated metrics.extra.reasoning_tokens for this step, including zero.
    reasoning_tokens: int | None = None

    @property
    def authored(self) -> bool:
        """The agent's own step in this session (not context copied from elsewhere)."""
        return self.source == "agent" and not self.copied

    def authored_surfaces(self) -> Iterator[Surface]:
        """Message, reasoning and every classified call argument, in that order."""
        yield Surface(Locator(self.index, Channel.MESSAGE), self.message)
        yield Surface(Locator(self.index, Channel.REASONING), self.reasoning)
        for call in self.calls:
            for field_index, (channel, content) in enumerate(call.fields):
                at = Locator(self.index, channel, call.index, field=field_index)
                yield Surface(at, content, call.tool, call.name)

    def observation_surfaces(self) -> Iterator[Surface]:
        for index, observation in enumerate(self.observations):
            at = Locator(self.index, Channel.OBSERVATION, observation=index)
            yield Surface(at, observation.content)

    def results_for(self, call: ToolCall) -> list[tuple[int, Observation]]:
        """Results linked by recorded ID or index, including a derived call's parent.

        Empty IDs never link calls to each other. The legacy lone-call fallback only
        includes observations with neither an ID nor an indexed association.
        """
        key = call.result_key
        parent_index = call.index if call.result_id is None else None
        if call.result_id:
            parents = [c for c in self.calls if c.result_id is None and c.id == key]
            if len(parents) == 1:
                parent_index = parents[0].index
        linked = [
            (j, o)
            for j, o in enumerate(self.observations)
            if (o.source_call_index is not None and o.source_call_index == parent_index)
            or (o.source_call_index is None and key and o.source_call_id == key)
        ]
        if linked or len(self.calls) != 1:
            return linked
        return [
            (j, o)
            for j, o in enumerate(self.observations)
            if o.source_call_id is None and o.source_call_index is None
        ]


@dataclass(frozen=True)
class Usage:
    """ATIF final_metrics: total_cost_usd and prompt/completion/cached token totals."""

    cost_usd: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None  # total_reasoning_tokens or reasoning_output_tokens
    # Prompt tokens written to the provider's cache (Anthropic cache creation), part of
    # the prompt tokens: final_metrics.extra.total_cache_creation_input_tokens.
    cache_write_tokens: int | None = None
    # Recorded counts are never normalized. Only verified exporter versions override this.
    completion_basis: Literal["includes_reasoning", "separate_visible"] = field(
        default="includes_reasoning", kw_only=True
    )


# A trace's results read as status-only when at least this many are recorded and this
# share of them is a bare status word.
MIN_STATUS_RESULTS = 5
# Fewer metered calls than this: the one call's usage is the run's own total.
MIN_METERED_CALLS = 2
STATUS_ONLY_SHARE = 0.9
STATUS_ONLY = frozenset(
    {
        "success",
        "failure",
        "failed",
        "error",
        "ok",
        "done",
        "true",
        "false",
        "completed",
        "null",
        "none",
        "✓",
        "",
    }
)
ACTION_CLAIM = re.compile(
    r"\b(?:created|wrote|written|ran|installed|implemented|fixed|saved|executed|built|"
    r"configured|updated|modified)\b",
    re.I,
)


@dataclass(frozen=True)
class Trace:
    schema_version: str | None
    steps: tuple[Step, ...]
    # final_metrics.extra.total_tool_use_tokens when reported as an integer.
    tool_use_tokens: int | None = None
    # final_metrics totals as recorded by the harness (None when absent/invalid).
    usage: Usage | None = None
    # Steps whose system/user message says earlier history was compacted/summarized away:
    # the recorded steps then cover only part of the session.
    compacted: tuple[int, ...] = ()
    # Steps where the harness recorded compacting the agent's context while keeping the
    # earlier steps in the file (fast-agent `extra.context_management.type: compaction`).
    # Nothing is missing for a reader of the trace, but a tool call pending at that point
    # can lose its result. Not a recording gap: `compacted` is the history-gone case.
    context_compactions: tuple[int, ...] = ()
    # Sum of agent steps' `llm_call_count` when recorded (None when never recorded).
    llm_calls: int | None = None
    # ATIF root `agent` block: (name, version, model_name), each None when absent.
    agent: tuple[str | None, str | None, str | None] = (None, None, None)
    # Bare `[REDACTED]` JSON values the loader read as null (a publisher redaction defect).
    redacted_values: int = 0
    # Token usage summed over agent steps' own metrics (None when no step recorded any),
    # and LLM calls without usage (usage.step_usage): the fallback when final_metrics
    # has no totals.
    step_usage: Usage | None = None
    calls_without_usage: int = 0
    # Token kinds ("input", "output", "cached") only some metered steps record: their
    # step sums are lower bounds, never complete totals.
    step_kinds_partial: tuple[str, ...] = ()
    # The last metered agent step's own usage, how many agent steps recorded usage, and
    # whether every metered step's prompt and completion tokens are at least the
    # previous one's (the step metrics could then be running totals; usage.last_call).
    last_step_usage: Usage | None = None
    metered_steps: int = 0
    step_usage_rising: bool = False
    # Explicit fast-agent provider retries (legacy "stream" names). These are
    # harness events, not proof that usage or agent history is missing.
    stream_retry_steps: int = 0
    stream_retry_attempts: int = 0
    # Of those retried attempts, how many failed before any output (no stream event, or
    # an HTTP error status before a stream opened), and the HTTP statuses recorded.
    retry_before_output: int = 0
    retry_statuses: tuple[int, ...] = ()
    # The harness's recorded termination error type (`extra.termination.error_type`, a
    # label-safe code such as ResponsesWebSocketError); None when absent or not an error.
    termination_error: str | None = None
    # The last agent step's recorded stop reason as a lowercase code (fast-agent
    # `extra.stop_reason: "LlmStopReason.SAFETY"` -> "safety"); None when absent.
    final_stop_reason: str | None = None
    # final_metrics.extra.llm_usage_calls_complete as the harness recorded it (None: absent).
    usage_calls_complete: bool | None = None
    # Root-only totals after reconciling declared child usage with embedded records.
    root_usage: Usage | None = None

    @cached_property
    def totals_match_last_call(self) -> bool:
        """final_metrics prompt and completion totals equal the last metered step's own,
        while two or more metered steps sum to more. With rising step metrics they may
        be running totals instead (`step_usage_rising`), where that is correct."""
        usage, last, steps = self.usage, self.last_step_usage, self.step_usage
        if usage is None or last is None or steps is None or self.metered_steps < MIN_METERED_CALLS:
            return False
        if usage.prompt_tokens is None or usage.completion_tokens is None:
            return False
        same = (usage.prompt_tokens, usage.completion_tokens) == (
            last.prompt_tokens,
            last.completion_tokens,
        )
        more = (steps.prompt_tokens or 0) > usage.prompt_tokens or (
            steps.completion_tokens or 0
        ) > usage.completion_tokens
        return same and more

    @property
    def authored_usage(self) -> Usage | None:
        """Prefer reconciled root totals; canonical metrics remain unchanged."""
        return self.root_usage or self.usage

    @cached_property
    def step_models(self) -> dict[str, int]:
        """Agent steps per recorded step model (authored steps only)."""
        counts: dict[str, int] = {}
        for step in self.steps:
            if step.authored and step.model_name:
                counts[step.model_name] = counts.get(step.model_name, 0) + 1
        return counts

    @cached_property
    def results_unrecorded(self) -> bool:
        """Tool results were exported as bare status words ("success"/"failure") instead of
        output (ACE on TB2.1: every one of 12k results): checks on what the agent received
        can't be answered."""
        texts = [
            o.content.text.strip().strip('"').lower() for s in self.steps for o in s.observations
        ]
        return len(texts) >= MIN_STATUS_RESULTS and sum(
            t in STATUS_ONLY for t in texts
        ) >= STATUS_ONLY_SHARE * len(texts)

    @property
    def actions_unrecorded(self) -> bool:
        """The agent reports doing work but not one tool call was recorded (a TB2.1 harness
        exported 105 rewarded traces as the instruction plus "Done. Files created: …")."""
        if self.tool_calls or not self.agent_steps:
            return False
        return any(ACTION_CLAIM.search(s.message.text) for s in self.steps if s.authored)

    @property
    def head_missing(self) -> bool:
        """The first recorded step is the agent's: the prompt wasn't exported. A TB2.1
        harness exported 86 traces that way, keeping only the last ~27 steps (renumbered
        from 1). Recall checks can't call anything unprimed then."""
        for step in self.steps:
            if step.source in ("system", "user"):
                return False
            if step.source == "agent":
                return True
        return False

    @property
    def recording_gaps(self) -> bool:
        """Parts of the session aren't in the trace: negatives must not read as clean.
        (`head_missing` isn't one: many exporters simply don't record the prompt.)"""
        return bool(self.compacted) or self.results_unrecorded or self.actions_unrecorded

    @cached_property
    def reasoning_exposure(self) -> str:
        """Recorded reasoning presence, never inferred completeness.

        ATIF's reasoning_content can contain either a summary or reasoning text.
        Character/token ratios cannot distinguish them. Missing text with reported
        reasoning tokens is tokens-only; without either, exposure is unknown.
        """
        steps = [s for s in self.steps if s.authored]
        if any(s.reasoning.text for s in steps):
            return "recorded"
        usage = self.authored_usage
        tokens = usage.reasoning_tokens if usage else None
        if tokens is None and (usage is None or usage.completion_tokens is None):
            metered = [s for s in steps if s.completion_tokens is not None]
            tokens = step_reasoning_tokens(metered)
        return "withheld" if tokens else "none"

    @property
    def reasoning_hidden(self) -> bool:
        """Reasoning tokens were reported but no reasoning text was recorded."""
        return self.reasoning_exposure == "withheld"

    def agent_surfaces(self) -> Iterator[Surface]:
        """Never recursively walks a step: observations and prompt text stay separate."""
        for step in self.steps:
            if step.authored:
                yield from step.authored_surfaces()

    def agent_calls(self) -> Iterator[tuple[Step, ToolCall]]:
        for step in self.steps:
            if step.authored:
                for call in step.calls:
                    yield step, call

    def observation_surfaces(self) -> Iterator[Surface]:
        """Explicit opt-in for plugins verifying outcomes, never default authored evidence."""
        for step in self.steps:
            if not step.copied:
                yield from step.observation_surfaces()

    @cached_property
    def step_numbers(self) -> tuple[int, ...]:
        """The step number a reviewer looks up: the recorded ATIF `step_id` (1..n), or the
        1-based position when it's absent/invalid. Locators keep 0-based positions."""
        return tuple(
            s.step_id if s.step_id is not None and s.step_id > 0 else s.index + 1
            for s in self.steps
        )

    @cached_property
    def agent_steps(self) -> int:
        return sum(s.authored for s in self.steps)

    @cached_property
    def tool_calls(self) -> int:
        return sum(1 for _ in self.agent_calls())

    @property
    def unrecognized_tool_calls(self) -> int:
        return sum(call.tool == "other" for _, call in self.agent_calls())


def step_reasoning_tokens(steps: list[Step]) -> int | None:
    """A split only for a fully metered, reasoning-accounted comparison subset."""
    if not steps or any(s.reasoning_tokens is None for s in steps):
        return None
    return sum(s.reasoning_tokens or 0 for s in steps)
