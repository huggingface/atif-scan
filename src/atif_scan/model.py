"""Small immutable ATIF projection. Raw text is memory-only and hidden from repr."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


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


@dataclass(frozen=True, order=True)
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
        for value in (self.step, self.call, self.observation, self.field):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("invalid locator index")
        if not isinstance(self.channel, Channel):
            raise ValueError("invalid locator channel")
        if self.span is not None and not (
            isinstance(self.span, tuple)
            and len(self.span) == 2
            and all(type(v) is int for v in self.span)
            and 0 <= self.span[0] <= self.span[1]
        ):
            raise ValueError("invalid locator span")


@dataclass(frozen=True)
class Content:
    text: str = field(default="", repr=False)
    understood: bool = True


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


@dataclass(frozen=True)
class Observation:
    source_call_id: str | None = field(repr=False)
    content: Content = field(repr=False)


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


@dataclass(frozen=True)
class Usage:
    """ATIF final_metrics: total_cost_usd and prompt/completion/cached token totals."""

    cost_usd: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None  # final_metrics.extra.total_reasoning_tokens


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
    # Sum of agent steps' `llm_call_count` when recorded (None when never recorded).
    llm_calls: int | None = None
    # ATIF root `agent` block: (name, version, model_name), each None when absent.
    agent: tuple[str | None, str | None, str | None] = (None, None, None)

    @property
    def reasoning_hidden(self) -> bool:
        """Reasoning tokens were reported but no reasoning text was recorded."""
        return bool(self.usage and self.usage.reasoning_tokens) and not any(
            s.reasoning.text for s in self.steps if s.source == "agent" and not s.copied
        )

    def agent_surfaces(self) -> Iterator[Surface]:
        """Never recursively walks a step: observations and prompt text stay separate."""
        for step in self.steps:
            if step.source != "agent" or step.copied:
                continue
            yield Surface(Locator(step.index, Channel.MESSAGE), step.message)
            yield Surface(Locator(step.index, Channel.REASONING), step.reasoning)
            for call in step.calls:
                for field_index, (channel, content) in enumerate(call.fields):
                    at = Locator(step.index, channel, call.index, field=field_index)
                    yield Surface(at, content, call.tool, call.name)

    def agent_calls(self) -> Iterator[tuple[Step, ToolCall]]:
        for step in self.steps:
            if step.source == "agent" and not step.copied:
                for call in step.calls:
                    yield step, call

    def observation_surfaces(self) -> Iterator[Surface]:
        """Explicit opt-in for plugins verifying outcomes, never default authored evidence."""
        for step in self.steps:
            if step.copied:
                continue
            for index, observation in enumerate(step.observations):
                yield Surface(
                    Locator(step.index, Channel.OBSERVATION, observation=index), observation.content
                )

    @property
    def agent_steps(self) -> int:
        return sum(s.source == "agent" and not s.copied for s in self.steps)

    @property
    def tool_calls(self) -> int:
        return sum(1 for _ in self.agent_calls())

    @property
    def unrecognized_tool_calls(self) -> int:
        return sum(call.tool == "other" for _, call in self.agent_calls())
