"""Small immutable ATIF projection. Raw text is memory-only and hidden from repr."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class Channel(StrEnum):
    MESSAGE = "message"
    REASONING = "reasoning"
    COMMAND = "command"
    PATH = "path"
    QUERY = "query"
    URL = "url"
    OBSERVATION = "observation"
    ARGUMENTS = "arguments"


@dataclass(frozen=True, order=True)
class Locator:
    step: int
    channel: Channel
    call: int | None = None
    observation: int | None = None

    def __post_init__(self) -> None:
        for value in (self.step, self.call, self.observation):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("invalid locator index")
        if not isinstance(self.channel, Channel):
            raise ValueError("invalid locator channel")


@dataclass(frozen=True)
class Content:
    text: str = field(default="", repr=False)
    understood: bool = True


@dataclass(frozen=True)
class Surface:
    at: Locator
    content: Content = field(repr=False)
    tool: str | None = None


@dataclass(frozen=True)
class ToolCall:
    index: int
    tool: str  # Normalized allowlisted name, not an arbitrary provider tool string.
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


@dataclass(frozen=True)
class Trace:
    schema_version: str | None
    steps: tuple[Step, ...]

    def agent_surfaces(self) -> Iterator[Surface]:
        """Never recursively walks a step: observations and prompt text stay separate."""
        for step in self.steps:
            if step.source != "agent" or step.copied:
                continue
            yield Surface(Locator(step.index, Channel.MESSAGE), step.message)
            yield Surface(Locator(step.index, Channel.REASONING), step.reasoning)
            for call in step.calls:
                for channel, content in call.fields:
                    yield Surface(Locator(step.index, channel, call.index), content, call.tool)

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
