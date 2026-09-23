"""Extension contract. Assessments are typed data, not arbitrary report dictionaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Protocol

from .model import Locator, Trace


def identifier(value: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.:/-]{0,127}", value) is None
    ):
        raise ValueError("invalid_identifier")
    return value


class Status(StrEnum):
    MATCH = "match"
    NO_MATCH = "no_match"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


class Severity(IntEnum):
    INFO = 0
    LOW = 25
    MEDIUM = 50
    HIGH = 75
    CRITICAL = 100


@dataclass(frozen=True)
class Context:
    task: str | None = None
    partial: bool = False

    def __post_init__(self) -> None:
        if self.task is not None:
            identifier(self.task)
        if type(self.partial) is not bool:
            raise ValueError("invalid_partial_flag")


@dataclass(frozen=True)
class CheckSpec:
    id: str
    severity: Severity = Severity.INFO
    version: str = "1"
    tasks: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        identifier(self.id)
        identifier(self.version)
        if not isinstance(self.severity, Severity) or not isinstance(self.tasks, frozenset):
            raise ValueError("invalid_check_spec")
        for task in self.tasks:
            identifier(task)


@dataclass(frozen=True)
class Detection:
    status: Status
    evidence: tuple[Locator, ...] = ()
    complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.status, Status) or type(self.complete) is not bool:
            raise ValueError("invalid_detection")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(x, Locator) for x in self.evidence
        ):
            raise ValueError("invalid_evidence")
        if self.status == Status.NO_MATCH and not self.complete:
            raise ValueError("incomplete_negative_must_be_unknown")
        if self.status in {Status.UNKNOWN, Status.ERROR} and self.complete:
            raise ValueError("unknown_or_error_cannot_be_complete")


class Detector(Protocol):
    spec: CheckSpec

    def evaluate(self, trace: Trace, context: Context) -> Detection: ...
