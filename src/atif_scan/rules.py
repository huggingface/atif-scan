"""Declarative expressions using strong Kleene three-valued logic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .checks import CheckSpec, Detection, Status, identifier

Truth = bool | None


class Expression(Protocol):
    def dependencies(self) -> frozenset[str]: ...
    def evaluate(self, results: Mapping[str, Detection]) -> Truth: ...


@dataclass(frozen=True)
class Ref:
    id: str

    def __post_init__(self) -> None:
        identifier(self.id)

    def dependencies(self) -> frozenset[str]:
        return frozenset({self.id})

    def evaluate(self, results: Mapping[str, Detection]) -> Truth:
        result = results[self.id]
        if result.status == Status.MATCH:
            return True
        if result.status == Status.NO_MATCH:
            return False
        return None


@dataclass(frozen=True)
class Not:
    child: Expression

    def dependencies(self) -> frozenset[str]:
        return self.child.dependencies()

    def evaluate(self, results: Mapping[str, Detection]) -> Truth:
        value = self.child.evaluate(results)
        return None if value is None else not value


@dataclass(frozen=True)
class All:
    children: tuple[Expression, ...]

    def dependencies(self) -> frozenset[str]:
        return frozenset().union(*(c.dependencies() for c in self.children))

    def evaluate(self, results: Mapping[str, Detection]) -> Truth:
        values = [c.evaluate(results) for c in self.children]
        return False if False in values else None if None in values else True


@dataclass(frozen=True)
class AnyOf:
    children: tuple[Expression, ...]

    def dependencies(self) -> frozenset[str]:
        return frozenset().union(*(c.dependencies() for c in self.children))

    def evaluate(self, results: Mapping[str, Detection]) -> Truth:
        values = [c.evaluate(results) for c in self.children]
        return True if True in values else None if None in values else False


@dataclass(frozen=True)
class Requires:
    """A review finding when antecedent is true AND consequent is false.

    It reports violations, not the truth of the material implication itself.
    Unknown/absent consequents cannot establish a violation.
    """

    antecedent: Expression
    consequent: Expression

    def dependencies(self) -> frozenset[str]:
        return self.antecedent.dependencies() | self.consequent.dependencies()

    def evaluate(self, results: Mapping[str, Detection]) -> Truth:
        return All((self.antecedent, Not(self.consequent))).evaluate(results)


@dataclass(frozen=True)
class Rule:
    spec: CheckSpec
    expression: Expression
