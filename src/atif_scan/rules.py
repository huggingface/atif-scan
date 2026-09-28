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
        # A check that doesn't apply to this task can't have matched: it's no evidence
        # either way, so it doesn't make "any of" unknown (TB4 merge: a task-scoped input
        # left every access roll-up unknown on traces with no tool calls at all).
        values = [
            c.evaluate(results)
            for c in self.children
            if not (isinstance(c, Ref) and results[c.id].status == Status.NOT_APPLICABLE)
        ]
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

    @property
    def dependencies(self) -> tuple[str, ...]:
        return tuple(sorted(self.expression.dependencies()))

    def evaluate(self, results: Mapping[str, Detection]) -> Detection:
        """Combine already-evaluated dependency results; cites matched dependency evidence."""
        truth = self.expression.evaluate(results)
        if truth is None:
            return Detection(Status.UNKNOWN, complete=False)
        if truth is False:
            # Short-circuit truth can be known even when an irrelevant input is unknown.
            return Detection(Status.NO_MATCH)
        deps = [results[d] for d in self.dependencies]
        evidence = dict.fromkeys(at for r in deps if r.status == Status.MATCH for at in r.evidence)
        return Detection(Status.MATCH, tuple(evidence), all(r.complete for r in deps))


@dataclass(frozen=True)
class Allowance:
    """A "positive" component: declares that matches of `covers` are expected.

    It never changes what a detector found. When `when` is true (or absent, meaning the
    allowance holds within its task scope), matched assessments it covers are reported as
    expected and excluded from the score. An unknown `when` does not apply the allowance:
    a finding is only excused on positive evidence. Rules see raw detector results, so to
    excuse a rule's match, cover the rule's own ID.
    """

    spec: CheckSpec
    covers: frozenset[str]
    when: Expression | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.covers, frozenset) or not self.covers:
            raise ValueError("allowance_must_cover_checks")
        for key in self.covers:
            identifier(key)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return tuple(sorted(self.when.dependencies())) if self.when is not None else ()

    def evaluate(self, results: Mapping[str, Detection]) -> Detection:
        if self.when is None:
            return Detection(Status.MATCH)
        return Rule(self.spec, self.when).evaluate(results)
