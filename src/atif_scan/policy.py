"""Small JSON rule language; no eval, execution, or implicit plugin discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .checks import CheckSpec, Severity
from .jsonval import as_list, as_str
from .rules import All, Allowance, AnyOf, Expression, Not, Ref, Requires, Rule

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

MAX_DEPTH = 32  # nesting limit for policy expressions
RULE_KEYS = frozenset({"id", "severity", "version", "tasks", "when", "title"})
RULE_REQUIRED = frozenset({"id", "severity", "when"})
ALLOWANCE_KEYS = frozenset({"id", "version", "tasks", "covers", "when", "title"})
REQUIRES_ARITY = 2  # `requires` takes exactly [antecedent, consequent]


def _requires(children: tuple[Expression, ...]) -> Expression:
    if len(children) != REQUIRES_ARITY:
        raise ValueError("invalid_expression")
    return Requires(children[0], children[1])


# Operators over a non-empty list of sub-expressions.
COMBINATORS: dict[str, Callable[[tuple[Expression, ...]], Expression]] = {
    "all": All,
    "any": AnyOf,
    "requires": _requires,
}


def expression(value: object, depth: int = 0) -> Expression:
    if depth > MAX_DEPTH:
        raise ValueError("expression_too_deep")
    if isinstance(value, str):
        return Ref(value)
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError("invalid_expression")
    op, argument = next(iter(value.items()))
    if op == "not":
        return Not(expression(argument, depth + 1))
    combine = COMBINATORS.get(op)
    if combine is None or not isinstance(argument, list) or not argument:
        raise ValueError("invalid_expression")
    return combine(tuple(expression(v, depth + 1) for v in argument))


def tasks(raw: Mapping[str, object]) -> frozenset[str]:
    value = raw.get("tasks", [])
    names = [t for t in as_list(value) if isinstance(t, str)]
    if not isinstance(value, list) or len(names) != len(value):
        raise ValueError("invalid_task_scope")
    return frozenset(names)


def title(raw: Mapping[str, object]) -> str:
    value = raw.get("title", "")
    if not isinstance(value, str):
        raise ValueError("invalid_check_title")
    return value


def _rule(raw: object) -> Rule:
    if not isinstance(raw, dict) or set(raw) - RULE_KEYS or not set(raw) >= RULE_REQUIRED:
        raise ValueError("invalid_rule")
    name = as_str(raw["severity"])
    severity = Severity.__members__.get(name.upper()) if name is not None else None
    if severity is None:
        raise ValueError("invalid_severity")
    spec = CheckSpec(raw["id"], severity, raw.get("version", "1"), tasks(raw), title(raw))
    return Rule(spec, expression(raw["when"]))


def _allowance(raw: object) -> Allowance:
    if not isinstance(raw, dict) or set(raw) - ALLOWANCE_KEYS:
        raise ValueError("invalid_allowance")
    covers = raw.get("covers")
    if "id" not in raw or not isinstance(covers, list) or not covers:
        raise ValueError("invalid_allowance")
    if any(not isinstance(c, str) for c in covers):
        raise ValueError("invalid_allowance")
    spec = CheckSpec(raw["id"], Severity.INFO, raw.get("version", "1"), tasks(raw), title(raw))
    when = expression(raw["when"]) if "when" in raw else None
    return Allowance(spec, frozenset(covers), when)


def load_rules(value: object) -> list[Rule | Allowance]:
    """Parse `{"rules": [...], "allow": [...]}`; either section may be omitted."""
    if not isinstance(value, dict) or not value or set(value) - {"rules", "allow"}:
        raise ValueError("invalid_policy")
    if any(not isinstance(v, list) for v in value.values()):
        raise ValueError("invalid_policy")
    rules = [_rule(raw) for raw in value.get("rules", [])]
    return [*rules, *(_allowance(raw) for raw in value.get("allow", []))]
