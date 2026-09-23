"""Small JSON rule language; no eval, execution, or implicit plugin discovery."""

from __future__ import annotations

from .checks import CheckSpec, Severity
from .rules import All, AnyOf, Expression, Not, Ref, Requires, Rule


def expression(value: object, depth: int = 0) -> Expression:
    if depth > 32:
        raise ValueError("expression_too_deep")
    if isinstance(value, str):
        return Ref(value)
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError("invalid_expression")
    op, argument = next(iter(value.items()))
    if op == "not":
        return Not(expression(argument, depth + 1))
    if op in {"all", "any", "requires"} and isinstance(argument, list) and argument:
        children = tuple(expression(v, depth + 1) for v in argument)
        if op == "all":
            return All(children)
        if op == "any":
            return AnyOf(children)
        if len(children) == 2:
            return Requires(*children)
    raise ValueError("invalid_expression")


def load_rules(value: object) -> list[Rule]:
    if (
        not isinstance(value, dict)
        or set(value) != {"rules"}
        or not isinstance(value["rules"], list)
    ):
        raise ValueError("invalid_policy")
    rules = []
    for raw in value["rules"]:
        if not isinstance(raw, dict) or set(raw) - {"id", "severity", "version", "tasks", "when"}:
            raise ValueError("invalid_rule")
        if not {"id", "severity", "when"} <= set(raw):
            raise ValueError("invalid_rule")
        tasks = raw.get("tasks", [])
        if not isinstance(tasks, list) or any(not isinstance(t, str) for t in tasks):
            raise ValueError("invalid_task_scope")
        try:
            severity = Severity[raw["severity"].upper()]
        except (KeyError, AttributeError):
            raise ValueError("invalid_severity") from None
        spec = CheckSpec(raw["id"], severity, raw.get("version", "1"), frozenset(tasks))
        rules.append(Rule(spec, expression(raw["when"])))
    return rules
