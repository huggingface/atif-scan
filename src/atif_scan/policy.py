"""Small JSON rule language; no eval, execution, or implicit plugin discovery."""

from __future__ import annotations

from .checks import CheckSpec, Severity
from .rules import All, Allowance, AnyOf, Expression, Not, Ref, Requires, Rule


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


def tasks(raw: dict) -> frozenset[str]:
    value = raw.get("tasks", [])
    if not isinstance(value, list) or any(not isinstance(t, str) for t in value):
        raise ValueError("invalid_task_scope")
    return frozenset(value)


def load_rules(value: object) -> list[Rule | Allowance]:
    """Parse `{"rules": [...], "allow": [...]}`; either section may be omitted."""
    if not isinstance(value, dict) or not value or set(value) - {"rules", "allow"}:
        raise ValueError("invalid_policy")
    if any(not isinstance(v, list) for v in value.values()):
        raise ValueError("invalid_policy")
    checks: list[Rule | Allowance] = []
    for raw in value.get("rules", []):
        if not isinstance(raw, dict) or set(raw) - {"id", "severity", "version", "tasks", "when"}:
            raise ValueError("invalid_rule")
        if not {"id", "severity", "when"} <= set(raw):
            raise ValueError("invalid_rule")
        try:
            severity = Severity[raw["severity"].upper()]
        except (KeyError, AttributeError):
            raise ValueError("invalid_severity") from None
        spec = CheckSpec(raw["id"], severity, raw.get("version", "1"), tasks(raw))
        checks.append(Rule(spec, expression(raw["when"])))
    for raw in value.get("allow", []):
        if not isinstance(raw, dict) or set(raw) - {"id", "version", "tasks", "covers", "when"}:
            raise ValueError("invalid_allowance")
        covers = raw.get("covers")
        if "id" not in raw or not isinstance(covers, list) or not covers:
            raise ValueError("invalid_allowance")
        if any(not isinstance(c, str) for c in covers):
            raise ValueError("invalid_allowance")
        spec = CheckSpec(raw["id"], Severity.INFO, raw.get("version", "1"), tasks(raw))
        when = expression(raw["when"]) if "when" in raw else None
        checks.append(Allowance(spec, frozenset(covers), when))
    return checks
