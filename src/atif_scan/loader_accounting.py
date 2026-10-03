"""Narrow explicitly declared fast-agent child accounting to root-only usage.

Child totals are accounting evidence, not scanned child history or provider bills.
Never subtract a declaration without matching embedded child records.
"""

from .jsonval import JsonObject, as_list, as_object, as_str, count
from .model import Usage


def _children(value: JsonObject) -> list[JsonObject]:
    children = as_list(value.get("subagent_trajectories"))
    ids = [as_str(as_object(child).get("trajectory_id")) for child in children]
    if not children or not all(ids) or len(set(ids)) != len(ids):
        return []
    return [as_object(child) for child in children]


def _child_sum(children: list[JsonObject], key: str, *, extra: bool = False) -> int | None:
    values: list[int] = []
    for child in children:
        metrics = as_object(child.get("final_metrics"))
        details = as_object(metrics.get("extra"))
        if not as_list(child.get("steps")) or details.get("llm_usage_calls_complete") is False:
            return None
        value = count((details if extra else metrics).get(key))
        if value is None:
            return None
        values.append(value)
    return sum(values)


def _root_kind(metrics: JsonObject, children: list[JsonObject], kind: str) -> int | None:
    extra = as_object(metrics.get("extra"))
    total = count(metrics.get(f"total_{kind}_tokens"))
    root = count(extra.get(f"root_{kind}_tokens"))
    declared = count(extra.get(f"subagent_{kind}_tokens"))
    embedded = _child_sum(children, f"total_{kind}_tokens")
    if total is None or root is None or declared is None or embedded is None:
        return None
    return root if declared == embedded and root + embedded == total else None


def root_usage(value: JsonObject) -> Usage | None:
    """Only a fully reconciled declared split permits root-scoped comparisons."""
    if as_object(value.get("agent")).get("name") != "fast-agent":
        return None
    children = _children(value)
    if not children:
        return None
    metrics = as_object(value.get("final_metrics"))
    kinds = [_root_kind(metrics, children, kind) for kind in ("prompt", "completion", "cached")]
    if any(kind is None for kind in kinds):
        return None
    reasoning = count(as_object(metrics.get("extra")).get("total_reasoning_tokens"))
    child_reasoning = _child_sum(children, "total_reasoning_tokens", extra=True)
    root_reasoning = (
        reasoning - child_reasoning
        if reasoning is not None and child_reasoning is not None and reasoning >= child_reasoning
        else None
    )
    return Usage(
        prompt_tokens=kinds[0],
        completion_tokens=kinds[1],
        cached_tokens=kinds[2],
        reasoning_tokens=root_reasoning,
    )
