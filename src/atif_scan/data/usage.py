"""Usage accounting read from a trajectory: per-step and final metrics, retries, agent
info, and fast-agent child accounting narrowed to root-only usage. Accounting evidence,
not provider bills.
"""

from __future__ import annotations

import re
from dataclasses import replace
from itertools import pairwise
from typing import TYPE_CHECKING

from .jsonval import JsonObject, as_list, as_object, as_str, count, is_object, number
from .model import Usage

if TYPE_CHECKING:
    from collections.abc import Mapping


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


def step_usage(raw_steps: list[object]) -> tuple[Usage | None, int, tuple[str, ...]]:
    """(summed per-step token usage, LLM calls without usage, token kinds only some
    metered steps record). Steps record `metrics.{prompt,completion,cached}_tokens`; the
    sum is the fallback when final_metrics has no totals (e.g. the harness died before
    writing them). A step without metrics leaves all its calls (`llm_call_count`, else 1)
    unmetered; a step with metrics but several calls is read as metering one of them (a
    retry's usage is often lost). A kind is complete only when every metered step records
    it: recorded on some steps it's a lower bound (a partial kind), on none it's unknown
    (None), never 0. None when no agent step recorded usage."""
    totals = [0, 0, 0]
    seen = missing = 0
    recorded = [0, 0, 0]  # metered steps recording each kind
    for raw in raw_steps:
        if not is_object(raw) or raw.get("source") != "agent":
            continue
        metrics = as_object(raw.get("metrics"))
        calls = count(raw.get("llm_call_count"))
        values = [count(metrics.get(k)) for k in STEP_TOKEN_KINDS]
        if values[0] is None and values[1] is None:
            missing += 1 if calls is None else calls
            continue
        seen += 1
        missing += max((calls or 1) - 1, 0)
        totals = [t + (v or 0) for t, v in zip(totals, values, strict=True)]
        recorded = [r + (v is not None) for r, v in zip(recorded, values, strict=True)]
    if not seen:
        return None, missing, ()
    kinds = [t if r else None for t, r in zip(totals, recorded, strict=True)]
    partial = tuple(
        name for name, r in zip(TOKEN_KIND_NAMES, recorded, strict=True) if 0 < r < seen
    )
    return Usage(None, *kinds), missing, partial


def last_call(raw_steps: list[object]) -> tuple[Usage | None, int, bool]:
    """(the last metered agent step's own usage, agent steps with usage, whether every
    metered step's prompt and completion tokens are both at least the previous step's).
    A step is metered as in step_usage: prompt or completion tokens recorded."""
    metered: list[list[int | None]] = []
    for raw in raw_steps:
        if not is_object(raw) or raw.get("source") != "agent":
            continue
        metrics = as_object(raw.get("metrics"))
        values = [count(metrics.get(k)) for k in STEP_TOKEN_KINDS]
        if values[0] is not None or values[1] is not None:
            metered.append(values)
    if not metered:
        return None, 0, False
    rising = all((b[n] or 0) >= (a[n] or 0) for a, b in pairwise(metered) for n in (0, 1))
    return Usage(None, *metered[-1]), len(metered), rising


FAST_AGENT_RETRY_SCHEMA = "fast-agent.retry/v1"


def stream_retries(raw_steps: list[object], final_metrics: object) -> tuple[int, int, bool | None]:
    """Count explicit fast-agent.retry/v1 provider retry markers and read the
    harness's usage-completeness flag. A marker does not establish whether a failure
    was mid-stream, whether usage was received, or whether output entered context.
    Legacy field names retain "stream" for report compatibility."""
    steps = attempts = 0
    for raw in raw_steps:
        if not is_object(raw) or raw.get("source") != "agent":
            continue
        retry = as_object(as_object(raw.get("extra")).get("retry"))
        tries = count(retry.get("provider_attempts"))
        if as_str(retry.get("schema")) == FAST_AGENT_RETRY_SCHEMA and tries and tries > 1:
            steps += 1
            attempts += tries - 1
    complete = as_object(as_object(final_metrics).get("extra")).get("llm_usage_calls_complete")
    return steps, attempts, complete if isinstance(complete, bool) else None


# A provider refusal recorded by the client: "Error code: 503 - {...}" (OpenAI/Anthropic
# SDKs). Only the status number is kept, never the message.
HTTP_STATUS = re.compile(r"\AError code: ([1-5]\d\d)\b")
HTTP_ERROR_MIN = 400


def _retry_status(retry: Mapping[str, object]) -> int | None:
    match = HTTP_STATUS.match(as_str(retry.get("error_message")) or "")
    return int(match.group(1)) if match else None


def _before_output(retry: Mapping[str, object]) -> bool:
    """The failed attempt returned nothing: no stream event arrived, or the provider
    refused the request with an HTTP error status before opening a stream. A count of
    events (even one) or an unknown count with no status is not this."""
    events = retry.get("stream_events_received")
    status = _retry_status(retry)
    return events == 0 or (events is None and status is not None and status >= HTTP_ERROR_MIN)


def retry_outcomes(raw_steps: list[object]) -> tuple[int, tuple[int, ...]]:
    """(fast-agent retried attempts that failed before any output, the HTTP statuses
    of all failed attempts that recorded one, sorted and unique)."""
    before, statuses = 0, set[int]()
    for raw in raw_steps:
        if not is_object(raw) or raw.get("source") != "agent":
            continue
        retry = as_object(as_object(raw.get("extra")).get("retry"))
        if as_str(retry.get("schema")) != FAST_AGENT_RETRY_SCHEMA:
            continue
        for attempt in as_list(retry.get("retries")):
            record = as_object(attempt)
            before += _before_output(record)
            if (status := _retry_status(record)) is not None:
                statuses.add(status)
    return before, tuple(sorted(statuses))


# How the report names each step token kind (usage fields: input, output, cache).
TOKEN_KIND_NAMES = ("input", "output", "cached")
STEP_TOKEN_KINDS = ("prompt_tokens", "completion_tokens", "cached_tokens")


def step_completion_tokens(metrics: object) -> int | None:
    return count(as_object(metrics).get("completion_tokens"))


def step_reasoning(metrics: object) -> int | None:
    """Do not invent zero or subtract an invalid split from completion tokens."""
    raw = as_object(metrics)
    completion = count(raw.get("completion_tokens"))
    reasoning = count(as_object(raw.get("extra")).get("reasoning_tokens"))
    if completion is None or reasoning is None or reasoning > completion:
        return None
    return reasoning


def model_label(v: object) -> str | None:
    """A label-safe agent/model name, else None (e.g. Claude Code's `<synthetic>`
    placeholder, or a name carrying query parameters, stays unknown)."""
    return v if isinstance(v, str) and re.fullmatch(r"[\w.:@/+-]{1,100}", v) else None


def agent_info(value: object) -> tuple[str | None, str | None, str | None]:
    """(name, version, model_name) from the ATIF root `agent` block, label-safe only."""
    agent = as_object(value)
    return (
        model_label(agent.get("name")),
        model_label(agent.get("version")),
        model_label(agent.get("model_name")),
    )


# Harness notices that earlier conversation history was replaced by a summary.
COMPACTED = re.compile(
    r"session is being continued from a previous conversation|\[COMPACTED HISTORY\]|"
    # Devin CLI: "You are continuing work from a previous conversation thread. Below is a
    # summary of the previous conversation thread".
    r"continuing work from a previous conversation thread|"
    r"summary of the previous conversation thread|"
    r"conversation history (?:was|has been) (?:compacted|summari[sz]ed)|"
    r"(?:ran|run) out of context\b[^.\n]{0,80}summary",
    re.I,
)


def _reasoning_tokens(extra: JsonObject) -> int | None:
    """Prefer the canonical field, including zero; Codex uses a historical alias.

    An absent/null canonical value permits fallback. A malformed non-null value
    remains unknown rather than being silently repaired from another field.
    """
    value = extra.get("total_reasoning_tokens")
    return count(extra.get("reasoning_output_tokens") if value is None else value)


def usage(
    metrics: object,
    agent: tuple[str | None, str | None, str | None] = (None, None, None),
) -> Usage | None:
    if not is_object(metrics):
        return None
    extra = as_object(metrics.get("extra"))
    found = Usage(
        number(metrics.get("total_cost_usd"), low=0),
        count(metrics.get("total_prompt_tokens")),
        count(metrics.get("total_completion_tokens")),
        count(metrics.get("total_cached_tokens")),
        _reasoning_tokens(extra),
        count(extra.get("total_cache_creation_input_tokens")),
    )
    if found == Usage():
        return None
    # Grok Build exports visible output and reasoning separately. This is a
    # harness convention, not a model-name or harness-version heuristic.
    if agent[0] == "grok-build":
        return replace(found, completion_basis="separate_visible")
    return found


# A JSON string (skipped as a whole) or a bare `[REDACTED]` token outside any string.
