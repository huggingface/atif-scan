"""Harbor Hub listing rows: validation and allowlisted run/trial facts (no I/O, no CLI).

Shared by `harbor_hub` (live listings) and `harbor_runs` (a listing saved next to a synced
job), so a local rescan reads exactly what a live one would.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ...data.jsonval import (
    Doc,
    JsonObject,
    as_list,
    as_object,
    count,
    identifier,
    is_object,
    number,
)
from .files import configured_agents, duration, late_trials, overrides, started, text_label

if TYPE_CHECKING:
    from collections.abc import Sequence

UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
# Hub values that become CLI arguments or folder names are validated first: an ID must be
# a UUID (never an option like `-o...`), a name a single path segment (no `/`, no `..`).
TRIAL_ID = re.compile(UUID)
TRIAL_NAME = re.compile(r"[A-Za-z0-9][\w.-]{0,127}")


def valid_row(row: Mapping[str, object]) -> bool:
    """A trial row whose ID is safe to pass to the CLI (see TRIAL_ID)."""
    return bool(TRIAL_ID.fullmatch(str(row.get("id") or "")))


def _checked(value: str | None) -> str | None:
    """The value if it's a valid identifier (see checks.identifier), else None."""
    if value is None:
        return None
    try:
        identifier(value)
    except ValueError:
        return None
    return value


def trial_meta(row: Mapping[str, object], retry: Mapping[str, object] | None = None) -> Doc:
    """Allowlisted per-trial facts from the Hub listing (numbers, codes, identifiers),
    with its place in a retry chain (`retry_chains`) when it has one."""
    task = str(row.get("task_name") or "").rsplit("/", 1)[-1] or None
    error = row.get("error_type")
    return {
        "hub_trial_id": _checked(str(row.get("id") or "") or None),
        "task": _checked(task),
        "reward": number(row.get("reward")),
        # An error that isn't a plain code is still an error.
        "error_type": (_checked(str(error)) or "other") if error else None,
        "status": _checked(str(row.get("status") or "") or None),
        "cost_usd": number(row.get("cost_usd"), 0),
        "input_tokens": count(row.get("input_tokens")),
        "cache_tokens": count(row.get("cache_tokens")),
        "output_tokens": count(row.get("output_tokens")),
        "duration_sec": duration(row),
        "overrides": overrides(row.get("config_values") or {}),
        **(retry or {}),
    }


# Harbor re-runs an errored trial as `<trial>__retry_<id>`; every attempt stays in the job,
# and the suffix-free name may belong to any attempt (not necessarily the first).
RETRY_SUFFIX = re.compile(r"__retry_[0-9a-fA-F]{4,}$")


def retry_base(name: str) -> str:
    return RETRY_SUFFIX.sub("", name)


def _chains(rows: Sequence[Mapping[str, object]]) -> list[list[Mapping[str, object]]]:
    """Each retry chain (two or more attempts of one trial) as listed."""
    by_base: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_base[retry_base(str(row.get("name") or ""))].append(row)
    named = any(RETRY_SUFFIX.search(str(r.get("name") or "")) for r in rows)
    return [chain for base, chain in by_base.items() if base and len(chain) > 1] if named else []


def _ordered(chain: list[Mapping[str, object]]) -> list[Mapping[str, object]] | None:
    """The chain by start time, or None when that order isn't known (a missing or
    tied start time): then no attempt can be called the last one."""
    times = [started(r) for r in chain]
    known = [t for t in times if t is not None]
    if len(known) < len(chain) or len(set(known)) < len(known):
        return None
    if len({t.tzinfo is None for t in known}) > 1:
        return None
    return [r for _, r in sorted(zip(known, chain, strict=True), key=lambda tr: tr[0])]


def retry_chains(rows: Sequence[Mapping[str, object]]) -> dict[str, Doc]:
    """{trial id: retry facts} for every attempt in a retry chain: `retry_attempts` (the
    chain's length) and `retry_superseded` (True for all but the last attempt; None when
    the chain's order is unknown, so every attempt stays scored)."""
    facts: dict[str, Doc] = {}
    for chain in _chains(rows):
        ordered = _ordered(chain)
        last = ordered[-1] if ordered else None
        for row in chain:
            superseded = None if last is None else row is not last
            facts[str(row.get("id"))] = {
                "retry_attempts": len(chain),
                "retry_superseded": superseded,
            }
    return facts


def retry_summary(rows: Sequence[Mapping[str, object]]) -> Doc | None:
    """What the job's retry chains replaced, so the assumption behind scoring only each
    chain's last attempt can be checked: superseded attempts should be infrastructure
    failures (an error, no usage). None when the job has no retry chains."""
    chains = _chains(rows)
    if not chains:
        return None
    facts = retry_chains(rows)
    superseded = [
        trial_meta(r) for r in rows if facts.get(str(r.get("id")), {}).get("retry_superseded")
    ]
    used = ("cost_usd", "input_tokens", "output_tokens")
    return {
        "chains": len(chains),
        "attempts": sum(len(c) for c in chains),
        "superseded": len(superseded),
        "unordered_chains": sum(1 for c in chains if _ordered(c) is None),
        "superseded_errors": dict(Counter(m["error_type"] or "none" for m in superseded)),
        # Against the assumption: an attempt that ended without an error, or did paid work.
        "superseded_without_error": sum(1 for m in superseded if not m["error_type"]),
        "superseded_with_usage": sum(1 for m in superseded if any(m.get(k) for k in used)),
    }


def job_datasets(config: JsonObject) -> list[JsonObject]:
    return [as_object(d) for d in as_list(config.get("datasets")) if is_object(d)]


UUID_V5 = 5


def _uuid_version(value: str) -> int | None:
    try:
        return uuid.UUID(value).version
    except ValueError:
        return None


def run_meta(job: str, show: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> Doc:
    config = as_object(show.get("config"))
    datasets = job_datasets(config)
    task_names = [n for d in datasets for n in as_list(d.get("task_names"))]
    return {
        "source": "harbor_hub",
        "job_id": job,
        "job_name": text_label(show.get("name")),
        "planned_trials": count(show.get("n_planned_trials")),
        "total_trials": count(show.get("n_total_trials")),
        "completed_trials": count(show.get("n_completed_trials")),
        "errored_trials": count(show.get("n_errors")),
        "listed_trials": len(rows),
        "datasets": [n for d in datasets if (n := text_label(d.get("name")))],
        "dataset_refs": [r for d in datasets if (r := text_label(d.get("ref")))],
        "config_task_names": len(task_names) or None,
        "n_attempts": count(config.get("n_attempts")),
        "configured_agents": configured_agents(config),
        "cost_usd": number(show.get("cost_usd"), 0),
        "overrides": overrides(config),
        # A deterministic (UUIDv5) job ID is minted by a tool that assembled the job from
        # other trials (e.g. a filtered mirror), not by a job run as launched.
        "constructed_id": _uuid_version(job) == UUID_V5,
        # The job's tasks, to check that jobs scanned together don't share a task.
        "tasks": sorted({t for r in rows if (t := trial_meta(r).get("task"))}),
        "late_trials": late_trials(rows),
        "retries": retry_summary(rows),
    }


# Written into each synced job folder so a later scan of the local copy (`atif-scan
# ~/.cache/atif-scan/harbor/<job>`) keeps the Hub's run facts: task, reward, error, cost.
SAVED_LISTING = "hub-listing.json"


def saved_listing(data: bytes) -> tuple[Doc | None, dict[str, Doc]]:
    """(run facts, {trial folder label: trial facts}) from a saved listing, re-validated
    exactly like a live listing; ({}, {}) when it isn't one."""
    try:
        value = json.loads(data.decode("utf-8"))
        job = str(value["job"]).lower()
        show, rows = value["show"], [r for r in value["rows"] if isinstance(r, Mapping)]
        if not TRIAL_ID.fullmatch(job) or not isinstance(show, Mapping):
            raise ValueError
    except (KeyError, TypeError, ValueError, RecursionError):
        return None, {}
    rows = [r for r in rows if valid_row(r)]
    chains = retry_chains(rows)
    facts = {trial_label(r): trial_meta(r, chains.get(str(r["id"]))) for r in rows}
    return run_meta(job, show, rows), facts


def trial_label(row: Mapping[str, object]) -> str:
    """The trial's folder and report label: its name if a safe segment, else its UUID."""
    name = str(row.get("name") or "")
    return name if TRIAL_NAME.fullmatch(name) else str(row["id"])
