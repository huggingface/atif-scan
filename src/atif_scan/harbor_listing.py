"""Harbor Hub listing rows: validation and allowlisted run/trial facts (no I/O, no CLI).

Shared by `harbor_hub` (live listings) and `harbor_runs` (a listing saved next to a synced
job), so a local rescan reads exactly what a live one would.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING

from .harbor_files import configured_agents, duration, late_trials, overrides, text_label
from .jsonval import Doc, JsonObject, as_list, as_object, count, identifier, is_object, number

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


def trial_meta(row: Mapping[str, object]) -> Doc:
    """Allowlisted per-trial facts from the Hub listing (numbers, codes, identifiers)."""
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
    return run_meta(job, show, rows), {trial_label(r): trial_meta(r) for r in rows}


def trial_label(row: Mapping[str, object]) -> str:
    """The trial's folder and report label: its name if a safe segment, else its UUID."""
    name = str(row.get("name") or "")
    return name if TRIAL_NAME.fullmatch(name) else str(row["id"])
