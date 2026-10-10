"""Datacurve's public DeepSWE trial index (`trials.json`), beside mirrored trials.

Datacurve publishes DeepSWE runs as a trial index
(`https://deepswe.datacurve.ai/artifacts/v1.1/trials.json`) and per-trial artifacts laid
out as Harbor trial folders (`<trial>/agent/trajectory.json`,
`<trial>/artifacts/model.patch`, `<trial>/verifier/...`). A mirror that keeps the index
(or a part of it) in the folder holding the trial folders is scanned like a Harbor
`trials.jsonl` ledger: the index is the run listing, so each trial gets its recorded
task, reward, error, cost, tokens and durations.

The index is `{"scope", "n_trials", "rows": [...]}`, one row per trial: `trial_name`
(the trial folder), `task_name` (full: Harbor cuts it to 32 characters in the folder
name), `source` (the benchmark, `deep-swe`), `harness`, `config` (the leaderboard
row: harness, model and effort), `reward`, `errored`, `error_category`, `cost_usd`,
`n_input_tokens` (cached included), `n_cache_tokens`, `n_output_tokens`,
`started_at`/`finished_at`, `agent_duration_seconds`, and more that isn't read. Only
allowlisted numbers, codes and identifiers are kept; free text (`exception`, `critique`,
`note`) never is. Unreadable or malformed indexes are unknown, never a negative result.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

from ..data.jsonval import Doc, as_list, as_object, as_str, count, identifier, number
from .harbor.files import LEDGER_TRIAL, duration

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

TRIAL_INDEX = "trials.json"
# The whole v1.1 index (31,617 trials, 70 configs) is 51 MB; a per-config part is ~0.8 MB.
INDEX_BYTES = 96 * 1024 * 1024
RUN_SOURCE = "datacurve_index"


def _label(value: object) -> str | None:
    text = as_str(value)
    if not text:
        return None
    try:
        return identifier(text)
    except ValueError:
        return None


def _error(row: Mapping[str, object]) -> str | None:
    category = _label(row.get("error_category"))
    if category:
        return category
    return "other" if row.get("errored") is True else None


def _trial_facts(row: Mapping[str, object]) -> Doc:
    meta = {
        "task": _label(row.get("task_name")),
        "reward": number(row.get("reward")),
        "error_type": _error(row),
        "cost_usd": number(row.get("cost_usd"), 0),
        "input_tokens": count(row.get("n_input_tokens")),
        "cache_tokens": count(row.get("n_cache_tokens")),
        "output_tokens": count(row.get("n_output_tokens")),
        "duration_sec": duration(row),
        "agent_duration_sec": number(row.get("agent_duration_seconds"), 0),
        "configured_agent": _label(row.get("harness")),
        "configured_model": _label(row.get("config")),
    }
    return {k: v for k, v in meta.items() if v is not None}


def index_rows(data: bytes | None) -> list[Doc] | None:
    """The index's rows (objects with a valid `trial_name`), or None when `data` isn't a
    Datacurve trial index."""
    if not data or len(data) > INDEX_BYTES:
        return None
    try:
        value = as_object(json.loads(data.decode("utf-8")))
    except (UnicodeError, ValueError, RecursionError):
        return None
    if "rows" not in value or "n_trials" not in value:
        return None
    rows = [as_object(r) for r in as_list(value.get("rows"))]
    return [r for r in rows if LEDGER_TRIAL.fullmatch(as_str(r.get("trial_name")) or "")]


def trial_index(
    data: bytes | None, present: Collection[str] | None = None
) -> tuple[Doc | None, dict[str, Doc]]:
    """(run facts, {trial folder: trial facts}) from a Datacurve trial index; (None, {})
    when it isn't one. With `present` (the trial folders beside the index), only the
    configs with a trial there are read, all of their trials, so an index of every
    config describes just the mirrored runs, and a trial of theirs that isn't in the
    folder stays in the run (unavailable, see `listed_without_trajectory`). A trial
    listed more than once is ambiguous and dropped (unknown, not a guess)."""
    rows = index_rows(data)
    if rows is None:
        return None, {}
    if present is not None:
        rows = _mirrored_configs(rows, present)
    names = Counter(str(r["trial_name"]) for r in rows)
    rows = [r for r in rows if names[str(r["trial_name"])] == 1]
    trials = {str(r["trial_name"]): _trial_facts(r) for r in rows}
    if not trials:
        return None, {}
    return _run_facts(rows, trials), trials


def _mirrored_configs(rows: list[Doc], present: Collection[str]) -> list[Doc]:
    """The rows of every config with a trial folder in `present` (a row without a
    config only when its own folder is there)."""
    configs = {r.get("config") for r in rows if r["trial_name"] in present} - {None}
    return [
        r
        for r in rows
        if r["trial_name"] in present or (r.get("config") is not None and r["config"] in configs)
    ]


def _run_facts(rows: list[Doc], trials: Mapping[str, Doc]) -> Doc:
    configs = sorted({c for r in rows if (c := _label(r.get("config")))})
    return {
        "source": RUN_SOURCE,
        # The leaderboard row, when the trials are one config's.
        "job_name": configs[0] if len(configs) == 1 else None,
        "configs": len(configs),
        "listed_trials": len(trials),
        "errored_trials": sum(1 for t in trials.values() if t.get("error_type")),
        "datasets": sorted({d for r in rows if (d := _label(r.get("source")))}),
        "tasks": sorted({t for f in trials.values() if (t := as_str(f.get("task")))}),
    }
