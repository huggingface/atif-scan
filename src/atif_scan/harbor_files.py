"""Harbor's own run files, for local and hf:// job folders (the Hub source has its own).

Harbor writes, per trial, `<trial>/result.json` (task name, verifier rewards, agent
token/cost totals, exception type, timing) and, per job, `<job>/config.json` (job name,
datasets with digests, n_attempts, timeout/resource settings) and `<job>/result.json`
(trial counts, token/cost totals). These are recorded run facts, like the Hub listing,
so they take precedence over inferences from folder names or trajectories. Only
allowlisted numbers, codes and identifiers are extracted.

Runs published by harbor-hf wrap the job folder: `<run>/run.json` declares the run's
token prices (`pricing`, $ per million input/cached/output tokens), and
`<run>/attempt-costs/<attempt id>.json` records each trial's cost next to the trace
(`attempt_id` is the trial result.json's `id`). Harnesses often leave cost out of the
trajectory and result.json, so these are where a run's cost is recorded.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from datetime import datetime

from .checks import identifier
from .jsonval import (
    Doc,
    JsonObject,
    as_list,
    as_object,
    as_str,
    count,
    is_object,
    load_object,
    number,
)

MAX_BYTES = 1024 * 1024
# Leaderboard static rules: these settings must be unset (multipliers may be 1.0).
OVERRIDE = re.compile(
    r"(?:^|[._])(?:\w*timeout_multiplier|override_timeout_sec|override_setup_timeout_sec|"
    r"max_timeout_sec|override_(?:cpus|gpus|memory_mb|storage_mb))$"
)


def _flatten(value: object, prefix: str = "") -> Iterator[tuple[str, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _flatten(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _flatten(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


# Settings that only change how the environment is provisioned (image builds, agent
# install): allowed "infrastructure replacements" for private runs. Everything else in
# OVERRIDE changes what the agent is given (time, resources) and so affects the score.
INFRA_OVERRIDE = re.compile(r"setup_timeout|build_timeout", re.I)
# Datasets whose tasks are the published benchmark (registry name or upstream repo).
CANONICAL_DATASETS = re.compile(
    r"^(?:terminal-bench/[\w.-]+|(?:https?://)?github\.com/(?:harbor-framework|laude-institute)/"
    r"terminal-bench[\w.-]*?(?:\.git)?)$",
    re.I,
)
# Free-text Harbor labels (names, refs, display strings): printable words and
# punctuation only, never a URL or control characters.
TEXT = re.compile(r"[\w .,:@/()$%+~#-]+")
GIT_REPO = re.compile(
    r"^(?:https?://)?(github\.com/[\w.-]+/[\w.-]+?)(?:\.git)?(?:@([0-9a-f]{7,40}))?$"
)


def override_kind(path: str) -> str:
    """`infrastructure` (provisioning only) or `scoring` (time/resources the agent gets)."""
    return "infrastructure" if INFRA_OVERRIDE.search(path) else "scoring"


def text_label(value: object, limit: int = 200) -> str | None:
    """The one validator for free-text labels from Harbor files or the Hub (truncated)."""
    if not isinstance(value, str) or "://" in value or not TEXT.fullmatch(value[:limit]):
        return None
    return value[:limit]


def duration(record: Mapping[str, object]) -> float | None:
    """Seconds from `started_at` to `finished_at` (ISO timestamps), else None."""
    try:
        start = datetime.fromisoformat(str(record["started_at"]))
        end = datetime.fromisoformat(str(record["finished_at"]))
    except (KeyError, TypeError, ValueError):
        return None
    return max((end - start).total_seconds(), 0.0)


# (name, ref, canonical) of a dataset, each None when unknown.
DatasetSource = tuple[str | None, str | None, bool | None]
UNKNOWN_DATASET: DatasetSource = (None, None, None)


def dataset_source(entry: Mapping[str, object]) -> DatasetSource:
    """(name, ref, canonical) for a job config dataset: a registry name, or a git repo
    (`repo: https://github.com/OWNER/REPO.git@COMMIT`, `path: tasks`)."""
    name = entry.get("name")
    if isinstance(name, str) and name:
        if text_label(name) is None:
            return UNKNOWN_DATASET
        return name, text_label(entry.get("ref")), bool(CANONICAL_DATASETS.match(name))
    repo = as_str(entry.get("repo"))
    match = GIT_REPO.match(repo.strip()) if repo is not None else None
    return _git_dataset(match, entry) if match else UNKNOWN_DATASET


def _git_dataset(match: re.Match[str], entry: Mapping[str, object]) -> DatasetSource:
    repo = str(match.group(1))
    path = as_str(entry.get("path")) or ""
    label = repo + (f"/{path.strip('/')}" if path else "")
    if text_label(label) != label:
        return UNKNOWN_DATASET
    return label, as_str(match.group(2)), bool(CANONICAL_DATASETS.match(repo))


def overrides(config: object) -> list[str]:
    """Setting paths the leaderboard requires unset (a multiplier of 1.0 is allowed)."""
    found = []
    for path, value in _flatten(config):
        if not OVERRIDE.search(path) or value is None:
            continue
        if path.endswith("multiplier") and value in (1, 1.0):
            continue
        found.append(re.sub(r"\[\d+\]", "[]", path))
    return sorted(set(found))


def _json(data: bytes | None) -> JsonObject:
    return load_object(data, MAX_BYTES)


def _label(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return identifier(value)
    except ValueError:
        return None


def primary_reward(rewards: object) -> float | None:
    """Harbor's rewards dict: the `reward` key, else a single numeric value."""
    if number(rewards) is not None:
        return number(rewards)
    if not isinstance(rewards, dict):
        return None
    rewards = as_object(rewards)
    if "reward" in rewards:
        return number(rewards["reward"])
    values = [number(v) for v in rewards.values()]
    return values[0] if len(values) == 1 else None


def trial_result(data: bytes | None) -> Doc:
    """Allowlisted facts from a trial's result.json ({} when absent or unreadable)."""
    d = _json(data)
    if not d or not ({"trial_name", "task_name"} & set(d)):
        return {}  # not a trial result (e.g. a job-level result.json)
    agent = as_object(d.get("agent_result"))
    verifier = as_object(d.get("verifier_result"))
    exception = as_object(d.get("exception_info"))
    attempt = as_str(d.get("id"))
    task = str(d.get("task_name") or "").rsplit("/", 1)[-1]
    meta = {
        "task": _label(task),
        "reward": primary_reward(verifier.get("rewards")),
        "error_type": (_label(exception.get("exception_type")) or "exception")
        if exception
        else None,
        "cost_usd": number(agent.get("cost_usd"), 0),
        "input_tokens": count(agent.get("n_input_tokens")),
        "cache_tokens": count(agent.get("n_cache_tokens")),
        "output_tokens": count(agent.get("n_output_tokens")),
        "duration_sec": duration(d),
        # Harbor's trial id (a UUID): the key of harbor-hf's attempt-costs file. Used for
        # that lookup only, never reported.
        "attempt_id": attempt if attempt and ATTEMPT_ID.fullmatch(attempt) else None,
    }
    return {k: v for k, v in meta.items() if v is not None}


TRIAL_LEDGER = "trials.jsonl"
LEDGER_BYTES = 16 * 1024 * 1024  # ~500 bytes per trial
LEDGER_TRIAL = re.compile(r"[A-Za-z0-9][\w.-]{0,127}")


def _ledger_row(line: str) -> tuple[str, Doc] | None:
    """(trial folder, allowlisted facts) from one ledger line; None when malformed."""
    try:
        row = as_object(json.loads(line))
    except (ValueError, RecursionError):
        return None
    name = as_str(row.get("trial_name"))
    if row.get("schema_version") != 1 or name is None or not LEDGER_TRIAL.fullmatch(name):
        return None
    error = row.get("error_type")
    meta = {
        "task": _label(str(row.get("task_name") or "").rsplit("/", 1)[-1]),
        "reward": number(row.get("reward")),
        "error_type": (_label(error) or "other") if error else None,
        "cost_usd": number(row.get("cost_usd"), 0),
        "input_tokens": count(row.get("input_tokens")),
        "cache_tokens": count(row.get("cached_input_tokens")),
        "output_tokens": count(row.get("output_tokens")),
        "duration_sec": duration(row),
    }
    return name, {k: v for k, v in meta.items() if v is not None}


def trial_ledger(data: bytes | None) -> dict[str, Doc]:
    """{trial folder: allowlisted facts} from a run's `trials.jsonl` ledger.

    One JSON object per trial (schema_version 1): `trial_name` (the trial folder),
    `task_name`, `reward`, `error_type`, `cost_usd`, `input_tokens` (incl. cached),
    `cached_input_tokens`, `output_tokens`, `started_at`/`finished_at`. These are
    recorded run facts, like a trial's result.json. Malformed lines are skipped; a
    trial listed more than once is ambiguous and dropped (unknown, not a guess).
    """
    if not data or len(data) > LEDGER_BYTES:
        return {}
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeError:
        return {}
    found: dict[str, Doc] = {}
    repeated: set[str] = set()
    for parsed in map(_ledger_row, lines):
        if parsed is None:
            continue
        name, facts = parsed
        if name in found:
            repeated.add(name)
        found[name] = facts
    return {k: v for k, v in found.items() if k not in repeated}


RUN_MANIFEST = "run.json"  # harbor-hf: <run>/run.json next to <run>/job/
RUN_MANIFEST_BYTES = 256 * 1024
ATTEMPT_COSTS = "attempt-costs"  # harbor-hf: <run>/attempt-costs/<attempt id>.json
ATTEMPT_COST_BYTES = 4096
ATTEMPT_ID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
PRICE_KINDS = ("uncached_input", "cached_input", "output")


def declared_prices(data: bytes | None) -> dict[str, float] | None:
    """{uncached_input, cached_input, output} $/M tokens a harbor-hf `run.json` declares
    (`pricing.{input,cached,output}_usd_per_million`), or None. `input` is the price of
    uncached input: cached tokens have their own rate. Only USD (or no currency) and
    finite, non-negative numbers; anything else is unknown, never a zero price."""
    d = _json(data) if data and len(data) <= RUN_MANIFEST_BYTES else {}
    pricing = d.get("pricing")
    if not is_object(pricing) or pricing.get("currency", "USD") != "USD":
        return None
    rates = [number(pricing.get(f"{k}_usd_per_million"), 0) for k in ("input", "cached", "output")]
    known = [r for r in rates if r is not None]
    if len(known) != len(PRICE_KINDS):
        return None
    return dict(zip(PRICE_KINDS, known, strict=True))


def attempt_cost(data: bytes | None, attempt_id: str, trial: str) -> float | None:
    """The cost a harbor-hf `attempt-costs/<attempt id>.json` records for this trial, or
    None (absent, null, malformed, or naming another attempt or trial folder)."""
    d = _json(data) if data and len(data) <= ATTEMPT_COST_BYTES else {}
    if d.get("attempt_id") != attempt_id or d.get("trial_name") != trial:
        return None
    return number(d.get("cost_usd"), 0)


def configured_agents(cfg: Mapping[str, object]) -> int | None:
    """How many agent/model entries a job config plans (a comparison job runs several,
    so several models are expected rather than a substitution)."""
    agents = cfg.get("agents")
    if not isinstance(agents, list):
        return None
    return len([a for a in agents if isinstance(a, dict)]) or None


def _listed_names(ev: JsonObject) -> Iterator[str]:
    """Trial names in one eval's `reward_stats` ({metric: {value: [names]}}) and
    `exception_stats` ({exception type: [names]})."""
    rewards = as_object(ev.get("reward_stats")).values()
    groups = [v for by_value in rewards for v in as_object(by_value).values()]
    groups += as_object(ev.get("exception_stats")).values()
    for values in groups:
        yield from (v for v in as_list(values) if isinstance(v, str) and v)


def job_listed_trials(result: bytes | None) -> frozenset[str] | None:
    """Trial folder names the job's own result.json accounts for (its per-eval
    `reward_stats` and `exception_stats`), or None when that can't be trusted: no
    listing, or fewer names than the job says it completed. Unknown, never "none"."""
    stats = as_object(_json(result).get("stats"))
    evals = stats.get("evals")
    if not is_object(evals):
        return None
    names = {name for ev in evals.values() for name in _listed_names(as_object(ev))}
    completed = count(stats.get("n_completed_trials"))
    if not names or (completed is not None and len(names) < completed):
        return None
    return frozenset(names)


def job_meta(config: bytes | None, result: bytes | None) -> Doc | None:
    """Run facts from a Harbor job folder's config.json/result.json (None if not a job)."""
    cfg, res = _json(config), _json(result)
    if not cfg or not ({"datasets", "agents", "n_attempts"} & set(cfg)):
        return None
    stats = as_object(res.get("stats"))
    datasets = [as_object(d) for d in as_list(cfg.get("datasets")) if is_object(d)]
    task_names = [n for d in datasets for n in as_list(d.get("task_names"))]
    sources = [dataset_source(d) for d in datasets]
    completed = count(stats.get("n_completed_trials"))
    return {
        "source": "harbor_job_folder",
        "job_id": _label(str(res.get("id") or "")),
        "job_name": text_label(cfg.get("job_name")),
        "planned_trials": count(res.get("n_total_trials")),
        "total_trials": count(res.get("n_total_trials")),
        "completed_trials": completed,
        "errored_trials": count(stats.get("n_errored_trials")),
        "listed_trials": None,
        "datasets": [n for n, _, _ in sources if n],
        "dataset_refs": [r for _, r, _ in sources if r],
        # False when the tasks came from a fork or copy (e.g. a git repo other than the
        # benchmark's own); None when unknown.
        "canonical_dataset": (
            None
            if not sources or any(c is None for _, _, c in sources)
            else all(c for _, _, c in sources)
        ),
        "config_task_names": len(task_names) or None,
        "n_attempts": count(cfg.get("n_attempts")),
        "configured_agents": configured_agents(cfg),
        "cost_usd": number(stats.get("cost_usd"), 0),
        "overrides": overrides(cfg),
    }
