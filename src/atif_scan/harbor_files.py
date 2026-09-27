"""Harbor's own run files, for local and hf:// job folders (the Hub source has its own).

Harbor writes, per trial, `<trial>/result.json` (task name, verifier rewards, agent
token/cost totals, exception type, timing) and, per job, `<job>/config.json` (job name,
datasets with digests, n_attempts, timeout/resource settings) and `<job>/result.json`
(trial counts, token/cost totals). These are recorded run facts, like the Hub listing,
so they take precedence over inferences from folder names or trajectories. Only
allowlisted numbers, codes and identifiers are extracted.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping
from datetime import datetime

from .checks import identifier

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
GIT_REPO = re.compile(
    r"^(?:https?://)?(github\.com/[\w.-]+/[\w.-]+?)(?:\.git)?(?:@([0-9a-f]{7,40}))?$"
)


def override_kind(path: str) -> str:
    """`infrastructure` (provisioning only) or `scoring` (time/resources the agent gets)."""
    return "infrastructure" if INFRA_OVERRIDE.search(path) else "scoring"


def dataset_source(entry: Mapping) -> tuple[str | None, str | None, bool | None]:
    """(name, ref, canonical) for a job config dataset: a registry name, or a git repo
    (`repo: https://github.com/OWNER/REPO.git@COMMIT`, `path: tasks`)."""
    name = entry.get("name")
    if isinstance(name, str) and name:
        return (
            name,
            entry.get("ref") if isinstance(entry.get("ref"), str) else None,
            bool(CANONICAL_DATASETS.match(name)),
        )
    repo = entry.get("repo")
    if isinstance(repo, str):
        m = GIT_REPO.match(repo.strip())
        if m:
            path = entry.get("path") if isinstance(entry.get("path"), str) else ""
            label = m.group(1) + (f"/{path.strip('/')}" if path else "")
            if not re.fullmatch(r"[\w.:@/-]{1,200}", label):
                return None, None, None
            return label, m.group(2), bool(CANONICAL_DATASETS.match(m.group(1)))
    return None, None, None


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


def _json(data: bytes | None) -> dict:
    if not data or len(data) > MAX_BYTES:
        return {}
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _number(value: object) -> float | None:
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _label(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return identifier(value)
    except ValueError:
        return None


def primary_reward(rewards: object) -> float | None:
    """Harbor's rewards dict: the `reward` key, else a single numeric value."""
    if _number(rewards) is not None:
        return _number(rewards)
    if not isinstance(rewards, dict):
        return None
    if "reward" in rewards:
        return _number(rewards["reward"])
    values = [_number(v) for v in rewards.values()]
    return values[0] if len(values) == 1 else None


def trial_result(data: bytes | None) -> dict:
    """Allowlisted facts from a trial's result.json ({} when absent or unreadable)."""
    d = _json(data)
    if not d or not ({"trial_name", "task_name"} & set(d)):
        return {}  # not a trial result (e.g. a job-level result.json)
    agent = d.get("agent_result") if isinstance(d.get("agent_result"), dict) else {}
    verifier = d.get("verifier_result") if isinstance(d.get("verifier_result"), dict) else {}
    exception = d.get("exception_info") if isinstance(d.get("exception_info"), dict) else None
    task = str(d.get("task_name") or "").rsplit("/", 1)[-1]
    duration = None
    try:
        start = datetime.fromisoformat(str(d["started_at"]))
        end = datetime.fromisoformat(str(d["finished_at"]))
        duration = max((end - start).total_seconds(), 0.0)
    except (KeyError, TypeError, ValueError):
        pass
    meta = {
        "task": _label(task),
        "reward": primary_reward(verifier.get("rewards")),
        "error_type": (_label(exception.get("exception_type")) or "exception")
        if exception
        else None,
        "cost_usd": _number(agent.get("cost_usd")),
        "input_tokens": _count(agent.get("n_input_tokens")),
        "cache_tokens": _count(agent.get("n_cache_tokens")),
        "output_tokens": _count(agent.get("n_output_tokens")),
        "duration_sec": duration,
    }
    return {k: v for k, v in meta.items() if v is not None}


def job_meta(config: bytes | None, result: bytes | None) -> dict | None:
    """Run facts from a Harbor job folder's config.json/result.json (None if not a job)."""
    cfg, res = _json(config), _json(result)
    if not cfg or not ({"datasets", "agents", "n_attempts"} & set(cfg)):
        return None
    stats = res.get("stats") if isinstance(res.get("stats"), dict) else {}
    datasets = [d for d in cfg.get("datasets") or [] if isinstance(d, dict)]
    task_names = [n for d in datasets for n in (d.get("task_names") or [])]
    sources = [dataset_source(d) for d in datasets]
    name = str(cfg.get("job_name") or "")
    completed = _count(stats.get("n_completed_trials"))
    return {
        "source": "harbor_job_folder",
        "job_id": _label(str(res.get("id") or "")),
        "job_name": name if re.fullmatch(r"[\w.:@/-]{1,200}", name) else None,
        "planned_trials": _count(res.get("n_total_trials")),
        "total_trials": _count(res.get("n_total_trials")),
        "completed_trials": completed,
        "errored_trials": _count(stats.get("n_errored_trials")),
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
        "n_attempts": _count(cfg.get("n_attempts")),
        "cost_usd": _number(stats.get("cost_usd")),
        "overrides": overrides(cfg),
    }
