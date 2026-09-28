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


def number(value: object, low: float | None = None) -> float | None:
    """A finite int/float (not bool), optionally at least `low`; anything else is None."""
    if type(value) in (int, float) and math.isfinite(value) and (low is None or value >= low):
        return float(value)
    return None


def count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def duration(record: Mapping) -> float | None:
    """Seconds from `started_at` to `finished_at` (ISO timestamps), else None."""
    try:
        start = datetime.fromisoformat(str(record["started_at"]))
        end = datetime.fromisoformat(str(record["finished_at"]))
    except (KeyError, TypeError, ValueError):
        return None
    return max((end - start).total_seconds(), 0.0)


def dataset_source(entry: Mapping) -> tuple[str | None, str | None, bool | None]:
    """(name, ref, canonical) for a job config dataset: a registry name, or a git repo
    (`repo: https://github.com/OWNER/REPO.git@COMMIT`, `path: tasks`)."""
    name = entry.get("name")
    if isinstance(name, str) and name:
        if text_label(name) is None:
            return None, None, None
        return name, text_label(entry.get("ref")), bool(CANONICAL_DATASETS.match(name))
    repo = entry.get("repo")
    if isinstance(repo, str):
        m = GIT_REPO.match(repo.strip())
        if m:
            path = entry.get("path") if isinstance(entry.get("path"), str) else ""
            label = m.group(1) + (f"/{path.strip('/')}" if path else "")
            if text_label(label) != label:
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
    except (UnicodeError, ValueError, RecursionError):
        return {}
    return value if isinstance(value, dict) else {}


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
    if "reward" in rewards:
        return number(rewards["reward"])
    values = [number(v) for v in rewards.values()]
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
    }
    return {k: v for k, v in meta.items() if v is not None}


def configured_agents(cfg: dict) -> int | None:
    """How many agent/model entries a job config plans (a comparison job runs several,
    so several models are expected rather than a substitution)."""
    agents = cfg.get("agents")
    if not isinstance(agents, list):
        return None
    return len([a for a in agents if isinstance(a, dict)]) or None


def job_meta(config: bytes | None, result: bytes | None) -> dict | None:
    """Run facts from a Harbor job folder's config.json/result.json (None if not a job)."""
    cfg, res = _json(config), _json(result)
    if not cfg or not ({"datasets", "agents", "n_attempts"} & set(cfg)):
        return None
    stats = res.get("stats") if isinstance(res.get("stats"), dict) else {}
    datasets = [d for d in cfg.get("datasets") or [] if isinstance(d, dict)]
    task_names = [n for d in datasets for n in (d.get("task_names") or [])]
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
