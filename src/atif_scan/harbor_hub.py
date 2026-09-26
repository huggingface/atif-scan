"""Harbor Hub jobs as inputs, through the user's installed and logged-in `harbor` CLI.

`harbor://jobs/<uuid>` or a pasted `https://hub.harborframework.com/jobs/<uuid>` URL:

1. `harbor hub job show <id> --json` for run facts (planned/completed/errored trials,
   dataset and digest, the task list and settings in the job config);
2. `harbor hub job trials <id> --json` (paged) for each trial's name, task, reward, error,
   cost and tokens, which the Hub records authoritatively;
3. trajectories: `harbor hub trial download <trial> --trajectory` per trial, in parallel
   (default), or one `harbor hub job download <id>` for the full archive (`full=True`).

The CLI is run without a shell, with fixed arguments and a validated UUID; nothing from a
trace is ever passed to it. Its stderr is withheld (it can carry URLs or tokens). Using
the CLI keeps the user's login and Harbor version, and relies only on its public
`--json` output rather than Harbor's internal Python API.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .checks import identifier
from .loader import TraceError, load_trace
from .sources import Source, SourceError

JOB = re.compile(
    r"^(?:harbor://jobs/|https?://hub\.harborframework\.com/jobs/)"
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
    r"(?:[/?#].*)?$"
)
PAGE_SIZE = 100
# Leaderboard static rules: these settings must be unset (multipliers may be 1.0).
OVERRIDE = re.compile(
    r"(?:^|[._])(?:\w*timeout_multiplier|override_timeout_sec|override_setup_timeout_sec|"
    r"max_timeout_sec|override_(?:cpus|gpus|memory_mb|storage_mb))$"
)


def is_harbor(value: str) -> bool:
    return value.startswith("harbor://") or value.startswith("https://hub.harborframework.com/")


def job_id(value: str) -> str:
    match = JOB.match(value)
    if match is None:
        raise SourceError("invalid_harbor_job_reference")
    return match.group(1).lower()


@dataclass(frozen=True)
class HarborCLI:
    exe: str
    timeout: float = 900.0

    @classmethod
    def find(cls) -> HarborCLI:
        exe = os.environ.get("ATIF_SCAN_HARBOR") or shutil.which("harbor")
        if not exe:
            raise SourceError("harbor_cli_not_found_install_and_login_to_harbor")
        return cls(exe)

    def run(self, *args: str) -> str:
        try:
            done = subprocess.run(
                [self.exe, *args], capture_output=True, text=True, timeout=self.timeout
            )
        except (OSError, subprocess.TimeoutExpired):
            raise SourceError("harbor_command_failed") from None
        if done.returncode != 0:
            raise SourceError("harbor_command_failed")  # stderr withheld
        return done.stdout

    def json(self, *args: str) -> object:
        output = self.run(*args, "--json")  # SourceError is a ValueError: keep it outside
        try:
            return json.loads(output)
        except ValueError:
            raise SourceError("harbor_returned_invalid_json") from None


def _number(value: object) -> float | None:
    return float(value) if type(value) in (int, float) else None


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _duration(row: Mapping) -> float | None:
    try:
        start = datetime.fromisoformat(str(row["started_at"]))
        end = datetime.fromisoformat(str(row["finished_at"]))
    except (KeyError, TypeError, ValueError):
        return None
    return max((end - start).total_seconds(), 0.0)


def _flatten(value: object, prefix: str = "") -> Iterator[tuple[str, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _flatten(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _flatten(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


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


def trial_meta(row: Mapping) -> dict:
    """Allowlisted per-trial facts from the Hub listing (numbers, codes, identifiers)."""
    task = str(row.get("task_name") or "").rsplit("/", 1)[-1] or None
    error = row.get("error_type")
    meta = {
        "hub_trial_id": str(row.get("id") or "") or None,
        "task": task,
        "reward": _number(row.get("reward")),
        "error_type": str(error) if error else None,
        "status": str(row.get("status") or "") or None,
        "cost_usd": _number(row.get("cost_usd")),
        "input_tokens": _count(row.get("input_tokens")),
        "cache_tokens": _count(row.get("cache_tokens")),
        "output_tokens": _count(row.get("output_tokens")),
        "duration_sec": _duration(row),
        "overrides": overrides(row.get("config_values") or {}),
    }
    for key in ("task", "error_type", "status", "hub_trial_id"):
        if meta[key] is not None:
            try:
                identifier(meta[key])
            except ValueError:
                meta[key] = None if key != "error_type" else "other"
    return meta


def run_meta(job: str, show: Mapping, rows: list[Mapping]) -> dict:
    config = show.get("config") if isinstance(show.get("config"), Mapping) else {}
    datasets = [d for d in config.get("datasets") or [] if isinstance(d, Mapping)]
    task_names = [n for d in datasets for n in (d.get("task_names") or [])]
    name = str(show.get("name") or "")
    return {
        "source": "harbor_hub",
        "job_id": job,
        "job_name": name if re.fullmatch(r"[\w.:@/-]{1,200}", name) else None,
        "planned_trials": _count(show.get("n_planned_trials")),
        "total_trials": _count(show.get("n_total_trials")),
        "completed_trials": _count(show.get("n_completed_trials")),
        "errored_trials": _count(show.get("n_errors")),
        "listed_trials": len(rows),
        "datasets": [str(d.get("name")) for d in datasets if d.get("name")],
        "dataset_refs": [str(d.get("ref")) for d in datasets if d.get("ref")],
        "config_task_names": len(task_names) or None,
        "n_attempts": _count(config.get("n_attempts")),
        "cost_usd": _number(show.get("cost_usd")),
        "overrides": overrides(config),
    }


def listing(cli: HarborCLI, job: str) -> tuple[dict, list[dict]]:
    show = cli.json("hub", "job", "show", job)
    if not isinstance(show, Mapping):
        raise SourceError("harbor_returned_invalid_json")
    rows: dict[str, dict] = {}
    page = 1
    while True:
        data = cli.json("hub", "job", "trials", job, "--limit", str(PAGE_SIZE), "--page", str(page))
        if not isinstance(data, Mapping):
            raise SourceError("harbor_returned_invalid_json")
        for row in data.get("items") or []:
            if isinstance(row, Mapping) and row.get("id"):
                rows[str(row["id"])] = dict(row)
        if page >= int(data.get("total_pages") or 1):
            break
        page += 1
    ordered = sorted(rows.values(), key=lambda r: str(r.get("name") or r["id"]))
    return run_meta(job, show, ordered), ordered


def _label(row: Mapping) -> str:
    try:
        return identifier(str(row.get("name")))
    except ValueError:
        return str(row["id"])


def _loader(path: Path) -> Callable:
    def load():
        if not path.is_file():
            raise TraceError("no_trajectory_downloaded")
        return load_trace(path)

    return load


def fetch(
    cli: HarborCLI, job: str, rows: list[Mapping], dest: Path, full: bool, workers: int = 8
) -> dict[str, Path]:
    """Download trajectories into `dest`; returns trial id -> expected trajectory path.

    Trajectory mode resumes: files already present are not downloaded again. A failed
    download leaves the trial unavailable (reported), never aborts the scan.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if full:
        target = dest / "job"
        if not target.exists():
            try:
                cli.run("hub", "job", "download", job, "-o", str(target))
            except SourceError:
                pass  # every trial then reports as unavailable
        found = {p.parent.parent.name: p for p in target.rglob("agent/trajectory.json")}
        return {str(r["id"]): found.get(_label(r), target / "missing") for r in rows}

    paths = {str(r["id"]): dest / _label(r) / "trajectory.json" for r in rows}

    def one(row: Mapping) -> None:
        path = paths[str(row["id"])]
        if path.is_file():
            return
        try:
            cli.run("hub", "trial", "download", str(row["id"]), "--trajectory", "-o", str(dest))
        except SourceError:
            return
        # Harbor names the folder after the trial; move it if it differs from our label.
        if not path.is_file():
            name = str(row.get("name") or "")
            candidate = dest / name / "trajectory.json" if name else None
            if candidate is not None and candidate.is_file() and candidate != path:
                path.parent.mkdir(parents=True, exist_ok=True)
                candidate.replace(path)

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, rows))
    return paths


def harbor_sources(
    value: str, dest: Path, full: bool = False, workers: int = 8, cli: HarborCLI | None = None
) -> tuple[list[Source], dict]:
    cli = cli or HarborCLI.find()
    job = job_id(value)
    run, rows = listing(cli, job)
    paths = fetch(cli, job, rows, dest / job, full, workers)
    sources = []
    for row in rows:
        meta = trial_meta(row)
        sources.append(
            Source(
                _label(row),
                _loader(paths[str(row["id"])]),
                lambda reward=meta["reward"]: reward,
                meta=meta,
            )
        )
    return sources, run


def inspect_job(value: str, cli: HarborCLI | None = None) -> dict:
    """Listing only (no downloads): run facts plus per-trial metadata records."""
    cli = cli or HarborCLI.find()
    run, rows = listing(cli, job_id(value))
    return {"run": run, "trials": [dict(trial_meta(r), input_id=_label(r)) for r in rows]}
