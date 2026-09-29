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
import threading
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .checks import identifier
from .harbor_files import (
    OVERRIDE,
    _flatten,
    configured_agents,
    duration,
    overrides,
    text_label,
)
from .jsonval import count, number
from .loader import TraceError, load_trace
from .sources import Source, SourceError, local_fingerprint, private_directory, private_tree

UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
JOB = re.compile(
    r"^(?:harbor://jobs/|https?://hub\.harborframework\.com/jobs/)(" + UUID + r")(?:[/?#].*)?$"
)
# Leaderboard rows: harbor://rows/<uuid> or any hub URL ending in .../rows/<uuid>.
ROW = re.compile(
    r"^(?:harbor://rows/|https?://hub\.harborframework\.com/\S*?/rows/)(" + UUID + r")(?:[/?#].*)?$"
)
# Hub values that become CLI arguments or folder names are validated first: an ID must be
# a UUID (never an option like `-o...`), a name a single path segment (no `/`, no `..`).
TRIAL_ID = re.compile(UUID)
TRIAL_NAME = re.compile(r"[A-Za-z0-9][\w.-]{0,127}")
ACCEPTED = (
    "harbor://jobs/<uuid>, harbor://rows/<uuid>, "
    "https://hub.harborframework.com/jobs/<uuid>, "
    "https://hub.harborframework.com/datasets/.../leaderboards/<lb>/rows/<uuid>"
)
# The Hub serves up to 1000 trials per page (a 2,225-trial job is 3 calls, not 23). A
# smaller server cap still pages correctly: `total_pages` drives the loop.
PAGE_SIZE = 1000
MAX_PAGES = 1000
MAX_JOB_LOOKUPS = 50

# Progress callbacks receive fixed text plus counts only: never IDs, names or URLs.
Progress = Callable[[str], None]


def _quiet(message: str) -> None:
    pass


def is_harbor(value: str) -> bool:
    return value.startswith("harbor://") or value.startswith("https://hub.harborframework.com/")


def job_id(value: str) -> str:
    match = JOB.match(value)
    if match is None:
        raise SourceError("invalid_harbor_reference: expected " + ACCEPTED)
    return match.group(1).lower()


def row_id(value: str) -> str | None:
    match = ROW.match(value)
    return match.group(1).lower() if match else None


def reference(value: str) -> tuple[str, str]:
    """("row" | "job", uuid) for a Hub reference; invalid ones fail before any CLI call."""
    row = row_id(value)
    return ("row", row) if row is not None else ("job", job_id(value))


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
                [self.exe, *args],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
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


def _pages(cli: HarborCLI, *args: str) -> Iterator[tuple[int, int, Mapping]]:
    """(page, total_pages, data) for a paged `--json` listing, at most MAX_PAGES."""
    for page in range(1, MAX_PAGES + 1):
        data = cli.json(*args, "--limit", str(PAGE_SIZE), "--page", str(page))
        if not isinstance(data, Mapping):
            raise SourceError("harbor_returned_invalid_json")
        pages = data.get("total_pages")
        pages = pages if type(pages) is int and pages > 0 else 1
        yield page, pages, data
        if page >= pages:
            return


def valid_row(row: Mapping) -> bool:
    """A trial row whose ID is safe to pass to the CLI (see TRIAL_ID)."""
    return bool(TRIAL_ID.fullmatch(str(row.get("id") or "")))


def trial_meta(row: Mapping) -> dict:
    """Allowlisted per-trial facts from the Hub listing (numbers, codes, identifiers)."""
    task = str(row.get("task_name") or "").rsplit("/", 1)[-1] or None
    error = row.get("error_type")
    meta = {
        "hub_trial_id": str(row.get("id") or "") or None,
        "task": task,
        "reward": number(row.get("reward")),
        "error_type": str(error) if error else None,
        "status": str(row.get("status") or "") or None,
        "cost_usd": number(row.get("cost_usd"), 0),
        "input_tokens": count(row.get("input_tokens")),
        "cache_tokens": count(row.get("cache_tokens")),
        "output_tokens": count(row.get("output_tokens")),
        "duration_sec": duration(row),
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
        "configured_agents": configured_agents(dict(config)),
        "cost_usd": number(show.get("cost_usd"), 0),
        "overrides": overrides(config),
    }


def listing(
    cli: HarborCLI,
    job: str,
    progress: Progress = _quiet,
    label: str = "job",
    want: set[str] | None = None,
    shows: dict[str, Mapping] | None = None,
) -> tuple[dict, list[dict]]:
    """Run facts and trial rows of one job. With `want` (a leaderboard row's trials),
    paging stops once every wanted trial has been listed: a row often holds one agent's
    share of a multi-agent job, so the rest of the job isn't needed. `shows` collects the
    raw `hub job show` record per job (for save_listing)."""
    progress(f"listing {label}")
    show = cli.json("hub", "job", "show", job)
    if not isinstance(show, Mapping):
        raise SourceError("harbor_returned_invalid_json")
    rows: dict[str, dict] = {}
    for page, pages, data in _pages(cli, "hub", "job", "trials", job):
        for row in data.get("items") or []:
            if isinstance(row, Mapping) and row.get("id"):
                rows[str(row["id"])] = dict(row)
        progress(f"listing {label} trials · page {page}/{pages} · {len(rows)} trials")
        if want is not None and want <= rows.keys():
            break
    ordered = sorted(rows.values(), key=lambda r: str(r.get("name") or r["id"]))
    if shows is not None:
        shows[job] = show
    return run_meta(job, show, ordered), ordered


# Written into each synced job folder so a later scan of the local copy (`atif-scan
# ~/.cache/atif-scan/harbor/<job>`) keeps the Hub's run facts: task, reward, error, cost.
SAVED_LISTING = "hub-listing.json"
ROW_KEYS = (
    "id",
    "name",
    "task_name",
    "reward",
    "error_type",
    "status",
    "cost_usd",
    "input_tokens",
    "cache_tokens",
    "output_tokens",
    "started_at",
    "finished_at",
)
SHOW_KEYS = (
    "name",
    "n_planned_trials",
    "n_total_trials",
    "n_completed_trials",
    "n_errors",
    "cost_usd",
)


def _override_values(config: object) -> dict:
    """Only the override settings `overrides()` looks at: configs can hold env and keys."""
    return {path: value for path, value in _flatten(config) if OVERRIDE.search(path)}


def _reduced(show: Mapping, rows: list[Mapping]) -> dict:
    """What run_meta/trial_meta read, and nothing else (they re-validate it on load)."""
    config = show.get("config") if isinstance(show.get("config"), Mapping) else {}
    datasets = [d for d in config.get("datasets") or [] if isinstance(d, Mapping)]
    agents = config.get("agents")
    return {
        "show": {
            **{k: show.get(k) for k in SHOW_KEYS},
            "config": {
                "datasets": [
                    {k: d.get(k) for k in ("name", "ref", "task_names")} for d in datasets
                ],
                "n_attempts": config.get("n_attempts"),
                "agents": [{} for a in agents if isinstance(a, dict)]
                if isinstance(agents, list)
                else None,
                **_override_values(config),
            },
        },
        "rows": [
            {
                **{k: r.get(k) for k in ROW_KEYS},
                "config_values": _override_values(r.get("config_values") or {}),
            }
            for r in rows
        ],
    }


def save_listing(dest: Path, job: str, show: Mapping | None, rows: list[Mapping]) -> None:
    """Atomically write the reduced listing into the job's sync folder (best effort).

    Rows already saved for the same job are kept (these rows win by trial id): several
    leaderboard rows can share one job, and each row sync lists only its own trials.
    """
    try:
        private_directory(dest)
        ids = {str(r.get("id")) for r in rows}
        try:
            old = json.loads((dest / SAVED_LISTING).read_text())
            kept = [
                r
                for r in old["rows"]
                if str(old["job"]).lower() == job.lower()
                and isinstance(r, Mapping)
                and valid_row(r)
                and str(r["id"]) not in ids
            ]
        except (OSError, KeyError, TypeError, ValueError, AttributeError, RecursionError):
            kept = []
        tmp = dest / f".{SAVED_LISTING}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps({"version": 1, "job": job, **_reduced(show or {}, kept + rows)}))
        tmp.chmod(0o600)
        tmp.replace(dest / SAVED_LISTING)
    except (OSError, TypeError, ValueError):
        pass  # a missing sidecar only means a later local rescan lacks Hub facts


def saved_listing(data: bytes) -> tuple[dict | None, dict[str, dict]]:
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
    return run_meta(job, show, rows), {_label(r): trial_meta(r) for r in rows}


def _label(row: Mapping) -> str:
    """The trial's folder and report label: its name if a safe segment, else its UUID."""
    name = str(row.get("name") or "")
    return name if TRIAL_NAME.fullmatch(name) else str(row["id"])


def _loader(path: Path) -> Callable:
    def load():
        if not path.is_file():
            raise TraceError("no_trajectory_downloaded")
        return load_trace(path)

    return load


def fetch(
    cli: HarborCLI,
    job: str,
    rows: list[Mapping],
    dest: Path,
    full: bool,
    workers: int = 8,
    refresh: bool = False,
    progress: Progress = _quiet,
) -> dict[str, Path]:
    """Download trajectories into `dest`; returns trial id -> expected trajectory path.

    Trajectory mode resumes: files already present are not downloaded again. A failed
    download leaves the trial unavailable (reported), never aborts the scan.
    """
    private_tree(dest)
    if full:
        target = dest / "job"
        if refresh and target.exists():
            shutil.rmtree(target)
        if not target.exists():
            progress(f"downloading job archive ({len(rows)} trials)")
            try:
                cli.run("hub", "job", "download", job, "-o", str(target))
            except SourceError:
                pass  # every trial then reports as unavailable
        private_tree(dest)
        found = {p.parent.parent.name: p for p in target.rglob("agent/trajectory.json")}
        return {str(r["id"]): found.get(_label(r), target / "missing") for r in rows}

    paths = {str(r["id"]): dest / _label(r) / "trajectory.json" for r in rows}
    todo = [r for r in rows if refresh or not paths[str(r["id"])].is_file()]
    local, done, failed = len(rows) - len(todo), 0, 0
    lock = threading.Lock()

    def report() -> None:
        progress(
            f"downloading trajectories {done}/{len(todo)}"
            + (f" · {local} already local" if local else "")
            + (f" · {failed} failed" if failed else "")
        )

    def one(row: Mapping) -> None:
        nonlocal done, failed
        ok = fetch_one(row)
        with lock:
            done += 1
            failed += not ok
            report()

    def fetch_one(row: Mapping) -> bool:
        path = paths[str(row["id"])]
        path.unlink(missing_ok=True)
        try:
            cli.run("hub", "trial", "download", str(row["id"]), "--trajectory", "-o", str(dest))
        except SourceError:
            path.unlink(missing_ok=True)
            return False
        # Harbor names the folder after the trial; move it if it differs from our label.
        if not path.is_file():
            name = str(row.get("name") or "")
            candidate = dest / name / "trajectory.json" if TRIAL_NAME.fullmatch(name) else None
            if candidate is not None and candidate.is_file() and candidate != path:
                path.parent.mkdir(parents=True, exist_ok=True)
                candidate.replace(path)
        return path.is_file()

    if todo:
        report()
        with ThreadPoolExecutor(max(1, workers)) as pool:
            list(pool.map(one, todo))
    private_tree(dest)
    return paths


def row_trials(cli: HarborCLI, row: str) -> list[str]:
    return [
        str(t["trial_id"])
        for _, _, data in _pages(cli, "hub", "leaderboard", "row", "trial", "list", row)
        for t in data.get("items") or []
        if isinstance(t, Mapping) and TRIAL_ID.fullmatch(str(t.get("trial_id") or ""))
    ]


def row_listing(
    cli: HarborCLI, row: str, progress: Progress = _quiet
) -> tuple[dict, list[tuple[str, list[dict]]]]:
    """A leaderboard row's facts and its trials grouped by job.

    The row lists trial IDs only; one `trial show` per job finds each job, whose listing
    then covers the rest of the row's trials in it (a row may hold a subset of a job,
    or trials from several jobs)."""
    progress("listing leaderboard row")
    show = cli.json("hub", "leaderboard", "row", "show", row)
    if not isinstance(show, Mapping):
        raise SourceError("harbor_returned_invalid_json")
    wanted = set(row_trials(cli, row))
    progress(f"leaderboard row lists {len(wanted)} trials · finding their jobs")
    remaining, jobs, job_runs, unresolved = set(wanted), [], [], set()
    lookups = 0
    while remaining and lookups < MAX_JOB_LOOKUPS:
        trial = min(remaining)
        lookups += 1
        detail = cli.json("hub", "trial", "show", trial)
        job = str(detail.get("job_id") or "") if isinstance(detail, Mapping) else ""
        if not JOB.match(f"harbor://jobs/{job}") or any(j == job for j, _ in jobs):
            remaining.discard(trial)
            unresolved.add(trial)
            continue
        run, rows = listing(cli, job, progress, f"job {len(jobs) + 1}", want=remaining)
        mine = [r for r in rows if str(r["id"]) in wanted]
        remaining -= {str(r["id"]) for r in mine}
        if trial in remaining:  # listed job didn't contain it
            remaining.discard(trial)
            unresolved.add(trial)
        jobs.append((job, mine))
        job_runs.append(run)
    unresolved |= remaining
    meta = show.get("metadata") if isinstance(show.get("metadata"), Mapping) else {}
    metrics = show.get("metrics") if isinstance(show.get("metrics"), Mapping) else {}

    def label(key: str) -> str | None:
        v = meta.get(key)
        v = v.get("label") if isinstance(v, Mapping) else v
        return text_label(v, 80)

    run = {
        "source": "harbor_leaderboard_row",
        "job_id": row,
        "job_name": f"leaderboard row #{show.get('rank')}"
        if show.get("rank")
        else "leaderboard row",
        "planned_trials": len(wanted),
        "total_trials": len(wanted),
        "completed_trials": None,
        "errored_trials": None,
        "listed_trials": len(wanted) - len(unresolved),
        "datasets": sorted({d for r in job_runs for d in r.get("datasets") or []}),
        "dataset_refs": sorted({d for r in job_runs for d in r.get("dataset_refs") or []}),
        "config_task_names": None,
        "n_attempts": max((r.get("n_attempts") or 0 for r in job_runs), default=0) or None,
        "cost_usd": number(metrics.get("total_cost_usd"), 0),
        "overrides": sorted({o for r in job_runs for o in r.get("overrides") or []}),
        "unresolved_trials": len(unresolved),
        "leaderboard": {
            "rank": count(show.get("rank")),
            "agent": label("agent_display"),
            "model": label("model_display"),
            "reasoning_effort": label("reasoning_effort"),
            "reported_accuracy": number(metrics.get("accuracy")),
            "reported_n_trials": count(metrics.get("n_trials")),
            "reported_cost_usd": number(metrics.get("total_cost_usd"), 0),
            "reported_reward_hacks_pct": number(metrics.get("reward_hacks")),
            "display_cost": text_label(metrics.get("display_cost"), 60),
            "jobs": [j for j, _ in jobs],
        },
    }
    return run, jobs


def _trial_sources(
    cli, job, rows, dest, full, workers, refresh, progress=_quiet, show=None
) -> list[Source]:
    rows = [r for r in rows if valid_row(r)]
    paths = fetch(cli, job, rows, dest / job, full, workers, refresh, progress)
    save_listing(dest / job, job, show, rows)
    sources = []
    for row in rows:
        meta = trial_meta(row)
        sources.append(
            Source(
                _label(row),
                _loader(paths[str(row["id"])]),
                lambda reward=meta["reward"]: reward,
                meta=meta,
                fingerprint=local_fingerprint(paths[str(row["id"])]),
                local=paths[str(row["id"])],
            )
        )
    return sources


def harbor_sources(
    value: str,
    dest: Path,
    full: bool = False,
    workers: int = 8,
    cli: HarborCLI | None = None,
    refresh: bool = False,
    progress: Progress = _quiet,
) -> tuple[list[Source], dict]:
    kind, ref = reference(value)
    cli = cli or HarborCLI.find()
    private_directory(dest)
    if kind == "row":
        run, jobs = row_listing(cli, ref, progress)
        sources = [
            s
            for job, rows in jobs
            for s in _trial_sources(cli, job, rows, dest, full, workers, refresh, progress)
        ]
        return sources, run
    job = ref
    shows: dict[str, Mapping] = {}
    run, rows = listing(cli, job, progress, shows=shows)
    sources = _trial_sources(cli, job, rows, dest, full, workers, refresh, progress, shows[job])
    return sources, run


def inspect_job(value: str, cli: HarborCLI | None = None) -> dict:
    """Listing only (no downloads): run facts plus per-trial metadata records."""
    kind, ref = reference(value)
    cli = cli or HarborCLI.find()
    if kind == "row":
        run, jobs = row_listing(cli, ref)
        rows = [r for _, job_rows in jobs for r in job_rows]
    else:
        run, rows = listing(cli, ref)
    trials = [dict(trial_meta(r), input_id=_label(r)) for r in rows if valid_row(r)]
    return {"run": run, "trials": trials}
