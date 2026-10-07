"""Harbor Hub jobs as inputs, through the user's installed and logged-in `harbor` CLI.

`harbor://jobs/<uuid>` or a pasted `https://hub.harborframework.com/jobs/<uuid>` URL:

1. `harbor hub job show <id> --json` for run facts (planned/completed/errored trials,
   dataset and digest, the task list and settings in the job config);
2. `harbor hub job trials <id> --json` (paged) for each trial's name, task, reward, error,
   cost and tokens, which the Hub records authoritatively;
3. trajectories: `harbor hub trial download <trial> --trajectory` per trial, in parallel
   (default), or one `harbor hub job download <id>` for the full archive (`full=True`);
4. with each trajectory, `harbor hub trial show <trial> --json` (~4 KB): phase timings,
   exception type and time, rewards. Reduced to the allowlisted fields of a Harbor trial
   result.json (`reduced_result`) and saved as `result.json` beside the trajectory, so a
   Hub trial reads like a local one (agent walltime, failed phase). The archive already
   has each trial's result.json.

The CLI is run without a shell, with fixed arguments and a validated UUID; nothing from a
trace is ever passed to it. Its stderr is withheld (it can carry URLs or tokens). Using
the CLI keeps the user's login and Harbor version, and relies only on its public
`--json` output rather than Harbor's internal Python API.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from ...data.jsonval import (
    Doc,
    JsonObject,
    as_list,
    as_object,
    as_str,
    count,
    is_object,
    number,
)
from ...data.loader import TraceError, load_trace
from ..inputs import Source, SourceError, local_fingerprint
from ..sync import private_directory, private_tree
from .files import (
    OVERRIDE,
    PHASES,
    _flatten,
    text_label,
    trial_result,
)
from .listing import (
    SAVED_LISTING,
    TRIAL_ID,
    TRIAL_NAME,
    UUID,
    job_datasets,
    retry_chains,
    run_meta,
    trial_label,
    trial_meta,
    valid_row,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ...data.model import Trace

JOB = re.compile(
    r"^(?:harbor://jobs/|https?://hub\.harborframework\.com/jobs/)(" + UUID + r")(?:[/?#].*)?$"
)
# Leaderboard rows: harbor://rows/<uuid> or any hub URL ending in .../rows/<uuid>.
ROW = re.compile(
    r"^(?:harbor://rows/|https?://hub\.harborframework\.com/\S*?/rows/)(" + UUID + r")(?:[/?#].*)?$"
)
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
    return str(match.group(1)).lower()


def row_id(value: str) -> str | None:
    match = ROW.match(value)
    return str(match.group(1)).lower() if match else None


def reference(value: str) -> tuple[str, str]:
    """("row" | "job", uuid) for a Hub reference; invalid ones fail before any CLI call."""
    row = row_id(value)
    return ("row", row) if row is not None else ("job", job_id(value))


class Harbor(Protocol):
    """What the Hub source needs from the `harbor` CLI: raw output and `--json` output
    of fixed subcommands. `HarborCLI` runs the real CLI; tests pass doubles that run
    nothing."""

    def run(self, *args: str) -> str: ...

    def json(self, *args: str) -> object: ...


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
            # No shell; fixed subcommands plus validated UUIDs and our own folder paths.
            done = subprocess.run(  # noqa: S603 - arguments are fixed or validated
                [self.exe, *args],
                check=False,
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


def _object(value: object) -> JsonObject:
    """A `--json` output that must be an object."""
    if not is_object(value):
        raise SourceError("harbor_returned_invalid_json")
    return value


def _pages(cli: Harbor, *args: str) -> Iterator[tuple[int, int, JsonObject]]:
    """(page, total_pages, data) for a paged `--json` listing, at most MAX_PAGES."""
    for page in range(1, MAX_PAGES + 1):
        data = _object(cli.json(*args, "--limit", str(PAGE_SIZE), "--page", str(page)))
        pages = count(data.get("total_pages")) or 1
        yield page, pages, data
        if page >= pages:
            return


def listing(
    cli: Harbor,
    job: str,
    progress: Progress = _quiet,
    label: str = "job",
    want: set[str] | None = None,
    shows: dict[str, JsonObject] | None = None,
) -> tuple[Doc, list[JsonObject]]:
    """Run facts and trial rows of one job. With `want` (a leaderboard row's trials),
    paging stops once every wanted trial has been listed: a row often holds one agent's
    share of a multi-agent job, so the rest of the job isn't needed. `shows` collects the
    raw `hub job show` record per job (for save_listing)."""
    progress(f"listing {label}")
    show = _object(cli.json("hub", "job", "show", job))
    rows: dict[str, JsonObject] = {}
    for page, pages, data in _pages(cli, "hub", "job", "trials", job):
        for row in as_list(data.get("items")):
            if is_object(row) and row.get("id"):
                rows[str(row["id"])] = dict(row)
        progress(f"listing {label} trials · page {page}/{pages} · {len(rows)} trials")
        if want is not None and want <= rows.keys():
            break
    ordered = sorted(rows.values(), key=lambda r: str(r.get("name") or r["id"]))
    if shows is not None:
        shows[job] = show
    return run_meta(job, show, ordered), ordered


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


def _override_values(config: object) -> JsonObject:
    """Only the override settings `overrides()` looks at: configs can hold env and keys."""
    return {path: value for path, value in _flatten(config) if OVERRIDE.search(path)}


def _reduced(show: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> Doc:
    """What run_meta/trial_meta read, and nothing else (they re-validate it on load)."""
    config = as_object(show.get("config"))
    datasets = job_datasets(config)
    agents = config.get("agents")
    return {
        "show": {
            **{k: show.get(k) for k in SHOW_KEYS},
            "config": {
                "datasets": [
                    {k: d.get(k) for k in ("name", "ref", "task_names")} for d in datasets
                ],
                "n_attempts": config.get("n_attempts"),
                "agents": [{} for a in agents if is_object(a)]
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


def save_listing(
    dest: Path, job: str, show: Mapping[str, object] | None, rows: Sequence[Mapping[str, object]]
) -> None:
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
        tmp.write_text(
            json.dumps({"version": 1, "job": job, **_reduced(show or {}, [*kept, *rows])})
        )
        tmp.chmod(0o600)
        tmp.replace(dest / SAVED_LISTING)
    except (OSError, TypeError, ValueError):
        pass  # a missing sidecar only means a later local rescan lacks Hub facts


def _loader(path: Path) -> Callable[[], Trace]:
    def load() -> Trace:
        if not path.is_file():
            raise TraceError("no_trajectory_downloaded")
        return load_trace(path)

    return load


def _fetch_archive(
    cli: Harbor,
    job: str,
    rows: Sequence[Mapping[str, object]],
    dest: Path,
    refresh: bool,
    progress: Progress,
) -> dict[str, Path]:
    """One `harbor hub job download` into `<dest>/job`, kept unless refreshing."""
    target = dest / "job"
    if refresh and target.exists():
        shutil.rmtree(target)
    if not target.exists():
        progress(f"downloading job archive ({len(rows)} trials)")
        with contextlib.suppress(SourceError):  # every trial then reports as unavailable
            cli.run("hub", "job", "download", job, "-o", str(target))
    private_tree(dest)
    found = {p.parent.parent.name: p for p in target.rglob("agent/trajectory.json")}
    return {str(r["id"]): found.get(trial_label(r), target / "missing") for r in rows}


def _fetch_trajectory(cli: Harbor, row: Mapping[str, object], path: Path, dest: Path) -> bool:
    """Download one trial's trajectory to `path`; False when it isn't there after."""
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


RESULT = "result.json"
RESULT_BYTES = 1024 * 1024
RESULT_TEXT = ("id", "trial_name", "task_name", "started_at", "finished_at")
PHASE_TIMES = ("started_at", "finished_at")
EXCEPTION_KEYS = ("exception_type", "occurred_at")


def _strings(record: Mapping[str, object], keys: Sequence[str]) -> Doc:
    return {k: v for k in keys if (v := as_str(record.get(k))) is not None}


def _same_time(a: object, b: object) -> bool:
    try:
        return datetime.fromisoformat(str(a)) == datetime.fromisoformat(str(b))
    except ValueError:
        return False


def reduced_result(show: Mapping[str, object]) -> Doc | None:
    """The fields of a Hub trial record (`hub trial show --json`) that Harbor's trial
    result.json has and `trial_result` reads: names, timings, exception type and time,
    numeric rewards. Owner and user names, the operator's paths, the run config and
    exception messages are dropped. The Hub stamps an exception's `occurred_at` with the
    trial's end, which says nothing about where it failed (a timeout would read as after
    the verifier): that time is dropped, leaving the failed phase unknown. None when it
    isn't a trial record."""
    if as_str(show.get("trial_name")) is None and as_str(show.get("task_name")) is None:
        return None
    result: Doc = _strings(show, RESULT_TEXT)
    for phase in PHASES:
        if is_object(times := show.get(phase)):
            result[phase] = _strings(times, PHASE_TIMES)
    rewards = as_object(as_object(show.get("verifier_result")).get("rewards"))
    numeric = {k: v for k, v in rewards.items() if number(v) is not None}
    if numeric:
        result["verifier_result"] = {"rewards": numeric}
    if is_object(exception := show.get("exception_info")):
        kept = _strings(exception, EXCEPTION_KEYS)
        if _same_time(kept.get("occurred_at"), result.get("finished_at")):
            del kept["occurred_at"]
        result["exception_info"] = kept
    return result


def _write_private(path: Path, doc: Doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(doc))
    tmp.chmod(0o600)
    tmp.replace(path)


def _fetch_result(cli: Harbor, row: Mapping[str, object], path: Path) -> bool:
    """Save one trial's reduced Hub record as `path` (result.json); False when the Hub
    gave none. A reply that isn't a trial record is saved as `{}` so a resumed sync
    doesn't ask again; a failed command is retried next time. Best effort: without it
    the trial's timings are unknown, not wrong."""
    try:
        result = reduced_result(_object(cli.json("hub", "trial", "show", str(row["id"]))))
        _write_private(path, result or {})
    except (SourceError, OSError):
        return False
    return result is not None


def _details(trajectory: Path) -> Callable[[], Doc]:
    """Lazy reader for the trial's result.json: beside the trajectory (synced) or one
    folder up (the job archive's `<trial>/agent/trajectory.json`)."""

    def read() -> Doc:
        for path in (trajectory.parent / RESULT, trajectory.parent.parent / RESULT):
            try:
                with path.open("rb") as handle:
                    if facts := trial_result(handle.read(RESULT_BYTES)):
                        return facts
            except OSError:
                continue
        return {}

    return read


@dataclass
class _Downloads:
    """Progress of parallel trajectory downloads (fixed text and counts only)."""

    total: int
    local: int
    progress: Progress
    done: int = 0
    failed: int = 0
    no_result: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def report(self) -> None:
        self.progress(
            f"downloading trajectories {self.done}/{self.total}"
            + (f" · {self.local} already local" if self.local else "")
            + (f" · {self.failed} failed" if self.failed else "")
            + (f" · {self.no_result} without timings" if self.no_result else "")
        )

    def finished(self, ok: bool, result: bool = True) -> None:
        with self.lock:
            self.done += 1
            self.failed += not ok
            self.no_result += not result
            self.report()


def fetch(
    cli: Harbor,
    job: str,
    rows: Sequence[Mapping[str, object]],
    dest: Path,
    full: bool,
    workers: int = 8,
    refresh: bool = False,
    progress: Progress = _quiet,
) -> dict[str, Path]:
    """Download trajectories (and each trial's reduced result.json) into `dest`;
    returns trial id -> expected trajectory path.

    Trajectory mode resumes: files already present are not downloaded again, and an
    older sync without result.json gets just those. A failed download leaves the trial
    unavailable (reported), never aborts the scan.
    """
    private_tree(dest)
    if full:
        return _fetch_archive(cli, job, rows, dest, refresh, progress)
    paths = {str(r["id"]): dest / trial_label(r) / "trajectory.json" for r in rows}

    def missing(row: Mapping[str, object]) -> tuple[bool, bool]:
        path = paths[str(row["id"])]
        return refresh or not path.is_file(), refresh or not (path.parent / RESULT).is_file()

    todo = [r for r in rows if any(missing(r))]
    downloads = _Downloads(len(todo), len(rows) - len(todo), progress)

    def one(row: Mapping[str, object]) -> None:
        path = paths[str(row["id"])]
        trajectory, result = missing(row)
        ok = _fetch_trajectory(cli, row, path, dest) if trajectory else True
        saved = _fetch_result(cli, row, path.parent / RESULT) if result else True
        downloads.finished(ok, saved)

    if todo:
        downloads.report()
        with ThreadPoolExecutor(max(1, workers)) as pool:
            list(pool.map(one, todo))
    private_tree(dest)
    return paths


def row_trials(cli: Harbor, row: str) -> list[str]:
    return [
        str(t["trial_id"])
        for _, _, data in _pages(cli, "hub", "leaderboard", "row", "trial", "list", row)
        for t in as_list(data.get("items"))
        if is_object(t) and TRIAL_ID.fullmatch(str(t.get("trial_id") or ""))
    ]


# A leaderboard row's trials grouped by job: (job id, that job's rows in the row).
RowJobs = list[tuple[str, list[JsonObject]]]


def _row_jobs(
    cli: Harbor, wanted: set[str], progress: Progress
) -> tuple[RowJobs, list[Doc], set[str]]:
    """(the row's trials by job, those jobs' run facts, trials no job accounted for)."""
    remaining = set(wanted)
    jobs: RowJobs = []
    job_runs: list[Doc] = []
    unresolved: set[str] = set()
    lookups = 0
    while remaining and lookups < MAX_JOB_LOOKUPS:
        trial = min(remaining)
        lookups += 1
        detail = as_object(cli.json("hub", "trial", "show", trial))
        job = str(detail.get("job_id") or "")
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
    return jobs, job_runs, unresolved | remaining


def _leaderboard(show: JsonObject, jobs: RowJobs) -> Doc:
    """The row's own leaderboard record: rank, labels and reported metrics."""
    meta = as_object(show.get("metadata"))
    metrics = as_object(show.get("metrics"))

    def label(key: str) -> str | None:
        value = meta.get(key)
        return text_label(as_object(value).get("label") if is_object(value) else value, 80)

    return {
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
    }


def row_listing(cli: Harbor, row: str, progress: Progress = _quiet) -> tuple[Doc, RowJobs]:
    """A leaderboard row's facts and its trials grouped by job.

    The row lists trial IDs only; one `trial show` per job finds each job, whose listing
    then covers the rest of the row's trials in it (a row may hold a subset of a job,
    or trials from several jobs)."""
    progress("listing leaderboard row")
    show = _object(cli.json("hub", "leaderboard", "row", "show", row))
    wanted = set(row_trials(cli, row))
    progress(f"leaderboard row lists {len(wanted)} trials · finding their jobs")
    jobs, job_runs, unresolved = _row_jobs(cli, wanted, progress)
    metrics = as_object(show.get("metrics"))
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
        "leaderboard": _leaderboard(show, jobs),
    }
    return run, jobs


def _trial_sources(
    cli: Harbor,
    job: str,
    rows: Sequence[JsonObject],
    dest: Path,
    full: bool,
    workers: int,
    refresh: bool,
    progress: Progress = _quiet,
    show: JsonObject | None = None,
) -> list[Source]:
    """Sources for a job's listed trials. Each trial's Hub facts are its run listing
    (`Source.meta`; see `facts` for how they combine with the trajectory's)."""
    rows = [r for r in rows if valid_row(r)]
    paths = fetch(cli, job, rows, dest / job, full, workers, refresh, progress)
    save_listing(dest / job, job, show, rows)
    chains = retry_chains(rows)
    return [
        Source(
            trial_label(row),
            _loader(paths[str(row["id"])]),
            meta=trial_meta(row, chains.get(str(row["id"]))),
            fingerprint=local_fingerprint(paths[str(row["id"])]),
            details=_details(paths[str(row["id"])]),
            local=paths[str(row["id"])],
        )
        for row in rows
    ]


def harbor_sources(
    value: str,
    dest: Path,
    full: bool = False,
    workers: int = 8,
    cli: Harbor | None = None,
    refresh: bool = False,
    progress: Progress = _quiet,
) -> tuple[list[Source], Doc]:
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
    shows: dict[str, JsonObject] = {}
    run, rows = listing(cli, job, progress, shows=shows)
    sources = _trial_sources(cli, job, rows, dest, full, workers, refresh, progress, shows[job])
    return sources, run


def inspect_job(value: str, cli: Harbor | None = None) -> Doc:
    """Listing only (no downloads): run facts plus per-trial metadata records."""
    kind, ref = reference(value)
    cli = cli or HarborCLI.find()
    if kind == "row":
        run, jobs = row_listing(cli, ref)
        rows = [r for _, job_rows in jobs for r in job_rows]
    else:
        run, rows = listing(cli, ref)
    rows = [r for r in rows if valid_row(r)]
    chains = retry_chains(rows)
    trials = [dict(trial_meta(r, chains.get(str(r["id"]))), input_id=trial_label(r)) for r in rows]
    return {"run": run, "trials": trials}
