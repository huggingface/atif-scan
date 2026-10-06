"""Which trials count: the canonical set of a run, with replaced trials kept as evidence.

A bench-run cohort re-runs infrastructure failures (finished, errored, never verified)
as single-trial *replacement* jobs (`runs/replacements/<id>/receipt.json`, chained by
`supersedes`). A release manifest (`bench-run.release/v1`) freezes the reported trials;
its `harbor_rows.trial_ids` are the set Harbor Hub shows. Either way the canonical set is
what a run is scored on; every replaced original and superseded link is still scanned,
for forensics, and labelled with its lineage, never counted.

Both readers fail closed: a receipt or manifest that names this run but doesn't verify
rejects the whole selection, as bench-run's dashboard does. Nothing here imports
bench-run, follows config paths outside the root, or reads trace content.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..data.jsonval import Doc, JsonObject, as_list, as_object, as_str, load_object
from .bench import LIMIT, _name, _object, _path, _read, load_run

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from .bench import BenchRun

REPLACEMENTS = ("runs", "replacements")
# bench-run's REPLACEMENT_KEYS: exactly these, no more.
RECEIPT_KEYS = {
    "schema_version",
    "kind",
    "run_id",
    "lineage",
    "upload",
    "configs",
    "sources_lock_sha256",
}
# Codex diagnostic cohorts' replacements: the same lineage, a distribution lock instead.
DIAGNOSTIC_RECEIPT_KEYS = (RECEIPT_KEYS - {"sources_lock_sha256"}) | {
    "arm",
    "distribution_lock_sha256",
}
RECEIPT_KINDS = {
    "replacement": RECEIPT_KEYS,
    "codex-diagnostic-replacement": DIAGNOSTIC_RECEIPT_KEYS,
}
RELEASE_SCHEMA = "bench-run.release/v1"
# A release lists every trial with file digests (~7 KB each): far above receipt sizes.
MANIFEST_LIMIT = 64 * 1024 * 1024
# Fixed codes from receipts (Harbor error types, phases): label-safe or dropped.
CODE = re.compile(r"[A-Za-z][\w.-]{0,80}")

# (job folder name, trial folder name): how a trial is identified across jobs.
TrialKey = tuple[str, str]


@dataclass(frozen=True)
class Selection:
    """The canonical trials of a run and the forensic ones, with allowlisted lineage."""

    kind: str  # "bench_run" or "release"
    name: str
    jobs: tuple[Path, ...]  # every job folder to scan: parents and replacement jobs
    # Canonical trials that replace another: lineage facts (codes and trial names).
    replacements: Mapping[TrialKey, Doc] = field(default_factory=dict)
    # Replaced originals and superseded links: kept for inspection, never counted.
    replaced: Mapping[TrialKey, Doc] = field(default_factory=dict)
    # A release's exact canonical set; None means "everything not replaced".
    canonical: frozenset[TrialKey] | None = None
    summary: tuple[Doc, ...] = ()  # one row per replacement, for the report
    pending: int = 0  # replacement slots whose last link hasn't run
    parents: frozenset[str] = frozenset()  # the runs' own job parts (folder names)

    @property
    def replacement_jobs(self) -> frozenset[str]:
        """Job folders holding replacements, not the runs' own job parts."""
        return frozenset(job.name for job in self.jobs) - self.parents

    def role(self, key: TrialKey) -> str:
        if key in self.replaced:
            return "replaced"
        if self.canonical is not None and key not in self.canonical:
            return "not_selected"
        return "canonical"

    def lineage(self, key: TrialKey) -> Doc:
        return dict(self.replaced.get(key) or self.replacements.get(key) or {})

    def document(self) -> Doc:
        return {
            "kind": self.kind,
            "name": self.name,
            "replacements": [dict(row) for row in self.summary],
            "pending": self.pending,
        }


def _code(value: object) -> str | None:
    text = as_str(value)
    return text if text and CODE.fullmatch(text) else None


def _task(trial: str) -> str:
    return trial.partition("__")[0]


def _sha(path: Path) -> str:
    return hashlib.sha256(_read(path)).hexdigest()


def _trial_result(job: Path, trial: str) -> JsonObject:
    path = job / trial / "result.json"
    return load_object(_read(path), LIMIT) if path.is_file() else {}


def _job_trials(job: Path) -> list[str]:
    if not job.is_dir():
        return []
    return sorted(
        p.name for p in job.iterdir() if p.is_dir() and not p.is_symlink() and "__" in p.name
    )


@dataclass
class _Replacement:
    rep_id: str
    lineage: JsonObject
    partition: str
    job: Path
    trial: str | None  # the replacement job's one trial (None: not started)


def _check_receipt(receipt: JsonObject, rep_id: str) -> None:
    if not (
        set(receipt) == RECEIPT_KINDS.get(as_str(receipt.get("kind")) or "")
        and receipt.get("schema_version") == 1
        and receipt.get("run_id") == rep_id
        and receipt.get("upload") is False
    ):
        raise ValueError("bench_replacement_receipt")


def _check_parent(run: BenchRun, lineage: JsonObject) -> str:
    """The receipt's partition, when its parent digests are this run's."""
    partition = as_str(lineage.get("partition"))
    if partition is None or partition not in run.partitions:
        raise ValueError("bench_replacement_partition")
    if (
        lineage.get("parent_receipt_sha256") != run.receipt_sha256
        or lineage.get("parent_config_sha256") != run.config_sha256[partition]
    ):
        raise ValueError("bench_replacement_parent_drift")
    return partition


def _replacement_job(
    root: Path, run: BenchRun, receipt: JsonObject, target: Path, partition: str
) -> Path:
    configs = as_object(receipt.get("configs"))
    name = f"{partition}.json"
    if set(configs) != {name} or _sha(target / name) != configs[name]:
        raise ValueError("bench_replacement_config_drift")
    job = _path(root, "jobs", _name(_object(target / name).get("job_name")))
    if job in run.jobs:
        raise ValueError("bench_replacement_job_alias")
    return job


def _receipt(root: Path, run: BenchRun, target: Path) -> _Replacement | None:
    """A replacement receipt naming `run` as parent, verified; None for other parents."""
    receipt = _object(target / "receipt.json")
    lineage = as_object(receipt.get("lineage"))
    if lineage.get("parent_run") != run.name:
        return None
    rep_id = _name(target.name)
    _check_receipt(receipt, rep_id)
    partition = _check_parent(run, lineage)
    job = _replacement_job(root, run, receipt, target, partition)
    task = _name(lineage.get("task"))
    _name(lineage.get("replaced_trial"))
    trials = _job_trials(job)
    if len(trials) > 1 or any(_task(t) != task for t in trials):
        raise ValueError("bench_replacement_attempts")
    return _Replacement(rep_id, lineage, partition, job, trials[0] if trials else None)


def _replaceable(job: Path, trial: str, lineage: JsonObject) -> bool:
    """Only finished, errored, never-verified trials (or a recorded verified crash)."""
    result = _trial_result(job, trial)
    if not result or _task(trial) != lineage.get("task") or not result.get("exception_info"):
        return False
    rewards = as_object(result.get("verifier_result")).get("rewards")
    reward = as_object(rewards).get("reward") if isinstance(rewards, dict) else rewards
    if lineage.get("verified_crash") is True:
        return reward == lineage.get("replaced_reward")
    return reward is None


def _row(rep: _Replacement, state: str, replaced: str) -> Doc:
    lineage = rep.lineage
    return {
        "id": rep.rep_id,
        "task": _task(replaced),
        "replaced_trial": replaced,
        "replaced_error": _code(lineage.get("exception_type")),
        "failure_phase": _code(lineage.get("failure_phase")),
        "supersedes": _code(lineage.get("supersedes")),
        "replacement_trial": rep.trial,
        "state": state,
    }


def _links(run: BenchRun, reps: dict[str, _Replacement]) -> dict[TrialKey, str]:
    """(job, replaced trial) -> the replacement that replaces it; validates each link."""
    links: dict[TrialKey, str] = {}
    for rep_id, rep in reps.items():
        earlier = as_str(rep.lineage.get("supersedes"))
        if earlier is not None and (
            earlier not in reps or earlier == rep_id or reps[earlier].partition != rep.partition
        ):
            raise ValueError("bench_replacement_chain")
        source = reps[earlier].job if earlier else run.partitions[rep.partition]
        trial = _name(rep.lineage.get("replaced_trial"))
        key = (source.name, trial)
        if key in links:
            raise ValueError("bench_duplicate_replacement")
        if not _replaceable(source, trial, rep.lineage):
            raise ValueError("bench_replaced_trial_not_infra")
        links[key] = rep_id
    return links


def _load(root: Path, run: BenchRun) -> dict[str, _Replacement]:
    base = _path(root, *REPLACEMENTS)
    reps: dict[str, _Replacement] = {}
    if not base.is_dir():
        return reps
    for target in sorted(base.iterdir()):
        if target.is_symlink():
            raise ValueError("bench_path_outside_root")
        if target.is_dir() and (target / "receipt.json").is_file():
            rep = _receipt(root, run, target)
            if rep is not None:
                reps[rep.rep_id] = rep
    return reps


@dataclass
class _Chains:
    """Resolved chains: canonical chain ends, replaced trials, unstarted slots."""

    reps: dict[str, _Replacement]
    links: dict[TrialKey, str]
    canonical: dict[TrialKey, Doc] = field(default_factory=dict)
    replaced: dict[TrialKey, Doc] = field(default_factory=dict)
    pending: int = 0

    def successor(self, rep: _Replacement) -> str | None:
        return self.links.get((rep.job.name, rep.trial)) if rep.trial else None

    def chain(self, head: str) -> list[str]:
        chain = [head]
        while (nxt := self.successor(self.reps[chain[-1]])) is not None:
            if nxt in chain:
                raise ValueError("bench_replacement_cycle")
            chain.append(nxt)
        return chain

    def resolve(self, key: TrialKey, head: str) -> None:
        chain = self.chain(head)
        end = self.reps[chain[-1]]
        self.replaced[key] = {
            **_row(self.reps[head], "", key[1]),
            "role": "replaced",
            "replaced_by": end.trial,
        }
        for link in chain[:-1]:  # superseded links that did run: evidence too
            rep = self.reps[link]
            if rep.trial:
                self.replaced[(rep.job.name, rep.trial)] = {
                    **_row(rep, "superseded", key[1]),
                    "role": "replaced",
                    "replaced_by": end.trial,
                }
        if end.trial is None:
            self.pending += 1
        else:
            row = _row(end, "", key[1])
            self.canonical[(end.job.name, end.trial)] = {**row, "role": "canonical"}

    def state(self, rep: _Replacement) -> str:
        if self.successor(rep):
            return "superseded"
        if rep.trial is None:
            return "not_started"
        return "finished" if _trial_result(rep.job, rep.trial).get("finished_at") else "running"


def replacements(root: Path, run: BenchRun) -> Selection:
    """The run's canonical set: parent trials, each replaced one swapped for the end of
    its replacement chain (pending when that end hasn't run). Replaced originals and
    superseded links stay in the scan as evidence."""
    reps = _load(root, run)
    chains = _Chains(reps, _links(run, reps))
    parents = {job.name for job in run.jobs}
    for key, head in chains.links.items():
        if key[0] in parents:  # intermediate links are reached from their chain head
            chains.resolve(key, head)
    rows = tuple(
        _row(rep, chains.state(rep), _name(rep.lineage.get("replaced_trial")))
        for rep in reps.values()
    )
    jobs = tuple(run.jobs) + tuple(rep.job for rep in reps.values() if rep.trial)
    return Selection(
        "bench_run",
        run.name,
        jobs,
        chains.canonical,
        chains.replaced,
        None,
        rows,
        chains.pending,
        frozenset(job.name for job in run.jobs),
    )


def _manifest(path: Path) -> JsonObject:
    with path.open("rb") as stream:
        data = stream.read(MANIFEST_LIMIT + 1)
    if len(data) > MANIFEST_LIMIT:
        raise ValueError("release_manifest_too_large")
    manifest = load_object(data, MANIFEST_LIMIT)
    if not manifest or manifest.get("schema") != RELEASE_SCHEMA:
        raise ValueError("release_unsupported_manifest")
    return manifest


def _hub_ids(manifest: JsonObject) -> set[str]:
    """Harbor Hub's leaderboard rows: the trial ids each row counts."""
    return {
        str(i)
        for row in as_list(as_object(manifest.get("harbor_rows")).get("rows"))
        for i in as_list(as_object(row).get("trial_ids"))
    }


def _link_info(rep_id: str, link: JsonObject) -> Doc:
    return {
        "id": rep_id,
        "replaced_trial": _name(link.get("replaced_trial")),
        "replaced_error": _code(link.get("replaced_error")),
        "failure_phase": _code(link.get("failure_phase")),
    }


@dataclass
class _Release:
    root: Path
    canonical: dict[TrialKey, Doc] = field(default_factory=dict)
    replaced: dict[TrialKey, Doc] = field(default_factory=dict)
    rows: list[Doc] = field(default_factory=list)
    jobs: list[Path] = field(default_factory=list)
    ids: set[str] = field(default_factory=set)
    parents: set[str] = field(default_factory=set)

    def job(self, name: object) -> Path:
        job = _path(self.root, "jobs", _name(name))
        if job not in self.jobs:
            self.jobs.append(job)
        return job

    def trial(self, raw: JsonObject, lineage: Mapping[str | None, JsonObject]) -> None:
        """One reported trial; its result.json must carry the manifest's Harbor id."""
        job, trial, trial_id = (
            self.job(raw.get("job")),
            _name(raw.get("trial")),
            as_str(raw.get("id")),
        )
        if trial_id is None or as_str(_trial_result(job, trial).get("id")) != trial_id:
            raise ValueError("release_trial_id_mismatch")
        self.ids.add(trial_id)
        info: Doc = {"role": "canonical"}
        rep_id = as_str(raw.get("replacement"))
        if rep_id is not None:
            if rep_id not in lineage:
                raise ValueError("release_lineage_missing")
            info |= _link_info(rep_id, lineage[rep_id])
        self.canonical[(job.name, trial)] = info

    def link(self, cohort: JsonObject, run: BenchRun, rep_id: str, link: JsonObject) -> None:
        """One replacement: its replaced trial (in the parent or a superseded link's job)."""
        partition = as_str(link.get("partition"))
        if partition is None or partition not in run.partitions:
            raise ValueError("release_lineage_partition")
        earlier = as_str(link.get("supersedes"))
        source = run.partitions[partition]
        if earlier is not None:
            jobs = (as_object(t) for t in as_list(cohort.get("trials")))
            name = next((t.get("job") for t in jobs if t.get("replacement") == earlier), earlier)
            source = _path(self.root, "jobs", _name(name))
        if source not in self.jobs:
            self.jobs.append(source)
        info = _link_info(rep_id, link)
        self.replaced[(source.name, info["replaced_trial"])] = {**info, "role": "replaced"}
        replacement = next((k[1] for k, v in self.canonical.items() if v.get("id") == rep_id), None)
        lineage = (as_object(x) for x in as_list(cohort.get("lineage")))
        superseded = any(as_str(x.get("supersedes")) == rep_id for x in lineage)
        self.rows.append(
            {
                **info,
                "task": _task(info["replaced_trial"]),
                "supersedes": _code(earlier),
                "replacement_trial": replacement,
                "state": "superseded" if superseded else _code(link.get("state")),
            }
        )

    def cohort(self, cohort: JsonObject) -> None:
        run = load_run(self.root, _name(cohort.get("cohort")))
        self.parents |= {job.name for job in run.jobs}
        lineage = {
            as_str(as_object(r).get("id")): as_object(r) for r in as_list(cohort.get("lineage"))
        }
        for raw in as_list(cohort.get("trials")):
            self.trial(as_object(raw), lineage)
        for rep_id, link in lineage.items():
            if rep_id is None:
                raise ValueError("release_lineage_partition")
            self.link(cohort, run, rep_id, link)


def release(root: Path, manifest_path: Path) -> Selection:
    """A release manifest's frozen set: its trials (cross-checked against Harbor Hub's
    `harbor_rows.trial_ids` and each trial's own result.json id) are canonical; every
    replaced original and superseded link it records is scanned as evidence."""
    manifest = _manifest(manifest_path)
    built = _Release(root)
    for raw in as_list(manifest.get("cohorts")):
        built.cohort(as_object(raw))
    hub = _hub_ids(manifest)
    if hub and hub != built.ids:
        raise ValueError("release_hub_trial_ids_mismatch")
    return Selection(
        "release",
        _name(manifest.get("release_id")),
        tuple(built.jobs),
        {k: v for k, v in built.canonical.items() if "replaced_trial" in v},
        built.replaced,
        frozenset(built.canonical),
        tuple(built.rows),
        0,
        frozenset(built.parents),
    )
