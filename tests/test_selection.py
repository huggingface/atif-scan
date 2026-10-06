"""Replacement chains and release manifests on synthetic bench-run trees (no real data)."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from atif_scan.cli import main
from atif_scan.sources.bench import load_run
from atif_scan.sources.selection import release, replacements

if TYPE_CHECKING:
    from pathlib import Path

    from atif_scan.data.jsonval import Doc


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def trial(
    job: Path,
    name: str,
    *,
    error: str | None = None,
    reward: float | None = None,
    trajectory: bool = True,
    trial_id: str | None = None,
) -> None:
    folder = job / name
    (folder / "agent").mkdir(parents=True)
    result: Doc = {
        "id": trial_id or f"id-{name}",
        "trial_name": name,
        "task_name": f"terminal-bench/{name.partition('__')[0]}",
        "finished_at": "2026-10-03T01:00:00Z",
        "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
    }
    if error:
        result["exception_info"] = {"exception_type": error}
    (folder / "result.json").write_text(json.dumps(result))
    if trajectory:
        steps = [
            {"step_id": 1, "source": "user", "message": "Do the synthetic task."},
            {"step_id": 2, "source": "agent", "message": "Done."},
        ]
        (folder / "agent" / "trajectory.json").write_text(
            json.dumps({"schema_version": "ATIF-v1.7", "steps": steps})
        )


def cohort(root: Path) -> Path:
    """A run 'demo' with one partition 'p': three trials, one an infra failure."""
    run = root / "runs" / "demo"
    run.mkdir(parents=True)
    config = json.dumps({"job_name": "demo-p"}).encode()
    (run / "p.json").write_bytes(config)
    receipt = json.dumps(
        {"schema_version": 1, "identity": {"run_id": "demo"}, "configs": {"p.json": sha(config)}}
    ).encode()
    (run / "receipt.json").write_bytes(receipt)
    job = root / "jobs" / "demo-p"
    trial(job, "alpha__a1", reward=1.0)
    trial(job, "beta__b1", reward=0.0)
    trial(job, "gamma__g1", error="SandboxError", trajectory=False)
    return run


def replacement(
    root: Path,
    rep: str,
    replaced: str,
    *,
    supersedes: str | None = None,
    runs: bool = True,
    reward: float | None = 1.0,
    **override: object,
) -> None:
    run = root / "runs" / "demo"
    target = root / "runs" / "replacements" / rep
    target.mkdir(parents=True)
    config = json.dumps({"job_name": rep}).encode()
    (target / "p.json").write_bytes(config)
    lineage = {
        "parent_run": "demo",
        "partition": "p",
        "task": replaced.partition("__")[0],
        "replaced_trial": replaced,
        "exception_type": "SandboxError",
        "failure_phase": "agent-or-setup",
        "parent_receipt_sha256": sha((run / "receipt.json").read_bytes()),
        "parent_config_sha256": sha((run / "p.json").read_bytes()),
        "supersedes": supersedes,
    }
    receipt = {
        "schema_version": 1,
        "kind": "replacement",
        "run_id": rep,
        "lineage": lineage,
        "upload": False,
        "configs": {"p.json": sha(config)},
        "sources_lock_sha256": "0" * 64,
    } | override
    (target / "receipt.json").write_text(json.dumps(receipt))
    (root / "jobs" / rep).mkdir(parents=True)
    if runs:
        task = replaced.partition("__")[0]
        error = "SandboxError" if reward is None else None
        trial(root / "jobs" / rep, f"{task}__{rep[-3:]}", reward=reward, error=error)


def test_a_replacement_takes_its_slot_and_the_original_stays_as_evidence(tmp_path: Path):
    cohort(tmp_path)
    replacement(tmp_path, "demo-rep-001", "gamma__g1")
    sel = replacements(tmp_path, load_run(tmp_path, "demo"))
    assert sel.role(("demo-p", "gamma__g1")) == "replaced"
    assert sel.role(("demo-rep-001", "gamma__001")) == "canonical"
    assert sel.lineage(("demo-p", "gamma__g1"))["replaced_by"] == "gamma__001"
    assert sel.replacement_jobs == {"demo-rep-001"}
    assert [r["state"] for r in sel.summary] == ["finished"]


def test_a_chain_ends_at_its_last_link(tmp_path: Path):
    cohort(tmp_path)
    replacement(tmp_path, "demo-rep-001", "gamma__g1", reward=None)  # failed again
    replacement(tmp_path, "demo-rep-002", "gamma__001", supersedes="demo-rep-001")
    sel = replacements(tmp_path, load_run(tmp_path, "demo"))
    assert sel.role(("demo-rep-001", "gamma__001")) == "replaced"  # superseded link
    assert sel.role(("demo-rep-002", "gamma__002")) == "canonical"
    assert {r["id"]: r["state"] for r in sel.summary} == {
        "demo-rep-001": "superseded",
        "demo-rep-002": "finished",
    }


def test_an_unstarted_replacement_leaves_the_slot_pending(tmp_path: Path):
    cohort(tmp_path)
    replacement(tmp_path, "demo-rep-001", "gamma__g1", runs=False)
    sel = replacements(tmp_path, load_run(tmp_path, "demo"))
    assert sel.pending == 1
    assert sel.lineage(("demo-p", "gamma__g1"))["replaced_by"] is None


@pytest.mark.parametrize(
    ("replaced", "override", "error"),
    [
        ("beta__b1", {}, "not_infra"),  # a verified trial is never replaceable
        ("gamma__g1", {"upload": True}, "replacement_receipt"),
        ("gamma__g1", {"extra_key": 1}, "replacement_receipt"),
    ],
)
def test_receipts_that_dont_verify_reject_the_whole_selection(
    tmp_path: Path, replaced: str, override: Doc, error: str
):
    cohort(tmp_path)
    replacement(tmp_path, "demo-rep-001", replaced, **override)
    with pytest.raises(ValueError, match=error):
        replacements(tmp_path, load_run(tmp_path, "demo"))


def test_a_receipt_for_another_version_of_the_parent_is_rejected(tmp_path: Path):
    cohort(tmp_path)
    replacement(tmp_path, "demo-rep-001", "gamma__g1")
    path = tmp_path / "runs" / "replacements" / "demo-rep-001" / "receipt.json"
    data = json.loads(path.read_text())
    data["lineage"]["parent_receipt_sha256"] = "f" * 64  # prepared against another receipt
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="parent_drift"):
        replacements(tmp_path, load_run(tmp_path, "demo"))


def test_two_replacements_of_one_trial_are_rejected(tmp_path: Path):
    cohort(tmp_path)
    replacement(tmp_path, "demo-rep-001", "gamma__g1")
    replacement(tmp_path, "demo-rep-002", "gamma__g1")
    with pytest.raises(ValueError, match="duplicate_replacement"):
        replacements(tmp_path, load_run(tmp_path, "demo"))


def scan(args: list[str], capsys) -> Doc:
    assert main([*args, "--format", "json", "--no-cache"]) == 0
    return json.loads(capsys.readouterr().out)


def test_run_scores_the_canonical_set_and_keeps_evidence(tmp_path: Path, capsys):
    cohort(tmp_path)
    replacement(tmp_path, "demo-rep-001", "gamma__g1")
    doc = scan(["--run", "demo", "--bench-root", str(tmp_path)], capsys)
    labels = sorted(i["input_id"] for i in doc["inputs"])
    assert labels == ["part-1/alpha__a1/agent", "part-1/beta__b1/agent", "part-2/gamma__001/agent"]
    [original] = doc["not_counted"]
    assert original["input_id"] == "part-1/gamma__g1/agent"
    assert original["input_error"] == "trajectory_missing"
    assert original["error_type"] == "SandboxError"
    assert original["selection"]["replaced_by"] == "gamma__001"
    assert doc["selection"]["replacements"][0]["replaced_trial"] == "gamma__g1"
    as_run = scan(["--run", "demo", "--bench-root", str(tmp_path), "--as-run"], capsys)
    assert "selection" not in as_run and len(as_run["inputs"]) == 2  # no trajectory for g1


def test_brief_reports_replacements_and_the_as_run_score(tmp_path: Path, capsys):
    cohort(tmp_path)
    replacement(tmp_path, "demo-rep-001", "gamma__g1")
    main(
        [
            "--run",
            "demo",
            "--bench-root",
            str(tmp_path),
            "--brief",
            "--format",
            "text",
            "--no-cache",
        ]
    )
    text = " ".join(capsys.readouterr().out.split())
    assert (
        "REPLACED 1 trial re-run after an infrastructure failure (bench-run demo): SandboxError 1"
        in text
    )
    assert "1 replaced trial kept as evidence, not scored" in text


def manifest(root: Path, hub_ids: list[str] | None = None) -> Path:
    trials = [
        {"job": "demo-p", "trial": "alpha__a1", "id": "id-alpha__a1", "replacement": None},
        {"job": "demo-p", "trial": "beta__b1", "id": "id-beta__b1", "replacement": None},
        {
            "job": "demo-rep-001",
            "trial": "gamma__001",
            "id": "id-gamma__001",
            "replacement": "demo-rep-001",
        },
    ]
    lineage = [
        {
            "id": "demo-rep-001",
            "partition": "p",
            "replaced_trial": "gamma__g1",
            "replaced_error": "SandboxError",
            "failure_phase": "agent-or-setup",
            "supersedes": None,
            "state": "finalized",
        }
    ]
    ids = hub_ids if hub_ids is not None else [t["id"] for t in trials]
    doc = {
        "schema": "bench-run.release/v1",
        "release_id": "demo-r1",
        "cohorts": [{"cohort": "demo", "trials": trials, "lineage": lineage}],
        "harbor_rows": {"rows": [{"trial_ids": ids}]},
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(doc))
    return path


def test_a_release_scans_exactly_its_trial_ids(tmp_path: Path, capsys):
    cohort(tmp_path)
    replacement(tmp_path, "demo-rep-001", "gamma__g1")
    path = manifest(tmp_path)
    sel = release(tmp_path, path)
    assert sel.canonical == {
        ("demo-p", "alpha__a1"),
        ("demo-p", "beta__b1"),
        ("demo-rep-001", "gamma__001"),
    }
    assert sel.role(("demo-p", "gamma__g1")) == "replaced"
    doc = scan(["--release", str(path), "--bench-root", str(tmp_path)], capsys)
    assert len(doc["inputs"]) == 3 and len(doc["not_counted"]) == 1
    assert doc["selection"]["kind"] == "release"


def test_a_release_whose_hub_ids_disagree_is_rejected(tmp_path: Path):
    cohort(tmp_path)
    replacement(tmp_path, "demo-rep-001", "gamma__g1")
    with pytest.raises(ValueError, match="hub_trial_ids_mismatch"):
        release(tmp_path, manifest(tmp_path, ["id-alpha__a1"]))


def test_a_release_trial_whose_result_id_differs_is_rejected(tmp_path: Path):
    cohort(tmp_path)
    replacement(tmp_path, "demo-rep-001", "gamma__g1")
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    data["cohorts"][0]["trials"][0]["id"] = "someone-elses-trial"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="trial_id_mismatch"):
        release(tmp_path, path)


def test_the_viewer_exports_replaced_trials_as_evidence(tmp_path: Path):
    cohort(tmp_path)
    replacement(tmp_path, "demo-rep-001", "gamma__g1")
    out = tmp_path / "viewer"
    assert main(["--run", "demo", "--bench-root", str(tmp_path), "--viewer", str(out)]) == 0
    text = (out / "data.js").read_text()
    data = json.loads(text[len("window.ATIF_VIEWER = ") : -2])
    roles = {t["input_id"]: t["selection"].get("role") for t in data["trials"]}
    assert roles["part-1/gamma__g1/agent"] == "replaced"
    assert roles["part-2/gamma__001/agent"] == "canonical"
    assert data["run"]["selection"]["not_counted"] == 1


def test_a_superseded_links_job_is_a_replacement_job_not_a_part(tmp_path: Path):
    cohort(tmp_path)
    replacement(tmp_path, "demo-rep-001", "gamma__g1", reward=None)
    replacement(tmp_path, "demo-rep-002", "gamma__001", supersedes="demo-rep-001")
    sel = replacements(tmp_path, load_run(tmp_path, "demo"))
    assert sel.replacement_jobs == {"demo-rep-001", "demo-rep-002"}
    assert sel.parents == {"demo-p"}


def test_the_private_desk_starts_with_replaced_trials_as_evidence(tmp_path: Path, monkeypatch):
    import importlib

    browse = importlib.import_module("atif_scan.cli.browse")
    cohort(tmp_path)
    replacement(tmp_path, "demo-rep-001", "gamma__g1")
    served = []
    monkeypatch.setattr(browse, "_serve", lambda desk, port: served.append(desk.overview()))
    assert main(["--run", "demo", "--bench-root", str(tmp_path), "--browse", "--no-cache"]) == 0
    assert len(served[0]["trials"]) == 4  # 3 canonical + the replaced original
