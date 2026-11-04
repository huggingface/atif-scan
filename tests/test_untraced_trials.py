"""Harbor job trial folders without a trajectory are trials of the run (synthetic only).

Regression: on a DeepSeek TB4 job (85 planned), three trials died of provider connection
errors before the agent ran. They had result.json and exception.txt but no trajectory,
so they never became inputs: the brief said "✓ no trial errored", counted them as
missing and left them out of the score (contradicting "errored without a reward = 0").
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from test_harbor_files import result_json, trajectory
from test_selection import cohort, replacement

from atif_scan.cli import main
from atif_scan.sources.inputs import SourceError, resolve

if TYPE_CHECKING:
    from pathlib import Path

    from atif_scan.data.jsonval import Doc


def job_folder(root: Path, planned: int = 6, errored: int = 1) -> Path:
    job = root / "job"
    job.mkdir()
    (job / "config.json").write_text(
        json.dumps({"job_name": "demo", "n_attempts": 1, "datasets": [{"name": "tb/x"}]})
    )
    stats = {"n_completed_trials": planned, "n_errored_trials": errored}
    (job / "result.json").write_text(json.dumps({"n_total_trials": planned, "stats": stats}))
    rows = [("alpha__a1", 1.0), ("alpha__a2", 0.0), ("beta__b1", 1.0), ("beta__b2", 0.0)]
    for name, reward in rows:
        trial = job / name
        (trial / "agent").mkdir(parents=True)
        (trial / "agent" / "trajectory.json").write_text(json.dumps(trajectory("ls")))
        (trial / "result.json").write_text(json.dumps(result_json(name[:-4], reward)))
    dead = job / "gamma__g1"  # died before the agent ran: no trajectory
    (dead / "agent").mkdir(parents=True)
    (dead / "result.json").write_text(json.dumps(result_json("gamma", error="RemoteProtocolError")))
    (dead / "exception.txt").write_text("synthetic traceback: never read")
    return job


def scan(args: list[str], capsys) -> tuple[int, Doc]:
    code = main([*args, "--format", "json"])
    return code, json.loads(capsys.readouterr().out)


def test_errored_trial_without_trajectory_is_present_errored_and_scored(tmp_path, capsys):
    job = job_folder(tmp_path)
    code, doc = scan([str(job), "--no-cache"], capsys)
    assert code == 0  # a recorded run fact, not unreadable input
    items = {i["input_id"]: i for i in doc["inputs"]}
    dead = items["gamma__g1/agent"]
    assert dead["input_status"] == "unavailable_or_invalid"
    assert dead["input_error"] == "no_trajectory_errored"
    assert (dead["task"], dead["error_type"], dead.get("reward")) == (
        "gamma",
        "RemoteProtocolError",
        None,
    )
    assert "never read" not in json.dumps(doc) and "SECRET" not in json.dumps(doc)
    _, brief = scan([str(job), "--brief", "--no-cache"], capsys)
    t = brief["overview"]["trials"]
    assert (t["present"], t["missing"], t["errored"], t["without_trajectory"]) == (5, 1, 1, 1)
    assert brief["overview"]["accuracy"][0] == 40.0  # 2 of 5: errored without a reward = 0
    main([str(job), "--brief", "--format", "text", "--no-cache"])
    out = capsys.readouterr().out
    assert "no trial errored" not in out
    assert "⚠ 1 trial errored (20.0%): RemoteProtocolError 1" in out
    assert "⚠ 5 of 6 planned trials are present" in out


def test_only_harbor_trial_folders_of_a_job_count(tmp_path, capsys):
    job = job_folder(tmp_path)
    (job / "notes").mkdir()  # not `<task>__<id>`
    (job / "notes" / "result.json").write_text(json.dumps(result_json("x", error="E")))
    (job / "delta__d1").mkdir()  # no result.json yet: not read as a trial here
    (job / "delta__d1" / "trial.log").write_text("synthetic")
    _, doc = scan([str(job), "--no-cache"], capsys)
    assert sorted(i["input_id"] for i in doc["inputs"] if i["input_status"] != "available") == [
        "gamma__g1/agent"
    ]
    # With neither a job config nor sibling trial configs, nothing is added.
    (job / "config.json").unlink()
    _, doc = scan([str(job), "--no-cache"], capsys)
    assert all(i["input_status"] == "available" for i in doc["inputs"])


def test_untraced_facts_are_read_afresh_on_every_scan(tmp_path, capsys):
    job = job_folder(tmp_path)
    cache = ["--cache", str(tmp_path / "cache")]
    scan([str(job), *cache], capsys)
    (job / "gamma__g1" / "result.json").write_text(json.dumps(result_json("gamma", reward=1.0)))
    (job / "gamma__g1" / "exception.txt").unlink()
    _, doc = scan([str(job), *cache], capsys)
    dead = next(i for i in doc["inputs"] if i["input_id"] == "gamma__g1/agent")
    assert dead["input_error"] == "no_trajectory" and dead["reward"] == 1.0
    assert "error_type" not in dead or dead["error_type"] is None


def test_run_selection_stamps_untraced_trials_without_duplicating_them(tmp_path, capsys):
    """--run: a replaced trial that left no trajectory is both a trial folder of its job
    and a replaced original; it is one input, labelled once, and not counted."""
    cohort(tmp_path)
    job = tmp_path / "jobs" / "demo-p"
    (job / "config.json").write_text(json.dumps({"job_name": "demo-p", "n_attempts": 1}))
    (job / "result.json").write_text(json.dumps({"n_total_trials": 3, "stats": {}}))
    replacement(tmp_path, "demo-rep-001", "gamma__g1")
    code, doc = scan(["--run", "demo", "--bench-root", str(tmp_path), "--no-cache"], capsys)
    assert code == 0
    labels = sorted(i["input_id"] for i in doc["inputs"])
    assert labels == ["part-1/alpha__a1/agent", "part-1/beta__b1/agent", "part-2/gamma__001/agent"]
    [original] = doc["not_counted"]
    assert original["input_id"] == "part-1/gamma__g1/agent"
    assert original["input_error"] == "no_trajectory_errored"
    assert original["selection"]["role"] == "replaced"
    # As run: the errored original is a trial of the job, scored 0.
    _, as_run = scan(
        ["--run", "demo", "--bench-root", str(tmp_path), "--as-run", "--no-cache"], capsys
    )
    assert len(as_run["inputs"]) == 3


def test_highlights_export_accepts_a_job_with_an_untraced_trial(tmp_path, capsys):
    """Regression (0.8.6-0.8.7): --highlights/--viewer refused any job folder holding a
    trial without a trajectory ("needs local traces"), since that trial has no local file."""
    job = job_folder(tmp_path)
    out = tmp_path / "highlights"
    assert main([str(job), "--highlights", str(out), "--no-cache"]) in (0, 1)
    err = capsys.readouterr().err
    assert "needs local traces" not in err
    assert (out / "data.js").exists()
    viewer = tmp_path / "viewer"
    assert main([str(job), "--viewer", str(viewer), "--no-cache"]) in (0, 1)
    assert "needs local traces" not in capsys.readouterr().err


def standalone_trial(root: Path, task: str, **facts) -> Path:
    """Reorganized trial receipts; the job config lives separately."""
    trial = root / f"{task}__x"
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(
        json.dumps({"task": {"path": f"tasks/{task}"}, "agent": {"name": "synthetic"}})
    )
    (trial / "result.json").write_text(json.dumps(result_json(task, **facts)))
    return trial


@pytest.mark.parametrize("trajectory_path", ["agent/trajectory.json", "trajectory.json"])
def test_standalone_mixed_trials_retain_missing_trace_and_result_facts(
    tmp_path, capsys, trajectory_path
):
    traced = standalone_trial(tmp_path, "alpha", reward=1.0)
    path = traced / trajectory_path
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(trajectory("ls")))
    standalone_trial(tmp_path, "beta", error="RemoteProtocolError", cost=0.25)
    # A separately stored job receipt must not supply configuration to these trials.
    job = tmp_path / "jobs" / "separate"
    job.mkdir(parents=True)
    (job / "config.json").write_text(json.dumps({"job_name": "separate", "n_attempts": 7}))
    (job / "result.json").write_text(json.dumps({"n_total_trials": 99, "stats": {}}))
    runs = []
    assert len(resolve([str(tmp_path)], runs=runs)) == 2
    assert runs == []
    code, doc = scan([str(tmp_path), "--no-cache"], capsys)
    assert code == 0
    items = {i["input_id"]: i for i in doc["inputs"]}
    assert len(items) == 2
    dead = items["beta__x/agent"]
    assert dead["input_status"] == "unavailable_or_invalid"
    assert dead["input_error"] == "no_trajectory_errored"
    assert (dead["task"], dead["error_type"], dead["cost_usd"]) == (
        "beta",
        "RemoteProtocolError",
        0.25,
    )
    assert dead.get("reward") is None
    assert dead["input_tokens"] == 1000
    assert "SECRET" not in json.dumps(doc)
    _, brief = scan([str(tmp_path), "--brief", "--no-cache"], capsys)
    trials = brief["overview"]["trials"]
    assert (trials["present"], trials["errored"], trials["without_trajectory"]) == (2, 1, 1)


def test_standalone_all_untraced_trials_preserve_rewards_errors_and_unknowns(tmp_path, capsys):
    standalone_trial(tmp_path, "alpha", reward=1.0)
    standalone_trial(tmp_path, "beta", error="AgentTimeoutError")
    unknown = standalone_trial(tmp_path, "gamma", tokens=(None, None, None))
    receipt = result_json("gamma")
    receipt.update(agent_result=None, verifier_result=None, exception_info=None)
    (unknown / "result.json").write_text(json.dumps(receipt))
    runs = []
    assert len(resolve([str(tmp_path)], runs=runs)) == 3
    assert runs == []
    code, doc = scan([str(tmp_path), "--no-cache"], capsys)
    assert code == 0
    items = {i["task"]: i for i in doc["inputs"]}
    assert len(items) == 3
    assert all(i["input_status"] == "unavailable_or_invalid" for i in items.values())
    assert items["alpha"]["input_error"] == "no_trajectory"
    assert items["alpha"]["reward"] == 1.0
    assert items["beta"]["input_error"] == "no_trajectory_errored"
    assert items["beta"]["error_type"] == "AgentTimeoutError"
    assert items["gamma"]["input_error"] == "no_trajectory"
    assert all(items["gamma"].get(k) is None for k in ("reward", "error_type", "cost_usd"))
    # An explicit alternative pattern must not silently select receipt-only trials.
    with pytest.raises(SourceError, match="no_files_match_pattern"):
        resolve([str(tmp_path)], pattern="other.json")


@pytest.mark.parametrize(
    "receipt",
    [
        {},
        {"n_total_trials": 3, "stats": {}},
        {"task_name": "gamma"},
        {"trial_name": "gamma__x", "task_name": "gamma"},
        {**result_json("gamma"), "trial_name": "different__x"},
        {**result_json("gamma"), "task_name": []},
        {**result_json("gamma"), "task_name": ""},
        {**result_json("gamma"), "agent_result": []},
        [],
        "not a receipt",
    ],
)
def test_standalone_unrelated_results_are_not_trials(tmp_path, receipt):
    trial = standalone_trial(tmp_path, "gamma")
    (trial / "result.json").write_text(json.dumps(receipt))
    with pytest.raises(SourceError, match="no_files_match_pattern"):
        resolve([str(tmp_path)])


def test_standalone_receipts_do_not_duplicate_job_trials(tmp_path):
    job = job_folder(tmp_path)
    standalone_trial(job, "delta", error="AgentTimeoutError")
    sources = resolve([str(job)])
    assert len(sources) == 6
    assert sum(s.label == "delta__x/agent" for s in sources) == 1


@pytest.mark.parametrize("oversized", [False, True])
def test_standalone_unreadable_result_is_not_recognized(tmp_path, oversized):
    trial = standalone_trial(tmp_path, "gamma")
    data = json.dumps(result_json("gamma")) + " " * (1024 * 1024) if oversized else "{"
    (trial / "result.json").write_text(data)
    with pytest.raises(SourceError, match="no_files_match_pattern"):
        resolve([str(tmp_path)])
