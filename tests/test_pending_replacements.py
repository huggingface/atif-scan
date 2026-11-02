"""Known replacement trials may exist before Harbor writes any trajectory or result."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from test_selection import cohort, replacement, scan, trial

from atif_scan.cli import main
from atif_scan.output.overview import overview

if TYPE_CHECKING:
    from pathlib import Path


def running_replacement(root: Path) -> Path:
    cohort(root)
    replacement(root, "demo-rep-001", "gamma__g1")
    folder = root / "jobs" / "demo-rep-001" / "gamma__001"
    (folder / "result.json").unlink()
    (folder / "agent" / "trajectory.json").unlink()
    (folder / "agent").rmdir()
    (folder / "trial.log").write_text("synthetic private log: never reported")
    return folder


def test_running_replacement_is_unavailable_unknown_and_unscored(tmp_path: Path, capsys):
    running_replacement(tmp_path)
    doc = scan(["--run", "demo", "--bench-root", str(tmp_path)], capsys)
    [pending] = [i for i in doc["inputs"] if i["input_id"] == "part-2/gamma__001/agent"]
    assert len(doc["inputs"]) == 3 and len(doc["not_counted"]) == 1
    assert pending["input_status"] == "unavailable_or_invalid"
    assert pending["input_error"] == "no_trajectory"
    assert pending.get("reward") is None and pending.get("error_type") is None
    assert pending["selection"]["role"] == "canonical"
    assert pending["selection"]["replaced_error"] == "SandboxError"
    assert {a["status"] for a in pending["assessments"]} <= {"unknown", "not_applicable"}
    assert any(a["status"] == "unknown" for a in pending["assessments"])
    assert doc["selection"]["replacements"][0]["state"] == "running"
    ov = overview(doc)
    assert ov["trials"]["without_trajectory"] == 1
    assert ov["trials"]["reward_unknown"] == 1
    assert ov["accuracy"][0] == 50.0  # unfinished replacement is not a scored failure
    serialized = json.dumps(doc)
    assert "synthetic private log" not in serialized and str(tmp_path) not in serialized


@pytest.mark.parametrize("reward", [None, 1.0])
def test_completed_replacement_without_trajectory_keeps_recorded_outcome(
    tmp_path: Path, capsys, reward: float | None
):
    running_replacement(tmp_path)
    job = tmp_path / "jobs" / "demo-rep-001"
    trial(
        job,
        "gamma__001",
        reward=reward,
        error="SandboxError" if reward is None else None,
        trajectory=False,
    )
    doc = scan(["--run", "demo", "--bench-root", str(tmp_path)], capsys)
    [missing] = [i for i in doc["inputs"] if i["input_id"] == "part-2/gamma__001/agent"]
    expected = "no_trajectory_errored" if reward is None else "no_trajectory"
    assert missing["input_error"] == expected and missing.get("reward") == reward
    assert missing.get("error_type") == ("SandboxError" if reward is None else None)
    assert doc["selection"]["replacements"][0]["state"] == "finished"
    assert overview(doc)["trials"]["reward_unknown"] == 0


def test_running_replacement_becoming_available_is_not_cached_as_missing(tmp_path: Path, capsys):
    running_replacement(tmp_path)
    args = [
        "--run",
        "demo",
        "--bench-root",
        str(tmp_path),
        "--cache",
        str(tmp_path / "cache"),
        "--format",
        "json",
    ]
    assert main(args) == 0
    before = json.loads(capsys.readouterr().out)
    assert before["inputs"][-1]["input_error"] == "no_trajectory"
    trial(tmp_path / "jobs" / "demo-rep-001", "gamma__001", reward=1.0)
    assert main(args) == 0
    after = json.loads(capsys.readouterr().out)
    [finished] = [i for i in after["inputs"] if i["input_id"] == "part-2/gamma__001/agent"]
    assert len(after["inputs"]) == 3
    assert finished["input_status"] == "available" and finished["reward"] == 1.0


def test_unstarted_replacement_still_has_a_pending_slot_without_an_invented_trial(
    tmp_path: Path, capsys
):
    cohort(tmp_path)
    replacement(tmp_path, "demo-rep-001", "gamma__g1", runs=False)
    doc = scan(["--run", "demo", "--bench-root", str(tmp_path)], capsys)
    assert doc["selection"]["pending"] == 1
    assert doc["selection"]["replacements"][0]["state"] == "not_started"
    assert len(doc["inputs"]) == 2 and len(doc["not_counted"]) == 1


def test_unmatched_custom_pattern_still_fails_for_a_known_replacement(tmp_path: Path, capsys):
    running_replacement(tmp_path)
    original = tmp_path / "jobs" / "demo-p" / "alpha__a1" / "agent"
    (original / "custom.json").write_bytes((original / "trajectory.json").read_bytes())
    code = main(["--run", "demo", "--bench-root", str(tmp_path), "--pattern", "custom.json"])
    assert code == 2
    assert "no_files_match_pattern" in capsys.readouterr().err


def test_unrecognized_empty_original_part_still_fails(tmp_path: Path, capsys):
    cohort(tmp_path)
    job = tmp_path / "jobs" / "demo-p"
    for path in job.rglob("trajectory.json"):
        path.unlink()
    assert main(["--run", "demo", "--bench-root", str(tmp_path)]) == 2
    assert "no_files_match_pattern" in capsys.readouterr().err


@pytest.mark.parametrize("errored", [False, True])
def test_missing_replacement_retains_reward_and_exception_sidecars(tmp_path: Path, capsys, errored):
    folder = running_replacement(tmp_path)
    (folder / "verifier").mkdir()
    (folder / "verifier" / "reward.json").write_text('{"reward": 1.0}')
    if errored:
        (folder / "exception.txt").write_text("synthetic private traceback: never reported")
    doc = scan(["--run", "demo", "--bench-root", str(tmp_path)], capsys)
    [missing] = [i for i in doc["inputs"] if i["input_id"] == "part-2/gamma__001/agent"]
    assert missing["reward"] == 1.0
    assert missing.get("error_type") == ("exception" if errored else None)
    assert missing["input_error"] == ("no_trajectory_errored" if errored else "no_trajectory")
    assert "private traceback" not in json.dumps(doc)


def test_missing_replacement_does_not_read_symlinked_result(tmp_path: Path, capsys):
    root = tmp_path / "bench"
    folder = running_replacement(root)
    outside = tmp_path / "outside"
    trial(outside, "unrelated__001", reward=1.0, trajectory=False)
    (folder / "result.json").symlink_to(outside / "unrelated__001" / "result.json")
    doc = scan(["--run", "demo", "--bench-root", str(root)], capsys)
    [missing] = [i for i in doc["inputs"] if i["input_id"] == "part-2/gamma__001/agent"]
    assert missing.get("reward") is None and missing.get("task") is None
    assert "unrelated" not in json.dumps(doc)
    assert overview(doc)["accuracy"][0] == 50.0
