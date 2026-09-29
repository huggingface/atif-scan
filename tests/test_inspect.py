"""--inspect: layout classification from names and sizes only (synthetic trees)."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from atif_scan.cli import main
from atif_scan.layout import document
from atif_scan.sources import list_input

# Deliberately not valid JSON: inspection must never read trace contents.
NOT_READ = "not json: inspection must not open this"


def write(root, files):
    for path in files:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(NOT_READ)


HARBOR_JOB = [
    "job.log",
    "config.json",
    "lock.json",
    "result.json",
    # A normal trial with a simulated user.
    "task-a__1/trial.log",
    "task-a__1/config.json",
    "task-a__1/result.json",
    "task-a__1/agent/trajectory.json",
    "task-a__1/agent/trajectory.summarization-1-summary.json",
    "task-a__1/user-agent/trajectory.json",
    "task-a__1/verifier/reward.txt",
    # A failed trial: no trajectory, an exception.
    "task-b__2/trial.log",
    "task-b__2/config.json",
    "task-b__2/result.json",
    "task-b__2/exception.txt",
    # A multi-step trial.
    "task-c__3/trial.log",
    "task-c__3/config.json",
    "task-c__3/result.json",
    "task-c__3/steps/build/agent/trajectory.json",
    "task-c__3/steps/build/verifier/reward.json",
]


def inspect(root, pattern="trajectory.json"):
    return document([list_input(str(root))], pattern)["inputs"][0]


def codes(item):
    return {a["code"]: a for a in item["anomalies"]}


def test_harbor_job_is_recognized_with_roles_and_anomalies(tmp_path):
    write(tmp_path, HARBOR_JOB)
    item = inspect(tmp_path)
    assert item["layout"] == "harbor_jobs"
    assert item["harbor"] == {
        "jobs": 1,
        "trials": 3,
        "trials_with_reward": 1,
        "multi_step_trials": 1,
    }
    assert item["would_scan"] == {
        "total": 3,
        "by_role": {"agent": 1, "simulated_user": 1, "step_agent": 1},
    }
    found = codes(item)
    assert found["simulated_user_trajectories_would_be_scanned"]["examples"] == [
        "task-a__1/user-agent"
    ]
    assert found["trials_without_agent_trajectory"]["examples"] == ["task-b__2"]
    assert found["trials_with_multiple_scanned_trajectories"]["examples"] == ["task-a__1"]
    assert found["trials_with_exception"]["examples"] == ["task-b__2"]
    assert found["unscanned_trajectory_like_files"]["examples"] == [
        "task-a__1/agent/trajectory.summarization-1-summary.json"
    ]


def test_single_harbor_trial_root(tmp_path):
    write(tmp_path, [p.split("/", 1)[1] for p in HARBOR_JOB if p.startswith("task-a__1/")])
    item = inspect(tmp_path)
    assert item["layout"] == "harbor_trials" and item["harbor"]["trials"] == 1
    assert item["would_scan"]["by_role"] == {"agent": 1, "simulated_user": 1}


def test_trajectory_folders_with_mixed_depth_and_companions(tmp_path):
    write(
        tmp_path,
        [
            "run/t1/trajectory.json",
            "run/t1/summary.json",
            "run/t2/trajectory.json",
            "run/replacements/r1/trials/t3/trajectory.json",
            "README.md",
        ],
    )
    item = inspect(tmp_path)
    assert item["layout"] == "trajectory_folders" and item["harbor"] is None
    assert item["folders"] == 3 and item["alongside"] == {"summary.json": 1}
    assert item["other_files"] == {"summary.json": 1, "README.md": 1}
    assert codes(item)["mixed_depths"]["examples"] == ["run/replacements/r1/trials/t3"]


def test_no_matches_is_reported_not_raised(tmp_path, capsys):
    write(tmp_path, ["a/summary.json"])
    item = inspect(tmp_path)
    assert item["layout"] == "no_trajectories" and "no_files_match_pattern" in codes(item)
    assert main(["--inspect", str(tmp_path), "--format", "json"]) == 0
    assert inspect(tmp_path, pattern="*.json")["would_scan"]["total"] == 1


def test_single_file_and_oversize(tmp_path, monkeypatch):
    write(tmp_path, ["x.json"])
    item = inspect(tmp_path / "x.json")
    assert item["layout"] == "single_file" and item["would_scan"]["total"] == 1
    monkeypatch.setattr("atif_scan.layout.MAX_BYTES", 3)
    assert codes(inspect(tmp_path / "x.json"))["oversize_files"]["count"] == 1


def test_cli_inspect_text_and_json_never_echo_root(tmp_path, capsys):
    write(tmp_path, HARBOR_JOB)
    assert main(["--inspect", str(tmp_path), "--format", "text"]) == 0
    text = capsys.readouterr().out
    assert "layout: harbor_jobs" in text and "simulated_user" in text
    assert main(["--inspect", str(tmp_path), "--format", "json"]) == 0
    raw = capsys.readouterr().out
    assert json.loads(raw)["kind"] == "inspection"
    assert str(tmp_path) not in text + raw and NOT_READ not in text + raw


@pytest.mark.parametrize("argv", [["--inspect"], ["--inspect", "--manifest", "m.json"]])
def test_cli_inspect_needs_paths(argv, capsys):
    assert main(argv) == 2


def test_cli_inspect_missing_path(tmp_path, capsys):
    assert main(["--inspect", str(tmp_path / "nope")]) == 2
    assert "path_not_found" in capsys.readouterr().err


def test_inspect_hub_listing_uses_sizes_without_opening(monkeypatch):
    from atif_scan import sources

    class ListingOnlyFS:
        files: ClassVar[dict[str, int]] = {
            "buckets/o/b/run/t1/trajectory.json": 10,
            "buckets/o/b/run/t1/summary.json": 2,
        }

        def info(self, path):
            return {"type": "directory"}

        def find(self, path, detail=False):
            return {p: {"type": "file", "size": n} for p, n in self.files.items()}

        def open(self, *args):  # pragma: no cover - must not be called
            raise AssertionError("inspection opened a file")

    monkeypatch.setattr(sources, "hf_filesystem", ListingOnlyFS)
    item = document([list_input("hf://buckets/o/b/run")], "trajectory.json")["inputs"][0]
    assert item["remote"] and item["bytes"] == 12
    assert item["would_scan"]["total"] == 1 and item["alongside"] == {"summary.json": 1}
