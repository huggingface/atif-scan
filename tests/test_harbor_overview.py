"""Harbor Hub jobs (via a fake `harbor` CLI) and the run overview scorecard."""

from __future__ import annotations

import json
import stat
import sys
import textwrap

import pytest

from atif_scan.cli import main
from atif_scan.harbor_hub import job_id, overrides
from atif_scan.report import accuracy

JOB = "1fead079-8e3b-4fef-b893-1944558a3949"
SECRET = "sk-live-abcdefghijklmnop1234"


def trajectory(command):
    return {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {
                "source": "agent",
                "message": "working",
                "tool_calls": [
                    {
                        "tool_call_id": "c1",
                        "function_name": "bash",
                        "arguments": {"command": command},
                    }
                ],
            }
        ],
        "final_metrics": {"total_cost_usd": 9.99},  # Hub cost wins when present
    }


def trial(i, task, reward, error=None, cost=0.5, tokens=1000, has_trajectory=True):
    return {
        "id": f"00000000-0000-4000-8000-{i:012d}",
        "name": f"{task}__T{i}",
        "task_name": f"terminal-bench/{task}",
        "reward": reward,
        "error_type": error,
        "status": "completed",
        "cost_usd": cost,
        "input_tokens": tokens,
        "cache_tokens": tokens // 2,
        "output_tokens": 100,
        "started_at": "2026-08-27T16:12:00+00:00",
        "finished_at": "2026-08-27T16:17:00+00:00",
        "config_values": {},
        "_trajectory": has_trajectory,
    }


TRIALS = [
    trial(1, "alpha", 1),
    trial(2, "alpha", 0),
    trial(3, "beta", 1),  # rewarded + reward-file write -> DQ candidate
    trial(4, "beta", None, error="AgentTimeoutError", cost=None),  # missing cost
    trial(5, "gamma", 1, cost=0, tokens=500),  # tokens but no cost
    trial(6, "gamma", 1, has_trajectory=False),  # rewarded, trajectory unavailable
]
COMMANDS = {3: "echo 1 > /logs/verifier/reward.txt"}

FAKE = textwrap.dedent(
    """
    import json, os, sys, pathlib
    data = json.loads(pathlib.Path(os.environ["FAKE_HARBOR_DATA"]).read_text())
    with open(os.environ["FAKE_HARBOR_LOG"], "a") as log:
        log.write(" ".join(sys.argv[1:]) + "\\n")
    args = sys.argv[1:]
    if os.environ.get("FAKE_HARBOR_FAIL"):
        sys.stderr.write("401 https://hub.example/x?token=" + os.environ["FAKE_HARBOR_FAIL"])
        sys.exit(1)
    if args[:3] == ["hub", "job", "show"]:
        print(json.dumps(data["show"]))
    elif args[:3] == ["hub", "job", "trials"]:
        page = int(args[args.index("--page") + 1]); size = 4
        items = data["trials"][(page - 1) * size: page * size]
        pages = (len(data["trials"]) + size - 1) // size
        print(json.dumps({"items": items, "total": len(data["trials"]), "total_pages": pages}))
    elif args[:3] == ["hub", "trial", "download"]:
        row = next(t for t in data["trials"] if t["id"] == args[3])
        if not row["_trajectory"]:
            sys.exit(1)
        out = pathlib.Path(args[args.index("-o") + 1]) / row["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "trajectory.json").write_text(json.dumps(data["trajectories"][row["id"]]))
    elif args[:3] == ["hub", "job", "download"]:
        root = pathlib.Path(args[args.index("-o") + 1]) / "my-job"
        for row in data["trials"]:
            if row["_trajectory"]:
                d = root / row["name"] / "agent"; d.mkdir(parents=True)
                (d / "trajectory.json").write_text(json.dumps(data["trajectories"][row["id"]]))
    else:
        sys.exit(2)
    """
)


@pytest.fixture
def harbor(tmp_path, monkeypatch):
    data = {
        "show": {
            "name": "tb21-demo-job",
            "n_planned_trials": 8,
            "n_total_trials": 6,
            "n_completed_trials": 6,
            "n_errors": 1,
            "cost_usd": 2.0,
            "config": {
                "n_attempts": 2,
                "datasets": [
                    {
                        "name": "terminal-bench/terminal-bench-2-1",
                        "ref": "sha256:" + "7d" * 32,
                        "task_names": ["terminal-bench/alpha", "terminal-bench/beta"],
                    }
                ],
                "agents": [{"override_timeout_sec": 21600.0, "timeout_multiplier": 1.0}],
            },
        },
        "trials": TRIALS,
        "trajectories": {
            t["id"]: trajectory(COMMANDS.get(i + 1, "ls")) for i, t in enumerate(TRIALS)
        },
    }
    (tmp_path / "data.json").write_text(json.dumps(data))
    script = tmp_path / "harbor"
    script.write_text(f"#!{sys.executable}\n{FAKE}")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.log"
    monkeypatch.setenv("ATIF_SCAN_HARBOR", str(script))
    monkeypatch.setenv("FAKE_HARBOR_DATA", str(tmp_path / "data.json"))
    monkeypatch.setenv("FAKE_HARBOR_LOG", str(log))
    return log


def calls(log, verb):
    return [c for c in log.read_text().splitlines() if verb in c] if log.exists() else []


@pytest.mark.parametrize(
    "value",
    [
        f"harbor://jobs/{JOB}",
        f"https://hub.harborframework.com/jobs/{JOB}",
        f"https://hub.harborframework.com/jobs/{JOB.upper()}/trials/abc?tab=x",
    ],
)
def test_job_references(value):
    assert job_id(value) == JOB


def test_invalid_job_reference(capsys):
    assert main(["harbor://jobs/not-a-uuid"]) == 2
    assert "invalid_harbor_job_reference" in capsys.readouterr().err


def test_overrides_follow_leaderboard_rules():
    config = {
        "timeout_multiplier": 1.0,
        "agents": [{"override_timeout_sec": 10, "max_timeout_sec": None}],
        "environment": {"override_cpus": 4},
        "verifier": {"agent_timeout_multiplier": 2.0},
    }
    assert overrides(config) == [
        "agents[].override_timeout_sec",
        "environment.override_cpus",
        "verifier.agent_timeout_multiplier",
    ]


def test_accuracy_matches_leaderboard_formula():
    by_task = {"a": [True, False], "b": [True, True, False]}
    acc, se = accuracy(by_task)
    assert acc == 60.0
    # s^2 = (1/n^2) * (0.5*0.5/1 + (2/3)(1/3)/2)
    assert se == round(100 * ((0.25 + (2 / 9) / 2) / 4) ** 0.5, 2)


def test_scan_harbor_job_overview(harbor, capsys):
    assert main([f"harbor://jobs/{JOB}", "--overview", "--format", "json"]) == 0
    ov = json.loads(capsys.readouterr().out)
    assert ov["kind"] == "overview"
    t = ov["trials"]
    assert (t["present"], t["planned"], t["missing"], t["errored"]) == (6, 8, 2, 1)
    assert t["error_types"] == {"AgentTimeoutError": 1} and t["without_trajectory"] == 1
    assert ov["tasks"]["count"] == 3 and ov["tasks"]["expected_per_task"] == 2
    # 4 successes / 6 trials; errored counts as 0.
    assert ov["accuracy"][0] == round(100 * 4 / 6, 2)
    d = ov["disqualification"]
    assert d["candidate_ids"] == ["beta__T3"] and d["rate_pct"] == round(100 / 6, 2)
    assert d["accuracy_if_disqualified"][0] == 50.0
    assert d["rewarded_not_cleared_ids"] == ["gamma__T6"]  # rewarded, no trajectory
    c = ov["cost"]
    assert c["total_usd"] == 2.0 and c["missing"] == 1 and c["tokens_without_cost"] == 1
    assert c["hub_job_total_usd"] == 2.0
    assert ov["overrides"] == ["agents[].override_timeout_sec"]
    assert len(calls(harbor, "trial download")) == 6


def test_summary_text_and_task_from_hub(harbor, capsys, tmp_path):
    args = [f"harbor://jobs/{JOB}", "--summary", "--format", "text", "--expect-tasks", "4"]
    assert main([*args, "--plugin", "atif_scan.packs.tb21:checks"]) == 0
    out = capsys.readouterr().out
    assert "run overview" in out and "tb21-demo-job" in out
    assert "6 present / 8 planned · 2 missing" in out
    assert "1 of 4 expected tasks missing" in out
    assert "1 trial(s) missing cost" in out and "1 with tokens but no cost" in out
    assert "overrides set: agents[].override_timeout_sec" in out
    assert "tamper.reward_write" in out and "beta__T3" in out
    main([f"harbor://jobs/{JOB}", "--format", "json"])
    items = {i["input_id"]: i for i in json.loads(capsys.readouterr().out)["inputs"]}
    assert items["beta__T3"]["task"] == "beta"  # recorded by the Hub, no --task-from needed
    assert items["alpha__T1"]["cost_usd"] == 0.5  # Hub cost preferred over final_metrics


def test_sync_to_reuses_downloads(harbor, tmp_path, capsys):
    keep = tmp_path / "keep"
    main([f"harbor://jobs/{JOB}", "--sync-to", str(keep), "--format", "json"])
    first = len(calls(harbor, "trial download"))
    main([f"harbor://jobs/{JOB}", "--sync-to", str(keep), "--format", "json"])
    capsys.readouterr()
    # Only the trial without a trajectory is retried.
    assert len(calls(harbor, "trial download")) - first == 1
    assert (keep / "harbor" / JOB / "alpha__T1" / "trajectory.json").is_file()


def test_full_archive_mode(harbor, capsys):
    assert main([f"harbor://jobs/{JOB}", "--full", "--overview", "--format", "json"]) == 0
    ov = json.loads(capsys.readouterr().out)
    assert ov["disqualification"]["candidate_ids"] == ["beta__T3"]
    assert calls(harbor, "job download") and not calls(harbor, "trial download")


def test_inspect_harbor_job_downloads_nothing(harbor, capsys):
    assert (
        main(["--inspect", f"https://hub.harborframework.com/jobs/{JOB}", "--format", "text"]) == 0
    )
    out = capsys.readouterr().out
    assert "6 present / 8 planned" in out and "DQ" not in out
    assert not calls(harbor, "download")


def test_harbor_errors_are_withheld(harbor, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_HARBOR_FAIL", SECRET)
    assert main([f"harbor://jobs/{JOB}"]) == 2
    captured = capsys.readouterr()
    assert "harbor_command_failed" in captured.err and SECRET not in captured.out + captured.err


def test_missing_harbor_cli(monkeypatch, capsys):
    monkeypatch.delenv("ATIF_SCAN_HARBOR", raising=False)
    monkeypatch.setenv("PATH", "")
    assert main([f"harbor://jobs/{JOB}"]) == 2
    assert "harbor_cli_not_found" in capsys.readouterr().err


def test_local_traces_use_final_metrics_and_exception_marker(tmp_path, capsys):
    trial_dir = tmp_path / "job" / "alpha__X1"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(trajectory("ls")))
    (trial_dir / "exception.txt").write_text("boom")
    assert (
        main([str(tmp_path / "job"), "--task-from", "trial-dir", "--overview", "--format", "json"])
        == 0
    )
    ov = json.loads(capsys.readouterr().out)
    assert ov["cost"]["total_usd"] == 9.99 and ov["trials"]["errored"] == 1
    assert ov["trials"]["error_types"] == {"exception": 1}
