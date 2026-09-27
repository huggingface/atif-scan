"""Harbor job folders: trial result.json and job config/result as recorded run facts."""

from __future__ import annotations

import json

import pytest

from atif_scan.cli import main
from atif_scan.harbor_files import job_meta, primary_reward, trial_result


def result_json(task, reward=None, error=None, cost=None, tokens=(1000, 500, 100)):
    return {
        "trial_name": f"{task}__x",
        "task_name": f"terminal-bench/{task}",
        "started_at": "2026-09-11T10:00:00+00:00",
        "finished_at": "2026-09-11T10:05:00+00:00",
        "agent_result": {
            "n_input_tokens": tokens[0],
            "n_cache_tokens": tokens[1],
            "n_output_tokens": tokens[2],
            "cost_usd": cost,
        },
        "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
        "exception_info": {"exception_type": error, "exception_message": "SECRET detail"}
        if error
        else None,
    }


def test_trial_result_facts():
    facts = trial_result(json.dumps(result_json("alpha", 1.0, cost=0.5)).encode())
    assert facts == {
        "task": "alpha",
        "reward": 1.0,
        "cost_usd": 0.5,
        "input_tokens": 1000,
        "cache_tokens": 500,
        "output_tokens": 100,
        "duration_sec": 300.0,
    }
    errored = trial_result(json.dumps(result_json("beta", error="AgentTimeoutError")).encode())
    assert errored["error_type"] == "AgentTimeoutError" and "reward" not in errored
    assert "SECRET" not in json.dumps(errored)  # messages are never extracted
    # A job-level result.json (no trial/task name) is not a trial result.
    assert trial_result(json.dumps({"id": "j", "n_total_trials": 5, "stats": {}}).encode()) == {}
    assert trial_result(b"not json") == {} and trial_result(None) == {}


@pytest.mark.parametrize(
    ("rewards", "expected"),
    [
        ({"reward": 0.5}, 0.5),
        ({"accuracy": 1}, 1.0),
        ({"a": 1, "b": 0}, None),
        (1, 1.0),
        (None, None),
    ],
)
def test_primary_reward(rewards, expected):
    assert primary_reward(rewards) == expected


def test_job_meta_and_overrides():
    config = {
        "job_name": "my-run",
        "n_attempts": 2,
        "agent_setup_timeout_multiplier": 2.0,
        "agents": [{"model_name": "m", "override_timeout_sec": 21600.0}],
        "datasets": [{"name": "terminal-bench/x", "ref": "sha256:abc", "task_names": ["a", "b"]}],
    }
    result = {
        "id": "1d6abb23-0000-4000-8000-000000000000",
        "n_total_trials": 4,
        "stats": {"n_errored_trials": 1, "cost_usd": None},
    }
    run = job_meta(json.dumps(config).encode(), json.dumps(result).encode())
    assert run["job_name"] == "my-run" and run["planned_trials"] == 4 and run["n_attempts"] == 2
    assert run["config_task_names"] == 2 and run["source"] == "harbor_job_folder"
    assert run["overrides"] == ["agent_setup_timeout_multiplier", "agents[].override_timeout_sec"]
    assert job_meta(b'{"unrelated": 1}', None) is None


def trajectory(command="ls", tokens=None):
    raw = {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {
                "source": "agent",
                "message": "x",
                "tool_calls": [
                    {
                        "tool_call_id": "c1",
                        "function_name": "bash",
                        "arguments": {"command": command},
                    }
                ],
            }
        ],
    }
    if tokens:
        raw["final_metrics"] = {"total_prompt_tokens": tokens, "total_completion_tokens": 1}
    return raw


def harbor_job(tmp_path, costs=(None, None, None)):
    job = tmp_path / "jobdir"
    job.mkdir()
    (job / "config.json").write_text(
        json.dumps(
            {
                "job_name": "my-run",
                "n_attempts": 2,
                "agents": [{"override_timeout_sec": 600}],
                "datasets": [{"name": "terminal-bench/x", "ref": "sha256:abc"}],
            }
        )
    )
    (job / "result.json").write_text(
        json.dumps({"id": "1d6abb23-0000-4000-8000-000000000000", "n_total_trials": 4, "stats": {}})
    )
    rows = [
        ("alpha", 1.0, None, "ls"),
        ("alpha", 0.0, None, "ls"),
        ("beta", None, "AgentTimeoutError", "ls"),
    ]
    for i, (task, reward, error, cmd) in enumerate(rows):
        trial = job / f"weird-folder-{i}"  # no <task>__ naming: tasks must come from result.json
        (trial / "agent").mkdir(parents=True)
        (trial / "agent" / "trajectory.json").write_text(json.dumps(trajectory(cmd, tokens=900)))
        (trial / "result.json").write_text(
            json.dumps(result_json(task, reward, error, cost=costs[i]))
        )
    return job


def test_local_job_folder_uses_recorded_facts(tmp_path, capsys):
    job = harbor_job(tmp_path)
    assert main([str(job), "--format", "json"]) == 0
    items = {i["input_id"]: i for i in json.loads(capsys.readouterr().out)["inputs"]}
    one = items["weird-folder-0/agent"]
    assert one["task"] == "alpha" and one["reward"] == 1.0 and one["input_tokens"] == 1000
    assert items["weird-folder-2/agent"]["error_type"] == "AgentTimeoutError"
    main([str(job), "--format", "text"])
    out = capsys.readouterr().out
    assert "my-run · job 1d6abb23" in out and "3 trials · 2 tasks × 1–2" in out
    assert "3/4 planned trials have a trajectory" in out
    assert "scoring overrides" in out and "agents[].override_timeout_sec" in out
    assert "no cost recorded for any trial" in out and "--price" in out


def test_price_estimates_unpriced_trials(tmp_path, capsys):
    job = harbor_job(tmp_path)
    main([str(job), "--brief", "--format", "json", "--price", "2,0.5,10"])
    ce = json.loads(capsys.readouterr().out)["cost_estimate"]
    # per trial: 500 uncached * 2 + 500 cached * 0.5 + 100 out * 10, per million tokens
    assert ce["method"] == "given --price" and ce["estimate_usd"] == round(3 * 2250 / 1e6, 2)
    with pytest.raises(SystemExit):
        main([str(job), "--price", "1,2"])


def test_unknown_tasks_are_reported_not_invented(tmp_path, capsys):
    root = tmp_path / "loose"
    for i in range(3):
        d = root / f"t{i}" / "agent"
        d.mkdir(parents=True)
        (d / "trajectory.json").write_text(json.dumps(trajectory()))
        (d.parent / "verifier").mkdir()
        (d.parent / "verifier" / "reward.txt").write_text("1")
    main([str(root), "--format", "text"])
    out = capsys.readouterr().out
    assert "tasks unknown (pass --task-from trial-dir or --task)" in out
    assert "RESULT     100.0% (3/3)" in out  # no per-task ± without tasks


def test_infrastructure_overrides_and_forked_tasks_are_told_apart():
    # Regression (a private DeepSeek run): setup-time overrides for a sandbox provider and a
    # fork of the task repo that only swaps images are infrastructure; the agent timeout
    # override changes the score.
    from atif_scan.brief import brief_text
    from atif_scan.harbor_files import dataset_source, job_meta, override_kind

    assert override_kind("agents[].override_setup_timeout_sec") == "infrastructure"
    assert override_kind("agent_setup_timeout_multiplier") == "infrastructure"
    assert override_kind("agents[].override_timeout_sec") == "scoring"
    fork = {
        "repo": "https://github.com/example/terminal-bench-2-1.git@75f5a2e66b2d",
        "path": "tasks",
    }
    assert dataset_source(fork) == (
        "github.com/example/terminal-bench-2-1/tasks",
        "75f5a2e66b2d",
        False,
    )
    upstream = {
        "repo": "https://github.com/harbor-framework/terminal-bench-2-1.git",
        "path": "tasks",
    }
    assert dataset_source(upstream)[2] is True
    assert dataset_source({"name": "terminal-bench/terminal-bench-2-1", "ref": "6"})[2] is True
    config = {
        "n_attempts": 5,
        "agent_setup_timeout_multiplier": 2.0,
        "agents": [{"override_timeout_sec": 21600.0, "override_setup_timeout_sec": 1800.0}],
        "datasets": [fork],
    }
    meta = job_meta(json.dumps(config).encode(), b"{}")
    assert meta["canonical_dataset"] is False
    doc = {"scanner_version": "dev", "inputs": [], "coverage": {}, "runs": [meta]}
    from atif_scan.brief import brief

    text = brief_text(brief(doc))
    assert "scoring overrides" in text and "agents[].override_timeout_sec" in text
    assert "infrastructure overrides (provisioning only)" in text
    assert "non-canonical source: github.com/example/terminal-bench-2-1/tasks" in text
