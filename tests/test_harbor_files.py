"""Harbor job folders: trial result.json and job config/result as recorded run facts."""

from __future__ import annotations

import json

import pytest
from test_brief import section

from atif_scan.cli import main
from atif_scan.sources.harbor.files import job_meta, primary_reward, trial_result


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
    assert run is not None
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
    flat = " ".join(out.split())  # unwrapped: phrases may span lines
    assert "RUN        job 1d6abb23 · my-run" in out and "3 trials of 2 tasks, 1–2 per task" in out
    assert "⚠ 3 of 4 planned trials have a trajectory" in out
    assert (
        "⚠ overrides that change the agent's time or resources (leaderboards require defaults):"
        " agents[].override_timeout_sec"
    ) in flat
    assert "COST       ⚠ no trial recorded a cost" in out
    assert "· price the tokens with --price U,C,O" in flat
    assert "= — total unknown: nothing could be priced" in flat


def test_price_estimates_unpriced_trials(tmp_path, capsys):
    job = harbor_job(tmp_path)
    main([str(job), "--brief", "--format", "json", "--price", "2,0.5,10"])
    ce = json.loads(capsys.readouterr().out)["cost_estimate"]
    # per trial: 500 uncached * 2 + 500 cached * 0.5 + 100 out * 10, per million tokens
    assert ce["method"] == "at the given --price" and ce["estimate_usd"] == round(3 * 2250 / 1e6, 2)
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
    score = section(out, "SCORE")
    assert score == "SCORE      100.0% · 3 of 3 scored trials rewarded"  # no ± without tasks


def test_infrastructure_overrides_and_forked_tasks_are_told_apart():
    # Regression (a private DeepSeek run): setup-time overrides for a sandbox provider and a
    # fork of the task repo that only swaps images are infrastructure; the agent timeout
    # override changes the score.
    from atif_scan.output.brief import brief_text
    from atif_scan.sources.harbor.files import dataset_source, job_meta, override_kind

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
    assert meta is not None
    assert meta["canonical_dataset"] is False
    doc = {"scanner_version": "dev", "inputs": [], "coverage": {}, "runs": [meta]}
    from atif_scan.output.brief import brief

    text = brief_text(brief(doc))
    settings = " ".join(section(text, "SETTINGS").split())  # unwrapped
    assert (
        "⚠ overrides that change the agent's time or resources (leaderboards require defaults):"
        " agents[].override_timeout_sec ·"
    ) in settings
    assert (
        "· provisioning-only overrides: agent_setup_timeout_multiplier,"
        " agents[].override_setup_timeout_sec"
    ) in settings
    assert (
        "⚠ tasks aren't from the benchmark's own source"
        " (github.com/example/terminal-bench-2-1/tasks)"
    ) in settings
    # No trials: no highest-priority breakdown with nothing after its label.
    assert "trials by their highest priority" not in text


def rerun_job(tmp_path, listing=True, completed=2):
    """A job whose result.json lists two trials; a third folder re-ran `alpha` (as when
    a resumed job writes into the folder while the first execution still runs)."""
    job = tmp_path / "rerun"
    job.mkdir(parents=True)
    (job / "config.json").write_text(json.dumps({"datasets": [{"name": "terminal-bench/x"}]}))
    evals = {
        "agent__model__x": {
            "n_trials": 2,
            "reward_stats": {"reward": {"1.0": ["alpha__a1"], "0.0": ["beta__b1"]}},
            "exception_stats": {},
        }
    }
    stats = {"n_completed_trials": completed, **({"evals": evals} if listing else {})}
    (job / "result.json").write_text(
        json.dumps(
            {"id": "2d6abb23-0000-4000-8000-000000000000", "n_total_trials": 2, "stats": stats}
        )
    )
    rows = [
        ("alpha__a1", "alpha", 1.0, 1.0, True),
        ("beta__b1", "beta", 0.0, 2.0, True),
        ("alpha__a2", "alpha", 0.0, 4.0, True),  # the unlisted rerun
        ("gamma__g2", "gamma", None, None, False),  # unlisted, died before a trajectory
    ]
    for folder, task, reward, cost, traced in rows:
        trial = job / folder
        trial.mkdir()
        if traced:
            (trial / "agent").mkdir()
            (trial / "agent" / "trajectory.json").write_text(json.dumps(trajectory("ls")))
        (trial / "result.json").write_text(
            json.dumps(
                result_json(task, reward, None if traced else "RemoteProtocolError", cost=cost)
            )
        )
    return job


def test_trials_missing_from_the_jobs_result_are_scored_and_flagged_as_a_rerun(tmp_path, capsys):
    """Regression: two executions of one job wrote into the same folder; the scan scored
    both silently and showed "✓ 126/89 planned trials"."""
    job = rerun_job(tmp_path)
    main([str(job), "--format", "json"])
    items = {i["input_id"]: i for i in json.loads(capsys.readouterr().out)["inputs"]}
    assert items["alpha__a2/agent"]["in_job_result"] is False
    assert items["alpha__a1/agent"]["in_job_result"] is True
    main([str(job), "--brief", "--format", "json"])
    b = json.loads(capsys.readouterr().out)
    rr = b["overview"]["reruns"]
    assert (rr["unlisted_trials"], rr["unlisted_with_trajectory"], rr["tasks_rerun"]) == (2, 1, 1)
    assert rr["listed"] == {"trials": 2, "scored": 2, "rewarded": 1, "cost_usd": 3.0}
    assert rr["unlisted"] == {"trials": 1, "scored": 1, "rewarded": 0, "cost_usd": 4.0}
    main([str(job), "--format", "text"])
    out = capsys.readouterr().out
    flat = " ".join(out.split())  # unwrapped: phrases may span lines
    assert "SCORE      33.3% ± 25.0 · 1 of 3 scored trials rewarded" in out  # all seen are scored
    assert "⚠ 3 of 2 planned trials have a trajectory" in out
    assert "⚠ 2 trial folders not listed in the job's result.json (1 with a trajectory)" in flat
    assert "; 1 task ran again: likely a rerun or resume" in flat
    assert (
        "· all are scored above · listed trials 1 of 2 rewarded (50.0%), $3.00"
        " · other trials 0 of 1 rewarded (0.0%), $4.00"
    ) in flat


def test_job_listing_missing_or_incomplete_is_unknown_not_clean(tmp_path, capsys):
    for kwargs in ({"listing": False}, {"completed": 3}):  # none / fewer names than completed
        job = rerun_job(tmp_path / next(iter(kwargs)), **kwargs)
        main([str(job), "--format", "json"])
        items = json.loads(capsys.readouterr().out)["inputs"]
        assert all(i["in_job_result"] is None for i in items)
        main([str(job), "--brief", "--format", "json"])
        assert json.loads(capsys.readouterr().out)["overview"]["reruns"] is None
        main([str(job), "--format", "text"])
        assert "not in the job's result.json" not in capsys.readouterr().out


def test_job_listed_trials_tolerates_malformed_stats():
    from atif_scan.sources.harbor.files import job_listed_trials

    def listed(stats):
        return job_listed_trials(json.dumps({"stats": stats}).encode())

    assert listed({"evals": {"e": {"reward_stats": [], "exception_stats": "x"}}}) is None
    assert listed({"evals": {"e": {"reward_stats": {"reward": {"1.0": ["a", 3, ""]}}}}}) == {"a"}
    assert listed({"evals": {"e": {"exception_stats": {"E": ["b"]}}, "f": None}}) == {"b"}
    assert job_listed_trials(b"not json") is None and job_listed_trials(None) is None


def test_overview_of_a_job_folder_without_a_job_id(tmp_path, capsys):
    # Regression (review follow-up): a job folder whose job result.json is missing (an
    # interrupted job) has no job ID, and --overview crashed on it.
    job = harbor_job(tmp_path)
    (job / "result.json").unlink()
    assert main([str(job), "--overview", "--format", "text"]) == 0
    out = capsys.readouterr().out
    assert "  job        my-run" in out and "harbor None" not in out
