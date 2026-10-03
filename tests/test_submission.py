"""Leaderboard submissions assembled from several Harbor jobs (TB2.1 PR #220 shape):
the submission file, jobs that must fit together, trials added to a job after it ran,
and the submission's filter checked against every trial. Synthetic fixtures only."""

from __future__ import annotations

import json

import pytest
from test_brief_cache import _item

from atif_scan.cli import main
from atif_scan.output.brief import brief, brief_text
from atif_scan.sources.harbor.files import late_trials, submission
from atif_scan.sources.harbor.listing import run_meta

V4 = "11111111-1111-4111-8111-111111111111"
V5 = "22222222-2222-5222-8222-222222222222"


def rows(starts, rewards=None, task="alpha"):
    rewards = rewards or [1] * len(starts)
    return [
        {
            "id": f"00000000-0000-4000-8000-{k:012d}",
            "name": f"{task}__t{k}",
            "task_name": f"terminal-bench/{task}",
            "reward": r,
            "started_at": f"2026-08-24T{h}:00+00:00",
            "finished_at": f"2026-08-24T{h}:30+00:00",
        }
        for k, (h, r) in enumerate(zip(starts, rewards, strict=True))
    ]


def test_submission_file_names_its_jobs_and_filter():
    doc = {
        "source_jobs": [
            f"https://hub.harborframework.com/jobs/{V4}",
            f"https://hub.harborframework.com/jobs/{V5.upper()}",
            f"https://hub.harborframework.com/jobs/{V4}",  # listed twice: scanned once
            "not a job",
        ],
        "source_filter": {"agent": "fast-agent", "model_name": "xai/grok-4.6", "x": "y"},
    }
    found = submission(json.dumps(doc).encode())
    assert found == {
        "jobs": [V4, V5],
        "filter": {"agent": "fast-agent", "model_name": "xai/grok-4.6"},
    }
    assert submission(b'{"source_jobs": []}') is None
    assert submission(b"not json") is None and submission(None) is None


def test_trials_started_long_after_the_rest_of_their_job_are_late():
    # Regression (PR #220 shard 02): two replacements started 7.2 h after the job ran,
    # and nothing in the report showed that trials had been added to it.
    starts = ["01", "01", "02", "02", "03", "03", "03", "04", "04", "04", "04", "12"]
    late = late_trials(rows(starts, [0] * 11 + [1]))
    assert late == {"trials": 1, "rewarded": 1, "gap_hours": 8.0}
    assert late_trials(rows(["01", "02", "03", "04"])) is None  # steady: no gap
    # A gap before most of the job isn't "added later": the tail must be small.
    assert late_trials(rows(["01", "09", "09", "09", "10", "10"])) is None
    missing = rows(["01", "12"])
    del missing[0]["started_at"]
    assert late_trials(missing) is None  # unknown start: never guessed


def test_run_facts_mark_constructed_ids_and_list_tasks():
    show = {"name": "shard", "config": {"n_attempts": 5}}
    meta = run_meta(V5, show, rows(["01", "01"]))
    assert meta["constructed_id"] is True and meta["tasks"] == ["alpha"]
    assert run_meta(V4, show, rows(["01", "01"]))["constructed_id"] is False


def _runs():
    return [
        {"source": "harbor_hub", "job_id": V4, "listed_trials": 3, "tasks": ["task0", "task1"]},
        {
            "source": "harbor_hub",
            "job_id": V5,
            "listed_trials": 3,
            "tasks": ["task2"],
            "constructed_id": True,
            "late_trials": {"trials": 2, "rewarded": 2, "gap_hours": 7.2},
        },
    ]


def _doc(runs, items, sub=None):
    doc = {"scanner_version": "dev", "inputs": items, "coverage": {}, "runs": runs}
    if sub:
        doc["submission"] = sub
    return doc


def test_brief_says_how_the_jobs_fit_together():
    items = [_item(i, "grok-4.6", 1.0) for i in range(6)]
    b = brief(_doc(_runs(), items))
    assert b["jobs"]["shared_tasks"] == [] and b["jobs"]["constructed"] == [V5]
    flat = " ".join(brief_text(b).split())
    assert "2 Harbor jobs, trials in each: 11111111 3 · 22222222 3" in flat
    assert "✓ the 2 jobs cover 3 tasks with no task in two jobs" in flat
    assert "⚠ job 22222222 has a constructed (UUIDv5) ID" in flat
    assert "⚠ 2 trials in job 22222222 started 7.2 h after the rest of it" in flat
    assert "2 were rewarded" in flat
    shared = _runs()
    shared[1]["tasks"] = ["task1", "task2"]
    flat = " ".join(brief_text(brief(_doc(shared, items))).split())
    assert "⚠ 1 task is in more than one job" in flat


def test_submission_filter_is_checked_against_every_trial():
    items = [_item(i, "grok-4.6", 1.0) for i in range(4)]
    for i in items:
        i.update(agent_name="fast-agent", agent_version="0.10.10")
    items[1]["model_name"] = None  # recorded no model: only agent and version checked
    items[2]["agent_version"] = "0.9.0"  # another version: doesn't match
    items[3].update(agent_name=None, agent_version=None, model_name=None)  # nothing recorded
    sub = {
        "name": "2026-08-24-demo",
        "jobs": [V4, V5],
        "filter": {
            "agent": "fast-agent",
            "agent_version": "0.10.10",
            "model_name": "xai/grok-4.6",  # provider prefix: compared without it
            "reasoning_effort": "medium",
        },
    }
    b = brief(_doc(_runs(), items, sub))
    check = b["submission"]
    assert (check["matched"], check["partial"], check["mismatched"], check["unrecorded"]) == (
        1,
        1,
        1,
        1,
    )
    flat = " ".join(brief_text(b).split())
    assert "RUN submission 2026-08-24-demo" in flat
    assert "⚠ 1 of 4 trials doesn't match the submission's filter" in flat
    assert "✓ 2 of 4 trials match the submission's filter (agent fast-agent" in flat
    assert "1 trial of those records only some of these fields" in flat
    assert "1 trial records no agent or model to check" in flat
    assert "reasoning effort medium isn't recorded per trial, so it isn't checked" in flat


def test_submission_option_adds_its_jobs_to_the_inputs(tmp_path, monkeypatch, capsys):
    path = tmp_path / "sub.json"
    path.write_text(json.dumps({"source_jobs": [f"https://hub.harborframework.com/jobs/{V4}"]}))
    seen = {}

    def inspect(args):  # stop before any Hub call: only the expansion is under test
        seen["paths"], seen["sub"] = args.paths, args.submission_doc
        return 0

    monkeypatch.setattr("atif_scan.cli.inspect", inspect)
    assert main(["--submission", str(path), "--inspect"]) == 0
    assert seen["paths"] == [f"harbor://jobs/{V4}"]
    assert seen["sub"] == {"name": "sub", "jobs": [V4], "filter": {}}
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(SystemExit):
        main(["--submission", str(bad), "--inspect"])
    assert "not a readable leaderboard submission" in capsys.readouterr().err
