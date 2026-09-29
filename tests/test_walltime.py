"""Walltime is summed per trial, with independent agent/full-trial coverage."""

from __future__ import annotations

import json

import pytest
from test_brief import section
from test_harbor_files import harbor_job

from atif_scan.cli import main
from atif_scan.harbor_files import duration, trial_result
from atif_scan.report import walltime_lines, walltime_totals


def interval(start, end):
    return {"started_at": start, "finished_at": end}


def timed_job(tmp_path):
    job = harbor_job(tmp_path)
    # The trials overlap: their durations must be added, not treated as job elapsed time.
    for index, (trial_minutes, agent_minutes) in enumerate(((5, 2), (10, 4), (None, None))):
        path = job / f"weird-folder-{index}" / "result.json"
        data = json.loads(path.read_text())
        data.pop("started_at")
        data.pop("finished_at")
        if trial_minutes is not None:
            assert agent_minutes is not None
            data.update(interval("2026-01-01T00:00:00Z", f"2026-01-01T00:{trial_minutes:02d}:00Z"))
            data["agent_execution"] = interval(
                "2026-01-01T00:01:00Z", f"2026-01-01T00:{agent_minutes + 1:02d}:00Z"
            )
        path.write_text(json.dumps(data))
    return job


@pytest.mark.parametrize("view", ["--brief", "--summary", "--overview"])
def test_walltime_totals_across_summary_views(tmp_path, capsys, view):
    job = timed_job(tmp_path)
    assert main([str(job), view, "--format", "json", "--no-cache"]) == 0
    doc = json.loads(capsys.readouterr().out)
    overview = doc if view == "--overview" else doc["overview"]
    timing = overview["walltime"]
    assert timing["trials"] == 3
    assert timing["agent"] == {"seconds": 360.0, "recorded_trials": 2}
    assert timing["trial"] == {"seconds": 900.0, "recorded_trials": 2}

    assert main([str(job), view, "--format", "text", "--no-cache"]) == 0
    text = capsys.readouterr().out
    text = section(text, "WALLTIME") if view == "--brief" else text
    assert "agent execution: 6m 0s" in text
    assert "full trial (setup + agent + verifier): 15m 0s" in text
    assert "2 of 3 trials" in text
    assert "not elapsed job time" in text


def test_agent_duration_is_not_inferred_from_full_trial_or_step_timestamps(tmp_path, capsys):
    job = timed_job(tmp_path)
    path = job / "weird-folder-1" / "result.json"
    data = json.loads(path.read_text())
    del data["agent_execution"]
    path.write_text(json.dumps(data))
    main([str(job), "--detail", "--format", "json", "--no-cache"])
    items = json.loads(capsys.readouterr().out)["inputs"]
    totals = walltime_totals(items)
    assert totals["agent"] == {"seconds": 120.0, "recorded_trials": 1}
    assert totals["trial"] == {"seconds": 900.0, "recorded_trials": 2}
    assert sum(i["agent_duration_sec"] is None for i in items) == 2


def test_result_timing_is_refreshed_even_when_trajectory_scan_is_cached(tmp_path, capsys):
    job = timed_job(tmp_path)
    args = [str(job), "--overview", "--format", "json", "--cache", str(tmp_path / "cache")]
    main(args)
    first = json.loads(capsys.readouterr().out)["walltime"]
    path = job / "weird-folder-0" / "result.json"
    data = json.loads(path.read_text())
    data["agent_execution"]["finished_at"] = "2026-01-01T00:04:00Z"
    path.write_text(json.dumps(data))
    main(args)
    second = json.loads(capsys.readouterr().out)["walltime"]
    assert second["agent"]["seconds"] == first["agent"]["seconds"] + 60
    assert second["trial"] == first["trial"]


def test_absent_and_zero_durations_are_distinct():
    empty = walltime_totals([{}, {"agent_duration_sec": None, "duration_sec": None}])
    assert empty["agent"]["seconds"] is None
    assert empty["trial"]["seconds"] is None
    assert "not recorded" in " ".join(walltime_lines(empty))
    zero = walltime_totals([{"duration_sec": 0.0}])
    assert zero["trial"] == {"seconds": 0.0, "recorded_trials": 1}
    assert zero["agent"]["seconds"] is None
    assert "full trial (setup + agent + verifier): 0s" in " ".join(walltime_lines(zero))


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"started_at": "2026-01-01T00:00:00Z"},
        interval("invalid", "2026-01-01T00:00:00Z"),
        interval("2026-01-01T00:01:00Z", "2026-01-01T00:00:00Z"),
        interval("2026-01-01T00:00:00", "2026-01-01T00:01:00Z"),
    ],
)
def test_invalid_or_unfinished_intervals_are_unknown_not_zero(record):
    assert duration(record) is None
    facts = trial_result(json.dumps({"task_name": "demo", "agent_execution": record}).encode())
    assert "agent_duration_sec" not in facts


def test_offsets_and_fractional_seconds_are_preserved():
    record = interval("2026-01-01T01:00:00.25+01:00", "2026-01-01T00:01:00.75Z")
    assert duration(record) == 60.5
    facts = trial_result(
        json.dumps({"task_name": "demo", "agent_execution": record, "untrusted": "SECRET"}).encode()
    )
    assert facts["agent_duration_sec"] == 60.5
    assert "SECRET" not in json.dumps(facts)
