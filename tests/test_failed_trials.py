"""Failed trials: what Harbor and fast-agent record about how they ended (synthetic)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from atif_scan import parse_trace
from atif_scan.sources.harbor.files import safety_details, trial_result

if TYPE_CHECKING:
    from pathlib import Path

    from atif_scan.data.jsonval import Doc


def phases(occurred_at: str) -> Doc:
    return {
        "trial_name": "t__x",
        "task_name": "terminal-bench/t",
        "exception_info": {"exception_type": "AgentSafetyStopError", "occurred_at": occurred_at},
        "environment_setup": {
            "started_at": "2026-10-03T01:30:00Z",
            "finished_at": "2026-10-03T01:30:10Z",
        },
        "agent_setup": {
            "started_at": "2026-10-03T01:30:10Z",
            "finished_at": "2026-10-03T01:31:00Z",
        },
        "agent_execution": {
            "started_at": "2026-10-03T01:31:00Z",
            "finished_at": "2026-10-03T01:31:07Z",
        },
        "verifier": {"started_at": "2026-10-03T01:31:09Z", "finished_at": "2026-10-03T01:31:19Z"},
    }


def test_failed_phase_and_durations_come_from_result_json():
    facts = trial_result(json.dumps(phases("2026-10-03T01:31:05Z")).encode())
    assert facts["failed_phase"] == "agent_execution"
    assert facts["setup_duration_sec"] == 60.0
    assert facts["verifier_duration_sec"] == 10.0
    # Harbor stamps the exception just after the agent phase closes
    after = trial_result(json.dumps(phases("2026-10-03T01:31:08Z")).encode())
    assert after["failed_phase"] == "after_agent_execution"


SAFETY = {
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Do the task."}]},
        {
            "role": "assistant",
            "content": [],
            "channels": {
                "fast-agent-safety-details": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "provider": "anthropic",
                                "reason": "refusal",
                                "category": "cyber",
                                "explanation": "Blocked under the usage policy. SECRET-PROSE",
                            }
                        ),
                    }
                ]
            },
        },
    ]
}


def test_safety_codes_are_read_without_the_explanation():
    codes = safety_details(json.dumps(SAFETY).encode())
    assert codes == {
        "safety_provider": "anthropic",
        "safety_reason": "refusal",
        "safety_category": "cyber",
    }
    assert "SECRET-PROSE" not in json.dumps(codes)


def test_safety_codes_must_be_codes():
    bad = json.loads(json.dumps(SAFETY))
    channel = bad["messages"][1]["channels"]["fast-agent-safety-details"][0]
    channel["text"] = json.dumps({"provider": "anthropic", "category": "rm -rf /; echo"})
    assert safety_details(json.dumps(bad).encode()) == {"safety_provider": "anthropic"}
    assert safety_details(b"not json") == {}


def scan_trial(trial: Path, capsys) -> Doc:
    from atif_scan.cli import main

    main([str(trial), "--format", "json", "--no-cache"])
    return json.loads(capsys.readouterr().out)["inputs"][0]


def test_a_safety_stopped_trial_reports_how_it_ended(tmp_path: Path, capsys):
    trial = tmp_path / "t__x"
    (trial / "agent").mkdir(parents=True)
    steps = [
        {"step_id": 1, "source": "user", "message": "Do it."},
        {
            "step_id": 2,
            "source": "agent",
            "message": "",
            "extra": {"stop_reason": "LlmStopReason.SAFETY"},
        },
    ]
    (trial / "agent" / "trajectory.json").write_text(
        json.dumps({"schema_version": "ATIF-v1.7", "steps": steps})
    )
    (trial / "agent" / "fast-agent-results.json").write_text(json.dumps(SAFETY))
    (trial / "result.json").write_text(json.dumps(phases("2026-10-03T01:31:05Z")))
    item = scan_trial(trial, capsys)
    assert item["error_type"] == "AgentSafetyStopError"
    assert item["final_stop_reason"] == "safety"
    assert (item["safety_provider"], item["safety_reason"], item["safety_category"]) == (
        "anthropic",
        "refusal",
        "cyber",
    )
    assert item["failed_phase"] == "agent_execution"
    assert "SECRET-PROSE" not in json.dumps(item)
    # Without an exception the companion file isn't read at all.
    clean = phases("2026-10-03T01:31:05Z")
    del clean["exception_info"]
    (trial / "result.json").write_text(json.dumps(clean))
    assert scan_trial(trial, capsys)["safety_category"] is None


def test_final_stop_reason_is_a_code():
    steps = [
        {"source": "user", "message": "Do it."},
        {"source": "agent", "message": "", "extra": {"stop_reason": "LlmStopReason.SAFETY"}},
    ]
    assert (
        parse_trace({"schema_version": "ATIF-v1.7", "steps": steps}).final_stop_reason == "safety"
    )
    steps[1]["extra"]["stop_reason"] = "<b>weird</b>"
    assert parse_trace({"schema_version": "ATIF-v1.7", "steps": steps}).final_stop_reason is None
