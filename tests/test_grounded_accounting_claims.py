"""Synthetic accounting records establish observations, not billing provenance."""

from atif_scan import Status
from atif_scan.detectors.integrity import (
    cost_missing,
    observation_pairing_unresolved,
    output_token_ratio,
)
from atif_scan.facts import trace_facts
from atif_scan.loader import parse_trace
from atif_scan.output.brief import brief, brief_text


def accounting_trace():
    return parse_trace(
        {
            "steps": [
                {
                    "source": "agent",
                    "message": "Working.",
                    "metrics": {"prompt_tokens": 100, "completion_tokens": 10},
                }
            ],
            "final_metrics": {
                "total_prompt_tokens": 200,
                "total_completion_tokens": 20,
            },
        }
    )


def report_row():
    return {
        **trace_facts(accounting_trace()),
        "input_id": "synthetic",
        "input_tokens": 200,
        "output_tokens": 20,
        "input_status": "available",
        "incomplete": True,
        "assessments": [],
    }


def render(row):
    return brief_text(brief({"scanner_version": "dev", "inputs": [row], "coverage": {}}))


def test_recorded_tokens_with_unknown_ratio_are_not_missing_usage():
    trace = accounting_trace()
    assert trace.usage is not None
    assert trace.usage.completion_tokens == 20
    assert output_token_ratio(trace).status is Status.UNKNOWN
    text = " ".join(render(report_row()).split())
    assert "lower bound cannot be checked" in text
    assert "read low by design" not in text


def test_final_step_gap_does_not_establish_unrecorded_calls():
    row = report_row()
    assert row["steps_vs_totals"] == "steps_short"
    assert row["tokens_outside_steps"] == 110
    text = " ".join(render(row).split())
    assert "the cause is not established by this comparison" in text
    assert "LLM calls made but not recorded" not in text


def test_matching_records_are_not_independent_billing_verification():
    row = report_row()
    row.update(
        recorded_vs_trajectory="same",
        cost_vs_trajectory="same",
        cost_usd=0.25,
        trajectory_cost_usd=0.25,
    )
    text = " ".join(render(row).split())
    assert "do not establish that every provider attempt was counted" in text
    assert "may share a source; they do not verify provider billing" in text


def test_atif_cost_absence_does_not_mean_harbor_cost_absence():
    assert cost_missing(accounting_trace()).status is Status.MATCH
    row = report_row()
    row["cost_usd"] = 0.25
    row["assessments"] = [
        {
            "id": "integrity.cost_missing",
            "status": "match",
            "kind": "detector",
            "severity": "low",
            "complete": True,
            "evidence": [],
        }
    ]
    summary = brief({"scanner_version": "dev", "inputs": [row], "coverage": {}})
    assert summary["recording"].get("integrity.cost_missing", 0) == 0


def test_shared_output_retains_unknown_attribution_not_missing_output_claim():
    trace = parse_trace(
        {
            "steps": [
                {
                    "source": "agent",
                    "tool_calls": [
                        {"tool_call_id": "a", "function_name": "bash_command", "arguments": {}},
                        {"tool_call_id": "b", "function_name": "bash_command", "arguments": {}},
                    ],
                    "observation": {"results": [{"content": "Synthetic shared terminal output"}]},
                }
            ]
        }
    )
    assert observation_pairing_unresolved(trace).status is Status.MATCH
    assert not trace.results_unrecorded
    row = report_row()
    row["recording_gaps"] = {"pairing_unresolved": 1}
    text = " ".join(render(row).split())
    assert "unresolved pairing concerns call attribution, not necessarily missing output" in text
    assert "Synthetic shared terminal output" not in text
