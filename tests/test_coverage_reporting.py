"""Coverage labels distinguish absent telemetry from missing behavioural evidence."""

from __future__ import annotations

import io
import json

import pytest

from atif_scan import (
    CheckSpec,
    Engine,
    Severity,
    builtin_detectors,
    document,
    parse_trace,
    report,
    to_text,
)
from atif_scan.report import (
    coverage_gaps,
    coverage_summary,
    filter_findings,
    footer,
    headline,
    render_rich,
    unresolved,
)

PRIVATE_TEXT = "private-trajectory-marker"


def recording():
    return {
        "steps": [
            {
                "step_id": 1,
                "source": "user",
                "message": "Write a friendly greeting for a visitor arriving at the town library.",
            },
            {"step_id": 2, "source": "agent", "message": f"Hello! {PRIVATE_TEXT}"},
        ],
        "final_metrics": {"total_steps": 2},
    }


def scanned(raw, checks=None):
    trace = parse_trace(raw)
    rows = Engine(builtin_detectors() if checks is None else checks).evaluate(trace)
    item = report(rows, trace.step_numbers)
    item.update(
        input_id="trial",
        input_status="available",
        partial=trace.recording_gaps,
        agent_steps=trace.agent_steps,
        tool_calls=trace.tool_calls,
        unrecognized_tool_calls=trace.unrecognized_tool_calls,
    )
    return item


def test_missing_telemetry_does_not_mean_missing_behaviour():
    item = scanned(recording())
    gaps = item["coverage_gaps"]
    assert item["incomplete"]  # backwards-compatible aggregate, not a new clean verdict
    assert not gaps["behavioural"] and gaps["telemetry"]
    rows = {a["id"]: a for a in item["assessments"]}
    assert rows["integrity.timestamp_regression"]["status"] == "unknown"
    assert rows["integrity.cost_missing"]["status"] == "unknown"
    assert rows["awareness.verifier"]["status"] == "no_match"
    assert "behavioural coverage complete" in headline(item)
    assert "telemetry checks unresolved" in headline(item)


@pytest.mark.parametrize("rich", [False, True])
def test_renderers_explain_telemetry_without_exposing_trace_text(rich):
    doc = document([scanned(recording())], "test")
    if rich:
        buffer = io.StringIO()
        render_rich(doc, file=buffer)
        text = buffer.getvalue()
    else:
        text = to_text(doc)
    assert "behavioural coverage complete" in text
    assert "Telemetry checks unresolved" in text
    assert "integrity.timestamp_regression" in text
    assert "integrity.cost_missing" in text
    assert "INCOMPLETE" not in text and PRIVATE_TEXT not in text
    assert doc["coverage"]["behavioural_incomplete"] == 0
    assert doc["coverage"]["telemetry_unresolved"] == 1


def test_sufficient_telemetry_has_neither_gap():
    raw = recording()
    for index, step in enumerate(raw["steps"]):
        step["timestamp"] = f"2026-01-01T00:00:0{index}Z"
    raw["final_metrics"] = {
        "total_prompt_tokens": 100,
        "total_completion_tokens": 8,
        "total_cost_usd": 0.01,
        "extra": {"total_reasoning_tokens": 0},
    }
    item = scanned(raw)
    assert not item["incomplete"]
    assert not any(item["coverage_gaps"].values())
    assert "telemetry checks unresolved" not in headline(item)


@pytest.mark.parametrize("defect", ["compaction", "unreadable", "missing_result"])
def test_behavioural_evidence_gaps_remain_incomplete(defect):
    raw = recording()
    if defect == "compaction":
        raw["steps"][0]["message"] += " [COMPACTED HISTORY] earlier turns removed"
        raw["steps"][1]["message"] += " The verifier checks the answer."
    else:
        raw["steps"][1]["tool_calls"] = [
            {
                "tool_call_id": "c1",
                "function_name": "web_search",
                "arguments": "not json" if defect == "unreadable" else {"query": "python docs"},
            }
        ]
    item = scanned(raw)
    assert item["coverage_gaps"]["behavioural"]
    assert "behavioural coverage incomplete" in headline(item)
    if defect == "compaction":
        # Matches can still be incomplete; they must appear among the reasons too.
        row = next(a for a in item["assessments"] if a["id"] == "awareness.verifier")
        assert row["status"] == "match" and not row["complete"]
        assert row["id"] in item["coverage_gaps"]["behavioural"]
        assert row["id"] in "\n".join(unresolved(item))


def test_telemetry_detector_failure_is_not_disguised_as_absent_metadata():
    class Broken:
        spec = CheckSpec("integrity.cost_missing")

        def evaluate(self, trace, context):
            raise ValueError(PRIVATE_TEXT)

    item = scanned(recording(), [Broken()])
    assert item["assessments"][0]["status"] == "error"
    assert item["coverage_gaps"]["behavioural"] == ["integrity.cost_missing"]
    assert not item["coverage_gaps"]["telemetry"]
    assert PRIVATE_TEXT not in json.dumps(item)


@pytest.mark.parametrize("legacy", [False, True])
def test_filters_preserve_full_scan_reasons_and_counts(legacy):
    item = scanned(recording())
    reasons = item["coverage_gaps"]
    if legacy:
        del item["coverage_gaps"]  # older saved reports still render accurately
    doc = document([item], "test")
    shown = filter_findings(doc, Severity.HIGH, checks=("awareness.*",))
    filtered = shown["inputs"][0]
    assert not any(a["id"].startswith("integrity.") for a in filtered["assessments"])
    assert filtered["coverage_gaps"] == reasons
    assert shown["coverage"] == doc["coverage"]
    assert "integrity.cost_missing" in "\n".join(unresolved(filtered))
    assert "behavioural coverage complete" in headline(filtered)


def test_run_counts_distinguish_overlapping_categories():
    raw = recording()
    raw["steps"][0]["message"] += " [COMPACTED HISTORY] earlier turns removed"
    doc = document([scanned(recording()), scanned(raw)], "test")
    assert doc["coverage"]["incomplete"] == 2
    assert doc["coverage"]["behavioural_incomplete"] == 1
    assert doc["coverage"]["telemetry_unresolved"] == 2
    text = to_text(doc)
    assert "1 behavioural coverage incomplete" in text
    assert "2 telemetry checks unresolved" in text


def test_legacy_partial_report_cannot_be_cleared_by_filtering():
    raw = recording()
    raw["steps"][0]["message"] += " [COMPACTED HISTORY] earlier turns removed"
    item = scanned(raw)
    del item["coverage_gaps"]
    item["assessments"] = [a for a in item["assessments"] if a["id"] == "integrity.cost_missing"]
    assert coverage_gaps(item)["behavioural"]
    assert "behavioural coverage incomplete" in headline(item)


def test_legacy_footer_recomputes_categories_when_assessments_are_available():
    item = scanned(recording())
    del item["coverage_gaps"]
    doc = {
        "inputs": [item],
        "coverage": {"inputs": 1, "available": 1, "incomplete": 1},
    }
    text = footer(doc)
    assert "0 behavioural coverage incomplete" in text
    assert "1 telemetry checks unresolved" in text


def test_legacy_aggregate_does_not_invent_coverage_categories():
    text = coverage_summary({"incomplete": 3}, 3)
    assert "3 scans with unresolved checks" in text
    assert "coverage categories unavailable" in text
    assert "behavioural coverage incomplete" not in text


def test_partial_recording_with_telemetry_does_not_claim_it_is_absent():
    raw = recording()
    for index, step in enumerate(raw["steps"]):
        step["timestamp"] = f"2026-01-01T00:00:0{index}Z"
    raw["final_metrics"] = {
        "total_prompt_tokens": 100,
        "total_completion_tokens": 8,
        "total_cost_usd": 0.01,
        "extra": {"total_reasoning_tokens": 0},
    }
    raw["steps"][0]["message"] += " [COMPACTED HISTORY] earlier turns removed"
    item = scanned(raw)
    assert "behavioural coverage incomplete" in headline(item)
    assert "telemetry checks unresolved" in headline(item)
    assert "telemetry unavailable" not in headline(item)


def test_incomplete_telemetry_match_stays_in_telemetry_category():
    raw = recording()
    raw["steps"][0]["message"] += " [COMPACTED HISTORY] earlier turns removed"
    raw["steps"][1]["timestamp"] = "invalid-timestamp"
    item = scanned(raw)
    row = next(a for a in item["assessments"] if a["id"] == "integrity.timestamp_invalid")
    assert row["status"] == "match" and not row["complete"]
    assert row["id"] in item["coverage_gaps"]["telemetry"]
    assert row["id"] not in item["coverage_gaps"]["behavioural"]
