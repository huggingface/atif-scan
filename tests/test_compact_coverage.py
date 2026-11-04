"""Synthetic regressions for repeated coverage walls in citation-focused views."""

from __future__ import annotations

import io
import json

import pytest
from test_coverage_reporting import PRIVATE_TEXT, recording, scanned

from atif_scan import Severity, document, to_text
from atif_scan.output.coverage import COVERAGE_NOTICE, cause_line, coverage_causes
from atif_scan.output.document import filter_findings
from atif_scan.output.summary import summary, summary_text
from atif_scan.output.views import render_rich


def unreadable_recording():
    raw = recording()
    raw["steps"][1]["tool_calls"] = [
        {"tool_call_id": "c1", "function_name": "mystery_tool", "arguments": None}
    ]
    return raw


@pytest.mark.parametrize("rich", [False, True])
def test_large_gap_lists_are_compact_and_explained_once(rich):
    item = scanned(unreadable_recording())
    doc = document([item, dict(item, input_id="trial-2")], "test")
    shown = filter_findings(doc, Severity.INFO, checks=("awareness.*",), hide_empty=True)
    if rich:
        buffer = io.StringIO()
        render_rich(shown, file=buffer)
        text = buffer.getvalue()
        # Rich wraps long prose; normalize whitespace for the explanation.
        text = " ".join(text.split())
    else:
        text = to_text(shown)
    assert text.count(COVERAGE_NOTICE) == 1
    assert text.count("by family:") == 2
    assert "tool inputs not fully understood at least 1" in text
    assert "lookup.benchmark_source" not in text
    assert "unrecognized tool calls 1" in text
    assert PRIVATE_TEXT not in text
    assert len(shown["inputs"]) == 2  # no unknown trace is silently dropped
    assert shown["coverage"] == doc["coverage"]
    for before, after in zip(doc["inputs"], shown["inputs"], strict=True):
        assert after["coverage_gaps"] == before["coverage_gaps"]
        assert after["coverage_causes"] == before["coverage_causes"]
        assert (after["score"], after["severity"]) == (before["score"], before["severity"])
    assert "lookup.benchmark_source" in json.dumps(shown)


@pytest.mark.parametrize("legacy", [False, True])
def test_cause_locations_deduplicate_checks_and_survive_selection(legacy):
    item = scanned(unreadable_recording())
    assert len(item["coverage_gaps"]["behavioural"]) > 12
    assert item["coverage_causes"]["tool_inputs"] == 1
    if legacy:
        del item["coverage_causes"]
    shown = filter_findings(document([item], "test"), Severity.INFO, checks=("awareness.*",))
    assert shown["inputs"][0]["coverage_causes"]["tool_inputs"] == 1


def test_no_cause_is_inferred_from_unrecognized_count_or_missing_details():
    assert coverage_causes({"unrecognized_tool_calls": 40, "incomplete": True}) == {}
    item = scanned(recording())
    assert "tool_inputs" not in coverage_causes(item)


def test_only_known_located_causes_are_exported():
    item = {
        "assessments": [
            {
                "kind": "detector",
                "complete": False,
                "unread": [
                    {"reason": PRIVATE_TEXT, "step": 1},
                    {"reason": "unreadable"},
                    {"reason": "result_not_recorded", "step": 2, "call": 0},
                    {"reason": "result_not_recorded", "step": 2, "call": 0},
                ],
            },
            {
                "kind": "context",
                "complete": False,
                "unread": [{"reason": "undecidable", "step": 1}],
            },
        ]
    }
    assert coverage_causes(item) == {"result_not_recorded": 1}


def test_summary_does_not_repeat_per_trace_coverage_lists():
    item = scanned(unreadable_recording())
    doc = document([item, dict(item, input_id="trial-2")], "test")
    shown = filter_findings(doc, Severity.INFO, checks=("awareness.*",))
    rolled = summary(shown)
    text = summary_text(rolled)
    assert "agent steps" not in text
    assert "2 behavioural coverage incomplete" in text
    assert rolled["coverage"] == doc["coverage"]


def test_capped_diagnostics_are_lower_bounds_not_exact_counts():
    raw = unreadable_recording()
    raw["steps"][1]["tool_calls"] = [
        {"tool_call_id": f"c{i}", "function_name": "mystery_tool", "arguments": None}
        for i in range(25)
    ]
    item = scanned(raw)
    assert item["coverage_causes"]["tool_inputs"] == 10
    assert "tool inputs not fully understood at least 10" in cause_line(item)


def test_recorded_but_unreadable_result_does_not_claim_result_absent():
    raw = recording()
    raw["steps"][1]["tool_calls"] = [
        {
            "tool_call_id": "c1",
            "function_name": "bash",
            "arguments": {"command": "kubectl get pods -o json"},
        }
    ]
    raw["steps"][1]["observation"] = {
        "results": [{"source_call_id": "c1", "content": [{"type": "unsupported"}]}]
    }
    item = scanned(raw)
    assert item["coverage_causes"]["result_not_recorded"] == 1
    assert "tool results unavailable or unresolved at least 1" in cause_line(item)
    assert "tool results not recorded" not in cause_line(item)
