"""Synthetic action envelopes only; report exports must remain counts-only."""

import json
from types import MappingProxyType

import pytest

from atif_scan.model import Content, Step, ToolCall, Trace
from atif_scan.output.brief import brief, brief_text
from atif_scan.output.document import RECORDING_OVERLAP, document, recording_gaps, report
from atif_scan.web_activity import WebActivity, call_activity, web_activity


def call(args=None, name="web_search_call", tool="web_search", call_id="w", parent=None):
    return ToolCall(0, tool, (), call_id, name, args, parent)


def trace(*calls):
    return Trace(None, (Step(0, "agent", False, Content(), Content(), calls),))


@pytest.mark.parametrize(
    ("args", "known", "unknown"),
    [
        ({"query": "synthetic private query"}, 1, 0),
        ({"queries": ["one", "two"]}, 2, 0),
        ({"query": "one", "queries": ["one", "two"]}, 2, 0),
        ({"query": "three", "queries": ("one", "two")}, 3, 0),
        ({"query": "one", "queries": ("one", "one")}, 2, 0),
        ({}, 0, 1),
        ({"query": False}, 0, 1),
        ({"query": ["one"]}, 0, 1),
        ({"queries": []}, 0, 1),
        ({"queries": "one"}, 0, 1),
        ({"queries": ("one", None, 1)}, 1, 1),
        ({"query": None, "queries": ("one",)}, 1, 1),
        ({"query": "one", "queries": None}, 1, 1),
    ],
)
def test_hosted_queries_are_known_lower_bound(args, known, unknown):
    result = call_activity(call(MappingProxyType({"action_type": "search", **args})))
    assert result.searches == 1
    assert result.known_queries == known
    assert result.unknown_query_actions == unknown


def test_find_open_and_backend_requests_do_not_count_as_searches():
    result = web_activity(
        trace(
            call({"action_type": "find_in_page", "query": "not a search"}),
            call({"action_type": "open_page", "url": "https://synthetic.invalid/private"}),
            call({"query": "ignored"}, name="provider_request", tool="other"),
        )
    )
    assert result == WebActivity(opens=1, finds=1)


def test_unknown_hosted_action_does_not_fall_back_to_search():
    assert call_activity(call(None)) == WebActivity(unknown_actions=1)
    assert call_activity(call({"action_type": ["search"]})) == WebActivity(unknown_actions=1)
    assert call_activity(call({}, name="web__run")) == WebActivity(unknown_actions=1)


def test_native_fetch_and_search_fallback():
    result = web_activity(
        trace(
            call(None, name="fetch", tool="web_fetch"),
            call(None, name="web_search"),
            call({"queries": ("a", "b")}, name="WebSearch"),
        )
    )
    assert result == WebActivity(searches=2, opens=1, known_queries=2, unknown_query_actions=1)


def test_structured_batch_and_nested_action():
    result = web_activity(
        trace(
            call(
                MappingProxyType(
                    {
                        "search_query": (MappingProxyType({"q": "one"}), {"q": False}),
                        "open": ({"ref_id": "private"},),
                        "find": ({"pattern": "private"},),
                        "backend_requests": ({"query": "ignored"},),
                    }
                ),
                name="web__run",
            ),
            call({"action": {"type": "search", "query": "two"}}),
        )
    )
    assert result == WebActivity(
        searches=3,
        opens=1,
        finds=1,
        known_queries=2,
        unknown_query_actions=1,
    )
    assert call_activity(call({"search_query": False}, name="web__run")) == WebActivity(
        searches=1,
        unknown_query_actions=1,
    )


def test_derived_alias_not_double_counted_but_code_mode_is_counted():
    assert web_activity(
        trace(
            call({"action_type": "search", "query": "one"}),
            call({"query": "one"}, name="web_search", call_id="alias", parent="w"),
        )
    ) == WebActivity(searches=1, known_queries=1)
    assert web_activity(
        trace(
            call({}, name="exec", tool="inert", call_id="program"),
            call({"query": "one"}, name="web_search", parent="program"),
        )
    ) == WebActivity(searches=1, known_queries=1)


def test_absent_trace_is_unknown_not_zero():
    assert web_activity(None) is None
    assert report(())["web_activity"] is None
    assert web_activity(trace()) == WebActivity()


def item(label, gaps=(), activity=None):
    row = report((), web_activity=activity)
    row.update(
        input_id=label,
        input_status="available",
        task="synthetic",
        reward=1,
        assessments=[
            {
                "id": check,
                "kind": "detector",
                "status": "match",
                "severity": "low",
                "score": 25,
                "complete": False,
                "evidence": [],
                "expected_by": [],
            }
            for check in gaps
        ],
    )
    row.pop("coverage_gaps")
    row.pop("recording_gaps")
    return row


def test_overlapping_gaps_are_one_union_warning_with_subgroups():
    reconstructed, unresolved, missing = RECORDING_OVERLAP.values()
    rows = [
        item("a", (reconstructed, missing)),
        item("b", (unresolved, missing)),
        item("c", (missing,)),
        item("d"),
    ]
    assert recording_gaps(rows) == {
        "trials": 3,
        "pairing_reconstructed": 1,
        "pairing_unresolved": 1,
        "web_results_not_recorded": 3,
    }
    doc = document(rows, "test")
    assert doc["coverage"]["recording_gaps"]["trials"] == 3
    b = brief(doc)
    text = " ".join(brief_text(b).split())
    assert "3 trials with recording gaps" in text
    assert "1 pairing links inferred" in text and "1 pairing unresolved" in text
    assert "3 web outputs missing" in text and "subgroups may overlap" in text
    assert "inferred links do not recover missing content" in text
    assert "source exposure is unknown in 3 trials" in text


def test_no_visible_awareness_is_not_a_clean_claim_with_missing_output():
    row = item("a", (RECORDING_OVERLAP["web_results_not_recorded"],))
    row["assessments"].append(
        {
            "id": "awareness.benchmark",
            "kind": "detector",
            "status": "no_match",
            "severity": "low",
            "score": None,
            "complete": True,
            "evidence": [],
            "expected_by": [],
        }
    )
    text = brief_text(brief(document([row], "test")))
    assert "no visible benchmark awareness" in text
    assert "web output evidence unavailable" in text
    assert "✓ no scanned trial shows benchmark awareness" not in text


def test_web_export_aggregate_and_no_raw_values():
    private = "synthetic-private-query"
    activity = web_activity(
        trace(
            call(
                {
                    "action_type": "search",
                    "query": private,
                    "url": "https://synthetic.invalid/private",
                    "entities": ["private-entity"],
                }
            )
        )
    )
    rows = [item("a", activity=activity), item("b")]
    doc = document(rows, "test")
    b = brief(doc)
    assert b["web_activity"] == {
        "traces_known": 1,
        "traces_unknown": 1,
        "searches": 1,
        "opens": 0,
        "finds": 0,
        "known_queries": 1,
        "unknown_query_actions": 0,
        "unknown_actions": 0,
    }
    encoded = json.dumps([doc, b]) + brief_text(b)
    assert private not in encoded
    assert "synthetic.invalid" not in encoded and "private-entity" not in encoded


def test_invalid_aggregate_activity_is_unknown_and_not_summed():
    row = item("a", activity=WebActivity(searches=1, known_queries=1))
    row["web_activity"]["searches"] = True
    row["web_activity"]["query"] = "private"
    result = brief(document([row], "test"))["web_activity"]
    assert result["traces_known"] == 0 and result["traces_unknown"] == 1
    assert result["searches"] == 0 and result["known_queries"] == 0
    assert "query" not in result


def test_full_scan_overlap_counts_survive_finding_filter():
    from atif_scan.checks import Severity
    from atif_scan.output.document import filter_findings

    row = item("a", tuple(RECORDING_OVERLAP.values()))
    doc = filter_findings(document([row], "test"), Severity.HIGH)
    assert not doc["inputs"][0]["assessments"]
    assert recording_gaps(doc["inputs"]) == {
        "trials": 1,
        "pairing_reconstructed": 1,
        "pairing_unresolved": 1,
        "web_results_not_recorded": 1,
    }
    assert brief(doc)["recording_gaps"]["trials"] == 1


def test_copied_and_non_agent_web_calls_are_not_counted():
    c = call({"action_type": "search", "query": "private"})
    t = Trace(
        None,
        (
            Step(0, "agent", True, Content(), Content(), (c,)),
            Step(1, "user", False, Content(), Content(), (c,)),
        ),
    )
    assert web_activity(t) == WebActivity()


def test_cli_exports_recorded_web_counts_without_query_text(tmp_path, capsys):
    from atif_scan.cli import main

    path = tmp_path / "synthetic.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "steps": [
                    {"source": "user", "message": "Synthetic documentation task."},
                    {
                        "source": "agent",
                        "tool_calls": [
                            {
                                "tool_call_id": "",
                                "function_name": "web_search_call",
                                "arguments": {
                                    "action_type": "search",
                                    "query": "synthetic-private-primary",
                                    "queries": [
                                        "synthetic-private-primary",
                                        "synthetic-private-second",
                                    ],
                                },
                            }
                        ],
                        "observation": {"results": [{}]},
                    },
                ],
            }
        )
    )
    assert main([str(path), "--format", "json", "--no-cache"]) == 0
    output = capsys.readouterr()
    assert "synthetic-private" not in output.out + output.err
    row = json.loads(output.out)["inputs"][0]
    assert row["web_activity"] == {
        "searches": 1,
        "opens": 0,
        "finds": 0,
        "known_queries": 2,
        "unknown_query_actions": 0,
        "unknown_actions": 0,
    }
    by_id = {a["id"]: a for a in row["assessments"]}
    assert by_id["integrity.web_results_not_recorded"]["status"] == "match"
    assert by_id["lookup.search_surfaced_benchmark"]["status"] == "unknown"


def test_empty_parent_ids_do_not_suppress_unrelated_derived_searches():
    activity = web_activity(
        trace(
            call({"action_type": "search", "query": "one"}, call_id=""),
            call({}, name="exec", tool="inert", call_id=""),
            call({"query": "two"}, name="web_search", call_id="#0", parent=""),
        )
    )
    assert activity == WebActivity(searches=2, known_queries=2)


def test_all_unavailable_activity_is_explicit_without_fabricated_zero_actions():
    text = brief_text(brief(document([item("a"), item("b")], "test")))
    assert "web activity unavailable in 2 traces" in text
    assert "recorded actions: 0 searches" not in text
