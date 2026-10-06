"""Synthetic web-gap navigation; never execute calls or fetch their targets."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from atif_scan import Context, Engine, builtin_detectors, parse_trace
from atif_scan.browser.session import Session
from atif_scan.browser.web import web_gap_details
from atif_scan.cli.scan import Scanner
from atif_scan.data.loader import load_trace
from atif_scan.data.web_gaps import web_gaps
from atif_scan.evidence.extract import read_segment
from atif_scan.sources.inputs import Source

if TYPE_CHECKING:
    from pathlib import Path

    from atif_scan.data.jsonval import Doc

CHECK = "integrity.web_results_not_recorded"


def call(key, arguments=None, name="web_search"):
    return {
        "tool_call_id": key,
        "function_name": name,
        "arguments": {"query": "synthetic-query"} if arguments is None else arguments,
    }


def result(key, content):
    return {"source_call_id": key, "content": content}


def raw_trace(calls, results):
    return {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {"step_id": 10, "source": "user", "message": "Synthetic task."},
            {
                "step_id": 42,
                "source": "agent",
                "tool_calls": calls,
                "observation": {"results": results},
            },
        ],
    }


def desk(tmp_path: Path, raw: Doc) -> tuple[Session, Doc]:
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(raw))
    source = Source("synthetic", lambda: load_trace(path), local=path)
    context = Context(task="synthetic-task")
    detector = next(c for c in builtin_detectors() if c.spec.id == CHECK)
    item = Scanner(Engine([detector]), None).item(source, context)
    return Session([(source, context)], [item], tmp_path / "feedback"), item


def test_each_missing_call_has_a_distinct_locator_and_gap_reason(tmp_path: Path):
    raw = raw_trace(
        [call("ok"), call("missing"), call("status"), call("empty")],
        [result("ok", "Synthetic body"), result("status", "completed"), result("empty", "")],
    )
    session, item = desk(tmp_path, raw)
    evidence = item["assessments"][0]["evidence"]
    assert [e["call"] for e in evidence] == [1, 2, 3]
    assert all(e["step_id"] == 42 and e["channel"] == "metadata" for e in evidence)
    assert item["assessments"][0]["version"] == "3"
    trial = session.trial("0")
    gaps = trial["findings"][0]["web_gaps"]
    assert [(g["call"], g["reason"]) for g in gaps] == [
        (1, "no_linked_result"),
        (2, "status_only"),
        (3, "unusable_content"),
    ]
    assert gaps[0]["result_location"] is None
    assert [g["result_location"]["index"] for g in gaps[1:]] == [1, 2]
    for gap in gaps:
        target = gap["call_location"]
        assert target["step"] == 42 and target["index"] == gap["call"]
        assert session.segment("0", **target)["text"] == "synthetic-query"
    # Existing counts and status classification remain call-based and unchanged.
    assert web_gaps(parse_trace(raw)).calls == 3
    assert item["assessments"][0]["severity"] == "low"


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("status: OK", "status_only"),
        ("Script completed\nWall time 1 seconds\nOutput:\n", "status_only"),
        ("", "unusable_content"),
        (None, "unusable_content"),
        ({"unsupported": "synthetic"}, "unusable_content"),
        (
            [{"type": "image_url", "image_url": {"url": "https://example.org/a.png"}}],
            "unusable_content",
        ),
    ],
)
def test_linked_gap_results_are_inspectable(tmp_path: Path, content, reason):
    session, _ = desk(tmp_path, raw_trace([call("w")], [result("w", content)]))
    gap = session.trial("0")["findings"][0]["web_gaps"][0]
    assert gap["reason"] == reason
    assert gap["result_location"]["index"] == 0
    page = session.segment("0", **gap["result_location"])
    assert page["part"] == "result"
    assert page["status"] in ("text", "empty", "unreadable", "media", "media_partial")


@pytest.mark.parametrize("content", ["HTTP 403 Forbidden", "Error: retrieval failed", "Page body"])
def test_explicit_errors_and_content_are_not_missing_results(tmp_path: Path, content):
    session, item = desk(tmp_path, raw_trace([call("w")], [result("w", content)]))
    assert item["assessments"][0]["status"] == "no_match"
    assert session.trial("0")["findings"] == []


def test_multipart_result_points_only_to_unusable_siblings(tmp_path: Path):
    session, _ = desk(
        tmp_path,
        raw_trace(
            [call("w")],
            [result("w", "Page body"), result("w", "pending"), result("w", None)],
        ),
    )
    gaps = session.trial("0")["findings"][0]["web_gaps"]
    assert [g["result_location"]["index"] for g in gaps] == [1, 2]
    assert [g["reason"] for g in gaps] == ["status_only", "unusable_content"]


def test_call_without_arguments_has_real_metadata_anchor_not_invented_text(tmp_path: Path):
    raw = raw_trace([call("w", {})], [])
    session, _ = desk(tmp_path, raw)
    trial = session.trial("0")
    gap = trial["findings"][0]["web_gaps"][0]
    assert gap["call_location"] == {"step": 42, "part": "call_info", "index": 0, "field": 0}
    assert {"part": "call_info", "index": 0, "field": 0} in trial["steps"][1]["fields"]
    page = session.segment("0", **gap["call_location"])
    assert page["metadata_only"] is True
    assert "Inspectable argument fields: 0." in page["text"]
    assert "not been recovered" in page["text"]
    for index, field in [(99, 0), (0, 1)]:
        with pytest.raises(ValueError):
            read_segment(parse_trace(raw), 42, "call_info", index=index, field=field)


def test_gap_details_do_not_export_queries_and_call_view_still_masks(tmp_path: Path):
    token = "sk-proj-syntheticCredential8274"
    raw = raw_trace([call("w", {"query": f"api_key={token}"})], [])
    session, _ = desk(tmp_path, raw)
    trial = session.trial("0")
    assert token not in json.dumps(trial)
    target = trial["findings"][0]["web_gaps"][0]["call_location"]
    assert token not in session.segment("0", **target)["text"]


def test_old_step_only_or_compacted_history_does_not_invent_a_call():
    trace = parse_trace(raw_trace([call("w")], []))
    assert web_gap_details(trace, [{"step": 42, "part": "message", "index": 0, "field": 0}]) == []
    assert web_gap_details(trace, [{"step": 999, "part": "call_info", "index": 0}]) == []
    assert web_gap_details(trace, [{"step": 42, "part": "call_info", "index": 99}]) == []


def test_code_mode_derived_call_follows_parent_result(tmp_path: Path):
    program = "const r = await tools.web__run({open: [{ref_id: 'https://example.org'}]});"
    raw = raw_trace([call("program", {"input": program}, "exec")], [result("program", "pending")])
    session, _ = desk(tmp_path, raw)
    gaps = session.trial("0")["findings"][0]["web_gaps"]
    assert gaps
    assert all(g["call"] > 0 and g["reason"] == "status_only" for g in gaps)
    assert all(g["result_location"]["index"] == 0 for g in gaps)


def test_inferred_pairing_warning_survives_navigation(tmp_path: Path):
    raw = raw_trace(
        [call("w"), call("b", {"command": "printf synthetic"}, "bash")],
        [{"content": "pending"}, result("b", "synthetic explicit")],
    )
    session, _ = desk(tmp_path, raw)
    gap = session.trial("0")["findings"][0]["web_gaps"][0]
    assert gap["pairing_reconstructed"] is True
    page = session.segment("0", **gap["result_location"])
    assert page["pairing_reconstructed"] is True


def test_unlinked_result_is_not_arbitrarily_attached_to_a_gap(tmp_path: Path):
    raw = raw_trace(
        [call("a"), call("b")],
        [result("foreign", "Some unrelated page body")],
    )
    session, _ = desk(tmp_path, raw)
    gaps = session.trial("0")["findings"][0]["web_gaps"]
    assert len(gaps) == 2
    assert all(g["reason"] == "no_linked_result" and g["result_location"] is None for g in gaps)
