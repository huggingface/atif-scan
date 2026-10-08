"""Synthetic private browser integration; no real traces or benchmark artifacts."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from atif_scan import CheckSpec, Context, Engine, RegexDetector, Severity
from atif_scan.browser.session import CACHE_SIZE, Session
from atif_scan.cli.scan import Scanner
from atif_scan.data.loader import load_trace
from atif_scan.sources.inputs import Source

if TYPE_CHECKING:
    from pathlib import Path

    from atif_scan.cli.inputs import Record
    from atif_scan.data.jsonval import Doc


def fixture(
    tmp_path: Path, *, label: str = "synthetic", message: str = "candidate"
) -> tuple[Record, Doc]:
    path = tmp_path / f"{label}.json"
    raw = {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {
                "step_id": 7,
                "source": "agent",
                "message": message,
                "tool_calls": [
                    {
                        "tool_call_id": "c1",
                        "function_name": "bash",
                        "arguments": {"command": "echo candidate"},
                    }
                ],
                "observation": {
                    "results": [{"source_call_id": "c1", "content": "synthetic result"}]
                },
            }
        ],
    }
    path.write_text(json.dumps(raw))
    source = Source(label, lambda: load_trace(path), local=path)
    context = Context(task="synthetic-task")
    engine = Engine([RegexDetector(CheckSpec("synthetic.check", Severity.HIGH, "1"), "candidate")])
    return (source, context), Scanner(engine, None).item(source, context)


def test_actual_item_shape_and_field_jumps(tmp_path: Path):
    record, item = fixture(tmp_path)
    item["citations"] = [{"text": "NOT EXPORTED"}]
    session = Session([record], [item], tmp_path / "feedback")
    summary = session.overview()["trials"][0]
    finding = summary["findings"][0]
    assert summary["id"] == "0"
    assert summary["task"] == "synthetic-task"
    assert finding["check_id"] == finding["title"] == "synthetic.check"
    assert finding["status"] == "match"
    assert finding["severity"] == "high"
    assert finding["feedback"] == {"verdict": "unreviewed", "note": ""}
    assert finding["locations"]
    assert "NOT EXPORTED" not in json.dumps(summary)
    for location in finding["locations"]:
        assert set(location) == {"step", "part", "index", "field"}
        assert location["step"] == 7
        assert "candidate" in session.segment("0", **location)["text"]
    trial = session.trial("0")
    assert trial["steps"][0]["role"] == "agent"
    assert {"part": "result", "index": 0, "field": 0} in trial["steps"][0]["fields"]


def test_credential_candidates_are_low_priority_and_still_masked(tmp_path: Path):
    from atif_scan.detectors.side_channel import CredentialExposure

    token = "sk-proj-syntheticCredential8274"
    record, _ = fixture(tmp_path, message=f"api_key={token}")
    detector = CredentialExposure()
    source, context = record
    item = Scanner(Engine([detector]), None).item(source, context)
    session = Session(
        [record],
        [item],
        tmp_path / "feedback",
        report={"checks": {detector.spec.id: {"title": detector.spec.title}}},
    )
    finding = session.overview()["trials"][0]["findings"][0]
    assert finding["severity"] == "low"
    assert finding["title"] == "Possible credential-like value"
    assert finding["check_version"] == "4"
    assert finding["status"] == "match"
    assert token not in session.segment("0", 7, "message")["text"]


def test_feedback_reload_does_not_clean_trial(tmp_path: Path):
    record, item = fixture(tmp_path)
    directory = tmp_path / "feedback"
    session = Session([record], [item], directory)
    before = session.overview()["trials"][0]
    value = session.save_feedback("0", "0", "false_positive", "reviewed synthetic case")
    after = Session([record], [item], directory).overview()["trials"][0]
    assert after["findings"][0]["feedback"] == value
    assert after["score"] == before["score"]
    assert after["severity"] == before["severity"]
    assert after["coverage"] == before["coverage"]
    assert after["findings"][0]["status"] == "match"
    # Returned documents cannot mutate internal evidence or feedback.
    after["findings"][0]["locations"].clear()
    assert session.overview()["trials"][0]["findings"][0]["locations"]


@pytest.mark.parametrize("change", ["version", "evidence", "input", "task", "trace"])
def test_feedback_is_bound_to_complete_identity(tmp_path: Path, change: str):
    record, item = fixture(tmp_path)
    directory = tmp_path / "feedback"
    Session([record], [item], directory).save_feedback("0", "0", "valid_signal", "")
    updated = deepcopy(item)
    if change == "version":
        updated["assessments"][0]["version"] = "2"
    elif change == "evidence":
        updated["assessments"][0]["evidence"][0]["span"] = [1, 2]
    elif change == "input":
        source = Source("different", record[0].load, local=record[0].local)
        record = (source, record[1])
        updated["input_id"] = "different"
    elif change == "task":
        updated["task"] = "different-task"
    else:
        path = record[0].local
        assert path is not None
        path.write_text(path.read_text() + "\n")
    reloaded = Session([record], [updated], directory).overview()["trials"][0]
    assert reloaded["findings"][0]["feedback"]["verdict"] == "unreviewed"


@pytest.mark.parametrize("operation", ["overview", "trial", "segment", "search", "save_feedback"])
def test_changed_source_rejected_even_when_cached(tmp_path: Path, operation: str):
    record, item = fixture(tmp_path)
    session = Session([record], [item], tmp_path / "feedback")
    session.trial("0")
    path = record[0].local
    assert path is not None
    path.write_text('{"steps":[]}')
    actions = {
        "overview": session.overview,
        "trial": lambda: session.trial("0"),
        "segment": lambda: session.segment("0", 7, "message"),
        "search": lambda: session.search("0", "candidate"),
        "save_feedback": lambda: session.save_feedback("0", "0", "unclear", ""),
    }
    with pytest.raises(ValueError, match="trace_source_changed"):
        actions[operation]()


def test_lazy_bounded_cache_retains_binding_after_eviction(tmp_path: Path):
    pairs = [fixture(tmp_path, label=f"trial-{i}") for i in range(CACHE_SIZE + 1)]
    session = Session([r for r, _ in pairs], [i for _, i in pairs], tmp_path / "feedback")
    assert not session._cache
    for i in range(len(pairs)):
        session.trial(str(i))
    assert len(session._cache) == CACHE_SIZE
    path = pairs[0][0][0].local
    assert path is not None
    path.unlink()
    with pytest.raises(ValueError, match="trace_source_changed"):
        session.trial("0")


def test_mask_before_paging_and_literal_search(tmp_path: Path):
    secret = "sk-" + "a" * 48
    message = f"prefix {secret} suffix [literal] " + "needle " * 41
    record, item = fixture(tmp_path, message=message)
    session = Session([record], [item], tmp_path / "feedback")
    full = session.segment("0", 7, "message")
    assert secret not in full["text"]
    assert secret[10:20] not in session.segment("0", 7, "message", offset=12)["text"]
    assert session.search("0", secret)["matches"] == []
    literal = session.search("0", "[literal]")
    assert len(literal["matches"]) == 1
    location = literal["matches"][0]
    assert session.segment("0", **location)["text"].startswith("[literal]")
    matches = session.search("0", "needle")
    assert len(matches["matches"]) == 40
    assert matches["truncated"] is True
    assert session.search("0", "not present") == {"matches": [], "truncated": False}


@pytest.mark.parametrize("query", ["", "x" * 151])
def test_query_limits(tmp_path: Path, query: str):
    record, item = fixture(tmp_path)
    session = Session([record], [item], tmp_path / "feedback")
    with pytest.raises(ValueError, match="invalid_search_query"):
        session.search("0", query)


def test_unavailable_unknown_is_not_clean_and_never_calls_loader(tmp_path: Path):
    def forbidden():
        raise AssertionError("must not load remote sources")

    source = Source("missing", forbidden)
    item: Doc = {
        "input_id": "missing",
        "input_status": "unavailable_or_invalid",
        "incomplete": True,
        "assessments": [
            {"id": "synthetic.check", "kind": "rule", "version": "1", "status": "unknown"}
        ],
    }
    session = Session([(source, Context())], [item], tmp_path / "feedback")
    trial = session.trial("0")
    assert trial["coverage"]["incomplete"] is True
    assert trial["steps"] == []
    assert trial["findings"][0]["status"] == "unknown"
    with pytest.raises(ValueError, match="trace_unavailable"):
        session.segment("0", 7, "message")


def test_bad_ids_and_indices(tmp_path: Path):
    record, item = fixture(tmp_path)
    session = Session([record], [item], tmp_path / "feedback")
    with pytest.raises(ValueError, match="unknown_trial"):
        session.trial("../synthetic.json")
    with pytest.raises(ValueError, match="unknown_finding"):
        session.save_feedback("0", "../0", "unclear", "")
    with pytest.raises(ValueError):
        session.segment("0", 7, "call", index=-1)
    with pytest.raises(ValueError):
        session.segment("0", 7, "unknown")
    with pytest.raises(ValueError, match="record_item_count_mismatch"):
        Session([], [item], tmp_path / "feedback")
    with pytest.raises(ValueError, match="record_item_mismatch"):
        Session([record], [{**item, "input_id": "wrong"}], tmp_path / "feedback")


@pytest.mark.parametrize("ancestor", [False, True])
def test_source_symlinks_rejected(tmp_path: Path, ancestor: bool):
    record, item = fixture(tmp_path)
    link = tmp_path / "link"
    if ancestor:
        link.symlink_to(tmp_path, target_is_directory=True)
        path = link / "synthetic.json"
    else:
        original = record[0].local
        assert original is not None
        link.symlink_to(original)
        path = link
    source = Source(record[0].label, record[0].load, local=path)
    with pytest.raises((OSError, ValueError)):
        Session([(source, record[1])], [item], tmp_path / "feedback")


def test_media_and_unknown_fields_remain_explicit(tmp_path: Path):
    record, item = fixture(tmp_path)
    path = record[0].local
    assert path is not None
    path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "step_id": 7,
                        "source": "agent",
                        "message": [{"type": "image", "source": {"url": "https://invalid.test"}}],
                        "reasoning_content": {"unknown": "not text"},
                    }
                ]
            }
        )
    )
    # Fresh report for the new synthetic trace.
    item = Scanner(Engine([]), None).item(record[0], record[1])
    session = Session([record], [item], tmp_path / "feedback")
    media = session.segment("0", 7, "message")
    assert media["media"] is True
    assert media["status"] == "media"
    unknown = session.segment("0", 7, "reasoning")
    assert unknown["understood"] is False
    assert unknown["status"] == "unreadable"


def test_exact_search_cap_is_not_truncated(tmp_path: Path):
    record, item = fixture(tmp_path, message="needle " * 40)
    session = Session([record], [item], tmp_path / "feedback")
    result = session.search("0", "needle")
    assert len(result["matches"]) == 40
    assert result["truncated"] is False


@pytest.mark.parametrize("initially_present", [False, True])
def test_invalid_or_missing_local_input_binding(tmp_path: Path, initially_present: bool):
    record, _ = fixture(tmp_path)
    path = record[0].local
    assert path is not None
    if initially_present:
        path.write_text("invalid JSON")
    else:
        path.unlink()
    item = Scanner(Engine([]), None).item(record[0], record[1])
    session = Session([record], [item], tmp_path / "feedback")
    assert session.trial("0")["steps"] == []
    assert session.overview()["trials"][0]["input_status"] == "unavailable_or_invalid"
    path.write_text('{"steps":[]}')
    with pytest.raises(ValueError, match="trace_source_changed"):
        session.overview()


def test_report_metadata_titles_and_feedback_context(tmp_path: Path):
    record, item = fixture(tmp_path)
    report = {
        "checks": {"synthetic.check": {"title": "Synthetic signal"}},
        "coverage": {"sync_failed_files": 2},
        "citations": ["NEVER A BROWSER OVERVIEW"],
    }
    session = Session([record], [item], tmp_path / "feedback", report=report)
    overview = session.overview()
    assert overview["report"] == {"coverage": {"sync_failed_files": 2}, "questions": {}}
    finding = overview["trials"][0]["findings"][0]
    assert finding["title"] == "Synthetic signal"
    session.save_feedback("0", finding["id"], "unclear", "synthetic review")
    row = json.loads((tmp_path / "feedback" / "feedback.jsonl").read_text())
    assert row["saved_at"]
    assert row["context"]["input_id"] == "synthetic"
    assert row["context"]["check_version"] == "1"
    assert row["context"]["check_id"] == "synthetic.check"
    assert len(row["context"]["trace_sha256"]) == 64
    assert str(tmp_path) not in json.dumps(row)


def test_digest_pins_available_and_absent_sources(tmp_path: Path):
    from atif_scan.browser.session import source_digests

    record, item = fixture(tmp_path)
    before = source_digests([record])
    assert before[0] is not None
    Session([record], [item], tmp_path / "feedback", expected_digests=before)
    assert record[0].local is not None
    record[0].local.unlink()
    assert source_digests([record]) == [None]
    unavailable = {**item, "input_status": "unavailable_or_invalid"}
    with pytest.raises(ValueError, match="trace_source_changed"):
        Session([record], [unavailable], tmp_path / "feedback", expected_digests=before)
