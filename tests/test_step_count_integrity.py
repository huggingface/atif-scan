"""Synthetic canonical step-count and conservative notes regressions."""

import json

import pytest

from atif_scan import Context, Engine, Severity, Status, builtin_detectors, parse_trace, report
from atif_scan.data.jsonval import Doc

CHECK = "integrity.step_count_mismatch"


def detector():
    return next(d for d in builtin_detectors() if d.spec.id == CHECK)


def recording(count: object = 3) -> Doc:
    return {
        "steps": [
            {"source": "user", "message": "Synthetic request"},
            {"source": "agent", "message": "Synthetic reply"},
        ],
        "final_metrics": {"total_steps": count},
    }


@pytest.mark.parametrize("count", [0, 1, 3, 9])
def test_mismatch_preserves_canonical_count(count):
    trace = parse_trace(recording(count))
    result = detector().evaluate(trace, Context())
    assert trace.total_steps == count
    assert result.status == Status.MATCH and result.complete
    assert dict(result.measure) == {"declared_steps": count, "recorded_steps": 2}


@pytest.mark.parametrize("bad", [None, True, False, -1, 2.0, 2.5, "2", [], {}])
def test_invalid_count(bad):
    trace = parse_trace(recording(bad))
    assert trace.total_steps is None
    result = detector().evaluate(trace, Context())
    assert result.status == Status.UNKNOWN and not result.complete
    assert [u.reason for u in result.unread] == ["step_count_not_recorded"]


@pytest.mark.parametrize("metrics", [None, {}, [], "invalid", {"extra": {"total_steps": 2}}])
def test_missing_canonical_count(metrics):
    raw = recording()
    raw["final_metrics"] = metrics
    assert detector().evaluate(parse_trace(raw), Context()).status == Status.UNKNOWN
    del raw["final_metrics"]
    assert detector().evaluate(parse_trace(raw), Context()).status == Status.UNKNOWN


@pytest.mark.parametrize("notes", ["Unrelated synthetic note", "Children included", " ", "\n\t"])
@pytest.mark.parametrize("count", [2, 3])
def test_nonempty_notes_do_not_validate_scope(notes, count):
    raw = recording(count)
    raw["notes"] = notes
    trace = parse_trace(raw)
    assert trace.has_notes is True
    result = detector().evaluate(trace, Context())
    if count == 2:
        assert result.status == Status.NO_MATCH and result.complete
        assert not result.unread
    else:
        assert result.status == Status.UNKNOWN and not result.complete
        assert [u.reason for u in result.unread] == ["step_count_scope_documented"]


@pytest.mark.parametrize("notes", ["", None, True, 1, {}, ["scope"]])
def test_empty_or_nonstring_notes_do_not_apply_exception(notes):
    raw = recording()
    raw["notes"] = notes
    trace = parse_trace(raw)
    assert trace.has_notes is False
    assert detector().evaluate(trace, Context()).status == Status.MATCH


def test_children_and_nested_notes_do_not_change_root_comparison():
    raw = recording()
    raw["subagent_trajectories"] = [
        {"steps": [{"source": "agent", "message": "Child"}], "notes": "Child scope"}
    ]
    raw["extra"] = {"notes": "Nested scope"}
    raw["steps"][0]["notes"] = "Step scope"
    raw["final_metrics"]["notes"] = "Metrics scope"
    trace = parse_trace(raw)
    assert len(trace.steps) == 2 and not trace.has_notes
    assert detector().evaluate(trace, Context()).status == Status.MATCH
    raw["final_metrics"]["total_steps"] = 2
    assert detector().evaluate(parse_trace(raw), Context()).status == Status.NO_MATCH


def test_empty_recording_equal_zero():
    raw = recording(0)
    raw["steps"] = []
    assert detector().evaluate(parse_trace(raw), Context()).status == Status.NO_MATCH


def test_report_allowlist_and_telemetry_classification():
    check = detector()
    assert check.spec.severity == Severity.LOW
    assert check.spec.title == "Declared step count differs from recorded steps"
    for notes in ("private-notes-marker", ""):
        raw = recording()
        raw["notes"] = notes
        trace = parse_trace(raw)
        item = report(Engine([check]).evaluate(trace), trace.step_numbers)
        exported = json.dumps(item)
        assert "private-notes-marker" not in exported
        assert "has_notes" not in exported and '"notes"' not in exported
        assert "private-notes-marker" not in repr(trace)
        assert item["coverage_gaps"]["behavioural"] == []
        assert item["coverage_gaps"]["telemetry"] == ([CHECK] if notes else [])
