"""Judge answers shown by the browser and viewer: allowlisted rows, never the reason."""

from __future__ import annotations

from atif_scan.browser.answers import answer_rows, question_labels

SECRET_REASON = "synthetic reason quoting trace text sk-proj-synthetic"


def item(*rows):
    return {"input_id": "synthetic", "answers": list(rows)}


HACK = {
    "question": "hack_hunt",
    "version": "9",
    "status": "answered",
    "answer": "attempted",
    "confidence": "medium",
    "mechanism": "benchmark_material",
    "steps": [4, 5],
    "reason": SECRET_REASON,
    "coverage": {"steps_read": 8, "share": 0.9, "basis": "steps", "reason": SECRET_REASON},
}
CLEAN_GUESS = {
    "question": "fabrication_hunt",
    "version": "5",
    "status": "answered",
    "answer": "absent",
    "confidence": "high",
    "steps": [2],
    "coverage": {"steps_read": 1, "share": 0.1, "basis": "steps"},
}
UNANSWERED = {"question": "history_probe", "version": "1", "status": "unanswered"}


def test_rows_keep_allowlisted_fields_and_never_the_reason():
    rows = answer_rows(item(CLEAN_GUESS, UNANSWERED, HACK))
    assert SECRET_REASON not in repr(rows)
    hack = rows[0]  # concerns first
    assert hack == {
        "question": "hack_hunt",
        "version": "9",
        "status": "answered",
        "answer": "attempted",
        "mechanism": "benchmark_material",
        "confidence": "medium",
        "steps": [4, 5],
        "read_share": 0.9,
        "thin": False,
        "concern": True,
    }
    guess = next(r for r in rows if r["question"] == "fabrication_hunt")
    assert guess["thin"] and not guess["concern"]  # a universal answer after reading 10%
    # Unanswered is shown as such, never as a negative.
    assert next(r for r in rows if r["question"] == "history_probe")["status"] == "unanswered"
    assert answer_rows({"input_id": "x"}) == []


def test_question_labels_come_from_the_catalogue():
    labels = question_labels([item(HACK, UNANSWERED), item({"question": "retired_id"})])
    assert set(labels) == {"hack_hunt", "history_probe"}
    assert labels["hack_hunt"]["universal"] == ["clean"]
    assert "other_run" in labels["history_probe"]["mechanisms"]
