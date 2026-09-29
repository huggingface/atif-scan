"""Synthetic regressions for recorded results and review presentation."""

from atif_scan.brief import brief, brief_text, colourise


def item(label, statuses, reward=1):
    return {
        "input_id": label,
        "input_status": "available",
        "incomplete": False,
        "task": "demo",
        "reward": reward,
        "cost_usd": 0.5,
        "assessments": [
            {
                "id": f"demo.check_{i}",
                "kind": "detector",
                "status": status,
                "severity": "high",
                "score": 75 if status == "match" else None,
                "expected_by": [],
                "evidence": [],
            }
            for i, status in enumerate(statuses)
        ],
    }


def summary(*items, dq="high"):
    return brief({"scanner_version": "dev", "inputs": list(items), "coverage": {}}, dq=dq)


def test_recorded_result_and_unique_review_candidates():
    b = summary(item("one", ["match", "match"]), item("two", ["unknown"]))
    text = brief_text(b)
    result = next(line for line in text.splitlines() if line.startswith("RESULT"))
    assert "100.0%" in result and "→" not in result and "zeroed" not in result
    assert "if 1 flagged success is zeroed (not a verdict)" in text
    assert "1 unique rewarded DQ candidate(s) (threshold high+" in text
    assert "1 rewarded trial(s) with evidence gaps, not cleared" in text
    assert "review priority, not a verdict" in text and "overlap" not in text
    assert "--judge-prompts DIR" in text
    assert "✗" not in text
    assert "accuracy" not in text.split("ADJUSTMENTS", 1)[1]
    assert "COST       $1.00 reported" in text
    assert "COVERAGE" in text


def test_unknown_is_not_clean_and_threshold_is_explicit():
    text = brief_text(summary(item("one", ["unknown"]), dq="medium"))
    assert "REVIEW     0 unique rewarded DQ candidate(s) (threshold medium+" in text
    assert "1 rewarded trial(s) with evidence gaps" in text
    assert "SCENARIO" not in text
    assert "no DQ candidates" not in text


def test_review_metadata_shows_counts_and_only_generic_next_steps():
    b = summary(item("one", []))
    b["review"] = {
        "scope": "rewarded",
        "selected": 3,
        "written": 4,
        "unavailable": 1,
        "question_ids": ["hack_hunt"],
        "directory": "/private/synthetic-location",
    }
    text = brief_text(b)
    assert "4 prompt(s) written · 3 selected · 1 skipped/unavailable" in text
    assert "scope rewarded" in text
    assert "no provider calls made" in text
    assert "tools/ask-fast-agent.sh --model MODEL --questions DIR --inspect-tool --jobs 8" in text
    assert "rerun same inputs with --answers DIR" in text
    assert "/private" not in text
    assert colourise(text).plain == text
    styled = colourise(text)
    for label in ("RESULT", "REVIEW"):
        offset = text.index(label)
        assert any(span.start <= offset < span.end for span in styled.spans)


def test_clean_run_does_not_need_review_section():
    assert "REVIEW" not in brief_text(summary(item("one", [])))


def test_reported_score_does_not_repeat_scenario():
    b = summary(item("one", ["match"]))
    b["runs"] = [{"leaderboard": {"reported_accuracy": 80.0}}]
    text = brief_text(b)
    assert "REPORTED   80.0%" in text
    assert "scan-adjusted" not in text
    assert text.count("zeroed") == 1
    assert colourise(text).plain == text


def test_model_mismatch_and_gaps_are_not_double_counted():
    items = [item("a", []), item("b", []), item("c", ["unknown"])]
    for i, model in zip(items, ["main", "main", "other"], strict=True):
        i["model_name"] = model
    b = summary(*items)
    d = b["overview"]["disqualification"]
    assert d["candidate_ids"] == ["c"]
    assert d["rewarded_not_cleared"] == 0
    assert items[2]["assessments"][0]["status"] == "unknown"


def test_review_metadata_does_not_export_private_paths():
    doc = {
        "scanner_version": "dev",
        "inputs": [item("one", [])],
        "coverage": {},
        "review": {
            "scope": "rewarded",
            "selected": 1,
            "written": 1,
            "unavailable": 0,
            "question_ids": ["hack_hunt"],
            "directory": "/private/synthetic",
        },
    }
    assert "directory" not in brief(doc)["review"]
