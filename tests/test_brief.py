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


def section(text, label):
    """One section of a brief: its label line and the lines under it, up to the blank
    line that ends it ("" when the section isn't there)."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.split(" ", 1)[0] == label), None)
    if start is None:
        return ""
    end = next((i for i in range(start + 1, len(lines)) if not lines[i]), len(lines))
    return "\n".join(lines[start:end])


def test_recorded_result_and_unique_review_candidates():
    b = summary(item("one", ["match", "match"]), item("two", ["unknown"]))
    text = brief_text(b)
    # The recorded result is stated as recorded; the scenario is a separate line.
    result = section(text, "SCORE").splitlines()[0]
    assert "100.0%" in result and "had failed" not in result and "if " not in result
    assert "50.0% ± 50.0 if the flagged rewarded trial had failed (a scenario, not a verdict)" in (
        section(text, "SCORE")
    )
    # Review candidates are counted once, per rewarded trial, with the threshold named.
    review = section(text, "REVIEW")
    assert "1 of 2 rewarded trials (50.0%) has a high or critical finding" in review
    assert "1 more rewarded trial can't be cleared, as their evidence is incomplete" in review
    assert "Findings set review priority, not verdicts." in text and "overlap" not in text
    assert "--judge-prompts DIR" in review
    assert "✗" not in text
    assert "accuracy" not in review
    assert "COST       $1.00 recorded" in text
    assert section(text, "EVIDENCE")


def test_unknown_is_not_clean_and_threshold_is_explicit():
    text = brief_text(summary(item("one", ["unknown"]), dq="medium"))
    review = section(text, "REVIEW")
    assert "REVIEW     ✓ no rewarded trial has a medium or higher finding" in review
    # Unknown evidence is not clean: the trial is still named as not cleared.
    assert "⚠ 1 rewarded trial can't be cleared, as their evidence is incomplete" in review
    assert "--judge-scope rewarded" in review
    assert "had failed" not in text
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
    assert "4 review prompts written for 3 selected trials (scope rewarded); 1 skipped" in text
    assert "scope rewarded" in text
    assert "nothing was sent anywhere" in text
    assert "tools/ask-fast-agent.sh --model MODEL --questions DIR" in text
    assert "--inspect-tool --jobs 8, then rerun with --answers DIR" in text
    assert "/private" not in text
    assert colourise(text).plain == text
    styled = colourise(text)
    for label in ("SCORE", "REVIEW"):
        offset = text.index(label)
        assert any(span.start <= offset < span.end for span in styled.spans)


def test_clean_run_does_not_need_review_section():
    # Nothing to review: REVIEW says so in one ✓ line, with no warning or next step.
    review = section(brief_text(summary(item("one", []))), "REVIEW")
    assert review == "REVIEW     ✓ no rewarded trial has a high or critical finding"


def test_reported_score_does_not_repeat_scenario():
    b = summary(item("one", ["match"]))
    b["runs"] = [{"leaderboard": {"reported_accuracy": 80.0}}]
    text = brief_text(b)
    assert "· the leaderboard reports 80.0%" in section(text, "SCORE")
    # A row without agent or model names: no empty gap where they would be.
    assert "RUN        Harbor job folder\n           leaderboard row · jobs unknown\n" in text
    assert "scan-adjusted" not in text
    assert text.count("had failed") == 1
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
