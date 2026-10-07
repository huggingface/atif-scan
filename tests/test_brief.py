"""Synthetic regressions for recorded results and review presentation."""

from atif_scan.output.brief import brief, brief_text, colourise


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
    assert "--questions DIR" in review
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
    assert "--question-scope rewarded" in review
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
    assert "atif-scan hunt --model MODEL --questions DIR" in text
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


def test_benchmark_awareness_is_counted_at_any_priority():
    # Awareness checks are mostly low/info, so the medium+ findings table never shows
    # them; the brief counts each stage of the funnel, with rewarded trials.
    def trial(label, matched, reward=1, excused=()):
        return {
            "input_id": label,
            "input_status": "available",
            "incomplete": False,
            "task": "demo",
            "reward": reward,
            "assessments": [
                {
                    "id": check,
                    "kind": "detector",
                    "status": "match",
                    "severity": "low",
                    # An excused finding carries no score (the engine's Assessment.counts).
                    "score": None if check in excused else 25,
                    "expected_by": ["allow.demo"] if check in excused else [],
                    "evidence": [],
                }
                for check in matched
            ],
        }

    items = [
        trial("a", ["awareness.benchmark", "awareness.named_benchmark", "lookup.benchmark_source"]),
        trial("b", ["awareness.benchmark"], reward=0),
        trial("c", ["awareness.verifier"]),  # talk about tests alone isn't awareness
        trial("d", ["awareness.benchmark"], excused=("awareness.benchmark",)),  # expected
    ]
    b = summary(*items)
    aw = b["awareness"]
    rows = {s["stage"]: (s["trials"], s["rewarded"]) for s in aw["stages"]}
    assert rows == {"noticed": (2, 1), "named": (1, 1), "searched": (1, 1)}
    assert "obtained" not in rows  # its checks didn't run: left out, not shown as 0
    assert aw["trials"] == 2 and aw["verifier_talk"]["trials"] == 1
    text = " ".join(brief_text(b).split())
    assert "AWARENESS 2 trials of 4 scanned (50.0%) show benchmark awareness" in text
    assert "2 1 remarked on being benchmarked" in text
    assert "1 1 looked the benchmark up" in text
    assert "1 trial (1 rewarded) talked about hidden tests or the verifier" in text


def _covered(styled, start, end, style):
    return any(
        span.start <= start and end <= span.end and style in str(span.style).split()
        for span in styled.spans
    )


def _every(styled, needle, style):
    """Every occurrence of `needle` is covered by one span of `style`."""
    plain, at, found = styled.plain, 0, False
    while (found_at := plain.find(needle, at)) >= 0:
        found = True
        if not _covered(styled, found_at, found_at + len(needle), style):
            return False
        at = found_at + len(needle)
    return found


def test_wrapped_context_line_stays_dim():
    """A · line stays dim on the rows wrap() continues under the mark.

    The row pattern only matches the line that starts with ·, so the continuation
    "traces, reasoning excluded (its tokens are reported separately)" rendered in
    the default colour. A wrapped ✓ line is not dimmed with it.
    """
    from atif_scan.output.brief_view.words import INFO, OK, wrap

    phrase = "reasoning excluded (its tokens are reported separately)"
    context = (
        f"{INFO} visible agent text per output token: median 3.02 characters"
        f" (p5-p95 2.58-3.71) over 434 traces, {phrase}"
    )
    longer = f"{INFO} " + "context stays dim across every wrapped row " * 6
    checked = f"{OK} " + "recorded totals match the trajectories' own " * 3
    text = (
        "\n".join(
            wrap(
                "TOKENS",
                [
                    "255.1M input (243.6M cached) · 1.7M output, from 434 of 434 trials",
                    context,
                    longer,
                    checked,
                ],
            )
        )
        + "\n"
    )
    styled = colourise(text)
    assert styled.plain == text
    # The explanation sits on its own row, not on the · row, and is still dim.
    assert any(phrase in line and "·" not in line for line in text.splitlines())
    assert _every(styled, phrase, "dim")
    assert sum("context stays dim" in line for line in text.splitlines()) >= 3
    assert _every(styled, "context stays dim across every wrapped row", "dim")
    check_wraps = [
        line for line in text.splitlines() if "recorded totals" in line and "✓" not in line
    ]
    assert check_wraps
    for row in check_wraps:
        at = text.index(row)
        assert not _covered(styled, at, at + len(row), "dim")


def test_sections_are_the_text_brief_unwrapped():
    from atif_scan.output.brief_view import sections

    b = summary(item("one", ["match", "match"]), item("two", ["unknown"]))
    text = brief_text(b)
    flat = " ".join(text.split())
    collected = sections(b)
    starts = {line.split()[0] for line in text.splitlines() if line[:1].isupper()}
    labels = [s["label"] for s in collected]
    assert labels and set(labels) <= starts
    assert "MORE" in starts and "MORE" not in labels
    for s in collected:
        for line in s["lines"]:
            assert " ".join(line["text"].split()) in flat
    assert brief_text(b) == text  # collecting doesn't change the text brief


def test_context_compactions_are_context_not_a_defect():
    one, two = item("one", []), item("two", [])
    one["context_compactions"] = 2
    text = brief_text(summary(one, two))
    evidence = section(text, "EVIDENCE")
    assert "agent context compacted in 1 trial (2 compactions)" in " ".join(evidence.split())
    assert "no recording defect detected" in evidence
