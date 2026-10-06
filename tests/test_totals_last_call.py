"""integrity.totals_are_last_call: final_metrics totals that are only the last model
call's usage (seen in indusagi runs cut off by a timeout or crash), so the run's token
and cost totals undercount. Synthetic traces only."""

from __future__ import annotations

from typing import TYPE_CHECKING

from atif_scan import Context, Status, builtin_detectors, parse_trace

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc

CHECK = "integrity.totals_are_last_call"


def check(trace: Doc):
    detector = next(d for d in builtin_detectors() if d.spec.id == CHECK)
    return detector.evaluate(parse_trace(trace), Context())


def trace(calls: list[tuple[int, int]], totals: tuple[int, int] | None) -> Doc:
    steps: list[Doc] = [{"source": "user", "message": "Fix /app."}]
    steps += [
        {
            "source": "agent",
            "message": "working",
            "metrics": {"prompt_tokens": p, "completion_tokens": c, "cached_tokens": 0},
        }
        for p, c in calls
    ]
    doc: Doc = {"schema_version": "ATIF-v1.6", "steps": steps}
    if totals is not None:
        doc["final_metrics"] = {
            "total_prompt_tokens": totals[0],
            "total_completion_tokens": totals[1],
        }
    return doc


CALLS = [(10_000, 400), (30_000, 900), (50_000, 30)]


def test_totals_equal_to_the_last_call_match_with_the_figures():
    result = check(trace(CALLS, (50_000, 30)))
    assert result.status == Status.MATCH and result.complete
    m = dict(result.measure)
    assert (m["prompt_tokens"], m["completion_tokens"]) == (50_000, 30)
    assert (m["step_prompt_tokens"], m["step_completion_tokens"]) == (90_000, 1_330)
    assert (m["last_prompt_tokens"], m["last_completion_tokens"]) == (50_000, 30)
    assert m["metered_steps"] == 3


def test_totals_that_sum_the_steps_are_clean():
    assert check(trace(CALLS, (90_000, 1_330))).status == Status.NO_MATCH


def test_a_single_call_is_its_own_total():
    result = check(trace([(50_000, 30)], (50_000, 30)))
    assert result.status == Status.NO_MATCH and result.complete


def test_rising_step_metrics_may_be_running_totals_so_unknown():
    """Each step's metrics at least the previous one's: they could be cumulative, where
    totals = last step is correct. Not a match, and not a clean result either."""
    rising = [(10_000, 100), (30_000, 900), (50_000, 1_000)]
    result = check(trace(rising, (50_000, 1_000)))
    assert result.status == Status.UNKNOWN
    assert [u.reason for u in result.unread] == ["step_metrics_may_be_cumulative"]


def test_without_totals_the_check_is_unknown():
    result = check(trace(CALLS, None))
    assert result.status == Status.UNKNOWN
    assert [u.reason for u in result.unread] == ["usage_not_recorded"]


def test_totals_matching_an_earlier_call_are_not_this_pattern():
    assert check(trace(CALLS, (30_000, 900))).status == Status.NO_MATCH


def test_output_ratio_uses_the_steps_when_totals_are_the_last_call():
    """Regression (indusagi timeouts): a whole run's text divided by one call's output
    tokens read as ~60 characters per token. The steps' own tokens are used instead."""
    doc = trace([(10_000, 100), (30_000, 120), (50_000, 30)], (50_000, 30))
    for step in doc["steps"][1:]:
        step["message"] = "x" * 400  # 1,200 characters over 250 step tokens: 4.8 per token
    detector = next(d for d in builtin_detectors() if d.spec.id == "integrity.output_token_ratio")
    result = detector.evaluate(parse_trace(doc), Context())
    assert result.status != Status.MATCH
    assert dict(result.measure)["output_tokens"] == 250
