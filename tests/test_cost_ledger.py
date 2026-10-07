"""The COST ledger and the TOKENS wording for calls without usage: every amount beyond
the recorded cost is a row saying why it is attributed. Synthetic rows only, shaped on
a fast-agent run billed through a subscription provider (cost $0 per call, the
harness's price-derived estimate recorded) whose provider failed a few image requests
before any output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from atif_scan.data.loader import parse_trace
from atif_scan.output.brief import brief, brief_text

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc


def attempt(**fields):
    return {"attempt": 1, "error_type": "InternalServerError", "reason": "provider_error", **fields}


def retried_step(*attempts):
    return {
        "source": "agent",
        "message": "working",
        "metrics": {"prompt_tokens": 100, "completion_tokens": 10},
        "extra": {
            "retry": {
                "schema": "fast-agent.retry/v1",
                "provider_attempts": len(attempts) + 1,
                "retries": list(attempts),
            }
        },
    }


def test_failed_before_output_needs_no_events_or_an_http_refusal():
    refused = attempt(error_message="Error code: 503 - synthetic", stream_events_received=None)
    silent = attempt(error_message="socket closed", stream_events_received=0)
    streamed = attempt(error_message="Error code: 500 - synthetic", stream_events_received=2)
    unknown = attempt(error_message="socket closed", stream_events_received=None)
    trace = parse_trace({"steps": [retried_step(refused, silent, streamed, unknown)]})
    assert trace.stream_retry_attempts == 4
    assert trace.retry_before_output == 2  # refused + silent; streamed and unknown are not
    assert trace.retry_statuses == (500, 503)
    # Only the status number is kept: the provider's message is never stored.
    assert "synthetic" not in repr(trace.retry_statuses)


def observed_row(n: int, cost: float, calls: int = 50, missed: int = 0) -> Doc:
    """A trial with fast-agent observed accounting: no bill, the harness's estimate."""
    return {
        "input_id": f"trial-{n}",
        "input_status": "available",
        "task": "synthetic",
        "reward": 1.0,
        "usage_basis": "run_observed" if missed else "run",
        "input_tokens": 200_000,
        "cache_tokens": 150_000,
        "output_tokens": 5_000,
        "llm_calls": calls,
        "calls_without_usage": missed,
        "stream_retry_attempts": missed,
        "retry_before_output": missed,
        "retry_statuses": [503] if missed else None,
        "chars_per_output_token": 2.0,
        "output_ratio_basis": "answer_only",
        "cost_usd": None,
        "accounting": {"cost_scope": "observed", "estimated_observed_cost_usd": cost},
        "assessments": [],
        "incomplete": False,
    }


def render(rows):
    b = brief({"scanner_version": "dev", "inputs": rows, "coverage": {}})
    return b, " ".join(brief_text(b).split())


def test_calls_failed_before_output_are_one_tokens_line_and_a_bounded_ledger_row():
    rows = [observed_row(n, 1.0) for n in range(6)]
    rows += [observed_row(n, 1.0, calls=51, missed=1) for n in range(6, 8)]
    b, text = render(rows)
    # TOKENS: one line says which trials, what is missing and why; no second retry line.
    assert (
        "⚠ 2 trials (all rewarded, none errored) miss usage for 1 model call each: the"
        " provider failed them before any output (HTTP 503) and the harness retried; their"
        " token totals are lower bounds"
    ) in text
    assert "had fast-agent provider failures" not in text
    # COST: the harness's estimate, the failed calls as a bound (1 call at $1/50 each),
    # and the total with its upper end.
    assert "no trial records a billed cost: the amounts below are estimates" in text
    assert "$8.00 8 trials: recorded tokens at list prices (the harness's estimate" in text
    assert (
        "+ ≤$0.04 2 trials: model calls without usage that failed before any output"
        " (HTTP 503): likely unbilled (at most each trial's own cost per call)"
    ) in text
    assert "= $8.00 estimated total, at most $8.04" in text
    ledger = b["cost_ledger"]
    assert [r["kind"] for r in ledger["rows"]] == ["observed_estimate", "failed_before_output"]
    assert ledger["rows"][1]["bound"] and ledger["total_usd"] == 8.0
    assert ledger["upper_usd"] == 8.04 and ledger["complete"]


def test_calls_lost_mid_stream_are_an_estimate_not_a_bound():
    rows = [observed_row(n, 1.0) for n in range(4)]
    mid = observed_row(4, 1.0, calls=51, missed=1)
    mid["retry_before_output"], mid["retry_statuses"] = 0, None
    _, text = render([*rows, mid])
    assert "+ $0.02 1 trial: model calls without recorded usage" in text
    assert "= $5.02 estimated total" in text and "at most" not in text
    assert "failed before any output" not in text


def test_without_additions_the_recorded_cost_stays_one_line():
    rows = [{**observed_row(n, 1.0), "cost_usd": 1.0, "accounting": None} for n in range(3)]
    b, text = render(rows)
    assert not b["cost_ledger"]["shown"]
    assert "COST $3.00 recorded · ✓ every trial with usage has a cost" in text
    assert "estimated total" not in text


def test_a_reported_partial_cost_is_kept_without_an_estimate():
    """Regression: with a harness-reported partial cost and no price-derived estimate,
    no ledger is drawn, and the reported amount must still be shown (not added)."""
    row = observed_row(0, 1.0)
    row["accounting"] = {"cost_scope": "observed", "reported_cost_usd": 0.5}
    _, text = render([row])
    assert "$0.50 partial reported observed cost (not a final bill)" in text


def test_the_ratio_spread_names_the_trials_left_out():
    rows = [observed_row(n, 1.0) for n in range(4)]
    stopped = {**observed_row(4, 1.0), "output_tokens": 90, "error_type": "AgentSafetyStopError"}
    silent = {**observed_row(5, 1.0), "chars_per_output_token": None, "output_ratio_basis": None}
    _, text = render([*rows, stopped, silent])
    assert "median 2.00 characters over 5 traces" not in text
    assert "over 4 traces" in text
    assert (
        "not in that spread: 2 trials with little or no agent output (under 1,000 output"
        " tokens); 1 of them errored: AgentSafetyStopError 1"
    ) in text


def test_each_ledger_row_counts_its_own_trials():
    """Regression: with calls lost both ways, each row named the whole group (3 trials
    twice). And step-sum trials, which TOKENS describes on their own line, aren't
    counted in the observed-bounds line."""
    rows = [observed_row(n, 1.0) for n in range(4)]
    rows.append(observed_row(4, 1.0, calls=51, missed=1))  # failed before output
    mid = observed_row(5, 1.0, calls=51, missed=1)
    mid["retry_before_output"], mid["retry_statuses"] = 0, None  # lost mid-stream
    partial = observed_row(6, 1.0, calls=51, missed=1)
    partial["usage_basis"] = "steps_partial"
    partial["retry_before_output"], partial["retry_statuses"] = 0, None
    b, text = render([*rows, mid, partial])
    kinds = {r["kind"]: r["trials"] for r in b["cost_ledger"]["rows"]}
    assert kinds["calls_without_usage"] == 2 and kinds["failed_before_output"] == 1
    assert (
        "⚠ 2 trials (all rewarded, none errored) miss usage for 1 model call each; 1 failed"
        " before any output (HTTP 503); their token totals are lower bounds"
    ) in text
