"""Accounting checks report what they compared (numbers and codes only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from atif_scan import Context, Status, builtin_detectors, parse_trace
from atif_scan.browser.findings import measure
from atif_scan.checks import Detection

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc


def check(check_id: str, trace: Doc):
    detector = next(d for d in builtin_detectors() if d.spec.id == check_id)
    return detector.evaluate(parse_trace(trace), Context())


def trace(final_metrics: Doc | None, message: str = "Done. " * 50) -> Doc:
    doc = {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {"source": "user", "message": "Fix /app."},
            {"source": "agent", "message": message, "llm_call_count": 1},
        ],
    }
    if final_metrics is not None:
        doc["final_metrics"] = final_metrics
    return doc


TOTALS = {"total_prompt_tokens": 40_103, "total_completion_tokens": 100}


def test_cost_missing_reports_the_totals_it_read():
    result = check("integrity.cost_missing", trace(TOTALS))
    assert result.status == Status.MATCH
    assert dict(result.measure) == {
        "prompt_tokens": 40_103,
        "completion_tokens": 100,
        "cost_usd": None,
    }


def test_tokens_per_call_is_reported_with_its_limit():
    result = check("integrity.tokens_exceed_recorded_calls", trace(TOTALS))
    assert result.status == Status.NO_MATCH
    m = dict(result.measure)
    assert m["per_call"] == 40_103.0
    assert m["limit_per_call"] == 2_000_000


def test_without_trajectory_totals_the_checks_say_why():
    for check_id in (
        "integrity.cost_missing",
        "integrity.tokens_exceed_recorded_calls",
        "integrity.output_token_ratio",
    ):
        result = check(check_id, trace(None))
        assert result.status == Status.UNKNOWN
        assert [u.reason for u in result.unread] == ["usage_not_recorded"], check_id


def test_output_ratio_reports_characters_tokens_and_bounds():
    result = check("integrity.output_token_ratio", trace(TOTALS))
    m = dict(result.measure)
    assert m["visible_chars"] == len("Done. " * 50)
    assert m["output_tokens"] == 100
    assert (m["min"], m["max"]) == (1.0, 8.0)
    if result.status == Status.UNKNOWN:  # no reasoning split: only the upper bound
        assert [u.reason for u in result.unread] == ["reasoning_tokens_not_split"]


@pytest.mark.parametrize(
    "bad", [(("note", "free text here"),), (("Name", 1),), (("x", float("nan")),), (("x", True),)]
)
def test_measures_cannot_carry_text(bad):
    with pytest.raises(ValueError, match="invalid_measure"):
        Detection(Status.MATCH, measure=bad)


def test_projection_drops_anything_but_numbers_and_codes():
    raw = {"measure": {"ok": 1, "basis": "answer_only", "leak": "rm -rf /", "Bad": 2, "none": None}}
    assert measure(raw) == {"ok": 1, "basis": "answer_only", "none": None}


def test_unmetered_calls_keep_the_ratio_unknown_and_say_so():
    # Regression (Astra TB2.1 bn-fit-modify): a plausible ratio over metered steps can't
    # clear a step whose model call reports no usage; the card must say that.
    doc = trace(None)
    doc["steps"][1]["metrics"] = {
        "prompt_tokens": 10,
        "completion_tokens": 100,
        "extra": {"reasoning_tokens": 0},
    }
    doc["steps"].append({"source": "agent", "message": "More.", "llm_call_count": 1})
    result = check("integrity.output_token_ratio", doc)
    assert result.status == Status.UNKNOWN
    assert [u.reason for u in result.unread] == ["calls_without_usage"]
    assert dict(result.measure)["calls_without_usage"] == 1
    assert dict(result.measure)["chars_per_token"] == 3.0


def retried(doc: Doc, failed: int) -> Doc:
    step = doc["steps"][1]
    step["llm_call_count"] = failed + 1
    step["metrics"] = {
        "prompt_tokens": 10,
        "completion_tokens": 100,
        "extra": {"reasoning_tokens": 0},
    }
    step["extra"] = {"retry": {"schema": "fast-agent.retry/v1", "provider_attempts": failed + 1}}
    doc["final_metrics"] = {"extra": {"llm_usage_calls_complete": False}}
    return doc


def test_failed_provider_attempts_retried_by_the_harness_do_not_block_the_ratio():
    # Regression (Astra TB2.1 break-filter-js-from-html): Copilot's WebSocket stream failed
    # three times and fast-agent retried; only the fourth attempt has usage. The failed
    # attempts' output was discarded, so the recorded text and tokens still compare.
    result = check("integrity.output_token_ratio", retried(trace(None), failed=3))
    assert result.status == Status.NO_MATCH
    assert dict(result.measure)["failed_retry_attempts"] == 3
    assert result.unread == ()


def test_termination_error_is_read_as_a_code_only():
    from atif_scan import parse_trace as parse

    doc = trace(None)
    doc["extra"] = {
        "termination": {
            "status": "error",
            "error_type": "ResponsesWebSocketError",
            "message": "WebSocket Responses request failed.",
        }
    }
    assert parse(doc).termination_error == "ResponsesWebSocketError"
    doc["extra"]["termination"]["error_type"] = "not a code; rm -rf /"
    assert parse(doc).termination_error is None
