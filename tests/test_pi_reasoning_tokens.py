"""Synthetic pi-harness usage: the run's reasoning total is `extra.reasoning_tokens`.

Regression: every pi trace left integrity.output_token_ratio unknown
(`reasoning_tokens_not_split`) although the split was recorded.
"""

import pytest

from atif_scan import Status, parse_trace
from atif_scan.detectors.integrity import output_ratio, output_token_ratio


def payload(reasoning=600, completion=1000, name="pi", canonical=None):
    extra = {"cache_write_tokens": 0, "reasoning_tokens": reasoning}
    if canonical is not None:
        extra["total_reasoning_tokens"] = canonical
    return {
        "schema_version": "ATIF-v1.6",
        "agent": {"name": name, "version": "0.84.4", "model_name": "synthetic/model"},
        "steps": [{"source": "agent", "message": "x" * 1200, "reasoning_content": "y" * 9000}],
        "final_metrics": {
            "total_prompt_tokens": 5000,
            "total_completion_tokens": completion,
            "total_cached_tokens": 0,
            "extra": extra,
        },
    }


def test_pi_split_makes_ratio_answer_only():
    trace = parse_trace(payload())
    assert trace.usage and trace.usage.reasoning_tokens == 600
    assert trace.usage.completion_tokens == 1000  # recorded usage is never rewritten
    ratio = output_ratio(trace)
    assert ratio and ratio.answer_only and ratio.low_verifiable
    assert (ratio.chars, ratio.tokens, ratio.value) == (1200, 400, 3)
    assert output_token_ratio(trace).status == Status.NO_MATCH


def test_pi_split_can_match_low_ratio():
    trace = parse_trace(payload(reasoning=100))  # 1200 chars / 900 visible tokens is fine
    assert output_token_ratio(trace).status == Status.NO_MATCH
    raw = payload(reasoning=0, completion=2000)
    assert output_token_ratio(parse_trace(raw)).status == Status.MATCH  # 0.6 chars/token


@pytest.mark.parametrize("reasoning", [None, True, -1, 1.5, "600", 1001])
def test_pi_unusable_split_stays_unknown(reasoning):
    trace = parse_trace(payload(reasoning=reasoning))
    assert trace.usage and trace.usage.reasoning_tokens is None
    assert output_token_ratio(trace).status == Status.UNKNOWN


@pytest.mark.parametrize(("canonical", "expected"), [(300, 300), ("bad", None)])
def test_canonical_field_wins(canonical, expected):
    trace = parse_trace(payload(canonical=canonical))
    assert trace.usage and trace.usage.reasoning_tokens == expected


@pytest.mark.parametrize("name", ["fast-agent", "Pi", "pi-agent", None])
def test_other_harnesses_ignore_the_alias(name):
    trace = parse_trace(payload(name=name))
    assert trace.usage and trace.usage.reasoning_tokens is None
    assert output_token_ratio(trace).status == Status.UNKNOWN
