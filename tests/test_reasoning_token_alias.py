"""Synthetic regression for Codex's final reasoning-token field alias."""

import pytest

from atif_scan import Status, parse_trace
from atif_scan.detectors.integrity import output_ratio, output_token_ratio
from atif_scan.facts import trace_facts


def trajectory(extra):
    return {
        "agent": {"name": "codex", "version": "synthetic", "model_name": "synthetic"},
        "steps": [{"source": "agent", "message": "x" * 300}],
        "final_metrics": {"total_completion_tokens": 1000, "extra": extra},
    }


def test_codex_alias_enables_visible_output_check_without_rewriting_counts():
    raw = trajectory({"reasoning_output_tokens": 900})
    trace = parse_trace(raw)
    assert trace.usage and trace.usage.reasoning_tokens == 900
    assert trace.usage.completion_tokens == 1000
    assert trace.reasoning_exposure == "withheld"
    ratio = output_ratio(trace)
    assert ratio and (ratio.tokens, ratio.chars, ratio.value) == (100, 300, 3)
    assert output_token_ratio(trace).status == Status.NO_MATCH
    facts = trace_facts(trace)
    assert facts["reasoning"] == "withheld"
    assert facts["output_ratio_basis"] == "answer_only"
    assert raw["final_metrics"]["extra"] == {"reasoning_output_tokens": 900}


@pytest.mark.parametrize("canonical", [0, 800])
def test_canonical_field_wins_even_when_zero(canonical):
    trace = parse_trace(
        trajectory({"total_reasoning_tokens": canonical, "reasoning_output_tokens": 900})
    )
    assert trace.usage and trace.usage.reasoning_tokens == canonical


def test_null_canonical_allows_alias():
    trace = parse_trace(
        trajectory({"total_reasoning_tokens": None, "reasoning_output_tokens": 900})
    )
    assert trace.usage and trace.usage.reasoning_tokens == 900


@pytest.mark.parametrize("invalid", [True, False, -1, 1.5, "900", [], {}])
def test_invalid_alias_stays_unknown(invalid):
    trace = parse_trace(trajectory({"reasoning_output_tokens": invalid}))
    assert trace.usage and trace.usage.reasoning_tokens is None
    assert output_token_ratio(trace).status == Status.UNKNOWN


@pytest.mark.parametrize("invalid", [True, -1, "900"])
def test_invalid_canonical_is_not_silently_repaired(invalid):
    trace = parse_trace(
        trajectory({"total_reasoning_tokens": invalid, "reasoning_output_tokens": 900})
    )
    assert trace.usage and trace.usage.reasoning_tokens is None


def test_zero_alias_is_recorded_and_missing_remains_unknown():
    zero = parse_trace(trajectory({"reasoning_output_tokens": 0}))
    missing = parse_trace(trajectory({}))
    assert zero.usage and zero.usage.reasoning_tokens == 0
    assert missing.usage and missing.usage.reasoning_tokens is None
