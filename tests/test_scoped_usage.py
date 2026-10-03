"""Synthetic child accounting and per-step reasoning evidence regressions."""

from copy import deepcopy

import pytest

from atif_scan import Status
from atif_scan.data.facts import trace_facts
from atif_scan.data.loader import parse_trace
from atif_scan.detectors.integrity import output_ratio, output_token_ratio


def child_trace():
    return {
        "agent": {"name": "fast-agent"},
        "steps": [
            {
                "source": "agent",
                "message": "x" * 120,
                "metrics": {"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 500},
            }
        ],
        "final_metrics": {
            "total_prompt_tokens": 1200,
            "total_completion_tokens": 120,
            "total_cached_tokens": 500,
            "extra": {
                "root_prompt_tokens": 1000,
                "root_completion_tokens": 100,
                "root_cached_tokens": 500,
                "subagent_prompt_tokens": 200,
                "subagent_completion_tokens": 20,
                "subagent_cached_tokens": 0,
                "total_reasoning_tokens": 90,
            },
        },
        "subagent_trajectories": [
            {
                "trajectory_id": "synthetic-summary",
                "steps": [{"source": "agent", "message": "Synthetic summary."}],
                "final_metrics": {
                    "total_prompt_tokens": 200,
                    "total_completion_tokens": 20,
                    "total_cached_tokens": 0,
                    "extra": {"total_reasoning_tokens": 10, "llm_usage_calls_complete": True},
                },
            }
        ],
    }


def test_embedded_usage_reconciles_without_changing_canonical_totals():
    trace = parse_trace(child_trace())
    facts = trace_facts(trace)
    assert facts["steps_vs_totals"] == "same"
    assert facts["tokens_outside_steps"] == 0
    assert facts["usage"]["input_tokens"] == 1200
    assert facts["usage"]["output_tokens"] == 120
    assert trace.usage and trace.usage.reasoning_tokens == 90
    ratio = output_ratio(trace)
    assert ratio and (ratio.chars, ratio.tokens, ratio.value) == (120, 20, 6)
    # Child text is not promoted into scanned root history.
    assert len(trace.steps) == 1


@pytest.mark.parametrize("defect", ["absent", "duplicate", "mismatch", "unknown", "incomplete"])
def test_unverified_children_cannot_explain_a_gap(defect):
    raw = child_trace()
    children = raw["subagent_trajectories"]
    if defect == "absent":
        children.clear()
    elif defect == "duplicate":
        children.append(deepcopy(children[0]))
    elif defect == "mismatch":
        children[0]["final_metrics"]["total_prompt_tokens"] = 199
    elif defect == "unknown":
        del children[0]["final_metrics"]["total_completion_tokens"]
    else:
        children[0]["final_metrics"]["extra"]["llm_usage_calls_complete"] = False
    trace = parse_trace(raw)
    assert trace.root_usage is None
    assert trace_facts(trace)["steps_vs_totals"] == "steps_short"


def test_reconciled_children_do_not_hide_a_root_step_gap():
    raw = child_trace()
    raw["steps"][0]["metrics"]["prompt_tokens"] = 900
    facts = trace_facts(parse_trace(raw))
    assert facts["steps_vs_totals"] == "steps_short"
    assert facts["tokens_outside_steps"] == 100


def metered_step(reasoning=80):
    return {
        "source": "agent",
        "message": "x" * 60,
        "metrics": {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "extra": {
                "reasoning_tokens": reasoning,
            },
        },
    }


def test_partial_usage_uses_step_reasoning_without_clearing_failed_attempts():
    raw = {
        "steps": [metered_step(), {"source": "agent", "message": "", "llm_call_count": 1}],
        "final_metrics": {"extra": {"llm_usage_calls_complete": False}},
    }
    trace = parse_trace(raw)
    facts = trace_facts(trace)
    ratio = output_ratio(trace)
    assert ratio and (ratio.chars, ratio.tokens, ratio.value) == (60, 20, 3)
    assert facts["output_ratio_basis"] == "answer_only"
    assert facts["usage_basis"] == "steps_partial"
    assert facts["calls_without_usage"] == 1
    assert trace.usage_calls_complete is False
    assert output_token_ratio(trace).status is Status.UNKNOWN
    assert trace.reasoning_exposure == "withheld"
    assert facts["steps_vs_totals"] is None


@pytest.mark.parametrize("invalid", [None, True, -1, 101, "80", 2.5])
def test_missing_or_invalid_step_split_keeps_lower_bound_unknown(invalid):
    trace = parse_trace({"steps": [metered_step(), metered_step(invalid)]})
    ratio = output_ratio(trace)
    assert ratio and ratio.tokens == 200 and not ratio.answer_only
    assert output_token_ratio(trace).status is Status.UNKNOWN


def test_step_split_zero_and_copied_unmetered_context():
    copied = metered_step(None)
    copied["is_copied_context"] = True
    trace = parse_trace({"steps": [metered_step(0), copied]})
    ratio = output_ratio(trace)
    assert ratio and ratio.answer_only and ratio.tokens == 100


def test_step_reasoning_does_not_repair_unknown_final_split():
    trace = parse_trace(
        {
            "steps": [metered_step()],
            "final_metrics": {"total_completion_tokens": 200},
        }
    )
    ratio = output_ratio(trace)
    assert ratio and not ratio.answer_only and ratio.tokens == 200


def test_multiple_children_reconcile_their_combined_usage():
    raw = child_trace()
    child = raw["subagent_trajectories"][0]
    child["final_metrics"]["total_prompt_tokens"] = 100
    child["final_metrics"]["total_completion_tokens"] = 10
    child["final_metrics"]["extra"]["total_reasoning_tokens"] = 5
    other = deepcopy(child)
    other["trajectory_id"] = "synthetic-summary-two"
    raw["subagent_trajectories"].append(other)
    trace = parse_trace(raw)
    assert trace_facts(trace)["steps_vs_totals"] == "same"
    ratio = output_ratio(trace)
    assert ratio and ratio.tokens == 20


def test_child_reasoning_unknown_does_not_invent_a_root_split():
    raw = child_trace()
    del raw["subagent_trajectories"][0]["final_metrics"]["extra"]["total_reasoning_tokens"]
    trace = parse_trace(raw)
    assert trace_facts(trace)["steps_vs_totals"] == "same"
    ratio = output_ratio(trace)
    assert ratio and not ratio.answer_only and ratio.tokens == 100
    assert output_token_ratio(trace).status is Status.UNKNOWN


def test_reasoning_summary_excluded_from_metered_subset():
    step = metered_step()
    step["reasoning_content"] = "Short summary."
    ratio = output_ratio(parse_trace({"steps": [step]}))
    assert ratio and ratio.chars == 60 and ratio.tokens == 20


def test_retry_with_one_usage_record_stays_incomplete():
    step = metered_step()
    step["llm_call_count"] = 2
    step["extra"] = {"retry": {"schema": "fast-agent.retry/v1", "provider_attempts": 2}}
    trace = parse_trace({"steps": [step]})
    facts = trace_facts(trace)
    assert facts["stream_retry_attempts"] == 1
    assert facts["calls_without_usage"] == 1
    assert facts["usage_basis"] == "steps_partial"
    assert facts["output_ratio_basis"] == "answer_only"
    assert output_token_ratio(trace).status is Status.UNKNOWN


@pytest.mark.parametrize("field", ["root_prompt_tokens", "subagent_prompt_tokens"])
def test_declared_split_must_match_canonical_and_embedded_totals(field):
    raw = child_trace()
    raw["final_metrics"]["extra"][field] += 1
    trace = parse_trace(raw)
    assert trace.root_usage is None
    assert trace_facts(trace)["steps_vs_totals"] == "steps_short"


def test_unknown_step_split_is_not_reported_as_absent_evidence():
    from atif_scan.output.brief import brief, brief_text

    trace = parse_trace({"steps": [metered_step(), metered_step(None)]})
    row = {
        **trace_facts(trace),
        "input_id": "synthetic",
        "input_tokens": 2000,
        "output_tokens": 200,
        "input_status": "available",
        "incomplete": True,
        "assessments": [],
    }
    text = " ".join(
        brief_text(brief({"scanner_version": "dev", "inputs": [row], "coverage": {}})).split()
    )
    assert "no usable reasoning split for this comparison" in text
    assert "no reasoning text or token split recorded" not in text


def test_all_reasoning_has_no_visible_output_denominator():
    assert output_ratio(parse_trace({"steps": [metered_step(100)]})) is None
