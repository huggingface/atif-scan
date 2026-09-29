"""Where each per-trial run fact comes from (facts.py): the precedence, pinned."""

from __future__ import annotations

from atif_scan.facts import listed_facts, recorded_facts, run_facts, trial_reward, trial_task

NO_TRACE = dict.fromkeys(
    ("usage", "usage_basis", "agent_name", "llm_calls"),
)


def test_reward_file_beats_listing_beats_result_json():
    listed, result = {"reward": 0.0}, {"reward": 1.0}
    assert trial_reward(0.5, listed, result) == 0.5
    assert trial_reward(None, listed, result) == 0.0
    assert trial_reward(None, {"reward": None}, result) == 1.0
    assert trial_reward(None, {}, {}) is None


def test_task_flag_beats_result_json_beats_listing_beats_folder():
    listed = {"task": "from-listing"}
    recorded = recorded_facts(listed, {"task": "from-result"})
    assert trial_task("flag", recorded, "folder") == "flag"
    assert trial_task(None, recorded, "folder") == "from-result"
    assert trial_task(None, recorded_facts(listed, {}), "folder") == "from-listing"
    assert trial_task(None, {}, "folder") == "folder"


def test_exception_marker_only_fills_a_missing_error_type():
    assert listed_facts({}, True, None) == {"error_type": "exception"}
    assert listed_facts({"error_type": "Timeout"}, True, None)["error_type"] == "Timeout"
    assert listed_facts({}, False, False) == {"in_job_result": False}
    recorded = recorded_facts(listed_facts({}, True, None), {"error_type": "AgentError"})
    assert run_facts(recorded, NO_TRACE)["error_type"] == "AgentError"


def test_recorded_tokens_beat_the_trajectory_and_attempt_cost_fills_a_missing_cost():
    traced = {
        **NO_TRACE,
        "usage": {"cost_usd": 9.0, "input_tokens": 9, "cache_tokens": 0, "output_tokens": 9},
        "usage_basis": "final_metrics",
    }
    recorded = {"input_tokens": 5, "output_tokens": 2, "attempt_cost_usd": 0.5}
    facts = run_facts(recorded, traced)
    assert (facts["input_tokens"], facts["cost_usd"], facts["usage_basis"]) == (5, 0.5, "run")
    assert facts["recorded_vs_trajectory"] == "differs" and facts["cost_records_agree"] is None
    facts = run_facts({"cost_usd": 1.0, "attempt_cost_usd": 1.005}, traced)
    assert facts["input_tokens"] == 9 and facts["usage_basis"] == "final_metrics"
    # Cost is decided separately from tokens: the recorded cost wins even when the tokens
    # come from the trajectory, and it's checked against the attempt cost and the
    # trajectory's own cost.
    assert facts["cost_usd"] == 1.0 and facts["cost_records_agree"] is True
    assert facts["cost_vs_trajectory"] == "differs"


def test_cost_is_decided_separately_from_tokens():
    # Regression (review): a recorded cost was replaced by the trajectory's (even None)
    # whenever the tokens came from the trajectory; and recorded tokens hid a trajectory
    # cost when no cost was recorded.
    def traced(cost):
        usage = {"cost_usd": cost, "input_tokens": 100, "cache_tokens": 0, "output_tokens": 10}
        return {**NO_TRACE, "usage": usage, "usage_basis": "final_metrics"}

    assert run_facts({"cost_usd": 12.0}, traced(None))["cost_usd"] == 12.0
    tokens = {"input_tokens": 100, "cache_tokens": 0, "output_tokens": 10}
    facts = run_facts(tokens, traced(3.0))
    assert (facts["cost_usd"], facts["usage_basis"]) == (3.0, "run")
    # harbor-hf's attempt cost beats the trajectory's, and agrees within a cent.
    facts = run_facts({"attempt_cost_usd": 3.0}, traced(3.004))
    assert (facts["cost_usd"], facts["cost_vs_trajectory"]) == (3.0, "same")
    facts = run_facts({"cost_usd": 12.0}, traced(3.0))
    assert (facts["cost_usd"], facts["cost_vs_trajectory"]) == (12.0, "differs")


def test_record_counting_uncached_input_is_a_convention_and_is_normalised():
    # Regression: the Harbor Hub records Claude Code's input_tokens as uncached input only
    # (prompt minus cache reads and cache writes). It was read as total input, so the brief
    # showed input smaller than its cached part and the cost fit lost uncached tokens; the
    # new token check then reported every trial as "differs".
    traced = {
        **NO_TRACE,
        "usage": {
            "cost_usd": None,
            "input_tokens": 130928,
            "cache_tokens": 112440,
            "output_tokens": 14458,
        },
        "usage_basis": "final_metrics",
        "cache_write_tokens": 18066,
    }
    hub = {"input_tokens": 422, "cache_tokens": 112440, "output_tokens": 14458}
    facts = run_facts(hub, traced)
    assert facts["recorded_vs_trajectory"] == "uncached_input"
    assert (facts["input_tokens"], facts["cache_tokens"]) == (130928, 112440)
    # Without cache writes (e.g. OpenAI-style records), input minus cache reads also fits.
    no_writes = {**traced, "cache_write_tokens": None}
    uncached = {**hub, "input_tokens": 130928 - 112440}
    assert run_facts(uncached, no_writes)["recorded_vs_trajectory"] == "uncached_input"
    # Anything else is a real disagreement, and the record is kept as it is.
    other = run_facts({**hub, "input_tokens": 999}, traced)
    assert (other["recorded_vs_trajectory"], other["input_tokens"]) == ("differs", 999)
    # An output mismatch is never excused by the input convention.
    assert run_facts({**hub, "output_tokens": 1}, traced)["recorded_vs_trajectory"] == "differs"
    assert run_facts({**hub, "input_tokens": 130928}, traced)["recorded_vs_trajectory"] == "same"
