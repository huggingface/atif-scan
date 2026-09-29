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
    assert facts["tokens_match_trajectory"] is False and facts["cost_records_agree"] is None
    facts = run_facts({"cost_usd": 1.0, "attempt_cost_usd": 1.005}, traced)
    assert facts["input_tokens"] == 9 and facts["usage_basis"] == "final_metrics"
    # Without recorded tokens the trajectory's cost is reported, and checked against
    # harbor-hf's attempt cost.
    assert facts["cost_usd"] == 9.0 and facts["cost_records_agree"] is False
