"""Synthetic Harbor observed accounting, via local result.json facts only."""

from __future__ import annotations

import json

import pytest
from test_cost_integrity import harbor_hf_run, result, step, trajectory

from atif_scan.cli import main
from atif_scan.data.facts import recorded_facts, run_facts, trace_facts
from atif_scan.data.loader import parse_trace
from atif_scan.output.brief import brief, brief_text
from atif_scan.output.estimates import (
    Pricing,
    choose_pricing,
    cost_estimate,
    price_check,
    unmetered_work,
)
from atif_scan.sources.harbor.files import trial_result


def accounting(**changes):
    return {
        "schema": "harbor.fast-agent.accounting/v1",
        "scope": "observed",
        "provider_usage_complete": False,
        "unknown_attempts": 1,
        "token_sources": {
            "prompt_tokens": "observed_lower_bound",
            "completion_tokens": "observed_steps",
            "cached_tokens": "unknown",
        },
        "tokens_partial": True,
        "cost_source": "price_derived",
        "cost_scope": "observed",
        "reported_cost_usd": None,
        "estimated_observed_cost_usd": 0.125,
        "estimate_basis": "step_tokens",
        **changes,
    }


def recorded(contract):
    raw = result(0, cost=0.125)
    raw["agent_result"]["metadata"] = {
        "fast_agent_accounting": contract,
        "private": "SYNTHETIC_PRIVATE_VALUE",
        "fast_agent_final_metrics": {
            "total_prompt_tokens": 999999,
            "private": "SYNTHETIC_PRIVATE_VALUE",
            "extra": {
                "accounting": {
                    "schema": "fast-agent.accounting/v1",
                    "scope": "observed",
                    "provider_usage_complete": False,
                    "observed_token_availability": {
                        "prompt_tokens": True,
                        "completion_tokens": True,
                        "cached_tokens": False,
                        "private": "SYNTHETIC_PRIVATE_VALUE",
                    },
                }
            },
        },
    }
    return raw


def traced():
    raw = trajectory([step((1000, 100, 500))], (1000, 100, 500))
    raw["final_metrics"]["total_cost_usd"] = 0.125
    return trace_facts(parse_trace(raw))


def facts(contract):
    rec = trial_result(json.dumps(recorded(contract)).encode())
    return run_facts({"attempt_cost_usd": 0.125, **rec}, traced())


def test_observed_counts_never_become_totals_or_bills():
    item = facts(accounting())
    assert item["input_tokens"] == 1000
    assert item["output_tokens"] == 100
    assert item["cache_tokens"] is None
    assert item["usage_basis"] == "run_observed"
    for key in (
        "cost_usd",
        "cost_records_agree",
        "cost_vs_trajectory",
        "recorded_vs_trajectory",
        "steps_vs_totals",
        "tokens_outside_steps",
    ):
        assert item[key] is None
    assert item["accounting"]["estimated_observed_cost_usd"] == 0.125
    extension = item["accounting"]["final_metrics_accounting"]
    assert extension["observed_token_availability"]["cached_tokens"] is False
    assert "SYNTHETIC_PRIVATE_VALUE" not in json.dumps(item)
    assert traced()["usage_basis"] == "final_metrics"  # canonical export unchanged


@pytest.mark.parametrize(
    "source", ["reported_final", "reported_steps", "price_derived", "unknown", None]
)
def test_only_final_reported_cost_is_a_bill(source):
    item = facts(accounting(cost_source=source, reported_cost_usd=0.2))
    assert item["cost_usd"] is None
    assert item["accounting"]["reported_cost_usd"] == 0.2


def test_complete_accounting_keeps_final_counts_and_reported_zero_cost():
    item = facts(
        accounting(
            provider_usage_complete=True,
            unknown_attempts=0,
            tokens_partial=False,
            token_sources=dict.fromkeys(
                ("prompt_tokens", "completion_tokens", "cached_tokens"), "final_total"
            ),
            cost_source="reported_final",
            cost_scope="final_total",
            reported_cost_usd=0,
        )
    )
    assert item["usage_basis"] == "run"
    assert item["recorded_vs_trajectory"] == "same"
    assert item["cost_usd"] == 0
    assert item["cache_tokens"] == 500


@pytest.mark.parametrize("bad", [True, -1, "private", [], {}, float("inf")])
def test_invalid_accounting_values_are_unknown(bad):
    item = facts(
        accounting(
            unknown_attempts=bad,
            reported_cost_usd=bad,
            estimated_observed_cost_usd=bad,
            token_sources={"prompt_tokens": bad},
            estimate_basis=bad,
        )
    )
    projected = item["accounting"]
    assert projected["unknown_attempts"] is None
    assert projected["reported_cost_usd"] is None
    assert projected["estimated_observed_cost_usd"] is None
    assert projected["estimate_basis"] is None
    assert item["input_tokens"] is None
    assert item["usage_basis"] == "run_observed"


@pytest.mark.parametrize("bad", [None, [], {}, {"schema": "future", "scope": "observed"}])
def test_absent_or_unknown_contract_preserves_legacy_facts(bad):
    item = facts(bad)
    assert "accounting" not in item
    assert item["usage_basis"] == "run"
    assert item["cost_usd"] == 0.125
    assert item["recorded_vs_trajectory"] == "same"


def test_unknown_completeness_and_missing_counts_do_not_fall_back():
    raw = recorded(accounting(provider_usage_complete=None, unknown_attempts=None))
    raw["agent_result"]["n_input_tokens"] = None
    rec = recorded_facts(
        {"input_tokens": 999999},
        trial_result(json.dumps(raw).encode()),
    )
    item = run_facts(rec, traced())
    assert item["input_tokens"] is None
    assert item["usage_basis"] == "run_observed"
    assert item["accounting"]["provider_usage_complete"] is None


def test_observed_counts_cannot_train_price_fit_or_be_price_checked():
    item = {
        "input_id": "synthetic",
        **facts(
            accounting(cost_source="reported_final", cost_scope="final_total", reported_cost_usd=1)
        ),
    }
    assert choose_pricing([item] * 30).source is None
    assert Pricing(rates=(1, 1, 1), source="declared").trial_cost(item) is None
    checked = price_check([item], (1, 1, 1))
    assert checked is not None and checked["compared"] == 0


def test_local_scan_and_cached_rescan_reread_accounting(tmp_path, capsys):
    traj = trajectory([step((1000, 100, 500))], (1000, 100, 500))
    run = harbor_hf_run(tmp_path, [(recorded(accounting()), traj)], attempt_costs={0: 0.125})
    cache = tmp_path / "cache"
    args = [str(run), "--format", "json", "--packs", "none", "--cache", str(cache)]
    main(args)
    doc = json.loads(capsys.readouterr().out)
    item = doc["inputs"][0]
    assert item["usage_basis"] == "run_observed"
    assert item["cost_usd"] is None
    text = brief_text(brief(doc))
    assert "not complete" in text
    assert "not final bills" in text
    assert "SYNTHETIC_PRIVATE_VALUE" not in json.dumps(doc)
    # Fresh result facts win even when the trajectory scan is cached.
    path = run / "job" / "alpha__t0" / "result.json"
    path.write_text(json.dumps(result(0, cost=0.5)))
    main(args)
    refreshed = json.loads(capsys.readouterr().out)["inputs"][0]
    assert refreshed["usage_basis"] == "run"
    assert refreshed["cost_usd"] == 0.5
    assert "accounting" not in refreshed


def test_producer_final_metrics_key_is_authoritative():
    raw = recorded(accounting())
    metadata = raw["agent_result"]["metadata"]
    metadata["final_metrics"] = {"extra": {"accounting": {"schema": "wrong", "scope": "observed"}}}
    item = run_facts(trial_result(json.dumps(raw).encode()), traced())
    extension = item["accounting"]["final_metrics_accounting"]
    assert extension["provider_usage_complete"] is False
    assert extension["observed_token_availability"] == {
        "prompt_tokens": True,
        "completion_tokens": True,
        "cached_tokens": False,
    }
    del metadata["fast_agent_final_metrics"]
    item = run_facts(trial_result(json.dumps(raw).encode()), traced())
    assert "final_metrics_accounting" not in item["accounting"]


@pytest.mark.parametrize("amount", [0, 1.25])
@pytest.mark.parametrize("source", ["price_derived", "reported_steps", "reported_final"])
def test_scoped_amounts_display_without_refitting(tmp_path, capsys, amount, source):
    contract = accounting(
        cost_source=source,
        estimated_observed_cost_usd=amount,
        reported_cost_usd=2.5,
    )
    traj = trajectory([step((1000, 100, 500))], (1000, 100, 500))
    run = harbor_hf_run(tmp_path, [(recorded(contract), traj)])
    main([str(run), "--format", "json", "--packs", "none", "--no-cache"])
    doc = json.loads(capsys.readouterr().out)
    # Neither declared rates nor explicit replacement prices may reprice these amounts.
    for price in (None, (900, 900, 900)):
        summary = brief(doc, price=price)
        text = " ".join(brief_text(summary).split())
        assert (
            f"${amount:,.2f} observed price-derived estimate (excludes unmetered attempts)" in text
        )
        assert "$2.50 partial reported observed cost (not a final bill)" in text
        assert "actual bill unknown" in text
        assert "every trial with usage has a cost" not in text
        assert "not added together" in text
        assert summary["cost_estimate"]["unpriced"] == 0
        assert summary["cost_estimate"]["estimate_usd"] is None
        assert summary["overview"]["cost"]["total_usd"] == 0


def test_generic_estimation_only_prices_unscoped_missing_costs():
    scoped = {"input_id": "scoped", **facts(accounting())}
    # Complete tokens do not turn a partial reported amount into a final bill.
    scoped["usage_basis"] = "run"
    ordinary = {**scoped, "input_id": "ordinary"}
    del ordinary["accounting"]
    pricing = Pricing(rates=(2, 0.5, 10), source="declared")
    estimate = cost_estimate([scoped, ordinary], pricing)
    assert estimate["unpriced_ids"] == ["ordinary"]
    assert estimate["estimate_usd"] == cost_estimate([ordinary], pricing)["estimate_usd"]
    assert pricing.trial_cost(scoped) is None
    scoped.update(input_tokens=None, output_tokens=None, cache_tokens=None, llm_calls=10)
    assert unmetered_work([scoped], pricing)["trials"] == 0


def test_scoped_sums_keep_reported_and_estimated_amounts_separate(tmp_path, capsys):
    traj = trajectory([step((1000, 100, 500))], (1000, 100, 500))
    records = [
        (recorded(accounting(estimated_observed_cost_usd=1.25)), traj),
        (recorded(accounting(estimated_observed_cost_usd=2.5, reported_cost_usd=0.75)), traj),
    ]
    run = harbor_hf_run(tmp_path, records)
    main([str(run), "--format", "json", "--packs", "none", "--no-cache"])
    summary = brief(json.loads(capsys.readouterr().out))
    assert summary["scoped_costs"] == {
        "trials": 2,
        "estimated_observed_cost_usd": 3.75,
        "reported_cost_usd": 0.75,
    }
    text = " ".join(brief_text(summary).split())
    assert "$3.75 observed price-derived estimate" in text
    assert "$0.75 partial reported observed cost" in text
