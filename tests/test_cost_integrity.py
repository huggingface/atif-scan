"""Costs recorded beside the trace (harbor-hf run.json prices, attempt-costs) and the
integrity of usage between result files, trajectories and prices. Synthetic only."""

from __future__ import annotations

import json
import re

import pytest
from test_brief import section

from atif_scan.brief import brief, brief_text
from atif_scan.cli import main
from atif_scan.estimates import choose_pricing, partial_usage, unmetered_work
from atif_scan.harbor_files import attempt_cost, declared_prices, trial_result
from atif_scan.loader import parse_trace
from atif_scan.sync import sync_cap

PRICING = {
    "currency": "USD",
    "input_usd_per_million": 2.0,
    "cached_usd_per_million": 0.5,
    "output_usd_per_million": 10.0,
}
IDS = [f"00000000-0000-4000-8000-00000000000{i}" for i in range(4)]


def step(tokens=None, calls=None, command="ls"):
    raw = {
        "source": "agent",
        "message": "x",
        "tool_calls": [
            {"tool_call_id": "c1", "function_name": "bash", "arguments": {"command": command}}
        ],
    }
    if tokens:
        raw["metrics"] = dict(
            zip(("prompt_tokens", "completion_tokens", "cached_tokens"), tokens, strict=True)
        )
    if calls is not None:
        raw["llm_call_count"] = calls
    return raw


def trajectory(steps, totals=None):
    raw = {"schema_version": "ATIF-v1.7", "steps": steps}
    if totals:
        raw["final_metrics"] = {
            "total_prompt_tokens": totals[0],
            "total_completion_tokens": totals[1],
            "total_cached_tokens": totals[2],
        }
    return raw


def result(i, tokens=(1000, 500, 100), cost=None):
    return {
        "id": IDS[i],
        "trial_name": f"alpha__t{i}",
        "task_name": "alpha",
        "agent_result": {
            "n_input_tokens": tokens[0] if tokens else None,
            "n_cache_tokens": tokens[1] if tokens else None,
            "n_output_tokens": tokens[2] if tokens else None,
            "cost_usd": cost,
        },
        "verifier_result": {"rewards": {"reward": 1.0}},
    }


def harbor_hf_run(tmp_path, trials, pricing=PRICING, attempt_costs=None):
    """<run>/run.json, <run>/job/{config,result}.json, <run>/job/<trial>/…,
    <run>/attempt-costs/<attempt id>.json, as harbor-hf publishes a run."""
    run = tmp_path / "run-x"
    job = run / "job"
    job.mkdir(parents=True)
    if pricing is not None:
        (run / "run.json").write_text(json.dumps({"schema_version": "v1", "pricing": pricing}))
    (job / "config.json").write_text(
        json.dumps({"job_name": "job", "n_attempts": 1, "datasets": [{"name": "d"}]})
    )
    (job / "result.json").write_text(json.dumps({"id": "j", "n_total_trials": len(trials)}))
    (run / "attempt-costs").mkdir()
    for i, (res, traj) in enumerate(trials):
        trial = job / f"alpha__t{i}"
        (trial / "agent").mkdir(parents=True)
        (trial / "agent" / "trajectory.json").write_text(json.dumps(traj))
        (trial / "result.json").write_text(json.dumps(res))
        (trial / "verifier").mkdir()
        (trial / "verifier" / "reward.txt").write_text("1")
        cost = (attempt_costs or {}).get(i)
        (run / "attempt-costs" / f"{IDS[i]}.json").write_text(
            json.dumps(
                {
                    "schema_version": "v1",
                    "attempt_id": IDS[i],
                    "trial_name": f"alpha__t{i}",
                    "cost_usd": cost,
                }
            )
        )
    return run


def scan(run, capsys, *extra):
    main([str(run), "--brief", "--format", "json", "--packs", "none", *extra])
    b = json.loads(capsys.readouterr().out)
    main([str(run), "--format", "text", "--packs", "none", *extra])
    return b, capsys.readouterr().out


def test_declared_prices_and_attempt_cost_parsing():
    assert declared_prices(json.dumps({"pricing": PRICING}).encode()) == {
        "uncached_input": 2.0,
        "cached_input": 0.5,
        "output": 10.0,
    }
    for bad in (
        {"pricing": {**PRICING, "currency": "EUR"}},
        {"pricing": {**PRICING, "output_usd_per_million": -1}},
        {"pricing": {**PRICING, "cached_usd_per_million": None}},
        {"pricing": "cheap"},
        {},
    ):
        assert declared_prices(json.dumps(bad).encode()) is None  # unknown, never $0
    assert declared_prices(b"not json") is None and declared_prices(None) is None
    doc = {"attempt_id": IDS[0], "trial_name": "alpha__t0", "cost_usd": 0.25}
    assert attempt_cost(json.dumps(doc).encode(), IDS[0], "alpha__t0") == 0.25
    # A file naming another attempt or trial folder, or a null cost, is no record.
    assert attempt_cost(json.dumps(doc).encode(), IDS[1], "alpha__t0") is None
    assert attempt_cost(json.dumps(doc).encode(), IDS[0], "alpha__t1") is None
    assert attempt_cost(json.dumps({**doc, "cost_usd": None}).encode(), IDS[0], "alpha__t0") is None
    # The attempt id is a lookup key only; it isn't a UUID -> not taken.
    assert trial_result(json.dumps(result(0)).encode())["attempt_id"] == IDS[0]
    assert "attempt_id" not in trial_result(json.dumps({**result(0), "id": "../x"}).encode())


def test_sync_includes_run_prices_and_attempt_costs_only():
    assert sync_cap("run-x/run.json") is not None
    assert sync_cap(f"run-x/attempt-costs/{IDS[0]}.json") == 4096
    assert sync_cap("run-x/attempt-costs/notes.txt") is None
    assert sync_cap("run-x/state.json") is None


def test_run_priced_at_declared_prices_when_no_cost_is_recorded(tmp_path, capsys):
    # Regression: a harbor-hf run records no cost in trajectories or result.json, but its
    # run.json declares prices. The brief said "no cost recorded for any trial".
    trials = [
        (result(i), trajectory([step((1000, 100, 500))] * 1, (1000, 100, 500))) for i in range(3)
    ]
    b, out = scan(harbor_hf_run(tmp_path, trials), capsys)
    ce, ci = b["cost_estimate"], b["cost_integrity"]
    # per trial: 500 uncached * 2 + 500 cached * 0.5 + 100 out * 10 = 2250 $/M-tokens
    assert ce["price_source"] == "declared" and ce["estimate_usd"] == round(3 * 2250 / 1e6, 2)
    assert ci["declared_prices"] == {"uncached_input": 2.0, "cached_input": 0.5, "output": 10.0}
    u = b["usage"]
    assert u["recorded_vs_trajectory"] == {"same": 3}
    assert u["steps_vs_totals"] == {"same": 3}
    assert u["trials_with_tokens"] == 3 and u["basis"] == {"run": 3}
    assert (
        "COST       $0.01 at the run's declared prices (run.json); no trial recorded a cost" in out
    )
    assert "· $2.00 uncached input · $0.50 cached input · $10.00 output, per M tokens" in out
    assert "no cost recorded for any trial" not in out
    assert "TOKENS     3k input (2k cached) · 300 output, from 3 of 3 trials" in out
    assert "✓ recorded totals match the trajectories' in 3 of 3 compared trials" in out
    assert "✓ trajectory totals equal the sum of their steps in 3 of 3 compared trials" in out
    # Token accounting comes first: cost is priced from it.
    assert out.index("TOKENS") < out.index("COST")
    # No recorded cost to check the prices against: said once, in the headline.
    assert "no recorded cost to check the declared prices against" not in out
    # A given --price still wins over the declared one, which then has nothing to check.
    b, out = scan(harbor_hf_run(tmp_path / "g", trials), capsys, "--price", "1,1,1")
    assert b["cost_estimate"]["price_source"] == "given" and "at the given --price" in out
    assert "· no recorded cost to check the declared prices against" in out


def test_recorded_costs_are_checked_against_declared_prices_and_each_other(tmp_path, capsys):
    ok = 2250 / 1e6
    trials = [
        (result(0, cost=ok), trajectory([step()])),  # fits the prices
        (result(1, cost=1.0), trajectory([step()])),  # doesn't
        (result(2, cost=ok), trajectory([step()])),  # attempt-costs disagrees
        (result(3), trajectory([step()])),  # only attempt-costs records it
    ]
    b, out = scan(harbor_hf_run(tmp_path, trials, attempt_costs={2: 0.5, 3: ok}), capsys)
    ci = b["cost_integrity"]
    assert ci["price_check"]["compared"] == 4 and ci["price_check"]["mismatched"] == 1
    assert ci["cost_records"] == {"compared": 1, "mismatched": 1}
    assert b["cost_estimate"]["unpriced"] == 0  # attempt-costs priced trial 3
    flat = " ".join(out.split())  # unwrapped: phrases may span lines
    assert "⚠ 1 of 4 recorded costs don't fit the declared prices ($1.01 recorded" in flat
    assert "⚠ 1 of 1 trial: result.json and attempt-costs record different costs" in flat


def test_result_tokens_that_differ_from_the_trajectory_are_flagged(tmp_path, capsys):
    trials = [
        (result(0), trajectory([step()], (1000, 100, 500))),
        (result(1), trajectory([step()], (999, 100, 500))),
    ]
    b, out = scan(harbor_hf_run(tmp_path, trials, pricing=None), capsys)
    assert b["usage"]["recorded_vs_trajectory"] == {"same": 1, "differs": 1}
    assert b["cost_integrity"]["declared_prices"] is None
    assert "⚠ recorded totals differ from the trajectory's in 1 of 2 trials" in out
    assert "recorded totals match" not in out


def test_trajectory_totals_that_differ_from_step_usage_are_flagged(tmp_path, capsys):
    # The trajectory's own totals against the sum of its steps' usage.
    steps = [step((1000, 100, 500), calls=1)] * 2
    trials = [
        (result(0, tokens=(2000, 1000, 200)), trajectory(steps, (2000, 200, 1000))),
        # Totals above the steps: 500 prompt tokens from calls not recorded as steps.
        (result(1, tokens=(2500, 1000, 200)), trajectory(steps, (2500, 200, 1000))),
        # Totals below the steps: they disagree outright.
        (result(2, tokens=(1500, 1000, 200)), trajectory(steps, (1500, 200, 1000))),
    ]
    b, out = scan(harbor_hf_run(tmp_path, trials, pricing=None), capsys)
    u = b["usage"]
    assert u["steps_vs_totals"] == {"same": 1, "steps_short": 1, "differs": 1}
    assert (u["tokens_outside_steps"], u["tokens_of_short_trials"]) == (500, 2700)
    flat = " ".join(out.split())  # unwrapped: phrases may span lines
    assert "⚠ 1 of 3 trajectories count 500 tokens (18.5% of theirs) outside their steps" in flat
    assert "⚠ 1 of 3 trajectories have totals that disagree with their steps" in flat
    assert "trajectory totals equal the sum of their steps" not in out
    # Steps that never record cached tokens leave them unknown, not a mismatching 0.
    raw = trajectory([step()], (100, 10, 50))
    raw["steps"][0]["metrics"] = {"prompt_tokens": 100, "completion_tokens": 10}
    usage = parse_trace(raw).step_usage
    assert usage is not None and usage.cached_tokens is None


def test_step_usage_is_the_fallback_when_totals_are_missing():
    # Regression: fast-agent traces whose harness wrote no usage totals (final_metrics
    # without tokens, result.json null) were reported as "did work but report no usage",
    # although every step records its own usage, except for a retried call.
    trace = parse_trace(trajectory([step((100, 10, 50)), step((200, 20, 150), calls=2), step()]))
    assert trace.usage is None
    usage = trace.step_usage
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens) == (300, 30)
    assert usage.cached_tokens == 200
    assert trace.calls_without_usage == 2  # the retried call + the unmetered step
    full = parse_trace(trajectory([step((100, 10, 50)), step((200, 20, 150), calls=1)]))
    assert full.calls_without_usage == 0


def test_partial_step_usage_is_priced_as_a_lower_bound(tmp_path, capsys):
    steps = [step((1000, 100, 500), calls=1)] * 3 + [step((1000, 100, 500), calls=2)]
    trials = [(result(i), trajectory(steps[:3], (3000, 300, 1500))) for i in range(3)]
    trials.append((result(3, tokens=None), trajectory(steps)))
    b, out = scan(harbor_hf_run(tmp_path, trials), capsys)
    pu = b["usage"]["partial"]
    assert pu["trials"] == 1 and pu["calls_without_usage"] == 1
    # 4 metered steps at 2250 $/M-tokens each, and one more call at that per-call cost.
    assert pu["estimate_usd"] == pytest.approx(4 * 2250 / 1e6 / 4, abs=1e-4)
    assert b["unmetered_work"]["trials"] == 0  # not "no usage": its steps record it
    flat = " ".join(out.split())  # unwrapped: phrases may span lines
    assert "⚠ 1 trial (1 rewarded) recorded no totals" in flat
    assert "missing 1 LLM call (a lower bound)" in flat
    assert "· est. +<$0.01 for the 1 LLM call without usage" in flat
    assert b["usage"]["basis"] == {"run": 3, "steps_partial": 1}


def test_unmetered_work_is_estimated_from_trials_priced_at_given_rates():
    # Regression: with no recorded cost anywhere, trials that did work but report no
    # usage got no estimate even when prices were known.
    items = [
        {
            "input_id": f"t{i}",
            "llm_calls": 10,
            "input_tokens": 1_000_000,
            "cache_tokens": 0,
            "output_tokens": 0,
            "cost_usd": None,
        }
        for i in range(25)
    ]
    items.append({"input_id": "dead", "llm_calls": 20, "tool_calls": 20, "cost_usd": None})
    assert unmetered_work(items)["estimate_usd"] is None
    um = unmetered_work(items, choose_pricing(items, given=(1.0, 0.0, 0.0)))
    assert um["trials"] == 1 and um["estimate_usd"] > 1.0  # 20 calls at $0.10+/call
    assert partial_usage(items) == {
        "trials": 0,
        "ids": [],
        "calls_without_usage": 0,
        "rewarded": 0,
        "estimate_usd": None,
    }


def test_findings_headline_counts_findings_and_trials_not_overlap():
    def item(i, checks):
        return {
            "input_id": f"t{i}",
            "input_status": "available",
            "incomplete": False,
            "reward": 1.0,
            "task": "a",
            "assessments": [
                {
                    "id": c,
                    "status": "match",
                    "kind": "detector",
                    "severity": "medium",
                    "score": 50,
                    "expected_by": [],
                    "evidence": [
                        {"step": s, "channel": "c", "call": 0, "observation": None, "field": None}
                        for s in steps
                    ],
                }
                for c, steps in checks
            ],
        }

    items = [
        item(0, [("access.test_path", [1, 2]), ("awareness.verifier", [3])]),
        item(1, [("access.test_path", [4])]),
        item(2, []),
    ]
    b = brief({"scanner_version": "dev", "inputs": items, "coverage": {}, "runs": []})
    assert (b["findings"]["total"], b["findings"]["trials"]) == (4, 2)
    text = brief_text(b)
    assert "· at any priority: 4 findings in 2 trials" in section(text, "FINDINGS")
    # Per check: trials (not findings, which overlap across checks) in the table.
    assert re.search(r"medium +2 +2  access\.test_path\n", text)
    assert b["findings"]["checks"]["access.test_path"]["events"] == 3
    assert "overlap" not in text


def test_rates_are_chosen_once_given_then_declared_then_fitted():
    from atif_scan.estimates import choose_pricing

    items = [
        {
            "input_id": f"t{i}",
            "input_tokens": 1_000_000 * (i + 1),
            "cache_tokens": 100_000 * (i % 4),
            "output_tokens": 1000 * (i % 7),
            "cost_usd": 2.0 * (i + 1) + 0.01 * (i % 5),
        }
        for i in range(25)
    ]
    given, declared = (1.0, 0.1, 3.0), (2.0, 0.2, 6.0)
    assert choose_pricing(items, given, declared).source == "given"
    assert choose_pricing(items, None, declared).rates == declared
    fitted = choose_pricing(items)
    assert fitted.source == "fit" and not fitted.known
    # Fitted rates never price a trial as a reference: only recorded or known costs do.
    unpriced = dict(items[0], cost_usd=None)
    assert fitted.trial_cost(unpriced) is None
    assert choose_pricing(items, None, declared).trial_cost(unpriced) == 2.0
    assert choose_pricing(items[:5]).rates is None  # too few priced trials: no rates
