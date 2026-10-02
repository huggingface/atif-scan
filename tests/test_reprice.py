"""tools/reprice.py: per-step catalog pricing, bands and lower bounds. Synthetic only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "reprice.py"
spec = importlib.util.spec_from_file_location("reprice", TOOL)
assert spec and spec.loader
reprice = importlib.util.module_from_spec(spec)
sys.modules["reprice"] = reprice  # dataclasses resolve their module by name
spec.loader.exec_module(reprice)


def rule(rid, tiers, rates, minimum=None, maximum=None, **match):
    raw = {
        "id": rid,
        "match": {"model": {"values": ["m-1"]}, "providers": ["responses"], "service_tiers": tiers},
        "rates": dict(zip(("input", "cache_read", "output"), rates, strict=True)),
        **match,
    }
    if minimum is not None or maximum is not None:
        raw["prompt_tokens"] = {
            k: v for k, v in (("minimum", minimum), ("maximum", maximum)) if v is not None
        }
    return raw


def catalog(*rules):
    return {
        "schema": "fast-agent.pricing/v1",
        "catalog_version": "t.1",
        "currency": "USD",
        "unit": "usd_per_million_tokens",
        "rules": list(rules),
    }


CATALOG = catalog(
    rule("std-short", ["standard"], ("2", "0.2", "10"), maximum=1000),
    rule("std-long", ["standard"], ("4", "0.4", "15"), minimum=1001),
    rule("flex-short", ["flex"], ("1", "0.1", "5"), maximum=1000),
    rule("flex-long", ["flex"], ("2", "0.2", "7.5"), minimum=1001),
)


def step(prompt, cached, output, calls=1, retry=None, model="m-1"):
    raw = {
        "source": "agent",
        "message": "x",
        "model_name": model,
        "llm_call_count": calls,
        "metrics": {"prompt_tokens": prompt, "cached_tokens": cached, "completion_tokens": output},
    }
    if retry:
        raw["extra"] = {"retry": {"schema": "fast-agent.retry/v1", "provider_attempts": retry}}
    return raw


def trace(*steps):
    return {"schema_version": "ATIF-v1.7", "steps": [{"source": "user", "message": "t"}, *steps]}


def test_each_step_is_priced_in_its_own_prompt_band():
    bands = reprice.load_bands(CATALOG, "m-1", "responses", "flex")
    assert [b.rule for b in bands] == ["flex-short", "flex-long"]
    row = reprice.reprice_trace(trace(step(1000, 400, 100), step(2000, 1500, 10)), bands, "m-1")
    short = (600 * 1 + 400 * 0.1 + 100 * 5) / 1e6
    long = (500 * 2 + 1500 * 0.2 + 10 * 7.5) / 1e6
    assert row["usd"] == pytest.approx(short + long)
    assert (row["priced_calls"], row["long_context_calls"], row["lower_bound"]) == (2, 1, False)


def test_stream_failures_and_unmetered_calls_make_a_lower_bound():
    bands = reprice.load_bands(CATALOG, "m-1", "responses", "standard")
    row = reprice.reprice_trace(
        trace(step(100, 0, 10), step(100, 0, 10, calls=2, retry=2)), bands, "m-1"
    )
    assert (row["unpriced_calls"], row["stream_failures"], row["lower_bound"]) == (1, 1, True)
    assert row["usd"] == pytest.approx(2 * (100 * 2 + 10 * 10) / 1e6)


@pytest.mark.parametrize(
    "bad",
    [
        catalog(
            rule("gap-a", ["flex"], ("1", "0.1", "5"), maximum=1000),
            rule("gap-b", ["flex"], ("1", "0.1", "5"), minimum=1002),
        ),
        catalog(rule("open", ["flex"], ("1", "0.1", "5"), maximum=1000)),
        catalog(rule("timed", ["flex"], ("1", "0.1", "5"), utc_weekdays=[1])),
        catalog(rule("neg", ["flex"], ("-1", "0.1", "5"))),
        catalog(),
        {**CATALOG, "unit": "usd_per_token"},
    ],
)
def test_catalogs_the_tool_cannot_apply_fail_closed(bad):
    if bad.get("rules") and "utc_weekdays" in bad["rules"][0]:
        bad["rules"][0]["match"]["utc_weekdays"] = bad["rules"][0].pop("utc_weekdays")
    with pytest.raises(reprice.RepriceError):
        reprice.load_bands(bad, "m-1", "responses", "flex")


def test_other_model_steps_and_bad_cache_counts_fail_closed():
    bands = reprice.load_bands(CATALOG, "m-1", "responses", "flex")
    with pytest.raises(reprice.RepriceError):
        reprice.reprice_trace(trace(step(10, 0, 1, model="other")), bands, "m-1")
    with pytest.raises(reprice.RepriceError):
        reprice.reprice_trace(trace(step(10, 11, 1)), bands, "m-1")
    ok = reprice.reprice_trace(trace(step(10, 0, 1, model="openai/m-1")), bands, "m-1")
    assert ok["priced_calls"] == 1
    spec = "responses.m-1?reasoning=high&service_tier=flex&web_search=off"  # Harbor-recovered label
    assert (
        reprice.reprice_trace(trace(step(10, 0, 1, model=spec)), bands, "m-1")["priced_calls"] == 1
    )
    with pytest.raises(reprice.RepriceError):
        reprice.reprice_trace(
            trace(step(10, 0, 1, model="responses.m-2?reasoning=high")), bands, "m-1"
        )


def test_cli_reports_numbers_only_and_compares_harbor_cost(tmp_path, capsys):
    job = tmp_path / "job"
    for i, steps in enumerate([[step(100, 50, 10)], [step(100, 50, 10, calls=2, retry=2)]]):
        trial = job / f"task-a__{i}"
        (trial / "agent").mkdir(parents=True)
        (trial / "agent" / "trajectory.json").write_text(json.dumps(trace(*steps)))
        (trial / "result.json").write_text(json.dumps({"agent_result": {"cost_usd": 0.5}}))
    cat = tmp_path / "catalog.json"
    cat.write_text(json.dumps(CATALOG))
    argv = [str(job), "--catalog", str(cat), "--model", "m-1", "--provider", "responses"]
    assert reprice.main([*argv, "--tier", "flex", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    j = out["jobs"]["job"]
    assert (j["trials"], j["exact_trials"], j["lower_bound_trials"]) == (2, 1, 1)
    assert j["repriced_is_lower_bound"] and j["harbor_recorded_usd"] == 1.0
    assert j["lower_bound_trial_folders"] == ["task-a__1"]
    assert out["catalog"]["rules"] == ["flex-short", "flex-long"]
    assert "not provider billing" in out["basis"]
    assert reprice.main([*argv, "--tier", "fast"]) == 2  # no rule for that tier


def test_strip_prefix_rules_apply_and_other_model_matchers_fail_closed():
    cat = catalog(
        rule("x-short", None, ("2", "0.5", "6"), maximum=1000),
        rule("x-long", None, ("4", "1", "12"), minimum=1001),
    )
    for r in cat["rules"]:
        r["match"]["model"]["strip_prefixes"] = ["xai."]
        r["match"]["providers"] = ["xai"]
        del r["match"]["service_tiers"]
    assert [b.rule for b in reprice.load_bands(cat, "m-1", "xai", "standard")] == [
        "x-short",
        "x-long",
    ]
    for r in cat["rules"]:
        r["match"]["model"]["strip_after"] = "?"
    with pytest.raises(reprice.RepriceError):
        reprice.load_bands(cat, "m-1", "xai", "standard")


@pytest.mark.parametrize(
    "value",
    ["NaN", "nan", "Infinity", "-Infinity", "1e309", float("nan"), float("inf"), float("-inf")],
)
@pytest.mark.parametrize("key", ["input", "cache_read", "output"])
def test_nonfinite_catalog_rates_fail_closed(key, value):
    with pytest.raises(reprice.RepriceError):
        reprice.rate({key: value}, key, "synthetic-rule")
