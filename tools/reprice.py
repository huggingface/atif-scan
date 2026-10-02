"""Reprice Harbor job trials from their ATIF steps with a fast-agent pricing catalog.

    uv run python tools/reprice.py JOB_DIR [JOB_DIR ...] --catalog pricing_catalog.json \\
        --model gpt-6-luna --provider responses --tier flex [--json]

The catalog is card-packs' `plugins/price-calculator/pricing_catalog.json`
(schema fast-agent.pricing/v1), read as data. Pass a pinned copy; its SHA-256 and
catalog_version are reported. Each metered agent step is priced on its own, in the
prompt-token band its own prompt (cached included) falls in, the way the price
calculator prices each provider call: uncached input, cache reads and output. The
recorded ATIF has no cache-write counts, so none are priced.

A trial's price is a lower bound, never a total, when some of its LLM calls recorded no
usage: fast-agent stream failures (explicit fast-agent.retry/v1 markers; the failed
attempts report nothing), other unmetered calls, or a compacted history. Harbor's
recorded `cost_usd` is shown beside it for comparison only. Everything here is a
list-rate estimate, not provider billing. Only numbers, rule IDs and trial folder names
are printed, never trace text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from atif_scan.jsonval import (
    Doc,
    JsonObject,
    as_list,
    as_object,
    as_str,
    count,
    is_object,
    load_object,
    number,
)
from atif_scan.loader import parse_trace

MAX_TRACE_BYTES = 256 * 1024 * 1024
MAX_CATALOG_BYTES = 4 * 1024 * 1024
PER_MILLION = 1_000_000
# Rule features this tool does not implement: a candidate rule using one fails closed.
UNSUPPORTED_MATCH = {"upstream_providers", "utc_weekdays", "utc_time_ranges"}
TIERS = ("standard", "flex", "fast", "batch")


class RepriceError(ValueError):
    pass


@dataclass(frozen=True)
class Band:
    rule: str
    minimum: int
    maximum: int | None
    input: float
    cache_read: float
    output: float

    def price(self, uncached: int, cached: int, output: int) -> float:
        return (
            uncached * self.input + cached * self.cache_read + output * self.output
        ) / PER_MILLION


@dataclass
class Tally:
    trials: int = 0
    exact: int = 0
    lower_bound: int = 0
    repriced_usd: float = 0.0
    recorded_usd: float = 0.0
    recorded_known: int = 0
    unpriced_calls: int = 0
    stream_failures: int = 0
    long_context_calls: int = 0
    priced_calls: int = 0
    tokens: dict[str, int] = field(
        default_factory=lambda: {"uncached_input": 0, "cached_input": 0, "output": 0}
    )
    lower_bound_trials: list[str] = field(default_factory=list)


def rate(rates: JsonObject, key: str, rule: str) -> float:
    value = number(rates.get(key) if key in rates else None, 0)
    if value is None:
        # Catalog rates are decimal strings ("0.05"); numbers are accepted too.
        try:
            value = float(as_str(rates.get(key)) or "")
        except ValueError as exc:
            raise RepriceError(f"rule {rule}: rate {key} missing or invalid") from exc
    if not math.isfinite(value):
        raise RepriceError(f"rule {rule}: non-finite rate {key}")
    if value < 0:
        raise RepriceError(f"rule {rule}: negative rate {key}")
    return value


def _applies(rule: JsonObject, model: str, provider: str, tier: str) -> bool:
    """Whether a catalog rule prices this model, provider and service tier."""
    match = as_object(rule.get("match"))
    matcher = as_object(match.get("model"))
    values = {v.casefold() for v in as_list(matcher.get("values")) if isinstance(v, str)}
    candidate = model.casefold()
    for prefix in as_list(matcher.get("strip_prefixes")):  # e.g. "xai.grok-4.7"
        if isinstance(prefix, str):
            candidate = candidate.removeprefix(prefix.casefold())
    providers = as_list(match.get("providers"))
    tiers = match.get("service_tiers")
    if candidate not in values or provider not in providers:
        return False
    if tiers is not None and tier not in as_list(tiers):
        return False
    if (
        set(match) & UNSUPPORTED_MATCH
        or "effective" in rule
        or set(matcher) - {"values", "strip_prefixes"}
    ):
        rid = as_str(rule.get("id")) or "?"
        raise RepriceError(f"rule {rid} uses matching this tool does not implement")
    return True


def _band(rule: JsonObject) -> Band:
    rid = as_str(rule.get("id")) or "?"
    prompt = as_object(rule.get("prompt_tokens"))
    rates = as_object(rule.get("rates"))
    return Band(
        rid,
        count(prompt.get("minimum")) or 0,
        count(prompt.get("maximum")),
        rate(rates, "input", rid),
        rate(rates, "cache_read", rid),
        rate(rates, "output", rid),
    )


def _check_coverage(bands: list[Band]) -> None:
    """Bands must tile every prompt size exactly once, from 0 upwards."""
    if bands[0].minimum != 0:
        raise RepriceError("catalog bands do not start at 0 prompt tokens")
    for low, high in pairwise(bands):
        if low.maximum is None or high.minimum != low.maximum + 1:
            raise RepriceError(f"catalog bands {low.rule} and {high.rule} overlap or leave a gap")
    if bands[-1].maximum is not None:
        raise RepriceError("catalog bands do not cover every prompt size")


def load_bands(catalog: JsonObject, model: str, provider: str, tier: str) -> list[Band]:
    if catalog.get("schema") != "fast-agent.pricing/v1":
        raise RepriceError("catalog schema is not fast-agent.pricing/v1")
    if catalog.get("unit") != "usd_per_million_tokens" or catalog.get("currency") != "USD":
        raise RepriceError("catalog unit/currency is not USD per million tokens")
    rules = [as_object(r) for r in as_list(catalog.get("rules"))]
    bands = sorted(
        (_band(r) for r in rules if _applies(r, model, provider, tier)), key=lambda b: b.minimum
    )
    if not bands:
        raise RepriceError(f"no catalog rule for {model} via {provider} at tier {tier}")
    _check_coverage(bands)
    return bands


def band_for(bands: list[Band], prompt: int) -> Band:
    return next(
        b for b in bands if prompt >= b.minimum and (b.maximum is None or prompt <= b.maximum)
    )


ROUTE_PROVIDERS = ("responses.", "codexresponses.", "copilot.", "openai.", "xai.", "anthropic.")


def same_model(name: str | None, model: str) -> bool:
    """Step model label vs --model. Harbor-recovered trajectories label steps with the
    fast-agent route spec (e.g. `responses.m?reasoning=x&service_tier=flex`)."""
    if name is None:
        return True
    label = name.casefold().split("?", 1)[0].rsplit("/", 1)[-1]
    for prefix in ROUTE_PROVIDERS:
        label = label.removeprefix(prefix)
    return label == model.casefold()


def reprice_trace(raw: JsonObject, bands: list[Band], model: str) -> Doc:
    """Price one trajectory's metered agent steps; count calls that recorded no usage."""
    trace = parse_trace(raw)  # validates structure; reads retry markers and compaction
    cost = 0.0
    unpriced = priced = long_calls = 0
    tokens = {"uncached_input": 0, "cached_input": 0, "output": 0}
    for step in as_list(raw.get("steps")):
        if not is_object(step) or step.get("source") != "agent":
            continue
        if not same_model(as_str(step.get("model_name")), model):
            raise RepriceError("a step ran another model than --model")
        metrics = as_object(step.get("metrics"))
        calls = count(step.get("llm_call_count"))
        prompt, cached = count(metrics.get("prompt_tokens")), count(metrics.get("cached_tokens"))
        output = count(metrics.get("completion_tokens"))
        if prompt is None or output is None:
            unpriced += 1 if calls is None else calls
            continue
        cached = cached or 0
        if cached > prompt:
            raise RepriceError("a step records more cached than prompt tokens")
        band = band_for(bands, prompt)
        cost += band.price(prompt - cached, cached, output)
        priced += 1
        long_calls += band is not bands[0]
        unpriced += max((calls or 1) - 1, 0)
        tokens["uncached_input"] += prompt - cached
        tokens["cached_input"] += cached
        tokens["output"] += output
    return {
        "usd": cost,
        "priced_calls": priced,
        "unpriced_calls": unpriced,
        "stream_failures": trace.stream_retry_attempts,
        "long_context_calls": long_calls,
        "lower_bound": bool(unpriced or trace.compacted),
        "tokens": tokens,
    }


def trial_dirs(job: Path) -> list[Path]:
    if job.is_symlink() or not job.is_dir():
        raise RepriceError(f"not a job folder: {job.name}")
    found = []
    for trial in sorted(job.iterdir()):
        trace = trial / "agent" / "trajectory.json"
        if trial.is_symlink() or not trial.is_dir():
            continue
        if trace.is_symlink():
            raise RepriceError(f"symlinked trajectory in {trial.name}")
        if trace.is_file():
            found.append(trial)
    return found


def recorded_cost(trial: Path) -> float | None:
    path = trial / "result.json"
    if path.is_symlink() or not path.is_file():
        return None
    result = load_object(path.read_bytes(), MAX_TRACE_BYTES)
    return number(as_object(result.get("agent_result")).get("cost_usd"), 0)


def reprice_job(job: Path, bands: list[Band], model: str) -> Tally:
    tally = Tally()
    for trial in trial_dirs(job):
        raw = load_object((trial / "agent" / "trajectory.json").read_bytes(), MAX_TRACE_BYTES)
        row = reprice_trace(raw, bands, model)
        tally.trials += 1
        tally.repriced_usd += row["usd"]
        tally.priced_calls += row["priced_calls"]
        tally.unpriced_calls += row["unpriced_calls"]
        tally.stream_failures += row["stream_failures"]
        tally.long_context_calls += row["long_context_calls"]
        for key, value in row["tokens"].items():
            tally.tokens[key] += value
        if row["lower_bound"]:
            tally.lower_bound += 1
            tally.lower_bound_trials.append(trial.name)
        else:
            tally.exact += 1
        recorded = recorded_cost(trial)
        if recorded is not None:
            tally.recorded_usd += recorded
            tally.recorded_known += 1
    return tally


def report(jobs: list[Path], catalog_path: Path, args: argparse.Namespace) -> Doc:
    data = catalog_path.read_bytes()
    catalog = load_object(data, MAX_CATALOG_BYTES)
    bands = load_bands(catalog, args.model, args.provider, args.tier)
    out: Doc = {
        "basis": "list-rate estimate from ATIF step metrics; not provider billing",
        "catalog": {
            "sha256": hashlib.sha256(data).hexdigest(),
            "version": as_str(catalog.get("catalog_version")),
            "rules": [b.rule for b in bands],
        },
        "model": args.model,
        "provider": args.provider,
        "tier": args.tier,
        "jobs": {},
    }
    for job in jobs:
        t = reprice_job(job, bands, args.model)
        out["jobs"][job.name] = {
            "trials": t.trials,
            "exact_trials": t.exact,
            "lower_bound_trials": t.lower_bound,
            # A total only when every trial is exact; otherwise a lower bound.
            "repriced_usd": round(t.repriced_usd, 6),
            "repriced_is_lower_bound": t.lower_bound > 0,
            "harbor_recorded_usd": round(t.recorded_usd, 6),
            "harbor_recorded_trials": t.recorded_known,
            "priced_calls": t.priced_calls,
            "long_context_calls": t.long_context_calls,
            "unpriced_calls": t.unpriced_calls,
            "stream_failures": t.stream_failures,
            "tokens": t.tokens,
            "lower_bound_trial_folders": t.lower_bound_trials,
        }
    return out


def summary(j: Doc) -> str:
    bound = "≥" if j["repriced_is_lower_bound"] else ""
    return (
        f"{bound}${j['repriced_usd']:,.4f} over {j['trials']} trials"
        f" ({j['lower_bound_trials']} lower-bound, {j['unpriced_calls']} unpriced calls,"
        f" {j['stream_failures']} stream failures,"
        f" {j['long_context_calls']} long-context calls)"
        f" · Harbor recorded ${j['harbor_recorded_usd']:,.4f}"
        f" ({j['harbor_recorded_trials']} trials)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("jobs", nargs="+", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--tier", choices=TIERS, default="standard")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        out = report(args.jobs, args.catalog, args)
    except (RepriceError, OSError, ValueError) as exc:
        print(f"reprice: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(out, indent=1))
        return 0
    cat = out["catalog"]
    print(
        f"{out['model']} via {out['provider']} · {out['tier']} · catalog {cat['version']}"
        f" ({cat['sha256'][:12]}) · rules {', '.join(cat['rules'])}"
    )
    print(out["basis"])
    for name, j in out["jobs"].items():
        print(f"  {name}: {summary(j)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
