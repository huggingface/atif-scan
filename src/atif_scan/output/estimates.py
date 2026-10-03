"""Run-level estimates for what the trajectories can't show directly.

Both are fitted on the run's own data and stay estimates; the brief labels them so.

- cost: unpriced trials (cost missing, or exactly $0 despite tokens) are priced with a
  least-squares fit of cost ~ uncached + cached + output tokens over the run's priced
  trials (a single per-token rate if the fit is ill-posed).
- activity outside ATIF: compacted trajectories record only their last context
  window, while their token totals cover the whole session. Earlier history may still
  be available in companion archives; the estimate is not source unavailability.
  Prompt tokens per recorded LLM call in uncompacted trials give a typical per-call
  context; dividing a compacted trial's prompt tokens by it estimates how many calls
  it really made. The median and
  p90 references give a range (validated against three Grok Build sessions whose full
  history was recoverable: recorded share 26.5%/1.6%/1.3% vs estimated 21-27%/0.7-0.9%/
  0.7-0.9%).
- unmetered work: trials that recorded agent work (LLM or tool calls) but report neither
  tokens nor cost, e.g. because the agent process died before writing usage. The run's
  total cost silently omits them, so they are counted and their cost is estimated from
  recorded LLM calls: cost ~ a*calls + b*calls^2 over the run's priced, uncompacted
  trials (each call resends a growing context). That is rough per trial but close to
  unbiased in aggregate; a single $/call ratio if the fit is ill-posed. When the run
  declares its prices (or --price gives them), trials priced at those rates are
  references too.
- declared prices: a run that declares its token prices (harbor-hf's run.json) is priced
  at them instead of a fit, and any cost it also recorded is checked against them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..data.jsonval import Doc

from ..data.accounting import has_scoped_cost

MIN_PRICED = 20
MIN_REFERENCES = 10
# Trials with fewer LLM calls say little about cost or context per call.
MIN_REFERENCE_CALLS = 5
# Pivots below this are treated as zero: the least-squares system is (near) singular.
SINGULAR = 1e-12
# Quantile of per-call prompt tokens used for the "fewer, larger calls" bound.
UPPER_QUANTILE = 0.9

# $/M tokens, in TOKEN_KINDS order: (uncached input, cached input, output).
Rates = tuple[float, float, float]
TOKEN_KINDS = ("uncached_input", "cached_input", "output")
PRICE_SOURCES = {"given": "given --price", "declared": "at the run's declared prices, run.json"}


def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; None if (near) singular."""
    n = len(b)
    m = [[*row, b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < SINGULAR:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col], strict=True)]
    return [m[i][n] / m[i][i] for i in range(n)]


def _least_squares(xs: list[list[float]], y: list[float]) -> list[float] | None:
    """Coefficients minimising |xs·c - y|² (normal equations); None if ill-posed."""
    k = len(xs[0])
    xtx = [[sum(r[a] * r[b] for r in xs) for b in range(k)] for a in range(k)]
    xty = [sum(r[a] * t for r, t in zip(xs, y, strict=True)) for a in range(k)]
    return _solve(xtx, xty)


def _features(item: Doc) -> list[float]:
    """A trial's (uncached input, cached input, output) tokens."""
    inp, cache, out = (item.get(k) or 0 for k in ("input_tokens", "cache_tokens", "output_tokens"))
    return [float(max(inp - cache, 0)), float(cache), float(out)]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0.0


def _price(rates: Sequence[float], row: list[float]) -> float:
    """Cost of one trial's (uncached, cached, output) tokens at $/M `rates`."""
    return sum(r * v / 1e6 for r, v in zip(rates, row, strict=True))


def _has_tokens(item: Doc) -> bool:
    return bool(item.get("input_tokens") or item.get("output_tokens"))


def _token_totals(rows: list[list[float]]) -> dict[str, int]:
    return {kind: int(sum(row[n] for row in rows)) for n, kind in enumerate(TOKEN_KINDS)}


@dataclass(frozen=True)
class Pricing:
    """The run's token rates, chosen once (`choose_pricing`) and used for every estimate.

    `source`: "given" (--price) or "declared" (run.json) rates are known; "fit" rates
    and an "average" flat $/token are fitted on recorded costs; None: no rates."""

    rates: Rates | None = None
    source: str | None = None
    method: str | None = None
    flat_rate: float | None = None  # "average": $ per token of any kind
    median_abs_error_usd: float | None = None

    @property
    def known(self) -> bool:
        return self.source in PRICE_SOURCES

    def price(self, row: list[float]) -> float | None:
        """Cost of (uncached, cached, output) tokens at these rates; None without rates."""
        if self.flat_rate is not None:
            return self.flat_rate * sum(row)
        return None if self.rates is None else _price(self.rates, row)

    def trial_cost(self, item: Doc) -> float | None:
        """A trial's recorded cost, else its tokens at known rates. Fitted rates never
        price a trial here: other estimates use these costs as references."""
        if item.get("usage_basis") == "run_observed" or has_scoped_cost(item):
            return None
        if item.get("cost_usd"):
            return float(item["cost_usd"])
        if self.known and self.rates is not None and _has_tokens(item):
            return _price(self.rates, _features(item))
        return None


def _priced(items: Sequence[Doc], other_model: frozenset[str]) -> list[tuple[list[float], float]]:
    """(tokens, recorded cost) of trials with both, on the run's own model."""
    return [
        (_features(i), float(i["cost_usd"]))
        for i in items
        if _has_tokens(i)
        and i.get("cost_usd")
        and i.get("usage_basis") != "run_observed"
        and i["input_id"] not in other_model
    ]


def _priced_other(items: Sequence[Doc], other_model: frozenset[str]) -> int:
    """Trials with tokens and a recorded cost that ran another model (never references)."""
    return sum(
        1 for i in items if _has_tokens(i) and i.get("cost_usd") and i["input_id"] in other_model
    )


def _fit_rates(priced: list[tuple[list[float], float]]) -> Pricing:
    """Per-token rates by least squares over priced trials, else one average $/token."""
    x = [row for row, _ in priced]
    y = [cost for _, cost in priced]
    # Scale columns to per-million tokens for a well-conditioned 3x3 system.
    rates = _least_squares([[v / 1e6 for v in row] for row in x], y)
    if rates is not None and all(r >= 0 for r in rates):
        fitted = Pricing(
            rates=(rates[0], rates[1], rates[2]),
            source="fit",
            method=f"per-token fit on {len(priced)} priced trials",
        )
    else:
        fitted = Pricing(
            source="average",
            method=f"average $/token over {len(priced)} priced trials",
            flat_rate=sum(y) / max(sum(sum(row) for row in x), 1.0),
        )
    errors = [abs((fitted.price(row) or 0.0) - t) for row, t in priced]
    return Pricing(
        fitted.rates,
        fitted.source,
        fitted.method,
        fitted.flat_rate,
        round(_median(errors), 3),
    )


def choose_pricing(
    items: Sequence[Doc],
    given: Rates | None = None,
    declared: Rates | None = None,
    other_model: frozenset[str] = frozenset(),
) -> Pricing:
    """The run's rates: `given` (--price), else `declared` (run.json), else fitted on
    the run's recorded costs. `other_model` trials ran another model (e.g. a fallback)
    at its prices, so they never train the fit: a run whose only priced trials were
    fallbacks gets no rates rather than wrong ones."""
    for rates, source in ((given, "given"), (declared, "declared")):
        if rates:
            return Pricing(rates=rates, source=source, method=PRICE_SOURCES[source])
    priced = _priced(items, other_model)
    if len(priced) >= MIN_PRICED:
        return _fit_rates(priced)
    other = _priced_other(items, other_model)
    return Pricing(
        method=f"not estimated: fewer than {MIN_PRICED} priced trials"
        + (f" on the run's model ({other} priced trial(s) ran another model)" if other else "")
    )


def cost_estimate(
    items: Sequence[Doc],
    pricing: Pricing | None = None,
    other_model: frozenset[str] = frozenset(),
) -> Doc:
    """Unpriced trials (tokens but no cost, or exactly $0) priced at the run's rates
    (`pricing`; fitted on `items` when not given). `other_model`: input IDs of trials
    that ran another model than the run's (their priced trials are counted apart)."""
    if pricing is None:
        pricing = choose_pricing(items, other_model=other_model)
    has_tokens = [(i, _features(i)) for i in items if _has_tokens(i)]
    candidates = [(i, row) for i, row in has_tokens if not has_scoped_cost(i)]
    unpriced = [row for i, row in candidates if not i.get("cost_usd")]
    result: Doc = {
        "unpriced": len(unpriced),
        "unpriced_ids": [i["input_id"] for i, _ in candidates if not i.get("cost_usd")],
        "no_usage": len(items) - len(has_tokens),
        "priced_other_model": _priced_other(items, other_model),  # left out of the fit
        "estimate_usd": None,
        "method": None,
        "median_abs_error_usd": None,
        "rates_per_mtok": None,
        "tokens": _token_totals([row for _, row in has_tokens]),
        "unpriced_tokens": _token_totals(unpriced),
    }
    if not unpriced:
        return result
    result["method"] = pricing.method
    if pricing.known:
        result["price_source"] = pricing.source
    if pricing.rates is not None:
        digits = None if pricing.known else 4  # fitted rates are rounded, known ones exact
        result["rates_per_mtok"] = {
            k: r if digits is None else round(r, digits)
            for k, r in zip(TOKEN_KINDS, pricing.rates, strict=True)
        }
    result["median_abs_error_usd"] = pricing.median_abs_error_usd
    costs = [pricing.price(row) for row in unpriced]
    if all(c is not None for c in costs):
        result["estimate_usd"] = round(sum(max(c or 0.0, 0.0) for c in costs), 2)
    return result


def price_check(items: Sequence[Doc], price: Rates | None) -> Doc | None:
    """Recorded costs against the run's declared prices: a recorded cost that the
    trial's own tokens at those prices don't explain (beyond 2% and $0.01) is a
    mismatch. None without prices; `compared` 0 when no trial recorded both."""
    if price is None:
        return None
    rows = [
        (float(i["cost_usd"]), _price(price, _features(i)))
        for i in items
        if i.get("cost_usd") and _has_tokens(i) and i.get("usage_basis") != "run_observed"
    ]
    mismatched = [(c, p) for c, p in rows if abs(c - p) > max(0.01, 0.02 * max(c, p))]
    return {
        "compared": len(rows),
        "mismatched": len(mismatched),
        "recorded_usd": round(sum(c for c, _ in rows), 2),
        "at_prices_usd": round(sum(p for _, p in rows), 2),
    }


def partial_usage(items: Sequence[Doc], pricing: Pricing | None = None) -> Doc:
    """Trials whose tokens come from step metrics with some LLM calls unmetered, or some
    token kinds on only some steps: their tokens (and cost) are lower bounds. With a cost
    (recorded, else at known rates), the unmetered calls are estimated at the trial's own
    metered cost per call (rough)."""
    pricing = pricing or Pricing()
    rows = [i for i in items if i.get("usage_basis") == "steps_partial"]
    estimate, priced = 0.0, 0
    for i in rows:
        missing = i.get("calls_without_usage") or 0
        metered = (i.get("llm_calls") or 0) - missing
        cost = pricing.trial_cost(i)
        if cost and metered > 0:
            estimate += cost * missing / metered
            priced += 1
    kinds: dict[str, int] = {}
    for i in rows:
        for kind in i.get("step_kinds_partial") or []:
            kinds[kind] = kinds.get(kind, 0) + 1
    return {
        "trials": len(rows),
        "ids": [i["input_id"] for i in rows],
        "calls_without_usage": sum(i.get("calls_without_usage") or 0 for i in rows),
        # Token kind -> trials where only some steps record it (a lower bound).
        "kinds_partial": dict(sorted(kinds.items())),
        "rewarded": sum(1 for i in rows if (i.get("reward") or 0) > 0),
        # Only when every such trial could be priced: a partial sum would read as whole.
        "estimate_usd": round(estimate, 4) if rows and priced == len(rows) else None,
        # Overlap in counts only: retry markers do not prove which calls lack usage.
        "stream_retry_calls": sum(
            min(i.get("stream_retry_attempts") or 0, i.get("calls_without_usage") or 0)
            for i in rows
        ),
    }


def stream_retries(items: Sequence[Doc]) -> Doc:
    """Trials with explicit fast-agent provider retries, regardless of usage basis.
    Markers alone do not establish absent usage or incomplete billing. Legacy report
    keys retain "stream" for compatibility."""
    rows = [i for i in items if (i.get("stream_retry_attempts") or 0) > 0]
    return {
        "trials": len(rows),
        "ids": [i["input_id"] for i in rows],
        "failed_attempts": sum(i["stream_retry_attempts"] for i in rows),
        "rewarded": sum(1 for i in rows if (i.get("reward") or 0) > 0),
        "without_totals": sum(1 for i in rows if i.get("usage_basis") == "steps_partial"),
    }


def _unmetered(item: Doc) -> bool:
    """Recorded agent work (LLM or tool calls), but no tokens and no cost at all."""
    worked = item.get("llm_calls") or item.get("tool_calls")
    return (
        bool(worked)
        and not _has_tokens(item)
        and item.get("cost_usd") is None
        and not has_scoped_cost(item)
    )


def _call_references(
    items: Sequence[Doc], pricing: Pricing, other_model: frozenset[str]
) -> list[tuple[float, float]]:
    """(LLM calls, cost) of whole-history, fully metered trials on the run's model."""
    refs = []
    for i in items:
        if (
            (i.get("llm_calls") or 0) < MIN_REFERENCE_CALLS
            or i.get("compacted")
            or i.get("usage_basis") == "steps_partial"
            or i["input_id"] in other_model
        ):
            continue
        if cost := pricing.trial_cost(i):
            refs.append((float(i["llm_calls"]), cost))
    return refs


def _fit_calls(refs: list[tuple[float, float]]) -> tuple[Callable[[float], float], str]:
    """cost ~ a*calls + b*calls² by least squares, else one average $/call."""
    xs = [[c / 100, (c / 100) ** 2] for c, _ in refs]  # per 100 calls: well-conditioned
    coef = _least_squares(xs, [y for _, y in refs])
    if coef is not None and all(k >= 0 for k in coef):
        a, b = coef

        def quadratic(calls: float) -> float:
            return a * calls / 100 + b * (calls / 100) ** 2

        return quadratic, f"cost ~ LLM calls + calls² fit on {len(refs)} priced trials"
    rate = sum(y for _, y in refs) / sum(c for c, _ in refs)

    def linear(calls: float) -> float:
        return rate * calls

    return linear, f"average $/LLM call over {len(refs)} priced trials"


def unmetered_work(
    items: Sequence[Doc],
    pricing: Pricing | None = None,
    other_model: frozenset[str] = frozenset(),
) -> Doc:
    """Trials with recorded agent work but no usage (tokens) and no cost at all, priced
    from LLM calls. References are trials with a cost (recorded, else at known rates);
    `other_model` trials (another model's prices) are left out of the fit."""
    rows = [i for i in items if _unmetered(i)]
    result: Doc = {
        "trials": len(rows),
        "ids": [i["input_id"] for i in rows],
        "llm_calls": sum(i.get("llm_calls") or 0 for i in rows),
        "tool_calls": sum(i.get("tool_calls") or 0 for i in rows),
        "duration_sec": round(sum(i.get("duration_sec") or 0 for i in rows), 1),
        "rewarded": sum(1 for i in rows if (i.get("reward") or 0) > 0),
        "errored": sum(1 for i in rows if i.get("error_type")),
        "estimate_usd": None,
        "method": None,
        "median_abs_error_usd": None,
    }
    if not rows:
        return result
    refs = _call_references(items, pricing or Pricing(), other_model)
    if len(refs) < MIN_PRICED:
        result["method"] = f"not estimated: fewer than {MIN_PRICED} priced trials with LLM calls"
        return result
    predict, result["method"] = _fit_calls(refs)
    result["median_abs_error_usd"] = round(_median([abs(predict(c) - y) for c, y in refs]), 3)
    result["estimate_usd"] = round(sum(predict(float(i.get("llm_calls") or 0)) for i in rows), 2)
    return result


def missing_activity(items: Sequence[Doc]) -> Doc:
    rows = [
        i
        for i in items
        if i.get("input_tokens") and i.get("llm_calls") and i.get("usage_basis") != "run_observed"
    ]
    compacted = [i for i in rows if i.get("compacted")]
    result: Doc = {
        "compacted": sum(1 for i in items if i.get("compacted")),
        "archives_available": sum(
            bool(i.get("compacted"))
            and (i.get("history_archive") or {}).get("status") == "available"
            for i in items
        ),
        "missing_calls_pct": None,
        "compacted_recorded_pct": None,
        "reference_trials": 0,
        "method": None,
    }
    if not compacted:
        return result
    refs = sorted(
        i["input_tokens"] / i["llm_calls"]
        for i in rows
        if not i.get("compacted") and i["llm_calls"] >= MIN_REFERENCE_CALLS
    )
    result["reference_trials"] = len(refs)
    if len(refs) < MIN_REFERENCES:
        result["method"] = f"not estimated: fewer than {MIN_REFERENCES} uncompacted trials"
        return result
    median, p90 = refs[len(refs) // 2], refs[min(len(refs) - 1, int(UPPER_QUANTILE * len(refs)))]

    def expected(item: Doc, per_call: float) -> float:
        if not item.get("compacted"):
            return float(item["llm_calls"])
        return max(float(item["llm_calls"]), float(item["input_tokens"]) / per_call)

    recorded = sum(i["llm_calls"] for i in rows)
    low = 1 - recorded / sum(expected(i, p90) for i in rows)  # fewer, larger calls
    high = 1 - recorded / sum(expected(i, median) for i in rows)
    c_recorded = sum(i["llm_calls"] for i in compacted)
    result["missing_calls_pct"] = [round(100 * low, 1), round(100 * high, 1)]
    result["compacted_recorded_pct"] = [
        round(100 * c_recorded / sum(expected(i, median) for i in compacted), 1),
        round(100 * c_recorded / sum(expected(i, p90) for i in compacted), 1),
    ]
    result["method"] = (
        f"token totals / per-call prompt tokens of {len(refs)} uncompacted trials (median-p90)"
    )
    return result
