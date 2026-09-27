"""Run-level estimates for what the trajectories can't show directly.

Both are fitted on the run's own data and stay estimates; the brief labels them so.

- cost: unpriced trials (cost missing, or exactly $0 despite tokens) are priced with a
  least-squares fit of cost ~ uncached + cached + output tokens over the run's priced
  trials (a single per-token rate if the fit is ill-posed).
- missing activity: trials whose history was compacted record only their last context
  window, while their token totals cover the whole session. Prompt tokens per recorded
  LLM call in uncompacted trials give a typical per-call context; dividing a compacted
  trial's prompt tokens by it estimates how many calls it really made. The median and
  p90 references give a range (validated against three Grok Build sessions whose full
  history was recoverable: recorded share 26.5%/1.6%/1.3% vs estimated 21-27%/0.7-0.9%/
  0.7-0.9%).
"""

from __future__ import annotations

MIN_PRICED = 20
MIN_REFERENCES = 10


def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; None if (near) singular."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col], strict=True)]
    return [m[i][n] / m[i][i] for i in range(n)]


def _features(item: dict) -> list[float]:
    inp, cache, out = (item.get(k) or 0 for k in ("input_tokens", "cache_tokens", "output_tokens"))
    return [float(max(inp - cache, 0)), float(cache), float(out)]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0.0


def cost_estimate(items: list[dict], price: tuple[float, float, float] | None = None) -> dict:
    has_tokens = [i for i in items if i.get("input_tokens") or i.get("output_tokens")]
    unpriced = [i for i in has_tokens if not i.get("cost_usd")]
    priced = [i for i in has_tokens if i.get("cost_usd")]
    result: dict = {
        "unpriced": len(unpriced),
        "unpriced_ids": [i["input_id"] for i in unpriced],
        "no_usage": sum(1 for i in items if not (i.get("input_tokens") or i.get("output_tokens"))),
        "estimate_usd": None,
        "method": None,
        "median_abs_error_usd": None,
        "rates_per_mtok": None,
    }
    tokens = [_features(i) for i in has_tokens]
    result["tokens"] = {
        "uncached_input": int(sum(t[0] for t in tokens)),
        "cached_input": int(sum(t[1] for t in tokens)),
        "output": int(sum(t[2] for t in tokens)),
    }
    if not unpriced:
        return result
    if price is not None:
        rates = list(price)
        result["method"] = "given --price"
        result["rates_per_mtok"] = dict(
            zip(("uncached_input", "cached_input", "output"), rates, strict=True)
        )
        result["estimate_usd"] = round(
            sum(
                sum(r * v / 1e6 for r, v in zip(rates, _features(i), strict=True)) for i in unpriced
            ),
            2,
        )
        return result
    if len(priced) < MIN_PRICED:
        result["method"] = f"not estimated: fewer than {MIN_PRICED} priced trials"
        return result
    x = [_features(i) for i in priced]
    y = [float(i["cost_usd"]) for i in priced]
    # Scale columns to per-million tokens for a well-conditioned 3x3 system.
    xs = [[v / 1e6 for v in row] for row in x]
    xtx = [[sum(r[a] * r[b] for r in xs) for b in range(3)] for a in range(3)]
    xty = [sum(r[a] * t for r, t in zip(xs, y, strict=True)) for a in range(3)]
    rates = _solve(xtx, xty)
    if rates is not None and all(r >= 0 for r in rates):
        predict = lambda row: sum(r * v / 1e6 for r, v in zip(rates, row, strict=True))  # noqa: E731
        result["method"] = f"per-token fit on {len(priced)} priced trials"
        result["rates_per_mtok"] = {
            "uncached_input": round(rates[0], 4),
            "cached_input": round(rates[1], 4),
            "output": round(rates[2], 4),
        }
    else:
        rate = sum(y) / max(sum(sum(row) for row in x), 1.0)
        predict = lambda row: rate * sum(row)  # noqa: E731
        result["method"] = f"average $/token over {len(priced)} priced trials"
    errors = [abs(predict(row) - t) for row, t in zip(x, y, strict=True)]
    result["median_abs_error_usd"] = round(_median(errors), 3)
    result["estimate_usd"] = round(sum(max(predict(_features(i)), 0.0) for i in unpriced), 2)
    return result


def missing_activity(items: list[dict]) -> dict:
    rows = [i for i in items if i.get("input_tokens") and i.get("llm_calls")]
    compacted = [i for i in rows if i.get("compacted")]
    result: dict = {
        "compacted": sum(1 for i in items if i.get("compacted")),
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
        if not i.get("compacted") and i["llm_calls"] >= 5
    )
    result["reference_trials"] = len(refs)
    if len(refs) < MIN_REFERENCES:
        result["method"] = f"not estimated: fewer than {MIN_REFERENCES} uncompacted trials"
        return result
    median, p90 = refs[len(refs) // 2], refs[min(len(refs) - 1, int(0.9 * len(refs)))]

    def expected(item: dict, per_call: float) -> float:
        if not item.get("compacted"):
            return float(item["llm_calls"])
        return max(float(item["llm_calls"]), item["input_tokens"] / per_call)

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
