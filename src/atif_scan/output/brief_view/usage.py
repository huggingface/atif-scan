"""TOKENS and COST: recorded usage checked against itself, then priced (declared or fitted
rates, estimates said as such), plus walltime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..document import COMPACTED_USAGE_EXPLANATION
from ..estimates import LAST_CALL_CHECK
from ..overview import _m, walltime_lines

if TYPE_CHECKING:
    from ...data.jsonval import Doc
from .evidence import USAGE_CHECKS
from .words import (
    CENT,
    INFO,
    OK,
    RATIO_BASIS,
    WARN,
    Lines,
    _present,
    _tally,
    _title_inline,
    counts,
    pct,
    plural,
    rate,
    spread,
    usd,
    wrap,
)


def _recorded_vs_trajectory(codes: dict[str, int]) -> Lines:
    compared = sum(codes.values())
    if not compared:
        return []
    same, uncached, differs = (codes.get(k, 0) for k in ("same", "uncached_input", "differs"))
    texts = []
    agree = same + uncached
    if differs:
        texts.append(
            f"{WARN} recorded totals differ from the trajectory's in {differs:,} of"
            f" {plural(compared, 'trial')}"
        )
    else:
        texts.append(
            f"{OK} recorded totals match the trajectories' in {agree:,} of"
            f" {plural(compared, 'compared trial')}"
        )
    texts.append(
        f"{INFO} matching token records do not establish that every provider attempt was counted"
    )
    if uncached:
        texts.append(
            f"{INFO} {plural(uncached, 'record')} count uncached input only (a harness"
            " convention): their input is shown with cache, from the trajectory"
        )
    return texts


def _steps_vs_totals(u: Doc) -> Lines:
    codes = _tally(u.get("steps_vs_totals"))
    compared = sum(codes.values())
    if not compared:
        return []
    short, differs = codes.get("steps_short", 0), codes.get("differs", 0)
    texts = []
    if not short and not differs:
        texts.append(
            f"{OK} trajectory totals reconcile with recorded usage in {compared:,} of"
            f" {plural(compared, 'compared trial')}"
        )
    if short:
        share = pct(u.get("tokens_outside_steps") or 0, u.get("tokens_of_short_trials") or 0)
        texts.append(
            f"{WARN} {short:,} of {plural(compared, 'trajectory', 'trajectories')} count"
            f" {_m(u.get('tokens_outside_steps') or 0)} tokens ({share} of theirs) outside"
            " their recorded step sums; the cause is not established by this comparison"
        )
    if differs:
        texts.append(
            f"{WARN} {differs:,} of {plural(compared, 'trajectory', 'trajectories')} have"
            " totals that disagree with their steps"
        )
    return texts


BASIS = {
    "run": "run records",
    "run_observed": "observed run counts (not complete totals)",
    "final_metrics": "trajectory totals",
    "steps": "step sums",
    "steps_partial": "partial step sums",
    "none": "an unknown source",
}


def _basis(u: Doc) -> Lines:
    """Where the token counts come from, unless all from the run's own records."""
    basis = _tally(u.get("basis"))
    if not basis or set(basis) == {"run"}:
        return []
    return [f"{INFO} counts from " + counts((BASIS.get(k, k), v) for k, v in basis.items())]


def _stream_retry_reason(explained: int, calls: int) -> str:
    """Retry context alongside missing usage; markers alone do not explain the gap."""
    if not explained:
        return ""
    share = "" if explained >= calls else f" ({explained:,} of them)"
    return f"; fast-agent also records provider retries{share}"


def _stream_retries(b: Doc) -> Lines:
    """Provider retry events, without assuming stream failure or absent usage."""
    sr = (b.get("usage") or {}).get("stream_retries") or {}
    if not sr.get("trials"):
        return []
    rewarded = f" ({sr['rewarded']:,} rewarded)" if sr["rewarded"] else ""
    return [
        f"{INFO} {plural(sr['trials'], 'trial')}{rewarded} had fast-agent provider"
        f" failures: {plural(sr['failed_attempts'], 'failed attempt')} retried by the"
        " harness (provider/transport events, not model behaviour). Retry markers alone"
        " do not establish missing usage or missing agent history"
    ]


def _usage_gaps(b: Doc) -> Lines:
    pu = (b.get("usage") or {}).get("partial") or {}
    um = b.get("unmetered_work") or {"trials": 0}
    texts = []
    observed = (b.get("usage") or {}).get("observed_accounting", 0)
    if observed:
        texts.append(
            f"{WARN} {plural(observed, 'trial')} retain observed run counts, not complete"
            " totals; total provider usage and billing are not established"
        )
    refusals = (b.get("usage") or {}).get("refusals_without_token_counts", 0)
    if refusals:
        texts.append(
            f"{INFO} {plural(refusals, 'trial')} with recorded refusal"
            " (AgentSafetyRefusalError) have no token counts; consumption remains unknown,"
            " even with recorded zero cost. This is not evidence of misconduct."
        )
    if pu.get("trials"):
        rewarded = f" ({pu['rewarded']:,} rewarded)" if pu["rewarded"] else ""
        gaps = []
        if pu["calls_without_usage"]:
            why = _stream_retry_reason(pu.get("stream_retry_calls") or 0, pu["calls_without_usage"])
            gaps.append(f"{plural(pu['calls_without_usage'], 'LLM call')} record no usage{why}")
        gaps += [
            f"{kind} tokens are missing on some steps in {plural(n, 'trial')}"
            for kind, n in (pu.get("kinds_partial") or {}).items()
        ]
        texts.append(
            f"{WARN} {plural(pu['trials'], 'trial')}{rewarded} recorded no totals, so their"
            " recorded step usage is retained as a lower bound; total provider usage and"
            " billing are not established: " + "; ".join(gaps)
        )
    if um["trials"]:
        why = ", ".join(
            f"{v:,} {k}" for k, v in (("errored", um["errored"]), ("rewarded", um["rewarded"])) if v
        )
        texts.append(
            f"{WARN} {plural(um['trials'], 'trial')}"
            + (f" ({why})" if why else "")
            + f" did work but recorded no usage: {plural(um['llm_calls'], 'LLM call')},"
            f" {plural(um['tool_calls'], 'tool call')}, {um['duration_sec'] / 60:,.1f} min"
        )
    return texts


def _text_ratio(b: Doc) -> Lines:
    return [
        f"{INFO} visible agent text per output token: median {r['median']:.2f} characters"
        f"{spread(r)} over {plural(r['traces'], 'trace')}, {RATIO_BASIS[basis]}"
        for basis, r in (b.get("output_ratio") or {}).items()
    ]


def _usage_title(b: Doc, check: str) -> str:
    if check == "integrity.incomplete_tool_generation":
        return "Incomplete tool generation failed"
    return _title_inline(b, check)


def _token_finding_texts(b: Doc, n: int) -> Lines:
    texts: Lines = []
    for check in USAGE_CHECKS:
        count = b["recording"].get(check, 0)
        compacted = (
            b.get("compacted_token_hits", 0)
            if check == "integrity.tokens_exceed_recorded_calls"
            else 0
        )
        if compacted:
            texts.append(
                f"{WARN} {plural(compacted, 'trial')} ({pct(compacted, n)}), compacted:"
                f" {COMPACTED_USAGE_EXPLANATION}"
            )
        if remaining := count - compacted:
            texts.append(
                f"{WARN} {plural(remaining, 'trial')} ({pct(remaining, n)}):"
                f" {_usage_title(b, check)}"
                + (
                    "; recorded usage may omit failed-attempt tokens"
                    if check == "integrity.incomplete_tool_generation"
                    else _last_call_tokens(b)
                    if check == LAST_CALL_CHECK
                    else ""
                )
            )
    return texts


def _last_call_tokens(b: Doc) -> str:
    """What the steps of trials with last-call totals add, in tokens."""
    tk = (b.get("last_call_totals") or {}).get("tokens") or {}
    if not tk:
        return ""
    return (
        f"; their steps record {_m(tk.get('uncached_input', 0) + tk.get('cached_input', 0))}"
        f" input ({_m(tk.get('cached_input', 0))} cached) · {_m(tk.get('output', 0))} output"
        " more, not in the totals above"
    )


def tokens_section(b: Doc) -> Lines:
    u, n = b.get("usage") or {}, _present(b)
    tk = b["cost_estimate"].get("tokens") or {}
    with_tokens = u.get("trials_with_tokens", 0)
    if not with_tokens:
        return wrap("TOKENS", [f"{WARN} no trial recorded token counts", *_usage_gaps(b)])
    head = (
        f"{_m(tk.get('uncached_input', 0) + tk.get('cached_input', 0))} input"
        f" ({_m(tk.get('cached_input', 0))} cached) · {_m(tk.get('output', 0))} output,"
        f" from {with_tokens:,} of {plural(n, 'trial')}"
    )
    checks = _basis(u) + _recorded_vs_trajectory(_tally(u.get("recorded_vs_trajectory")))
    checks += _steps_vs_totals(u)
    if not u.get("recorded_vs_trajectory") and not u.get("steps_vs_totals"):
        checks.append(f"{INFO} no second record of the tokens to check them against")
    checks += _token_finding_texts(b, n)
    return wrap("TOKENS", [head, *checks, *_usage_gaps(b), *_stream_retries(b), *_text_ratio(b)])


def _rates_text(ce: Doc) -> str:
    r = ce["rates_per_mtok"]
    return (
        f"{rate(r['uncached_input'])} uncached input · {rate(r['cached_input'])} cached input"
        f" · {rate(r['output'])} output, per M tokens"
    )


def _cost_head(b: Doc) -> Lines:
    """The headline: what was recorded, and what the unrecorded part adds."""
    ce, n = b["cost_estimate"], _present(b)
    total = b["overview"]["cost"]["total_usd"]
    with_usage = n - ce["no_usage"]
    scoped = (b.get("scoped_costs") or {}).get("trials")
    if not ce["unpriced"]:
        qualifier = (
            "final bills; actual bill unknown for observed-cost trials"
            if scoped
            else f"· {OK} every trial with usage has a cost"
        )
        return [f"{usd(total)} recorded {qualifier}"]
    source = ce.get("price_source")
    if ce["unpriced"] == with_usage and not total:
        return _no_cost_recorded(ce, source)
    head = f"{usd(total)} recorded" + _other_model_spend(b)
    missing = (
        f"{WARN} {plural(ce['unpriced'], 'trial')} ({pct(ce['unpriced'], n)}) with usage but"
        " no cost"
    )
    if ce["estimate_usd"] is None:
        return [head, missing + _unestimated(ce)]
    whole = total + ce["estimate_usd"]
    how = "at the run's declared prices" if source == "declared" else ce["method"]
    error = (
        f", median error {usd(ce['median_abs_error_usd'])} per trial"
        if ce.get("median_abs_error_usd") is not None
        else ""
    )
    scope = "excluding observed-cost trials" if scoped else "in all"
    return [
        head,
        f"{missing}: est. +{usd(ce['estimate_usd'])} ({how}{error}) → est. {usd(whole)} {scope}",
    ]


def _no_cost_recorded(ce: Doc, source: str | None) -> Lines:
    """No trial recorded a cost: priced at the declared or given rates, if any."""
    if source == "declared":
        return [
            f"{usd(ce['estimate_usd'])} at the run's declared prices (run.json); no trial"
            " recorded a cost",
            f"{INFO} {_rates_text(ce)}",
        ]
    if source == "given":
        return [
            f"est. {usd(ce['estimate_usd'])} at the given --price; no trial recorded a cost",
            f"{INFO} {_rates_text(ce)}",
        ]
    return [
        f"{WARN} no trial recorded a cost; price the tokens with --price U,C,O"
        " ($ per M uncached input, cached input, output)"
    ]


def _unestimated(ce: Doc) -> str:
    if ce.get("priced_other_model"):
        ut = ce["unpriced_tokens"]
        return (
            f": not estimated, since only {plural(ce['priced_other_model'], 'trial')} on another"
            " model recorded a price; their"
            f" {_m(ut['uncached_input'] + ut['cached_input'])} input"
            f" ({_m(ut['cached_input'])} cached) and {_m(ut['output'])} output tokens can be"
            " priced with --price U,C,O"
        )
    method = ce["method"]  # estimates.py already words it "not estimated: <why>"
    return f" ({method})" if method.startswith("not estimated") else f": not estimated ({method})"


def _other_model_spend(b: Doc) -> str:
    mm, total = b["overview"].get("model_mismatch") or {}, b["overview"]["cost"]["total_usd"]
    if not (b["cost_estimate"].get("priced_other_model") and mm.get("cost_usd")):
        return ""
    if abs(mm["cost_usd"] - total) < CENT:
        return ", all of it on another model"
    return f", {usd(mm['cost_usd'])} of it on another model"


def _last_call_cost(b: Doc) -> Lines:
    lc = b.get("last_call_totals") or {}
    if not lc.get("trials"):
        return []
    who = f"{plural(lc['trials'], 'trial')} whose totals cover only their last model call"
    if lc.get("estimate_usd") is None:
        return [f"{WARN} the {who} undercount, not estimated ({lc.get('method')})"]
    return [
        f"{WARN} est. +{usd(lc['estimate_usd'])} from the step metrics of {who}, not in the"
        f" total ({lc['method']})"
    ]


def _gap_costs(b: Doc) -> Lines:
    pu = (b.get("usage") or {}).get("partial") or {}
    um = b.get("unmetered_work") or {"trials": 0}
    texts = _last_call_cost(b)
    if pu.get("trials") and pu.get("estimate_usd") is not None:
        texts.append(
            f"{INFO} est. +{usd(pu['estimate_usd'])} for the"
            f" {plural(pu['calls_without_usage'], 'LLM call')} without usage (each trial's own"
            " cost per call)"
        )
    if um["trials"]:
        if um["estimate_usd"] is not None:
            texts.append(
                f"{WARN} est. +{usd(um['estimate_usd'])} for the"
                f" {plural(um['trials'], 'trial')} without usage, not in the total (rough:"
                f" {um['method']}, median error {usd(um['median_abs_error_usd'])} per trial)"
            )
        else:
            texts.append(
                f"{WARN} the {plural(um['trials'], 'trial')} without usage"
                f" {'is' if um['trials'] == 1 else 'are'}n't in the total"
                f" ({um['method']})"
            )
    return texts


def _price_checks(b: Doc) -> Lines:
    ci = b.get("cost_integrity") or {}
    pc = ci.get("price_check")
    texts = []
    if pc and pc["mismatched"]:
        texts.append(
            f"{WARN} {pc['mismatched']:,} of {plural(pc['compared'], 'recorded cost')} don't"
            f" fit the declared prices ({usd(pc['recorded_usd'])} recorded,"
            f" {usd(pc['at_prices_usd'])} at those prices)"
        )
    elif pc and pc["compared"]:
        texts.append(
            f"{OK} recorded costs fit the declared prices ({plural(pc['compared'], 'trial')})"
        )
    elif pc and b["cost_estimate"].get("price_source") != "declared":
        texts.append(f"{INFO} no recorded cost to check the declared prices against")
    return texts + _cost_record_checks(ci)


def _cost_record_checks(ci: Doc) -> Lines:
    """Compare cost copies without implying independent billing verification."""
    texts: Lines = []
    cr = ci.get("cost_records") or {}
    vt = {str(k): int(v) for k, v in (ci.get("cost_vs_trajectory") or {}).items()}
    compared = sum(vt.values())
    if vt.get("differs"):
        texts.append(
            f"{WARN} {vt['differs']:,} of {plural(compared, 'trial')}: the recorded cost"
            " differs from the trajectory's own (the recorded one is used)"
        )
    elif compared:
        texts.append(
            f"{OK} recorded costs match the trajectories' own in {compared:,} of"
            f" {plural(compared, 'compared trial')}"
        )
    if compared:
        texts.append(
            f"{INFO} matching cost records may share a source; they do not verify provider billing"
        )
    if cr.get("mismatched"):
        texts.append(
            f"{WARN} {cr['mismatched']:,} of {plural(cr['compared'], 'trial')}: result.json and"
            " attempt-costs record different costs (result.json is used)"
        )
    return texts


def walltime_section(b: Doc) -> Lines:
    return wrap(
        "WALLTIME",
        [f"{INFO} {line}" for line in walltime_lines(b["overview"].get("walltime") or {})],
    )


def _scoped_cost_notes(b: Doc) -> Lines:
    scoped = b.get("scoped_costs") or {}
    if not scoped.get("trials"):
        return []
    notes = [
        f"{WARN} actual bill unknown for {plural(scoped['trials'], 'trial')};"
        " observed amounts are not final bills"
    ]
    estimate = scoped.get("estimated_observed_cost_usd")
    reported = scoped.get("reported_cost_usd")
    if estimate is not None:
        notes.append(
            f"{usd(estimate)} observed price-derived estimate (excludes unmetered attempts)"
        )
    if reported is not None:
        notes.append(f"{usd(reported)} partial reported observed cost (not a final bill)")
    if estimate is not None and reported is not None:
        notes.append(f"{INFO} scoped amounts may overlap; not added together")
    return notes


def cost_section(b: Doc) -> Lines:
    return wrap("COST", [*_cost_head(b), *_scoped_cost_notes(b), *_gap_costs(b), *_price_checks(b)])
