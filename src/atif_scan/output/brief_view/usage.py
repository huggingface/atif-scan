"""TOKENS and COST: recorded usage checked against itself, then priced (declared or fitted
rates, estimates said as such), plus walltime."""

from __future__ import annotations

import textwrap
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
    LABEL,
    OK,
    RATIO_BASIS,
    WARN,
    WIDTH,
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
    "run_observed": "observed lower bounds (some calls without usage)",
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
    return (
        f"; fast-agent also records provider retries{share} (provider/transport events, not"
        " model behaviour)"
    )


def _stream_retries(b: Doc) -> Lines:
    """Provider retry events, without assuming stream failure or absent usage. Trials
    whose retried calls also lack usage are said once, by _missed_calls."""
    sr = (b.get("usage") or {}).get("stream_retries") or {}
    # Trials with calls missing usage are described once, with their retries, above.
    covered = set((b.get("missed_calls") or {}).get("ids") or [])
    trials = len(set(sr.get("ids") or []) - covered) if sr.get("ids") else sr.get("trials", 0)
    if not trials:
        return []
    return [
        f"{INFO} {plural(trials, 'other trial')} had fast-agent provider failures retried by"
        " the harness (provider/transport events, not model behaviour); their usage is"
        " recorded"
        if covered
        else f"{INFO} {plural(trials, 'trial')}"
        + (f" ({sr['rewarded']:,} rewarded)" if sr["rewarded"] else "")
        + f" had fast-agent provider failures: {plural(sr['failed_attempts'], 'failed attempt')}"
        " retried by the harness (provider/transport events, not model behaviour). Retry"
        " markers alone do not establish missing usage or missing agent history"
    ]


def _outcome(rewarded: int, errored: int, trials: int) -> str:
    """'(4 rewarded, none errored)'-style context for a group of trials."""
    parts = [f"{rewarded:,} rewarded" if rewarded < trials else "all rewarded"]
    parts.append(f"{errored:,} errored" if errored else "none errored")
    return f" ({', '.join(parts)})"


def _status_text(statuses: list[int]) -> str:
    return f" (HTTP {', '.join(str(s) for s in statuses)})" if statuses else ""


def _missed_calls(b: Doc) -> Lines:
    """Trials that kept observed counts because some model calls have no usage: what
    is missing and why, in one line."""
    mc = (b.get("missed_calls") or {}).get("observed") or {}
    if not mc.get("trials"):
        return []
    calls, before = mc["calls"], mc["before_output"]
    who = f"{plural(mc['trials'], 'trial')}{_outcome(mc['rewarded'], mc['errored'], mc['trials'])}"
    if before == calls:
        why = (
            f": the provider failed {'it' if calls == 1 else 'them'} before any output"
            f"{_status_text(mc['statuses'])} and the harness retried"
        )
    elif before:
        why = f"; {before:,} failed before any output{_status_text(mc['statuses'])}"
    else:
        why = ""
    each = " each" if calls == mc["trials"] and calls > 1 else ""
    count_text = "1 model call" if each else plural(calls, "model call")
    return [
        f"{WARN} {who} miss usage for {count_text}{each}{why}; their token totals are lower bounds"
    ]


def _usage_gaps(b: Doc) -> Lines:
    pu = (b.get("usage") or {}).get("partial") or {}
    um = b.get("unmetered_work") or {"trials": 0}
    texts = []
    described = ((b.get("missed_calls") or {}).get("observed") or {}).get("trials", 0)
    observed = (b.get("usage") or {}).get("observed_accounting", 0) - described
    if observed > 0:
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
    texts = [
        f"{INFO} visible agent text per output token: median {r['median']:.2f} characters"
        f"{spread(r)} over {plural(r['traces'], 'trace')}, {RATIO_BASIS[basis]}"
        for basis, r in (b.get("output_ratio") or {}).items()
    ]
    out = b.get("output_ratio_left_out") or {}
    if texts and out.get("trials"):
        errored = out.get("errored") or {}
        why = (
            f"; {sum(errored.values()):,} of them errored: {counts(errored.items())}"
            if errored
            else ""
        )
        texts.append(
            f"{INFO} not in that spread: {plural(out['trials'], 'trial')} with little or no"
            f" agent output (under {out['min_tokens']:,} output tokens){why}"
        )
    return texts


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
    gaps = [*_missed_calls(b), *_usage_gaps(b), *_stream_retries(b)]
    return wrap("TOKENS", [head, *checks, *gaps, *_text_ratio(b)])


def _rates_text(ce: Doc) -> str:
    r = ce["rates_per_mtok"]
    return (
        f"{rate(r['uncached_input'])} uncached input · {rate(r['cached_input'])} cached input"
        f" · {rate(r['output'])} output, per M tokens"
    )


def _cost_head(b: Doc) -> Lines:
    """The recorded cost when nothing is attributed beyond it (else: the ledger)."""
    total = b["overview"]["cost"]["total_usd"]
    if (b.get("scoped_costs") or {}).get("trials"):
        return [f"{usd(total)} recorded final bills; actual bill unknown for observed-cost trials"]
    return [
        f"{usd(total)} recorded{_other_model_spend(b)} · {OK} every trial with usage has a cost"
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


LEDGER_LABELS = {
    "recorded": "recorded cost",
    "observed_estimate": "recorded tokens at list prices (the harness's estimate, not a bill)",
    "unpriced": "tokens without a recorded cost",
    "last_call": "step tokens beyond totals that cover only the last model call",
    "calls_without_usage": "model calls without recorded usage",
    "failed_before_output": "model calls without usage that failed before any output{status}:"
    " likely unbilled",
    "unmetered": "work with no usage recorded at all",
}


def _ledger_amount(row: Doc) -> str:
    if row["usd"] is None:
        return "—"
    return ("≤" if row["bound"] else "") + usd(row["usd"])


def _ledger_method(row: Doc) -> str:
    method = row.get("method")
    if not method or row["kind"] in ("recorded", "observed_estimate"):
        return ""
    # estimates.py words unestimated methods "not estimated: <why>".
    why = method.removeprefix("not estimated: ")
    if row["usd"] is None:
        return f" (not estimated: {why})"
    error = row.get("median_abs_error_usd")
    return (
        f" ({why}" + (f", median error {usd(error)} per trial" if error is not None else "") + ")"
    )


LEDGER_AMOUNT = 10  # "≤$1,234.56"
LEDGER_TEXT = WIDTH - LABEL - LEDGER_AMOUNT - 4  # sign, spaces


def _ledger_row(sign: str, amount: str, text: str) -> Lines:
    """One pre-formatted ledger row: the amount right-aligned, the text wrapped under
    itself (pre-formatted lines are kept as they are by `wrap`)."""
    body = textwrap.wrap(text, LEDGER_TEXT, break_long_words=False) or [""]
    pad = " " * (LEDGER_AMOUNT + 4)
    return [f"  {sign} {amount:>{LEDGER_AMOUNT}}  {body[0]}", *(f" {pad} {t}" for t in body[1:])]


def _ledger_lines(b: Doc) -> Lines:
    """Recorded amounts, then each addition with why it's attributed, then the total."""
    ledger = b.get("cost_ledger") or {}
    rows = ledger.get("rows") or []
    if not ledger.get("shown"):
        return []
    status = _status_text((b.get("missed_calls") or {}).get("statuses") or [])
    lines: Lines = []
    for n, row in enumerate(rows):
        label = LEDGER_LABELS[row["kind"]].format(status=status)
        text = (
            "no cost recorded"
            if row["kind"] == "recorded" and not row["trials"]
            else f"{plural(row['trials'], 'trial')}: {label}{_ledger_method(row)}"
        )
        lines += _ledger_row(" " if n < ledger["base_rows"] else "+", _ledger_amount(row), text)
    total, upper = ledger["total_usd"], ledger["upper_usd"]
    if not ledger["complete"] and not total:
        return [*lines, *_ledger_row("=", "—", "total unknown: nothing could be priced")]
    lines += _ledger_row(
        "=",
        ("" if ledger["complete"] else "≥") + usd(total),
        "estimated total"
        + (f", at most {usd(upper)}" if upper > total else "")
        + ("" if ledger["complete"] else "; some additions aren't estimated"),
    )
    return lines


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


def _unpriced_headline(b: Doc) -> str:
    ce, n = b["cost_estimate"], _present(b)
    if ce["unpriced"] == n - ce["no_usage"] and not b["overview"]["cost"]["total_usd"]:
        return f"{WARN} no trial recorded a cost"
    share = pct(ce["unpriced"], n)
    return f"{WARN} {plural(ce['unpriced'], 'trial')} ({share}) with usage but no cost"


def _cost_headline(b: Doc) -> str:
    """What the recorded amounts are, above the ledger."""
    scoped = (b.get("scoped_costs") or {}).get("trials")
    if scoped and not b["overview"]["cost"]["total_usd"]:
        return f"{WARN} no trial records a billed cost: the amounts below are estimates"
    if scoped:
        return f"{WARN} {plural(scoped, 'trial')} record only observed amounts, not bills"
    if b["cost_estimate"]["unpriced"]:
        return _unpriced_headline(b)
    return f"{OK} every trial with usage has a cost"


def _pricing_notes(ce: Doc) -> Lines:
    """The rates unpriced tokens were priced at, or how to price them."""
    if not ce["unpriced"]:
        return []
    if ce.get("price_source") and ce.get("rates_per_mtok"):
        return [f"{INFO} {_rates_text(ce)}"]
    if ce["estimate_usd"] is None and ce.get("rates_per_mtok") is None:
        return [
            f"{INFO} price the tokens with --price U,C,O ($ per M uncached input, cached input,"
            " output)"
            + (f"; unpriced tokens{_unestimated(ce)}" if ce.get("priced_other_model") else "")
        ]
    return []


def _cost_status(b: Doc) -> Lines:
    texts = [_cost_headline(b)]
    if other := _other_model_spend(b):
        texts.append(f"{INFO} recorded cost{other}")
    return texts + _reported_note(b) + _pricing_notes(b["cost_estimate"])


def _reported_note(b: Doc) -> Lines:
    """A harness-reported partial cost: shown, never added to the estimate it may overlap."""
    scoped = b.get("scoped_costs") or {}
    if (reported := scoped.get("reported_cost_usd")) is None:
        return []
    overlap = (
        "; it may overlap the estimate, so it isn't added"
        if scoped.get("estimated_observed_cost_usd") is not None
        else ""
    )
    return [f"{INFO} {usd(reported)} partial reported observed cost (not a final bill{overlap})"]


def cost_section(b: Doc) -> Lines:
    """The recorded cost, or a ledger of recorded amounts and attributed additions."""
    if ledger := _ledger_lines(b):
        return wrap("COST", [*_cost_status(b), *ledger, *_price_checks(b)])
    return wrap("COST", [*_cost_head(b), *_reported_note(b), *_price_checks(b)])
