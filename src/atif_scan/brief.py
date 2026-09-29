"""The one-screen run integrity report (`--brief`, and the default text view for a run).

What a benchmarker needs first: what was scanned, the result and what it would become
after the potential adjustments (disqualification candidates, unpriced trials, missing
trace activity), trace/cost integrity, and counted findings. Detail stays behind flags
(--summary, --cite, --detail). Built only from the allowlisted report document.

The text view is one small renderer per section (`SECTIONS`), each reading the brief
document only.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING

from .estimates import (
    Pricing,
    choose_pricing,
    cost_estimate,
    missing_activity,
    partial_usage,
    price_check,
    unmetered_work,
)
from .harbor_files import PRICE_KINDS, override_kind
from .jsonval import as_str
from .packs import BUNDLED
from .questions import tally
from .report import (
    MIN_ATTEMPTS_FOR_SE,
    RANK,
    STYLE,
    _m,
    events,
    index_text,
    is_counted,
    overview,
    review_metadata,
    review_text,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from rich.text import Text

    from .estimates import Rates
    from .jsonval import Doc

OK, WARN, BAD, INFO = "✓", "⚠", "✗", "·"
PAD = f"{'':<10}"  # the label column, blank on continuation lines
SEVERITIES = ("critical", "high", "medium", "low", "info", "none", "unavailable")


def _counted(item: Doc) -> list[Doc]:
    return [a for a in item["assessments"] if is_counted(a)]


def _behaviour(check: str) -> bool:
    """A behaviour check, as opposed to a recording-integrity one."""
    return not check.startswith("integrity.")


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile of sorted `values` (so the median of 2 is their mean)."""
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


# Below this many traces a p5–p95 band is just the extremes; show min–max instead.
MIN_TRACES_FOR_PERCENTILES = 20


RATIO_BASIS = {
    "answer_only": "excl. reasoning (reasoning tokens reported)",
    "all_text": "incl. recorded reasoning (summaries read lower)",
    "visible_only": "no reasoning text recorded: reasoning models read low by design",
}
REASONING_LABELS = {
    "full": "full",
    "recorded": "recorded",
    "summarised": "summarised",
    "withheld": "withheld (tokens only)",
    "none": "not exposed",
}


def reasoning_exposure(items: Sequence[Doc]) -> dict[str, int]:
    """Scanned traces per `reasoning` exposure code, most to least exposed."""
    counts = Counter(i.get("reasoning") for i in items if i.get("input_status") == "available")
    return {k: counts[k] for k in REASONING_LABELS if counts[k]}


def _ratio_basis(item: Doc) -> str | None:
    basis = as_str(item.get("output_ratio_basis"))
    if basis == "all_text" and item.get("reasoning") in ("withheld", "none"):
        # No reasoning text, and its tokens (if any) weren't subtracted: reasoning is
        # in the output tokens but not the text, so it reads low. Expected by design.
        return "visible_only"
    return basis


def output_ratios(items: Sequence[Doc]) -> Doc | None:
    """Run-level spread of authored characters per completion token, per basis."""
    by_basis: dict[str, list[float]] = {}
    for item in items:
        value, basis = item.get("chars_per_output_token"), _ratio_basis(item)
        if isinstance(value, int | float) and basis in RATIO_BASIS:
            by_basis.setdefault(basis, []).append(float(value))
    if not by_basis:
        return None
    result: Doc = {}
    for basis in (b for b in RATIO_BASIS if b in by_basis):
        values = sorted(by_basis[basis])
        result[basis] = {
            "traces": len(values),
            "median": _quantile(values, 0.5),
            "p5": _quantile(values, 0.05),
            "p95": _quantile(values, 0.95),
            "min": values[0],
            "max": values[-1],
        }
    return result


@dataclass
class _Tally:
    """Counted findings over a run's trials."""

    behaviour: Counter[str] = field(default_factory=Counter)  # check -> trials
    # check -> distinct evidence locations over those trials
    events: Counter[str] = field(default_factory=Counter)
    integrity: Counter[str] = field(default_factory=Counter)  # recording checks -> trials
    by_severity: Counter[str] = field(default_factory=Counter)  # trials by highest priority
    flagged_trials: int = 0  # trials with at least one behaviour finding

    def add(self, item: Doc) -> None:
        counted = _counted(item)
        behaviour = [a for a in counted if _behaviour(a["id"])]
        self.flagged_trials += bool(behaviour)
        for a in counted:
            check = a["id"]
            if check == "integrity.cost_missing" and item.get("cost_usd") is not None:
                continue  # the trajectory lacks cost, but the source (e.g. Hub) has it
            if _behaviour(check):
                self.behaviour[check] += 1
                self.events[check] += events(a)
            else:
                self.integrity[check] += 1
        if item.get("input_status") != "available":
            # Never scanned (no trajectory, unreadable): unknown, not "none".
            self.by_severity["unavailable"] += 1
            return
        worst = max((a["severity"] for a in behaviour), key=RANK.__getitem__, default="none")
        self.by_severity[worst] += 1


def _findings(items: Sequence[Doc]) -> tuple[Doc, Doc]:
    """(recording checks -> trials, behaviour findings) for the brief."""
    tally_ = _Tally()
    for item in items:
        tally_.add(item)
    severity_of = {
        a["id"]: a["severity"] for item in items for a in item["assessments"] if a.get("severity")
    }

    def order(counts: Counter[str]) -> list[str]:
        # Ties by ID: counting order follows set iteration (string hashing).
        return sorted(counts, key=lambda c: (-RANK[severity_of[c]], -counts[c], c))

    recording = {check: tally_.integrity[check] for check in order(tally_.integrity)}
    findings = {
        # Each check's distinct evidence locations, summed: one finding per location.
        "total": sum(tally_.events.values()),
        "trials": tally_.flagged_trials,
        "traces_by_highest_severity": {
            k: tally_.by_severity[k] for k in SEVERITIES if tally_.by_severity[k]
        },
        "checks": {
            c: {
                "severity": severity_of[c],
                "traces": tally_.behaviour[c],
                "events": tally_.events[c],
            }
            for c in order(tally_.behaviour)
        },
    }
    return recording, findings


def _agents(items: Sequence[Doc]) -> dict[str, int]:
    """`agent / version / model` -> trials, most common first."""
    agents = Counter(
        " / ".join(
            p for p in (i.get("agent_name"), i.get("agent_version"), i.get("model_name")) if p
        )
        for i in items
        if i.get("agent_name") or i.get("model_name")
    )
    return dict(agents.most_common())


def _suggested_packs(items: Sequence[Doc], runs: Sequence[Doc]) -> list[str]:
    """A bundled pack the run's dataset has but the scan didn't load (e.g. --packs none)."""
    loaded = {a["id"].split(".", 1)[0] for item in items for a in item["assessments"]}
    datasets = " ".join(d for run in runs for d in run.get("datasets") or [])
    return [
        pack.plugin
        for pack in BUNDLED
        if pack.datasets is not None and pack.datasets.search(datasets) and pack.name not in loaded
    ]


def _usage(items: Sequence[Doc], pricing: Pricing) -> Doc:
    """Token accounting first: every cost figure is priced from these tokens."""
    with_tokens = [i for i in items if _has_tokens(i)]
    return {
        "trials_with_tokens": len(with_tokens),
        # Where each trial's tokens come from: the run's records (result.json, Hub,
        # ledger), the trajectory's totals, or its steps' own usage.
        "basis": dict(Counter(i.get("usage_basis") or "none" for i in with_tokens)),
        "run_vs_trajectory": _agreement(items, "tokens_match_trajectory"),
        "trajectory_vs_steps": _agreement(items, "step_tokens_match"),
        "partial": partial_usage(items, pricing),
    }


def brief(
    doc: Doc,
    dq: str = "high",
    min_trials: int | None = None,
    expect_tasks: int | None = None,
    price: Rates | None = None,
) -> Doc:
    """The brief document. `price`: --price $/M rates; else the run's declared prices;
    else rates fitted on its recorded costs (`choose_pricing`), chosen once for every
    cost estimate."""
    items = doc["inputs"]
    ov = overview(doc, dq, min_trials, expect_tasks=expect_tasks)
    other_model = frozenset((ov.get("model_mismatch") or {}).get("trial_ids") or [])
    declared = declared_prices(ov["runs"])
    declared_rates = _rates(declared)
    pricing = choose_pricing(items, price, declared_rates, other_model)
    recording, findings = _findings(items)
    return {
        "schema_version": 1,
        "kind": "integrity_brief",
        "review": review_metadata(doc),
        "packs": doc.get("packs") or [],
        "suggested_packs": _suggested_packs(items, ov["runs"]),
        # Reviewer answers to --questions prompts (annotations only; never DQ math).
        "answers": tally(items),
        "scanner_version": doc["scanner_version"],
        "runs": ov["runs"],
        "agents": _agents(items),
        "overview": ov,
        "dq_threshold": dq,
        "cost_estimate": cost_estimate(items, pricing, other_model),
        "unmetered_work": unmetered_work(items, pricing, other_model),
        "usage": _usage(items, pricing),
        "cost_integrity": {
            "declared_prices": declared,
            "price_check": price_check(items, declared_rates),
            "cost_records": _agreement(items, "cost_records_agree"),
        },
        "tasks_known": sum(1 for i in items if i.get("task")),
        "missing_activity": missing_activity(items),
        "output_ratio": output_ratios(items),
        "reasoning": reasoning_exposure(items),
        "recording": recording,
        "findings": findings,
    }


def declared_prices(runs: Sequence[Doc]) -> dict[str, float] | None:
    """The token prices the run declares (harbor-hf's run.json), when every run that
    declares any declares the same; None when absent or conflicting."""
    found = {
        tuple(r["declared_prices"].get(k) for k in PRICE_KINDS)
        for r in runs
        if r.get("declared_prices")
    }
    if len(found) != 1 or None in (rates := next(iter(found))):
        return None
    return dict(zip(PRICE_KINDS, rates, strict=True))


def _rates(prices: dict[str, float] | None) -> Rates | None:
    """Declared prices as rates, in PRICE_KINDS order."""
    if not prices:
        return None
    uncached, cached, output = (prices[k] for k in PRICE_KINDS)
    return (uncached, cached, output)


def _has_tokens(item: Doc) -> bool:
    return bool(item.get("input_tokens") or item.get("output_tokens"))


def _agreement(items: Sequence[Doc], key: str) -> dict[str, int]:
    """{compared, mismatched} over trials where `key` is True/False (None: not compared)."""
    values = [i.get(key) for i in items if i.get(key) is not None]
    return {"compared": len(values), "mismatched": sum(1 for v in values if not v)}


# --- Text view: one renderer per section, each `(brief, trials present) -> lines` ---------

# Below a cent, a positive amount reads `<$0.01`; within a cent, two costs are the same.
HALF_CENT = 0.005
CENT = 0.01
# Harbor error types shown on the coverage line; the rest are counted.
ERROR_KINDS_SHOWN = 3
# Medium+ behaviour checks listed under FINDINGS; the rest are in --summary.
CHECKS_SHOWN = 6
# A missing-calls estimate at or above this % of the run's calls is an adjustment.
MISSING_CALLS_NOTABLE_PCT = 1

RUN_KINDS = {"harbor_hub": "harbor", "harbor_leaderboard_row": "row"}


def _usd(value: float) -> str:
    """Dollars to the cent; a positive amount under a cent reads `<$0.01`, not $0.00."""
    return "<$0.01" if 0 < value < HALF_CENT else f"${value:,.2f}"


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "—"


def _labelled(label: str, texts: Iterable[str]) -> list[str]:
    """Lines under one label: the first carries it, the rest leave the column blank."""
    return [f"{label if j == 0 else '':<10} {text}" for j, text in enumerate(texts)]


def _unmetered(b: Doc) -> Doc:
    return b.get("unmetered_work") or {"trials": 0}


def _partial(b: Doc) -> Doc:
    return (b.get("usage") or {}).get("partial") or {}


# HEADER: what was scanned


def _run_line(r: Doc) -> str:
    ref = (r.get("dataset_refs") or [""])[0][:15]
    kind = RUN_KINDS.get(r.get("source") or "", "job")
    where = f"{kind} {r['job_id'][:8]}" if r.get("job_id") else "Harbor job folder"
    name = r.get("job_name")
    return (f"{name} · {where}" if name and name != "job" else where) + (
        f" · {', '.join(r['datasets'])}@{ref}" if r.get("datasets") else ""
    )


def _leaderboard_line(lb: Doc) -> str:
    who = " / ".join(x for x in (lb.get("agent"), lb.get("model")) if x)
    effort = f" ({lb['reasoning_effort']})" if lb.get("reasoning_effort") else ""
    jobs = ", ".join(j[:8] for j in lb.get("jobs") or [])
    return f"leaderboard  {who}{effort} · jobs {jobs or '?'}"


def _shape(b: Doc, n: int) -> str:
    k = b["overview"]["tasks"]
    shape = f"{n} trials"
    if not b["tasks_known"]:
        return shape + " · tasks unknown (pass --task-from trial-dir or --task)"
    if k["count"]:
        low, high = k["min_trials"], k["max_trials"]
        per = f"{low}" if low == high else f"{low}–{high}"
        shape += f" · {k['count']} tasks × {per}"
    return shape


def _header_lines(b: Doc, n: int) -> list[str]:
    ov, agents = b["overview"], b["agents"]
    lines = [f"atif-scan {b['scanner_version']} · run integrity", ""]
    if ov.get("sync_failed_files"):
        lines.append(
            f"SYNC       ⚠ {ov['sync_failed_files']} file(s) unavailable; report incomplete"
        )
    lines += [_run_line(r) for r in b["runs"]]
    lines += [_leaderboard_line(lb) for r in b["runs"] if (lb := r.get("leaderboard"))]
    if agents:
        shown = (f"{a} ({m})" if len(agents) > 1 else a for a, m in agents.items())
        lines.append("agent  " + " · ".join(shown))
    return [*lines, _shape(b, n), ""]


# RESULT is recorded; SCENARIO is hypothetical; REPORTED is the leaderboard's; REVIEW


def _result_lines(b: Doc, n: int) -> list[str]:
    ov = b["overview"]
    t, d = ov["trials"], ov["disqualification"]
    if not ov["accuracy"]:
        return ["RESULT     rewards unknown (no verifier output or Hub record)"]
    acc, se = ov["accuracy"]
    scored = n - t["reward_unknown"]
    # The per-task SE needs tasks, and attempts to vary: with one attempt per task
    # it is 0 by construction (not estimable), so it is left out.
    has_se = b["tasks_known"] and (ov["tasks"].get("max_trials") or 0) >= MIN_ATTEMPTS_FOR_SE
    spread = f" ± {se:.1f}" if has_se else ""
    lines = [f"RESULT     {acc:.1f}%{spread} ({round(acc * scored / 100)}/{scored})"]
    if d and d["candidates"]:
        adj, adj_se = d["accuracy_if_disqualified"]
        adj_spread = f" ± {adj_se:.1f}" if has_se else ""
        noun = "success is" if d["candidates"] == 1 else "successes are"
        lines.append(
            f"SCENARIO   {adj:.1f}%{adj_spread} if {d['candidates']} flagged {noun}"
            " zeroed (not a verdict)"
        )
    return lines


def _reported(lb: Doc, n: int) -> str:
    rep = f"REPORTED   {lb['reported_accuracy']:.1f}%"
    if lb.get("reported_reward_hacks_pct"):
        rep += (
            f" after the leaderboard's reward-hack DQs"
            f" ({lb['reported_reward_hacks_pct']:.1f}% of trials)"
        )
    if lb.get("reported_n_trials") is not None:
        same = lb["reported_n_trials"] == n
        rep += f" · {OK if same else WARN} {lb['reported_n_trials']} trials"
    if lb.get("reported_cost_usd") is not None:
        rep += f" · ${lb['reported_cost_usd']:,.2f}"
    if lb.get("display_cost") and "partial" in lb["display_cost"]:
        rep += f" ({lb['display_cost']})"
    return rep


def _reported_lines(b: Doc, n: int) -> list[str]:
    lines: list[str] = []
    for r in b["runs"]:
        lb = r.get("leaderboard")
        if not lb or lb.get("reported_accuracy") is None:
            continue
        lines.append(_reported(lb, n))
        if r.get("unresolved_trials"):
            lines.append(f"{PAD} {WARN} {r['unresolved_trials']} row trial(s) not found in any job")
    return lines


def _review_lines(b: Doc, n: int) -> list[str]:
    return review_text(b["overview"], b.get("review"), b.get("dq_threshold", "high"))


# COVERAGE


def _errored(t: Doc, n: int) -> str:
    text = f"{OK if not t['errored'] else WARN} {t['errored']} errored ({_pct(t['errored'], n)})"
    kinds = list(t["error_types"].items())
    if kinds:  # Harbor's recorded exception types, not trajectory evidence
        shown = ", ".join(f"{kind} {count}" for kind, count in kinds[:ERROR_KINDS_SHOWN])
        more = len(kinds) - ERROR_KINDS_SHOWN
        text += f": {shown}" + (f", +{more} more" if more > 0 else "")
    return text


def _coverage_parts(b: Doc, n: int) -> list[str]:
    t, k = b["overview"]["trials"], b["overview"]["tasks"]
    parts = []
    if t["planned"] is not None:
        folder = any(r.get("source") == "harbor_job_folder" for r in b["runs"])
        what = "planned trials have a trajectory" if folder else "planned trials present"
        over = t["present"] > t["planned"]  # more than planned: retries, reruns, merges
        mark = WARN if t["missing"] or over else OK
        parts.append(f"{mark} {t['present']}/{t['planned']} {what}")
    if k["expected_tasks"]:
        mark = OK if not k["missing_tasks"] else WARN
        parts.append(f"{mark} {k['count']}/{k['expected_tasks']} tasks")
    if k["expected_per_task"]:
        below = len(k["below_expected"])
        parts.append(
            f"{OK if not below else WARN} {below} task(s) below {k['expected_per_task']} trials"
        )
    parts.append(_errored(t, n))
    if t["without_trajectory"]:
        parts.append(f"{WARN} {t['without_trajectory']} without trajectory")
    return parts


def _listed_share(p: Doc) -> str:
    """`rewarded/scored (pct), $cost` of one part of a job's trials."""
    pct = f" ({100 * p['rewarded'] / p['scored']:.1f}%)" if p["scored"] else ""
    cost = f", ${p['cost_usd']:,.2f}" if p["cost_usd"] is not None else ""
    return f"{p['rewarded']}/{p['scored']}{pct}{cost}"


def _rerun_lines(rr: Doc | None) -> list[str]:
    if not rr or not rr["unlisted_trials"]:
        return []
    line = (
        f"{PAD} {WARN} {rr['unlisted_trials']} trial folder(s) not in the job's"
        f" result.json ({rr['unlisted_with_trajectory']} with a trajectory)"
    )
    if rr["tasks_rerun"]:
        line += f" · {rr['tasks_rerun']} task(s) run again: likely a rerun or resume of the job"
    return [
        line,
        f"{PAD} {INFO} all scored above · job's listed trials {_listed_share(rr['listed'])}"
        f" · other trials {_listed_share(rr['unlisted'])}",
    ]


def _coverage_lines(b: Doc, n: int) -> list[str]:
    return [
        "COVERAGE   " + " · ".join(_coverage_parts(b, n)),
        *_rerun_lines(b["overview"].get("reruns")),
    ]


# TRACES: recording integrity + missing-activity estimate

RECORDING_WARN = frozenset(
    {
        "integrity.observation_pairing_reconstructed",
        "integrity.observation_pairing_unresolved",
        "integrity.tokens_exceed_recorded_calls",
        "integrity.output_token_ratio",
        "integrity.web_results_not_recorded",
        "integrity.redacted_values",
        "integrity.agent_steps_missing",
    }
)


RECORDING_LABELS = {
    "integrity.history_compacted": "compacted history (only the last context window recorded)",
    "integrity.tool_results_not_recorded": "tool results exported as status words only",
    "integrity.actions_not_recorded": "work claimed but no tool calls recorded",
    "integrity.trace_head_missing": "trace starts mid-session (no prompt recorded)",
    "integrity.timestamp_missing": "traces with untimestamped agent steps",
    "integrity.tokens_exceed_recorded_calls": "token totals exceed what recorded calls could use",
    "integrity.cost_missing": "tokens recorded but no cost (anywhere)",
    "integrity.timestamp_smearing": "timestamps stamped at export",
    "integrity.timestamp_regression": "timestamps going backwards",
    "integrity.orphan_observation": "tool results without a matching call",
    "integrity.observation_pairing_reconstructed": (
        "call/result links inferred by position (not verified)"
    ),
    "integrity.observation_pairing_unresolved": (
        "tool results that couldn't be paired with their calls (left unlinked)"
    ),
    "integrity.agent_only_fields": "agent-only fields on system/user steps",
    "integrity.output_token_ratio": "recorded text doesn't fit reported output tokens",
    "integrity.web_results_not_recorded": "web searches/fetches without recorded results or URLs",
    "integrity.redacted_values": "bare [REDACTED] values (invalid JSON, read as unknown)",
    "integrity.agent_steps_missing": "no agent steps recorded",
}


def _compacted_text(ma: Doc, n: int) -> str | None:
    if not ma["compacted"]:
        return None
    est = ""
    if ma["missing_calls_pct"]:
        lo, hi = ma["missing_calls_pct"]
        if hi >= MISSING_CALLS_NOTABLE_PCT:
            est = f" — est. {lo:.0f}–{hi:.0f}% of the run's LLM calls are not in its trajectories"
        else:
            r_lo, r_hi = ma["compacted_recorded_pct"]
            est = (
                f" — those trials recorded est. {r_lo:.0f}–{r_hi:.0f}% of their calls"
                " (<1% of the run)"
            )
    return f"{WARN} {ma['compacted']} ({_pct(ma['compacted'], n)}) compacted history{est}"


def _recording_texts(b: Doc, n: int) -> list[str]:
    compacted = _compacted_text(b["missing_activity"], n)
    return [
        *([compacted] if compacted else []),
        *(
            f"{WARN if check in RECORDING_WARN else INFO}"
            f" {RECORDING_LABELS.get(check, check)}: {count} ({_pct(count, n)})"
            for check, count in b["recording"].items()
            if check != "integrity.history_compacted"
        ),
    ]


def _ratio_line(basis: str, r: Doc) -> str:
    k = r["traces"]
    if k == 1:
        spread = ""
    elif k < MIN_TRACES_FOR_PERCENTILES:
        spread = f" (range {r['min']:.2f}–{r['max']:.2f})"
    else:
        spread = f" (p5–p95 {r['p5']:.2f}–{r['p95']:.2f})"
    return (
        f"{PAD} {INFO} chars/output token: median {r['median']:.2f}{spread}"
        f" over {k} trace{'' if k == 1 else 's'}, {RATIO_BASIS[basis]}"
    )


def _traces_lines(b: Doc, n: int) -> list[str]:
    lines = _labelled("TRACES", _recording_texts(b, n)) or [
        f"TRACES     {OK} no recording defects detected"
    ]
    if exposure := b.get("reasoning"):
        # A model property, not a recording defect: shown for reading findings, never warned.
        shown = " · ".join(f"{REASONING_LABELS[k]} {v} ({_pct(v, n)})" for k, v in exposure.items())
        lines.append(
            f"{PAD} {INFO} reasoning: {shown}; withheld or summarised reasoning is by"
            " design for many models, and checks read the recorded text"
        )
    lines += [_ratio_line(basis, r) for basis, r in (b.get("output_ratio") or {}).items()]
    d = b["overview"]["disqualification"]
    if d and d["rewarded_not_cleared"]:
        why = " · ".join(f"{reason} {count}" for reason, count in d["not_cleared_reasons"].items())
        lines.append(
            f"{PAD} → {d['rewarded_not_cleared']} rewarded trial(s) can't be cleared: {why}"
        )
    return lines


# USAGE (token accounting), then COST priced from it


def _usage_checks(u: Doc) -> list[str]:
    checks = []
    for agree, what in (
        (u.get("run_vs_trajectory") or {}, "run records = trajectory totals"),
        (u.get("trajectory_vs_steps") or {}, "trajectory totals = sum of step usage"),
    ):
        if agree.get("mismatched"):
            checks.append(f"{WARN} {what}: {agree['mismatched']} of {agree['compared']} differ")
        elif agree.get("compared"):
            checks.append(f"{OK} {what} ({agree['compared']} trials)")
    return checks


def _usage_gap_lines(pu: Doc, um: Doc) -> list[str]:
    """Trials whose usage is partial (summed from steps) or missing despite work."""
    lines: list[str] = []
    if pu.get("trials"):
        lines.append(
            f"{PAD} {WARN} {pu['trials']} trial(s) report no usage totals"
            + (f" ({pu['rewarded']} rewarded)" if pu["rewarded"] else "")
            + f": their steps' usage is summed, {pu['calls_without_usage']} LLM call(s)"
            " without usage (a lower bound)"
        )
    if um["trials"]:
        why = ", ".join(
            f"{v} {k}" for k, v in (("errored", um["errored"]), ("rewarded", um["rewarded"])) if v
        )
        lines.append(
            f"{PAD} {WARN} {um['trials']} trial(s) did work but report no usage"
            + (f" ({why})" if why else "")
            + f": {um['llm_calls']:,} LLM calls, {um['tool_calls']:,} tool calls,"
            f" {um['duration_sec'] / 60:,.1f} min"
        )
    return lines


def usage_lines(b: Doc, n: int) -> list[str]:
    """Token accounting: how many trials have token counts, where they come from, and
    whether independent records of them agree. Cost is priced from these tokens."""
    u, um, pu = b.get("usage") or {}, b["unmetered_work"], _partial(b)
    tk = b["cost_estimate"].get("tokens") or {}
    rt, ts = u.get("run_vs_trajectory") or {}, u.get("trajectory_vs_steps") or {}
    bad = um["trials"] or pu.get("trials") or rt.get("mismatched") or ts.get("mismatched")
    checks = _usage_checks(u)
    return [
        f"USAGE      {WARN if bad else OK} tokens for {u.get('trials_with_tokens', 0)}/{n} trials:"
        f" {_m(tk.get('uncached_input', 0) + tk.get('cached_input', 0))} input"
        f" ({_m(tk.get('cached_input', 0))} cached) · {_m(tk.get('output', 0))} output",
        f"{PAD} " + " · ".join(checks)
        if checks
        else f"{PAD} {INFO} no second record of the tokens to check them against",
        *_usage_gap_lines(pu, um),
    ]


def _other_model_spend(b: Doc) -> str:
    """How much of the reported total went to trials on another model."""
    mm, total = b["overview"].get("model_mismatch") or {}, b["overview"]["cost"]["total_usd"]
    if not (b["cost_estimate"].get("priced_other_model") and mm.get("cost_usd")):
        return ""
    spent = mm["cost_usd"]
    if abs(spent - total) < CENT:
        return " (all of it on another model)"
    return f" (${spent:,.2f} of it on another model)"


def _other_model_notes(ce: Doc) -> list[str]:
    ut = ce["unpriced_tokens"]
    return [
        f"{PAD} {WARN} not estimated: only {ce['priced_other_model']} trial(s) on"
        " another model are priced, and their prices aren't this model's",
        f"{PAD} {INFO} unpriced tokens:"
        f" {_m(ut['uncached_input'] + ut['cached_input'])} input"
        f" ({_m(ut['cached_input'])} cached) · {_m(ut['output'])} output"
        " · estimate with --price U,C,O ($/M)",
    ]


def _unpriced_cost(b: Doc, n: int, line: str) -> tuple[str, list[str]]:
    """The COST line when some trials with usage have no cost: their estimate."""
    ce, total = b["cost_estimate"], b["overview"]["cost"]["total_usd"]
    line += _other_model_spend(b)
    line += f" · {WARN} {ce['unpriced']} unpriced trial(s) ({_pct(ce['unpriced'], n)})"
    if ce["estimate_usd"] is None and ce.get("priced_other_model"):
        return line, _other_model_notes(ce)
    if ce["estimate_usd"] is None:
        return line + f" ({ce['method']})", []
    corrected = total + ce["estimate_usd"]
    share = 100 * ce["estimate_usd"] / corrected if corrected else 0
    how = " at the run's declared prices" if ce.get("price_source") == "declared" else ""
    return (
        line + f" → est. +${ce['estimate_usd']:,.2f}{how} (≈ ${corrected:,.2f}, +{share:.1f}%)",
        [],
    )


def _known_rates_cost(ce: Doc) -> str:
    """No cost recorded at all, priced at the declared or given rates."""
    r = ce["rates_per_mtok"]
    rates = f"${r['uncached_input']:g}/${r['cached_input']:g}/${r['output']:g} per M"
    if ce["price_source"] == "declared":
        return (
            f"COST       {WARN} no cost recorded · ${ce['estimate_usd']:,.2f} at the run's"
            f" declared prices {rates} (uncached/cached/output, run.json)"
        )
    return (
        f"COST       {WARN} no cost recorded · est. ${ce['estimate_usd']:,.2f} at the"
        f" given --price {rates} (uncached/cached/output)"
    )


def _cost_headline(b: Doc, n: int) -> tuple[str, list[str]]:
    """The COST line, and notes to show under it."""
    ce, total = b["cost_estimate"], b["overview"]["cost"]["total_usd"]
    if ce["unpriced"] and ce["unpriced"] == n - ce["no_usage"] and ce["estimate_usd"] is None:
        return (
            f"COST       {WARN} no cost recorded for any trial (trajectories or run metadata)"
            " · estimate from the tokens with --price U,C,O ($/M)"
        ), []
    if ce["unpriced"] and not total and ce.get("price_source"):
        return _known_rates_cost(ce), []
    line = f"COST       ${total:,.2f} reported"
    if ce["unpriced"]:
        return _unpriced_cost(b, n, line)
    every = "every trial with usage priced" if _unmetered(b)["trials"] else "every trial priced"
    return f"{line} · {OK} {every}", []


def _gap_cost_lines(b: Doc, total: float) -> list[str]:
    """Estimated costs of the usage gaps: unmetered calls and trials without usage."""
    um, pu = b["unmetered_work"], _partial(b)
    lines: list[str] = []
    if pu.get("trials") and pu.get("estimate_usd") is not None:
        lines.append(
            f"{PAD} {WARN} {pu['calls_without_usage']} LLM call(s) without usage → est."
            f" +{_usd(pu['estimate_usd'])} (at each trial's own cost per call)"
        )
    if um["trials"] and um["estimate_usd"] is not None:
        whole = total + um["estimate_usd"]
        share = 100 * um["estimate_usd"] / whole if whole else 0
        lines.append(
            f"{PAD} {WARN} {um['trials']} trial(s) without usage → est."
            f" +${um['estimate_usd']:,.2f} (≈ ${whole:,.2f}, +{share:.1f}%) not in the total"
        )
    elif um["trials"]:
        lines.append(
            f"{PAD} {WARN} {um['trials']} trial(s) without usage: not in the total ({um['method']})"
        )
    return lines


def _price_check_lines(ci: Doc) -> list[str]:
    """Recorded costs against the declared prices, and against each other."""
    lines: list[str] = []
    pc = ci.get("price_check")
    if pc and pc["mismatched"]:
        lines.append(
            f"{PAD} {WARN} {pc['mismatched']} of {pc['compared']} recorded cost(s) don't fit"
            f" the declared prices (${pc['recorded_usd']:,.2f} recorded vs"
            f" ${pc['at_prices_usd']:,.2f} at those prices)"
        )
    elif pc and pc["compared"]:
        lines.append(f"{PAD} {OK} recorded costs fit the declared prices ({pc['compared']} trials)")
    elif pc:
        lines.append(f"{PAD} {INFO} no recorded cost to check the declared prices against")
    cr = ci.get("cost_records") or {}
    if cr.get("mismatched"):
        lines.append(
            f"{PAD} {WARN} {cr['mismatched']} of {cr['compared']} trial(s): result.json and"
            " attempt-costs record different costs (result.json used)"
        )
    return lines


def cost_integrity_lines(b: Doc, total: float) -> list[str]:
    """Costs for the usage gaps, and whether recorded costs hold together: against the
    run's declared prices, and against each other."""
    return _gap_cost_lines(b, total) + _price_check_lines(b.get("cost_integrity") or {})


def _cost_lines(b: Doc, n: int) -> list[str]:
    ce = b["cost_estimate"]
    line, notes = _cost_headline(b, n)
    idle = ce["no_usage"] - _unmetered(b)["trials"]  # no usage and no recorded work either
    if idle and ce["no_usage"] < n:
        line += f" · {idle} without usage data (no recorded work)"
    return [line, *notes, *cost_integrity_lines(b, b["overview"]["cost"]["total_usd"])]


# FINDINGS


def _findings_lines(b: Doc, n: int) -> list[str]:
    f, ix = b["findings"], b["overview"].get("finding_index")
    sev = " · ".join(f"{s} {v}" for s, v in f["traces_by_highest_severity"].items())
    lines = [
        f"FINDINGS   {f['total']:,} finding(s) across {f['trials']:,} trial(s)"
        f" · trials by highest review priority: {sev or 'none'}"
    ]
    if ix:
        lines.append(f"{PAD} index: {index_text(ix)} (review load, not a verdict)")
    top = [(cid, v) for cid, v in f["checks"].items() if RANK[v["severity"]] >= RANK["medium"]]
    for cid, v in top[:CHECKS_SHOWN]:
        spread = f"{v['events']:,} finding(s) across " if "events" in v else ""
        lines.append(f"{PAD} {WARN} {v['severity']:<8} {cid} · {spread}{v['traces']} trial(s)")
    if len(top) > CHECKS_SHOWN:
        lines.append(f"{PAD}   +{len(top) - CHECKS_SHOWN} more medium+ checks (see --summary)")
    return lines


# SETTINGS, MODEL, ANSWERS, PACKS


def _model_lines(mm: Doc | None, n: int) -> list[str]:
    if not mm:
        return []
    others = ", ".join(f"{m} {k}" for m, k in mm["other_models"].items())
    switched = f" · {len(mm['switched_ids'])} switched mid-trial" if mm.get("switched_ids") else ""
    return [
        f"MODEL      {WARN} critical  {len(mm['trial_ids'])} trial(s)"
        f" ({_pct(len(mm['trial_ids']), n)}) ran another model than {mm['expected']}:"
        f" {others} · {len(mm['rewarded_ids'])} rewarded · ${mm['cost_usd']:,.2f}{switched}",
        f"{PAD} fallback or substitution: those rewards and costs aren't this model's",
    ]


def _settings_lines(b: Doc, n: int) -> list[str]:
    ov = b["overview"]
    scoring = [o for o in ov["overrides"] if override_kind(o) == "scoring"]
    infra = [o for o in ov["overrides"] if override_kind(o) == "infrastructure"]
    overrides = []
    if scoring:
        overrides.append(
            f"{WARN} scoring overrides (change the agent's time or resources; "
            "leaderboards require defaults): " + ", ".join(scoring)
        )
    if infra:
        overrides.append("· infrastructure overrides (provisioning only): " + ", ".join(infra))
    lines = _labelled("SETTINGS", overrides)
    if not ov["overrides"] and b["runs"]:
        lines.append(f"SETTINGS   {OK} no leaderboard-forbidden overrides")
    lines += _model_lines(ov.get("model_mismatch"), n)
    sources = [
        f"{WARN} tasks from a non-canonical source: "
        + ", ".join(r.get("datasets") or ["?"])
        + " (diff it against the benchmark: tools/task_diff.py)"
        for r in b["runs"]
        if r.get("canonical_dataset") is False
    ]
    # The SETTINGS label goes on the section's first labelled line only.
    labelled_before = scoring or infra or ov.get("model_mismatch")
    return lines + _labelled("" if labelled_before else "SETTINGS", sources)


def _answer_lines(b: Doc, n: int) -> list[str]:
    answers = b.get("answers") or {}
    if not answers:
        return []
    shown = (
        f"{question}: "
        + " · ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        for question, counts in sorted(answers.items())
    )
    return [
        *_labelled("ANSWERS", shown),
        f"{PAD} reviewer annotations only; DQ candidates unchanged",
    ]


def _pack_lines(b: Doc, n: int) -> list[str]:
    return [
        f"PACKS      {OK} {pack['pack']} loaded (recognised by {pack['reason']})"
        for pack in b.get("packs") or []
    ] + [
        f"PACKS      {WARN} task pack for this dataset not loaded: --plugin {pack}"
        for pack in b.get("suggested_packs") or []
    ]


# ADJUSTMENTS summary


def _cost_adjustment(ce: Doc) -> str | None:
    if not ce["estimate_usd"]:
        return None
    if ce.get("price_source") == "declared":
        return (
            f"cost ${ce['estimate_usd']:,.2f} for {ce['unpriced']} trial(s) without a recorded"
            " cost, at the run's declared prices (run.json)"
        )
    error = ce["median_abs_error_usd"]
    return (
        f"cost +${ce['estimate_usd']:,.2f} for {ce['unpriced']} unpriced trials"
        f" ({ce['method']}" + (f", median error ${error})" if error is not None else ")")
    )


def _unmetered_adjustment(um: Doc) -> str | None:
    if not um["trials"]:
        return None
    estimate = (
        f" → est. +${um['estimate_usd']:,.2f} (rough: {um['method']},"
        f" median error ${um['median_abs_error_usd']}/trial)"
        if um["estimate_usd"] is not None
        else f" ({um['method']})"
    )
    rewarded = (
        f"; {um['rewarded']} rewarded, counted in RESULT without a cost" if um["rewarded"] else ""
    )
    return (
        f"cost: {um['trials']} trial(s) with {um['llm_calls']:,} LLM calls report no usage"
        + estimate
        + rewarded
    )


def _partial_adjustment(pu: Doc) -> str | None:
    if pu.get("estimate_usd") is None:
        return None
    return (
        f"cost +{_usd(pu['estimate_usd'])} for {pu['calls_without_usage']} LLM call(s)"
        f" without usage in {pu['trials']} trial(s) (rough: each trial's own cost per call)"
    )


def _traces_adjustment(ma: Doc) -> str | None:
    if not ma["missing_calls_pct"] or ma["missing_calls_pct"][1] < MISSING_CALLS_NOTABLE_PCT:
        return None
    lo, hi = ma["missing_calls_pct"]
    return (
        f"traces: ~{lo:.0f}–{hi:.0f}% of LLM calls unrecorded;"
        f" findings on {ma['compacted']} compacted trials are partial"
    )


def _adjustment_lines(b: Doc, n: int) -> list[str]:
    adjustments = [
        a
        for a in (
            _cost_adjustment(b["cost_estimate"]),
            _unmetered_adjustment(_unmetered(b)),
            _partial_adjustment(_partial(b)),
            _traces_adjustment(b["missing_activity"]),
        )
        if a
    ]
    lines = [""]
    if adjustments:
        lines.append("ADJUSTMENTS (estimates for review, not verdicts)")
        lines += [f"  · {a}" for a in adjustments]
    else:
        lines.append(f"ADJUSTMENTS {OK} none suggested")
    return [
        *lines,
        "",
        "more: --summary (per-check detail) · --cite high (evidence)"
        " · --detail (per trace) · --format json",
    ]


SECTIONS: tuple[Callable[[Doc, int], list[str]], ...] = (
    _header_lines,
    _result_lines,
    _reported_lines,
    _review_lines,
    _coverage_lines,
    _traces_lines,
    usage_lines,
    _cost_lines,
    _findings_lines,
    _settings_lines,
    _answer_lines,
    _pack_lines,
    _adjustment_lines,
)


def brief_text(b: Doc) -> str:
    """The brief as text: each section's lines in `SECTIONS` order."""
    n = b["overview"]["trials"]["present"]
    return "\n".join(line for section in SECTIONS for line in section(b, n)) + "\n"


# --- Colour (terminal only) --------------------------------------------------------------
# Styles are applied to the plain text by pattern, so the coloured and plain views can
# never say different things. rich honours NO_COLOR and disables colour when piped.

SEVERITY_STYLE = {**STYLE, "none": "dim", "unavailable": "yellow"}
PATTERNS = [
    (r"^atif-scan .*$", "bold"),
    (
        r"^(RESULT|SCENARIO|REVIEW|REPORTED|COVERAGE|TRACES|COST|FINDINGS|SETTINGS|MODEL|PACKS|ANSWERS|ADJUSTMENTS)\b",
        "bold cyan",
    ),
    (r"^agent\b", "dim"),
    (r"^leaderboard\b", "dim"),
    (r"✓", "bold green"),
    (r"⚠", "bold yellow"),
    (r"✗", "bold red"),
    (r"→", "bold magenta"),
    (r"^ {11}· .*$", "dim"),
    (r"(?<=RESULT     )\d+\.\d+%( ± \d+\.\d+)?", "bold"),
    (r"(?<=→  )\d+\.\d+%( ± \d+\.\d+)?", "bold magenta"),
    (r"(?<=→ )\d+\.\d+%", "bold magenta"),
    (r"est\. [^·(]*?\d[\d.,–%$]*", "magenta"),
    (r"\b[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+\b(?= ·)", "bold"),
    (r"\((?:estimates for review, not verdicts)\)", "dim italic"),
    (r"^more: .*$", "dim"),
]
COMPILED = [(re.compile(pattern), style) for pattern, style in PATTERNS] + [
    # Severity words where they are severities: "critical 38" and "✗ high  check".
    (re.compile(rf"(?<=[✗⚠] ){word}\b|\b{word}(?= \d)"), style)
    for word, style in SEVERITY_STYLE.items()
]


def colourise(text: str) -> Text:
    """The brief as a rich Text with styles applied by pattern."""
    from rich.text import Text  # noqa: PLC0415 - rich is optional, imported only to colour

    out = Text()
    for line in text.splitlines(keepends=True):
        styled = Text(line)
        for pattern, style in COMPILED:
            styled.highlight_regex(pattern, style)
        out.append_text(styled)
    return out


def print_brief(text: str, file: IO[str] | None = None) -> None:
    """Coloured on a terminal (rich), plain otherwise."""
    try:
        from rich.console import Console  # noqa: PLC0415 - rich is optional
    except ImportError:
        print(text, end="", file=file)
        return
    console = Console(file=file, highlight=False, soft_wrap=True)
    if not console.is_terminal:
        print(text, end="", file=file)
        return
    console.print(colourise(text), end="")
