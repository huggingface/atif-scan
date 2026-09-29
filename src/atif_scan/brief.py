"""The one-screen run integrity report (`--brief`, and the default text view for a run).

What a benchmarker needs first: what was scanned, the result and what it would become
after the potential adjustments (disqualification candidates, unpriced trials, missing
trace activity), trace/cost integrity, and counted findings. Detail stays behind flags
(--summary, --cite, --detail). Built only from the allowlisted report document.
"""

from __future__ import annotations

import re
from collections import Counter

from .estimates import cost_estimate, missing_activity, unmetered_work
from .harbor_files import override_kind
from .packs import BUNDLED
from .questions import tally
from .report import (
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

OK, WARN, BAD, INFO = "✓", "⚠", "✗", "·"


def _counted(item: dict) -> list[dict]:
    return [a for a in item["assessments"] if is_counted(a)]


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


def reasoning_exposure(items: list[dict]) -> dict[str, int]:
    """Scanned traces per `reasoning` exposure code, most to least exposed."""
    counts = Counter(i.get("reasoning") for i in items if i.get("input_status") == "available")
    return {k: counts[k] for k in REASONING_LABELS if counts[k]}


def output_ratios(items: list[dict]) -> dict | None:
    """Run-level spread of authored characters per completion token, per basis."""
    by_basis: dict[str, list[float]] = {}
    for item in items:
        value, basis = item.get("chars_per_output_token"), item.get("output_ratio_basis")
        if basis == "all_text" and item.get("reasoning") in ("withheld", "none"):
            # No reasoning text, and its tokens (if any) weren't subtracted: reasoning is
            # in the output tokens but not the text, so it reads low. Expected by design.
            basis = "visible_only"
        if isinstance(value, (int, float)) and basis in RATIO_BASIS:
            by_basis.setdefault(basis, []).append(float(value))
    if not by_basis:
        return None
    result = {}
    for basis in (b for b in RATIO_BASIS if b in by_basis):
        values = sorted(by_basis[basis])
        n = len(values)
        result[basis] = {
            "traces": n,
            "median": _quantile(values, 0.5),
            "p5": _quantile(values, 0.05),
            "p95": _quantile(values, 0.95),
            "min": values[0],
            "max": values[-1],
        }
    return result


def brief(doc: dict, dq: str = "high", min_trials=None, expect_tasks=None, price=None) -> dict:
    items = doc["inputs"]
    ov = overview(doc, dq, min_trials, expect_tasks=expect_tasks)
    other_model = frozenset((ov.get("model_mismatch") or {}).get("trial_ids") or [])
    agents = Counter(
        " / ".join(
            p for p in (i.get("agent_name"), i.get("agent_version"), i.get("model_name")) if p
        )
        for i in items
        if i.get("agent_name") or i.get("model_name")
    )
    behaviour = Counter()  # check -> traces, excluding recording-integrity checks
    behaviour_events = Counter()  # check -> distinct evidence locations over those traces
    integrity = Counter()
    by_severity = Counter()
    for item in items:
        counted = _counted(item)
        for a in counted:
            check = a["id"]
            if check == "integrity.cost_missing" and item.get("cost_usd") is not None:
                continue  # the trajectory lacks cost, but the source (e.g. Hub) has it
            if check.startswith("integrity."):
                integrity[check] += 1
            else:
                behaviour[check] += 1
                behaviour_events[check] += events(a)
        if item.get("input_status") != "available":
            # Never scanned (no trajectory, unreadable): unknown, not "none".
            by_severity["unavailable"] += 1
            continue
        by_severity[
            max(
                (a["severity"] for a in counted if not a["id"].startswith("integrity.")),
                key=RANK.__getitem__,
                default="none",
            )
        ] += 1
    severity_of = {
        a["id"]: a["severity"] for item in items for a in item["assessments"] if a.get("severity")
    }
    loaded = {a["id"].split(".", 1)[0] for item in items for a in item["assessments"]}
    datasets = " ".join(d for run in ov["runs"] for d in run.get("datasets") or [])
    # A bundled pack the run's dataset has but the scan didn't load (e.g. --packs none).
    suggested = [
        pack.plugin
        for pack in BUNDLED
        if pack.datasets is not None and pack.datasets.search(datasets) and pack.name not in loaded
    ]
    return {
        "schema_version": 1,
        "kind": "integrity_brief",
        "review": review_metadata(doc),
        "packs": doc.get("packs") or [],
        "suggested_packs": suggested,
        # Reviewer answers to --questions prompts (annotations only; never DQ math).
        "answers": tally(items),
        "scanner_version": doc["scanner_version"],
        "runs": ov["runs"],
        "agents": dict(agents.most_common()),
        "overview": ov,
        "dq_threshold": dq,
        "cost_estimate": cost_estimate(items, price, other_model),
        "unmetered_work": unmetered_work(items, other_model),
        "tasks_known": sum(1 for i in items if i.get("task")),
        "missing_activity": missing_activity(items),
        "output_ratio": output_ratios(items),
        "reasoning": reasoning_exposure(items),
        "recording": {
            check: integrity[check]
            # Ties by ID: counting order follows set iteration (string hashing).
            for check in sorted(integrity, key=lambda c: (-RANK[severity_of[c]], -integrity[c], c))
        },
        "findings": {
            "traces_by_highest_severity": {
                k: by_severity[k]
                for k in ("critical", "high", "medium", "low", "info", "none", "unavailable")
                if by_severity[k]
            },
            "checks": {
                c: {"severity": severity_of[c], "traces": n, "events": behaviour_events[c]}
                for c, n in sorted(
                    behaviour.items(), key=lambda kv: (-RANK[severity_of[kv[0]]], -kv[1], kv[0])
                )
            },
        },
    }


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "—"


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


def brief_text(b: dict) -> str:
    ov, ce, ma = b["overview"], b["cost_estimate"], b["missing_activity"]
    um = b.get("unmetered_work") or {"trials": 0}
    t, k, d, c = ov["trials"], ov["tasks"], ov["disqualification"], ov["cost"]
    n = t["present"]
    lines = [f"atif-scan {b['scanner_version']} · run integrity", ""]
    if ov.get("sync_failed_files"):
        lines.append(
            f"SYNC       ⚠ {ov['sync_failed_files']} file(s) unavailable; report incomplete"
        )
    for r in b["runs"]:
        ref = (r.get("dataset_refs") or [""])[0][:15]
        kind = {"harbor_hub": "harbor", "harbor_leaderboard_row": "row"}.get(r.get("source"), "job")
        where = f"{kind} {r['job_id'][:8]}" if r.get("job_id") else "Harbor job folder"
        name = r.get("job_name")
        lines.append(
            (f"{name} · {where}" if name and name != "job" else where)
            + (f" · {', '.join(r['datasets'])}@{ref}" if r.get("datasets") else "")
        )
    for r in b["runs"]:
        lb = r.get("leaderboard")
        if lb:
            who = " / ".join(x for x in (lb.get("agent"), lb.get("model")) if x)
            effort = f" ({lb['reasoning_effort']})" if lb.get("reasoning_effort") else ""
            jobs = ", ".join(j[:8] for j in lb.get("jobs") or [])
            lines.append(f"leaderboard  {who}{effort} · jobs {jobs or '?'}")
    if b["agents"]:
        lines.append(
            "agent  "
            + " · ".join(
                f"{a} ({m})" if len(b["agents"]) > 1 else a for a, m in b["agents"].items()
            )
        )
    shape = f"{n} trials"
    if not b["tasks_known"]:
        shape += " · tasks unknown (pass --task-from trial-dir or --task)"
    elif k["count"]:
        per = (
            f"{k['min_trials']}"
            if k["min_trials"] == k["max_trials"]
            else f"{k['min_trials']}–{k['max_trials']}"
        )
        shape += f" · {k['count']} tasks × {per}"
    lines += [shape, ""]

    # RESULT is recorded; SCENARIO is hypothetical.
    if ov["accuracy"]:
        acc, se = ov["accuracy"]
        succ = round(acc * (n - t["reward_unknown"]) / 100)
        # The per-task SE needs tasks, and attempts to vary: with one attempt per task
        # it is 0 by construction (not estimable), so it is left out.
        has_se = b["tasks_known"] and (k.get("max_trials") or 0) >= 2
        spread = f" ± {se:.1f}" if has_se else ""
        line = f"RESULT     {acc:.1f}%{spread} ({succ}/{n - t['reward_unknown']})"
        lines.append(line)
        if d and d["candidates"]:
            adj, adj_se = d["accuracy_if_disqualified"]
            adj_spread = f" ± {adj_se:.1f}" if has_se else ""
            noun = "success is" if d["candidates"] == 1 else "successes are"
            lines.append(
                f"SCENARIO   {adj:.1f}%{adj_spread} if {d['candidates']} flagged {noun}"
                " zeroed (not a verdict)"
            )
    else:
        lines.append("RESULT     rewards unknown (no verifier output or Hub record)")

    for r in b["runs"]:
        lb = r.get("leaderboard")
        if not lb or lb.get("reported_accuracy") is None:
            continue
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
        lines.append(rep)
        if r.get("unresolved_trials"):
            lines.append(
                f"{'':<10} {WARN} {r['unresolved_trials']} row trial(s) not found in any job"
            )

    lines += review_text(ov, b.get("review"), b.get("dq_threshold", "high"))

    # COVERAGE
    cov = []
    if t["planned"] is not None:
        folder = any(r.get("source") == "harbor_job_folder" for r in b["runs"])
        what = "planned trials have a trajectory" if folder else "planned trials present"
        over = t["present"] > t["planned"]  # more than planned: retries, reruns, merges
        mark = WARN if t["missing"] or over else OK
        cov.append(f"{mark} {t['present']}/{t['planned']} {what}")
    if k["expected_tasks"]:
        cov.append(
            f"{OK if not k['missing_tasks'] else WARN} {k['count']}/{k['expected_tasks']} tasks"
        )
    if k["expected_per_task"]:
        below = len(k["below_expected"])
        cov.append(
            f"{OK if not below else WARN} {below} task(s) below {k['expected_per_task']} trials"
        )
    errored = f"{OK if not t['errored'] else WARN} {t['errored']} errored ({_pct(t['errored'], n)})"
    kinds = list(t["error_types"].items())
    if kinds:  # Harbor's recorded exception types, not trajectory evidence
        shown = ", ".join(f"{kind} {count}" for kind, count in kinds[:3])
        errored += f": {shown}" + (f", +{len(kinds) - 3} more" if len(kinds) > 3 else "")
    cov.append(errored)
    if t["without_trajectory"]:
        cov.append(f"{WARN} {t['without_trajectory']} without trajectory")
    lines.append("COVERAGE   " + " · ".join(cov))
    rr = ov.get("reruns")
    if rr and rr["unlisted_trials"]:
        lst, un = rr["listed"], rr["unlisted"]
        line = (
            f"{'':<10} {WARN} {rr['unlisted_trials']} trial folder(s) not in the job's"
            f" result.json ({rr['unlisted_with_trajectory']} with a trajectory)"
        )
        if rr["tasks_rerun"]:
            line += f" · {rr['tasks_rerun']} task(s) run again: likely a rerun or resume of the job"
        lines.append(line)

        def share(p: dict) -> str:
            pct = f" ({100 * p['rewarded'] / p['scored']:.1f}%)" if p["scored"] else ""
            cost = f", ${p['cost_usd']:,.2f}" if p["cost_usd"] is not None else ""
            return f"{p['rewarded']}/{p['scored']}{pct}{cost}"

        lines.append(
            f"{'':<10} {INFO} all scored above · job's listed trials {share(lst)}"
            f" · other trials {share(un)}"
        )

    # TRACES: recording integrity + missing-activity estimate
    rec = b["recording"]
    first = True
    if ma["compacted"]:
        est = ""
        if ma["missing_calls_pct"]:
            lo, hi = ma["missing_calls_pct"]
            if hi >= 1:
                est = (
                    f" — est. {lo:.0f}–{hi:.0f}% of the run's LLM calls are not in its trajectories"
                )
            else:
                r_lo, r_hi = ma["compacted_recorded_pct"]
                est = (
                    f" — those trials recorded est. {r_lo:.0f}–{r_hi:.0f}% of their calls"
                    " (<1% of the run)"
                )
        lines.append(
            f"TRACES     {WARN} {ma['compacted']} ({_pct(ma['compacted'], n)})"
            f" compacted history{est}"
        )
        first = False
    for check, count in rec.items():
        if check == "integrity.history_compacted":
            continue
        mark = (
            WARN
            if check
            in (
                "integrity.observation_pairing_reconstructed",
                "integrity.observation_pairing_unresolved",
                "integrity.tokens_exceed_recorded_calls",
                "integrity.output_token_ratio",
                "integrity.web_results_not_recorded",
                "integrity.redacted_values",
                "integrity.agent_steps_missing",
            )
            else INFO
        )
        label = RECORDING_LABELS.get(check, check)
        lines.append(f"{'TRACES' if first else '':<10} {mark} {label}: {count} ({_pct(count, n)})")
        first = False
    if first:
        lines.append(f"TRACES     {OK} no recording defects detected")
    if exposure := b.get("reasoning"):
        # A model property, not a recording defect: shown for reading findings, never warned.
        shown = " · ".join(f"{REASONING_LABELS[k]} {v} ({_pct(v, n)})" for k, v in exposure.items())
        lines.append(
            f"{'':<10} {INFO} reasoning: {shown}; withheld or summarised reasoning is by"
            " design for many models, and checks read the recorded text"
        )
    for basis, r in (b.get("output_ratio") or {}).items():
        k = r["traces"]
        if k == 1:
            spread = ""
        elif k < MIN_TRACES_FOR_PERCENTILES:
            spread = f" (range {r['min']:.2f}–{r['max']:.2f})"
        else:
            spread = f" (p5–p95 {r['p5']:.2f}–{r['p95']:.2f})"
        lines.append(
            f"{'':<10} {INFO} chars/output token: median {r['median']:.2f}{spread}"
            f" over {k} trace{'' if k == 1 else 's'}, {RATIO_BASIS[basis]}"
        )
    if d and d["rewarded_not_cleared"]:
        why = " · ".join(f"{reason} {count}" for reason, count in d["not_cleared_reasons"].items())
        lines.append(
            f"{'':<10} → {d['rewarded_not_cleared']} rewarded trial(s) can't be cleared: {why}"
        )

    # COST
    cost_notes: list[str] = []
    line = f"COST       ${c['total_usd']:,.2f} reported"
    tk = ce.get("tokens") or {}
    if ce["unpriced"] and ce["unpriced"] == n - ce["no_usage"] and ce["estimate_usd"] is None:
        line = (
            f"COST       {WARN} no cost recorded for any trial (trajectories or run metadata)\n"
            f"{'':<10} {INFO} tokens: {_m(tk.get('uncached_input', 0) + tk.get('cached_input', 0))}"
            f" input ({_m(tk.get('cached_input', 0))} cached) · {_m(tk.get('output', 0))} output"
            " · estimate with --price U,C,O ($/M)"
        )
    elif ce["unpriced"] and not c["total_usd"] and ce["method"] == "given --price":
        r = ce["rates_per_mtok"]
        line = (
            f"COST       {WARN} no cost recorded · est. ${ce['estimate_usd']:,.2f} at the given"
            f" --price ${r['uncached_input']:g}/${r['cached_input']:g}/${r['output']:g} per M"
            " (uncached/cached/output)"
        )
    elif ce["unpriced"]:
        mm = ov.get("model_mismatch") or {}
        if ce.get("priced_other_model") and mm.get("cost_usd"):
            spent = mm["cost_usd"]
            line += (
                " (all of it on another model)"
                if abs(spent - c["total_usd"]) < 0.01
                else f" (${spent:,.2f} of it on another model)"
            )
        line += f" · {WARN} {ce['unpriced']} unpriced trial(s) ({_pct(ce['unpriced'], n)})"
        if ce["estimate_usd"] is None and ce.get("priced_other_model"):
            ut = ce["unpriced_tokens"]
            cost_notes += [
                f"{'':<10} {WARN} not estimated: only {ce['priced_other_model']} trial(s) on"
                " another model are priced, and their prices aren't this model's",
                f"{'':<10} {INFO} unpriced tokens:"
                f" {_m(ut['uncached_input'] + ut['cached_input'])} input"
                f" ({_m(ut['cached_input'])} cached) · {_m(ut['output'])} output"
                " · estimate with --price U,C,O ($/M)",
            ]
        elif ce["estimate_usd"] is not None:
            corrected = c["total_usd"] + ce["estimate_usd"]
            share = 100 * ce["estimate_usd"] / corrected if corrected else 0
            line += f" → est. +${ce['estimate_usd']:,.2f} (≈ ${corrected:,.2f}, +{share:.1f}%)"
        else:
            line += f" ({ce['method']})"
    elif um["trials"]:
        line += f" · {OK} every trial with usage priced"
    else:
        line += f" · {OK} every trial priced"
    idle = ce["no_usage"] - um["trials"]  # no usage and no recorded work either
    if idle and ce["no_usage"] < n:
        line += f" · {idle} without usage data (no recorded work)"
    lines.append(line)
    lines += cost_notes
    if um["trials"]:
        # Real work the reported total silently leaves out.
        why = ", ".join(
            f"{v} {k}" for k, v in (("errored", um["errored"]), ("rewarded", um["rewarded"])) if v
        )
        line = (
            f"{'':<10} {WARN} {um['trials']} trial(s) did work but report no usage or cost"
            + (f" ({why})" if why else "")
            + f": {um['llm_calls']:,} LLM calls, {um['tool_calls']:,} tool calls,"
            f" {um['duration_sec'] / 60:,.1f} min"
        )
        if um["estimate_usd"] is not None:
            total = c["total_usd"] + um["estimate_usd"]
            share = 100 * um["estimate_usd"] / total if total else 0
            line += (
                f" → est. +${um['estimate_usd']:,.2f} (≈ ${total:,.2f}, +{share:.1f}%)"
                " not in the total"
            )
        else:
            line += f" · not in the total ({um['method']})"
        lines.append(line)

    # FINDINGS
    f = b["findings"]
    sev = " · ".join(f"{s} {v}" for s, v in f["traces_by_highest_severity"].items())
    lines.append(
        f"FINDINGS   traces by highest review priority: {sev or 'none'}; check counts overlap"
    )
    if ix := ov.get("finding_index"):
        lines.append(f"{'':<10} index: {index_text(ix)} (review load, not a verdict)")
    top = [(cid, v) for cid, v in f["checks"].items() if RANK[v["severity"]] >= RANK["medium"]]
    for cid, v in top[:6]:
        spread = f"{v['events']} event(s) across " if "events" in v else ""
        lines.append(f"{'':<10} {WARN} {v['severity']:<8} {cid} · {spread}{v['traces']} trace(s)")
    if len(top) > 6:
        lines.append(f"{'':<10}   +{len(top) - 6} more medium+ checks (see --summary)")

    # SETTINGS
    scoring = [o for o in ov["overrides"] if override_kind(o) == "scoring"]
    infra = [o for o in ov["overrides"] if override_kind(o) == "infrastructure"]
    label = "SETTINGS"
    if scoring:
        lines.append(
            f"{label:<10} {WARN} scoring overrides (change the agent's time or resources; "
            "leaderboards require defaults): " + ", ".join(scoring)
        )
        label = ""
    if infra:
        lines.append(
            f"{label:<10} · infrastructure overrides (provisioning only): " + ", ".join(infra)
        )
        label = ""
    if not ov["overrides"] and b["runs"]:
        lines.append(f"SETTINGS   {OK} no leaderboard-forbidden overrides")
    mm = ov.get("model_mismatch")
    if mm:
        others = ", ".join(f"{m} {k}" for m, k in mm["other_models"].items())
        lines.append(
            f"MODEL      {WARN} critical  {len(mm['trial_ids'])} trial(s)"
            f" ({_pct(len(mm['trial_ids']), n)}) ran another model than {mm['expected']}:"
            f" {others} · {len(mm['rewarded_ids'])} rewarded · ${mm['cost_usd']:,.2f}"
            + (f" · {len(mm['switched_ids'])} switched mid-trial" if mm.get("switched_ids") else "")
        )
        lines.append(
            f"{'':<10} fallback or substitution: those rewards and costs aren't this model's"
        )
        label = ""
    for r in b["runs"]:
        if r.get("canonical_dataset") is False:
            lines.append(
                f"{label:<10} {WARN} tasks from a non-canonical source: "
                + ", ".join(r.get("datasets") or ["?"])
                + " (diff it against the benchmark: tools/task_diff.py)"
            )
            label = ""
    for i, (question, counts) in enumerate(sorted((b.get("answers") or {}).items())):
        shown = " · ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        lines.append(f"{'ANSWERS' if i == 0 else '':<10} {question}: {shown}")
    if b.get("answers"):
        lines.append(f"{'':<10} reviewer annotations only; DQ candidates unchanged")
    for pack in b.get("packs") or []:
        lines.append(f"PACKS      {OK} {pack['pack']} loaded (recognised by {pack['reason']})")
    for pack in b.get("suggested_packs") or []:
        lines.append(f"PACKS      {WARN} task pack for this dataset not loaded: --plugin {pack}")

    # ADJUSTMENTS summary
    adj = []
    if ce["estimate_usd"]:
        adj.append(
            f"cost +${ce['estimate_usd']:,.2f} for {ce['unpriced']} unpriced trials"
            f" ({ce['method']}"
            + (
                f", median error ${ce['median_abs_error_usd']})"
                if ce["median_abs_error_usd"] is not None
                else ")"
            )
        )
    if um["trials"]:
        adj.append(
            f"cost: {um['trials']} trial(s) with {um['llm_calls']:,} LLM calls report no usage"
            + (
                f" → est. +${um['estimate_usd']:,.2f} (rough: {um['method']},"
                f" median error ${um['median_abs_error_usd']}/trial)"
                if um["estimate_usd"] is not None
                else f" ({um['method']})"
            )
            + (
                f"; {um['rewarded']} rewarded, counted in RESULT without a cost"
                if um["rewarded"]
                else ""
            )
        )
    if ma["missing_calls_pct"] and ma["missing_calls_pct"][1] >= 1:
        lo, hi = ma["missing_calls_pct"]
        adj.append(
            f"traces: ~{lo:.0f}–{hi:.0f}% of LLM calls unrecorded;"
            f" findings on {ma['compacted']} compacted trials are partial"
        )
    lines.append("")
    if adj:
        lines.append("ADJUSTMENTS (estimates for review, not verdicts)")
        lines += [f"  · {a}" for a in adj]
    else:
        lines.append(f"ADJUSTMENTS {OK} none suggested")
    lines += [
        "",
        "more: --summary (per-check detail) · --cite high (evidence)"
        " · --detail (per trace) · --format json",
    ]
    return "\n".join(lines) + "\n"


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


def colourise(text: str):
    """The brief as a rich Text with styles applied by pattern."""
    from rich.text import Text

    out = Text()
    for line in text.splitlines(keepends=True):
        styled = Text(line)
        for pattern, style in COMPILED:
            styled.highlight_regex(pattern, style)
        out.append_text(styled)
    return out


def print_brief(text: str, file=None) -> None:
    """Coloured on a terminal (rich), plain otherwise."""
    try:
        from rich.console import Console
    except ImportError:
        print(text, end="", file=file)
        return
    console = Console(file=file, highlight=False, soft_wrap=True)
    if not console.is_terminal:
        print(text, end="", file=file)
        return
    console.print(colourise(text), end="")
