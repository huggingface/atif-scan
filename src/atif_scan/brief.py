"""The one-screen run integrity report (`--brief`, and the default text view for a run).

What a benchmarker needs first: what was scanned, the result and what it would become
after the potential adjustments (disqualification candidates, unpriced trials, missing
trace activity), trace/cost integrity, and counted findings. Detail stays behind flags
(--summary, --cite, --detail). Built only from the allowlisted report document.
"""

from __future__ import annotations

from collections import Counter

from .estimates import cost_estimate, missing_activity
from .report import RANK, _m, overview

OK, WARN, BAD, INFO = "✓", "⚠", "✗", "·"


def _counted(item: dict) -> list[dict]:
    return [a for a in item["assessments"] if a.get("score") is not None]


def brief(doc: dict, dq: str = "high", min_trials=None, expect_tasks=None, price=None) -> dict:
    items = doc["inputs"]
    ov = overview(doc, dq, min_trials, expect_tasks=expect_tasks)
    agents = Counter(
        " / ".join(
            p for p in (i.get("agent_name"), i.get("agent_version"), i.get("model_name")) if p
        )
        for i in items
        if i.get("agent_name") or i.get("model_name")
    )
    behaviour = Counter()  # check -> traces, excluding recording-integrity checks
    integrity = Counter()
    by_severity = Counter()
    for item in items:
        counted = _counted(item)
        for check in {a["id"] for a in counted}:
            if check == "integrity.cost_missing" and item.get("cost_usd") is not None:
                continue  # the trajectory lacks cost, but the source (e.g. Hub) has it
            (integrity if check.startswith("integrity.") else behaviour)[check] += 1
        worst = max(
            (RANK[a["severity"]] for a in counted if not a["id"].startswith("integrity.")),
            default=None,
        )
        by_severity[next((k for k, v in RANK.items() if v == worst), "none")] += 1
    severity_of = {
        a["id"]: a["severity"] for item in items for a in item["assessments"] if a.get("severity")
    }
    return {
        "schema_version": 1,
        "kind": "integrity_brief",
        "scanner_version": doc["scanner_version"],
        "runs": ov["runs"],
        "agents": dict(agents.most_common()),
        "overview": ov,
        "cost_estimate": cost_estimate(items, price),
        "tasks_known": sum(1 for i in items if i.get("task")),
        "missing_activity": missing_activity(items),
        "recording": {
            check: integrity[check]
            for check in sorted(integrity, key=lambda c: (-RANK[severity_of[c]], -integrity[c]))
        },
        "findings": {
            "traces_by_highest_severity": {
                k: by_severity[k]
                for k in ("critical", "high", "medium", "low", "info", "none")
                if by_severity[k]
            },
            "checks": {
                c: {"severity": severity_of[c], "traces": n}
                for c, n in sorted(
                    behaviour.items(), key=lambda kv: (-RANK[severity_of[kv[0]]], -kv[1])
                )
            },
        },
    }


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "—"


RECORDING_LABELS = {
    "integrity.history_compacted": "compacted history (only the last context window recorded)",
    "integrity.reasoning_not_recorded": "reasoning produced but not recorded",
    "integrity.timestamp_missing": "steps without timestamps",
    "integrity.tokens_exceed_recorded_calls": "token totals exceed what recorded calls could use",
    "integrity.cost_missing": "tokens recorded but no cost (anywhere)",
    "integrity.timestamp_smearing": "timestamps stamped at export",
    "integrity.timestamp_regression": "timestamps going backwards",
    "integrity.orphan_observation": "tool results without a matching call",
    "integrity.agent_only_fields": "agent-only fields on system/user steps",
}


def brief_text(b: dict) -> str:
    ov, ce, ma = b["overview"], b["cost_estimate"], b["missing_activity"]
    t, k, d, c = ov["trials"], ov["tasks"], ov["disqualification"], ov["cost"]
    n = t["present"]
    lines = [f"atif-scan {b['scanner_version']} · run integrity", ""]
    for r in b["runs"]:
        ref = (r.get("dataset_refs") or [""])[0][:15]
        kind = "harbor" if r.get("source") == "harbor_hub" else "job"
        where = f"{kind} {r['job_id'][:8]}" if r.get("job_id") else "Harbor job folder"
        name = r.get("job_name")
        lines.append(
            (f"{name} · {where}" if name and name != "job" else where)
            + (f" · {', '.join(r['datasets'])}@{ref}" if r.get("datasets") else "")
        )
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

    # RESULT and its adjustment
    if ov["accuracy"]:
        acc, se = ov["accuracy"]
        succ = round(acc * (n - t["reward_unknown"]) / 100)
        spread = f" ± {se:.1f}" if b["tasks_known"] else ""  # per-task SE needs tasks
        line = f"RESULT     {acc:.1f}%{spread} ({succ}/{n - t['reward_unknown']})"
        if d and d["candidates"]:
            adj, adj_se = d["accuracy_if_disqualified"]
            adj_spread = f" ± {adj_se:.1f}" if b["tasks_known"] else ""
            line += (
                f"  →  {adj:.1f}%{adj_spread} if {d['candidates']} DQ candidate(s) are disqualified"
            )
        elif d is not None:
            line += "  ·  no DQ candidates"
        lines.append(line)
    else:
        lines.append("RESULT     rewards unknown (no verifier output or Hub record)")

    # COVERAGE
    cov = []
    if t["planned"] is not None:
        folder = any(r.get("source") == "harbor_job_folder" for r in b["runs"])
        what = "planned trials have a trajectory" if folder else "planned trials present"
        cov.append(f"{OK if not t['missing'] else WARN} {t['present']}/{t['planned']} {what}")
    if k["expected_tasks"]:
        cov.append(
            f"{OK if not k['missing_tasks'] else WARN} {k['count']}/{k['expected_tasks']} tasks"
        )
    if k["expected_per_task"]:
        below = len(k["below_expected"])
        cov.append(
            f"{OK if not below else WARN} {below} task(s) below {k['expected_per_task']} trials"
        )
    cov.append(
        f"{OK if not t['errored'] else WARN} {t['errored']} errored ({_pct(t['errored'], n)})"
    )
    if t["without_trajectory"]:
        cov.append(f"{WARN} {t['without_trajectory']} without trajectory")
    lines.append("COVERAGE   " + " · ".join(cov))

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
            in ("integrity.reasoning_not_recorded", "integrity.tokens_exceed_recorded_calls")
            else INFO
        )
        label = RECORDING_LABELS.get(check, check)
        lines.append(f"{'TRACES' if first else '':<10} {mark} {label}: {count} ({_pct(count, n)})")
        first = False
    if first:
        lines.append(f"TRACES     {OK} no recording defects detected")
    if d and d["rewarded_not_cleared"]:
        lines.append(
            f"{'':<10} → {d['rewarded_not_cleared']} rewarded trial(s) can't be cleared"
            " (partial or missing traces)"
        )

    # COST
    line = f"COST       ${c['total_usd']:,.2f} reported"
    tk = ce.get("tokens") or {}
    if ce["unpriced"] and ce["unpriced"] == n - ce["no_usage"] and ce["estimate_usd"] is None:
        line = (
            f"COST       {WARN} no cost recorded for any trial (trajectories or run metadata)"
            f" · tokens: {_m(tk.get('uncached_input', 0) + tk.get('cached_input', 0))} input"
            f" ({_m(tk.get('cached_input', 0))} cached) · {_m(tk.get('output', 0))} output"
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
        line += f" · {WARN} {ce['unpriced']} unpriced trial(s) ({_pct(ce['unpriced'], n)})"
        if ce["estimate_usd"] is not None:
            corrected = c["total_usd"] + ce["estimate_usd"]
            share = 100 * ce["estimate_usd"] / corrected if corrected else 0
            line += f" → est. +${ce['estimate_usd']:,.2f} (≈ ${corrected:,.2f}, +{share:.1f}%)"
        else:
            line += f" ({ce['method']})"
    else:
        line += f" · {OK} every trial priced"
    if ce["no_usage"] and ce["no_usage"] < n:
        line += f" · {ce['no_usage']} without usage data"
    lines.append(line)

    # FINDINGS
    f = b["findings"]
    sev = " · ".join(f"{s} {v}" for s, v in f["traces_by_highest_severity"].items())
    lines.append(f"FINDINGS   traces by highest severity: {sev or 'none'}")
    top = [(cid, v) for cid, v in f["checks"].items() if RANK[v["severity"]] >= RANK["medium"]]
    for cid, v in top[:6]:
        lines.append(
            f"{'':<10} {BAD if RANK[v['severity']] >= RANK['high'] else WARN}"
            f" {v['severity']:<8} {cid} · {v['traces']} trace(s)"
        )
    if len(top) > 6:
        lines.append(f"{'':<10}   +{len(top) - 6} more medium+ checks (see --summary)")

    # SETTINGS
    if ov["overrides"]:
        lines.append(
            f"SETTINGS   {WARN} overrides set: {', '.join(ov['overrides'])}"
            " (leaderboards require defaults)"
        )
    elif b["runs"]:
        lines.append(f"SETTINGS   {OK} no leaderboard-forbidden overrides")

    # ADJUSTMENTS summary
    adj = []
    if d and d["candidates"]:
        acc, _ = ov["accuracy"]
        adj.append(
            f"accuracy {acc:.1f}% → {d['accuracy_if_disqualified'][0]:.1f}%"
            f" ({d['candidates']} DQ candidates, review with --cite high)"
        )
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
