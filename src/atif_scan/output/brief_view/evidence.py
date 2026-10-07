"""EVIDENCE: whether the trials and their recordings are complete enough to rely on (missing or
errored trials, reruns, compaction, web results, recording gaps)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...data.jsonval import Doc
from .words import (
    ERROR_KINDS_SHOWN,
    INFO,
    MISSING_CALLS_NOTABLE_PCT,
    OK,
    REASONING,
    WARN,
    Lines,
    _present,
    _title_inline,
    counts,
    has,
    pct,
    plural,
    usd,
    was,
    wrap,
)

# Recording checks shown under TOKENS or COST instead (they concern usage or cost).
USAGE_CHECKS = (
    "integrity.tokens_exceed_recorded_calls",
    "integrity.totals_are_last_call",
    "integrity.output_token_ratio",
    "integrity.incomplete_tool_generation",
)


COST_CHECKS = ("integrity.cost_missing",)


# Recording defects that limit what the checks can see (⚠); the rest are context (·).
LIMITING = frozenset(
    {
        "integrity.observation_pairing_reconstructed",
        "integrity.observation_pairing_unresolved",
        "integrity.web_results_not_recorded",
        "integrity.web_input_unresolved",
        "integrity.redacted_values",
        "integrity.agent_steps_missing",
        "integrity.tool_results_not_recorded",
        "integrity.actions_not_recorded",
        "integrity.subagent_unrecorded",
    }
)


def _trial_texts(b: Doc) -> Lines:
    t, k, n = b["overview"]["trials"], b["overview"]["tasks"], _present(b)
    texts: Lines = []
    if t["planned"] is not None:
        folder = any(r.get("source") == "harbor_job_folder" for r in b["runs"])
        what = "have a trajectory" if folder else "are present"
        over = t["present"] > t["planned"]  # more than planned: retries, reruns, merges
        mark = WARN if t["missing"] or over else OK
        texts.append(f"{mark} {n:,} of {plural(t['planned'], 'planned trial')} {what}")
    if k["expected_tasks"]:
        mark = WARN if k["missing_tasks"] else OK
        texts.append(f"{mark} {k['count']:,} of {plural(k['expected_tasks'], 'expected task')}")
    if k["expected_per_task"] and k["below_expected"]:
        texts.append(
            f"{WARN} {plural(len(k['below_expected']), 'task')} with fewer than"
            f" {plural(k['expected_per_task'], 'trial')}"
        )
    if t["without_trajectory"]:
        texts.append(f"{WARN} {plural(t['without_trajectory'], 'trial')} without a trajectory")
    kinds = list(t["error_types"].items())
    if t["errored"]:
        shown = counts(kinds[:ERROR_KINDS_SHOWN])
        more = (
            f" · {len(kinds) - ERROR_KINDS_SHOWN} more kinds"
            if len(kinds) > ERROR_KINDS_SHOWN
            else ""
        )
        texts.append(
            f"{WARN} {plural(t['errored'], 'trial')} errored ({pct(t['errored'], n)}):"
            f" {shown}{more}"
        )
    else:
        texts.append(f"{OK} no trial errored")
    return texts


def _listed_share(p: Doc) -> str:
    share = f" ({pct(p['rewarded'], p['scored'])})" if p["scored"] else ""
    cost = f", {usd(p['cost_usd'])}" if p["cost_usd"] is not None else ""
    return f"{p['rewarded']:,} of {p['scored']:,} rewarded{share}{cost}"


def _rerun_texts(rr: Doc | None) -> Lines:
    if not rr or not rr["unlisted_trials"]:
        return []
    text = (
        f"{WARN} {plural(rr['unlisted_trials'], 'trial folder')} not listed in the job's"
        f" result.json ({rr['unlisted_with_trajectory']:,} with a trajectory)"
    )
    if rr["tasks_rerun"]:
        text += f"; {plural(rr['tasks_rerun'], 'task')} ran again: likely a rerun or resume"
    return [
        text,
        f"{INFO} all are scored above · listed trials {_listed_share(rr['listed'])}"
        f" · other trials {_listed_share(rr['unlisted'])}",
    ]


def _compacted_text(ma: Doc, n: int) -> str | None:
    if not ma["compacted"]:
        return None
    text = (
        f"{WARN} {plural(ma['compacted'], 'trial')} ({pct(ma['compacted'], n)})"
        f" {has(ma['compacted'])} compacted ATIF history: earlier segments are not in the"
        " scanned trajectory, so their findings are partial"
    )
    if ma["missing_calls_pct"]:
        lo, hi = ma["missing_calls_pct"]
        if hi >= MISSING_CALLS_NOTABLE_PCT:
            text += (
                f"; est. {lo:.0f}–{hi:.0f}% of the run's LLM calls are outside"
                " its scanned trajectories"
            )
        else:
            r_lo, r_hi = ma["compacted_recorded_pct"]
            text += f"; they recorded est. {r_lo:.0f}–{r_hi:.0f}% of their LLM calls"
    available = ma.get("archives_available", 0)
    text += (
        f"; {available} local Grok Markdown companion archive(s) available but not scanned"
        if available
        else "; supported Grok Markdown archives not found or not checked"
        " (other formats and source availability unknown)"
    )
    return text


def _overlapping_recording_texts(b: Doc) -> Lines:
    gaps = b.get("recording_gaps") or {}
    if not gaps.get("trials"):
        return []
    groups = [
        f"{gaps[key]:,} {label}"
        for key, label in (
            ("pairing_reconstructed", "pairing links inferred"),
            ("pairing_unresolved", "pairing unresolved"),
            ("web_results_not_recorded", "web outputs missing"),
        )
        if gaps.get(key)
    ]
    return [
        f"{WARN} {plural(gaps['trials'], 'trial')} with recording gaps: "
        + " · ".join(groups)
        + " (subgroups may overlap)",
        *(
            [
                f"{INFO} unresolved pairing concerns call attribution, not necessarily missing"
                " output; a result may cover a command batch"
            ]
            if gaps.get("pairing_unresolved")
            else []
        ),
        *(
            [
                f"{WARN} {gaps['web_calls_without_usable_result']:,} recorded web calls"
                f" have no usable result across {plural(gaps['web_call_gap_trials'], 'trial')}",
                f"{INFO} affected calls: {gaps['web_calls_missing_result_ids']:,} missing"
                f" result IDs; {gaps['web_calls_no_emitted_contents']:,} with no emitted"
                " contents (overlapping counts). Shared parents count once; calls are not"
                " web actions or backend requests."
                " Explicit retrieval errors are recorded outcomes.",
            ]
            if gaps.get("web_calls_without_usable_result")
            else []
        ),
        f"{INFO} inferred links do not recover missing content; source exposure"
        " remains unknown where web outputs are missing",
    ]


def _recording_texts(b: Doc) -> Lines:
    n = _present(b)
    texts: Lines = [c] if (c := _compacted_text(b["missing_activity"], n)) else []
    texts += _overlapping_recording_texts(b)
    for check, count in b["recording"].items():
        if check in ("integrity.history_compacted", *USAGE_CHECKS, *COST_CHECKS) or (
            b.get("recording_gaps")
            and check
            in {
                "integrity.observation_pairing_reconstructed",
                "integrity.observation_pairing_unresolved",
                "integrity.web_results_not_recorded",
            }
        ):
            continue
        mark = WARN if check in LIMITING else INFO
        texts.append(
            f"{mark} {plural(count, 'trial')} ({pct(count, n)}): {_title_inline(b, check)}"
        )
    if not texts:
        texts.append(f"{OK} no recording defect detected")
    if (cc := b.get("context_compactions") or {}).get("trials"):
        # Not a gap: the steps stay in the file; the agent worked from a summary after it.
        texts.append(
            f"{INFO} agent context compacted in {plural(cc['trials'], 'trial')}"
            f" ({plural(cc['total'], 'compaction')}); earlier steps stay in the recording, the"
            " agent continued from a summary"
        )
    if ic := b.get("image_checks"):
        # What --image-model resolved; images it couldn't read keep their checks unknown.
        read = ic.get("all", 0) + ic.get("no_text", 0)
        unread = ic["images"] - read
        texts.append(
            f"{INFO} images transcribed for blocked checks in {plural(ic['trials'], 'trial')}:"
            f" {read} of {ic['images']} read"
            + (f", {unread} still unread" if unread else "")
            + (f" · {ic['sensitive']} flagged sensitive" if ic.get("sensitive") else "")
            + (f" · {ic['instructions']} with instructions" if ic.get("instructions") else "")
        )
    if exposure := b.get("reasoning"):
        # A property of the model and its API, not a recording defect: never a warning.
        shown = " · ".join(f"{REASONING[k]} {v:,}" for k, v in exposure.items())
        texts.append(
            f"{INFO} reasoning: {shown} (many models withhold or summarise it by design;"
            " checks read what is recorded)"
        )
    return texts


def _job_texts(b: Doc) -> Lines:
    """How several jobs fit together, and trials added to a job after it ran."""
    jobs = b.get("jobs") or {}
    texts: Lines = []
    if jobs.get("count", 0) > 1 and jobs.get("with_tasks") == jobs["count"]:
        shared = jobs["shared_tasks"]
        if shared:
            texts.append(
                f"{WARN} {plural(len(shared), 'task')} {'is' if len(shared) == 1 else 'are'} in"
                " more than one job: their trials come from separate runs of the task, so"
                " check none were picked by outcome"
            )
        else:
            texts.append(
                f"{OK} the {jobs['count']} jobs cover {plural(jobs['tasks'], 'task')} with no"
                " task in two jobs"
            )
    for job in jobs.get("constructed") or []:
        texts.append(
            f"{WARN} job {job[:8]} has a constructed (UUIDv5) ID: its trials were assembled"
            " into it (e.g. a filtered mirror), not run as one job"
        )
    for late in jobs.get("late") or []:
        texts.append(
            f"{WARN} {plural(late['trials'], 'trial')} in job {str(late['job_id'])[:8]}"
            f" started {late['gap_hours']:g} h after the rest of it (added later, e.g."
            f" replacements); {late['rewarded']:,} {was(late['rewarded'])} rewarded"
        )
    return texts


def evidence_section(b: Doc) -> Lines:
    return wrap(
        "EVIDENCE",
        [
            *_trial_texts(b),
            *_job_texts(b),
            *_rerun_texts(b["overview"].get("reruns")),
            *_recording_texts(b),
        ],
    )
