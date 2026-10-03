"""The run brief as text: the default view of a scanned run.

Written to be read top to bottom by someone deciding whether a run's score stands:

    RUN       what was scanned
    SCORE     the recorded result, and what it would be without the flagged successes
    REVIEW    which rewarded trials need a human, and why
    FINDINGS  what the checks found, by review priority, in plain words
    AWARENESS whether the agent worked out it was being benchmarked, stage by stage
    EVIDENCE  whether the trials and their recordings are complete enough to rely on
    WALLTIME  summed agent execution and full-trial time, with coverage counts
    TOKENS    token accounting: recorded usage, checked against each other
    COST      priced from those tokens
    SETTINGS  run configuration that affects comparability
    MORE      where the detail is

House rules, so every line means exactly one thing:

- Every count names its unit and, where a share matters, its denominator ("47 of 419
  rewarded trials"); plurals are real words, never "trial(s)".
- One mark vocabulary: ✓ checked and fine · ⚠ needs attention · · context. Estimates say
  "est." and how they were made; nothing is stated twice.
- A section that has nothing to report says so in one ✓ line, or is left out when it
  can't apply (no rewards → no REVIEW).
- Lines fit in WIDTH columns; long ones wrap under their own text, never under the label.

Everything here reads the brief document only (brief.brief), which is built from the
allowlisted report: no trace text can reach this view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...sources.harbor.files import override_kind

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...data.jsonval import Doc
from .evidence import evidence_section, web_activity_section
from .run import awareness_section, findings_section, review_section, run_section, score_section
from .usage import cost_section, tokens_section, walltime_section
from .words import INFO, OK, WARN, Lines, plural, wrap


def settings_section(b: Doc) -> Lines:
    ov = b["overview"]
    scoring = [o for o in ov["overrides"] if override_kind(o) == "scoring"]
    infra = [o for o in ov["overrides"] if override_kind(o) == "infrastructure"]
    body: Lines = []
    if scoring:
        body.append(
            f"{WARN} overrides that change the agent's time or resources (leaderboards"
            " require defaults): " + ", ".join(scoring)
        )
    if infra:
        body.append(f"{INFO} provisioning-only overrides: " + ", ".join(infra))
    if not ov["overrides"] and b["runs"]:
        body.append(f"{OK} no override a leaderboard forbids")
    for r in b["runs"]:
        if r.get("canonical_dataset") is False:
            body.append(
                f"{WARN} tasks aren't from the benchmark's own source ("
                + ", ".join(r.get("datasets") or ["unknown"])
                + "): diff them with tools/task_diff.py"
            )
    body += _submission_texts(b)
    body += [
        f"{OK} task pack {p['pack']} loaded (recognised by {p['reason'].replace('_', ' ')})"
        for p in b.get("packs") or []
    ]
    body += [
        f"{WARN} this dataset's task pack isn't loaded: --plugin {p}"
        for p in b.get("suggested_packs") or []
    ]
    return wrap("SETTINGS", body)


FILTER_WORDS = (("agent", "agent"), ("agent_version", "version"), ("model_name", "model"))


def _submission_texts(b: Doc) -> Lines:
    """The submission file's source_filter, checked against each trial's own record."""
    sub = b.get("submission")
    if not sub:
        return []
    want = sub.get("filter") or {}
    shown = " · ".join(f"{word} {want[k]}" for k, word in FILTER_WORDS if want.get(k))
    total = sum(sub.get(k, 0) for k in ("matched", "partial", "mismatched", "unrecorded"))
    texts = []
    if sub["mismatched"]:
        texts.append(
            f"{WARN} {sub['mismatched']:,} of {plural(total, 'trial')}"
            f" {'doesn' if sub['mismatched'] == 1 else 'don'}'t match the"
            f" submission's filter ({shown})"
        )
    agree = sub["matched"] + sub.get("partial", 0)
    if agree:
        texts.append(
            f"{OK} {agree:,} of {plural(total, 'trial')} match the submission's filter ({shown})"
        )
    if sub.get("partial"):
        texts.append(
            f"{INFO} {plural(sub['partial'], 'trial')} of those"
            f" {'records' if sub['partial'] == 1 else 'record'} only some of these"
            " fields (e.g. no model in the header), so only those were checked"
        )
    if sub["unrecorded"]:
        texts.append(
            f"{INFO} {plural(sub['unrecorded'], 'trial')}"
            f" {'records' if sub['unrecorded'] == 1 else 'record'} no agent or model to check"
        )
    if want.get("reasoning_effort"):
        texts.append(
            f"{INFO} reasoning effort {want['reasoning_effort']} isn't recorded per trial,"
            " so it isn't checked"
        )
    return texts


def more_section(b: Doc) -> Lines:
    return wrap(
        "MORE",
        ["--summary every check · --cite high evidence · --detail each trial · --format json"],
    )


SECTIONS: tuple[Callable[[Doc], Lines], ...] = (
    run_section,
    score_section,
    review_section,
    findings_section,
    awareness_section,
    web_activity_section,
    evidence_section,
    walltime_section,
    tokens_section,
    cost_section,
    settings_section,
    more_section,
)


def brief_text(b: Doc) -> str:
    """The brief as text: a two-line header, then each non-empty section."""
    head = [
        f"atif-scan {b['scanner_version']} · run integrity report",
        f"Findings set review priority, not verdicts.  {OK} checked  {WARN} needs attention"
        f"  {INFO} context  est. estimate",
    ]
    blocks = [lines for section in SECTIONS if (lines := section(b))]
    return "\n\n".join("\n".join(block) for block in [head, *blocks]) + "\n"
