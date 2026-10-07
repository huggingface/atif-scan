"""Static trajectory viewer export: allowlisted findings plus whole-field masked text.

Unlike reports, this export deliberately contains trace text so a reviewer can publish
an individually checked trajectory. Every field is masked as a whole (best effort)
before it is written; highlights are only offsets proven against that masked text.
Nothing in a trace is executed, fetched or interpreted as markup by the viewer.
"""

from __future__ import annotations

import json
import os
import secrets
from copy import deepcopy
from importlib import resources
from typing import TYPE_CHECKING

from ..data.jsonval import as_object, as_str
from ..data.loader import load_bytes
from ..evidence.cite import mask, trace_secrets
from ..evidence.extract import _segment_content, _segment_status
from ..output.brief_view import sections
from ..review.coverage import UNIVERSAL
from .findings import apply_titles, fields, findings, first_span
from .focus import masked_focus
from .session import pinned_bytes
from .web import web_gap_details

if TYPE_CHECKING:
    from pathlib import Path

    from ..cli.inputs import Record
    from ..data.jsonval import Doc
    from ..data.model import Step, Trace
    from ..data.model import Trace as TraceType
    from ..review.catalogue import Question

FORMAT = "atif-scan-viewer/1"
ASSETS = ("index.html", "viewer.css", "evidence.js", "viewer.js")
DATA_FILE = "data.js"
MAX_NAME = 60
WEB_GAPS = "integrity.web_results_not_recorded"
TRIAL_KEYS = ("input_id", "task", "reward", "input_status", "severity", "score")
# Per-trial run facts, all allowlisted report fields (labels, counts, amounts).
FACT_KEYS = (
    "agent_name",
    "agent_version",
    "model_name",
    "error_type",
    "input_tokens",
    "cache_tokens",
    "output_tokens",
    "cost_usd",
    "duration_sec",
    "agent_duration_sec",
    "llm_calls",
    "agent_steps",
    "tool_calls",
    "context_compactions",
    "stream_retry_attempts",
    "termination_error",
    "final_stop_reason",
    "setup_duration_sec",
    "verifier_duration_sec",
    "failed_phase",
    "safety_provider",
    "safety_reason",
    "safety_category",
)
COVERAGE_KEYS = ("incomplete", "coverage_gaps", "recording_gaps", "input_error")
# --run/--release lineage: whether the trial counts, and what it replaced (codes, names).
SELECTION_KEYS = (
    "role",
    "id",
    "replaced_trial",
    "replaced_error",
    "failure_phase",
    "replaced_by",
    "replacement_trial",
    "supersedes",
    "state",
)


def _name(value: str, known: frozenset[str]) -> str:
    """A tool name is trace text: masked, single-line and bounded like any label."""
    text = " ".join(mask(value, known).split())
    return text if len(text) <= MAX_NAME else text[: MAX_NAME - 1] + "…"


def _label(step: Step, location: Doc, known: frozenset[str], provenance: Doc) -> str:
    part, index = as_str(location["part"]), location["index"]
    if part in ("message", "reasoning"):
        return part
    if part == "result":
        source = provenance.get("source_call_index")
        return f"result {index}" + (f" · call {source}" if source is not None else "")
    call = step.calls[index]
    name = _name(call.name, known) or call.tool
    if part == "call_info":
        return f"call {index} · {name} · call record"
    return f"call {index} · {name} · {provenance.get('channel', 'argument')}"


def _wanted(step: Step, location: Doc, status: str, targeted: bool) -> bool:
    """Hide only empty message/reasoning slots and redundant call records."""
    if targeted:
        return True
    if location["part"] == "call_info":
        call = step.calls[location["index"]]
        return not any(c.understood or c.text or c.media for _, c in call.fields)
    return status != "empty_or_absent"


def _field(step: Step, location: Doc, known: frozenset[str]) -> Doc:
    content, provenance = _segment_content(
        step, location["part"], location["index"], location["field"]
    )
    return {
        **{key: location[key] for key in ("part", "index", "field")},
        "label": _label(step, location, known, provenance),
        "status": _segment_status(content, location["part"]),
        "text": mask(content.text, known),
        **provenance,
    }


def _steps(trace: Trace, targeted: list[Doc], known: frozenset[str]) -> list[Doc]:
    by_number = dict(zip(trace.step_numbers, trace.steps, strict=True))
    shown: dict[int, list[Doc]] = {n: [] for n in by_number}
    for location, _ in fields(trace):
        step = by_number[location["step"]]
        field = _field(step, location, known)
        if _wanted(step, location, field["status"], location in targeted):
            shown[location["step"]].append(field)
    return [{"step": n, "role": s.source, "fields": shown[n]} for n, s in by_number.items()]


def _focus(trace: Trace, item: Doc, finding: Doc, location: Doc, known: frozenset[str]) -> Doc:
    if location["step"] not in trace.step_numbers:
        return {"focus_status": "unlocated", "highlight_start": None, "highlight_end": None}
    step = trace.steps[trace.step_numbers.index(location["step"])]
    try:
        content, _ = _segment_content(step, location["part"], location["index"], location["field"])
    except ValueError:
        return {"focus_status": "unlocated", "highlight_start": None, "highlight_end": None}
    span = first_span(item, finding, location) if location["part"] != "call_info" else None
    return masked_focus(content.text, span, known)


def _trial_findings(
    trace: Trace | None, item: Doc, report: Doc, known: frozenset[str]
) -> list[Doc]:
    found = findings(item)
    apply_titles(found, report)
    for finding in found:
        if trace is not None:
            if finding["check_id"] == WEB_GAPS and finding["status"] == "match":
                finding["web_gaps"] = web_gap_details(trace, finding["locations"])
            finding["locations"] = [
                {**loc, **_focus(trace, item, finding, loc, known)} for loc in finding["locations"]
            ]
        del finding["_identity"]
    return found


def trial_bundle(trial_id: str, item: Doc, trace: Trace | None, report: Doc) -> Doc:
    """One trial: allowlisted scan fields, findings with proven highlights, masked steps."""
    known = trace_secrets(trace) if trace is not None else frozenset()
    found = _trial_findings(trace, item, report, known)
    targeted = [
        {key: loc[key] for key in ("step", "part", "index", "field")}
        for finding in found
        for loc in finding["locations"]
    ]
    targeted += [
        {key: u["location"][key] for key in ("step", "part", "index", "field")}
        for finding in found
        for u in finding["unread"]
        if u["location"] is not None
    ]
    targeted += [
        gap[key]
        for finding in found
        for gap in finding.get("web_gaps", [])
        for key in ("call_location", "result_location")
        if gap.get(key) is not None
    ]
    steps = _steps(trace, targeted, known) if trace is not None else []
    return deepcopy(
        {
            "id": trial_id,
            **{key: item.get(key) for key in TRIAL_KEYS},
            "coverage": {key: item.get(key) for key in COVERAGE_KEYS},
            "facts": {key: item.get(key) for key in FACT_KEYS},
            "selection": {
                key: value
                for key in SELECTION_KEYS
                if (value := as_object(item.get("selection")).get(key)) is not None
            },
            "findings": found,
            "steps": steps,
        }
    )


def run_facts(b: Doc) -> Doc:
    """Run headline numbers from the brief document, plus its sections as text lines.

    Only these fields leave the brief: harness/model rows, job names, trial and score
    counts, token totals, walltime and cost amounts with their basis. Sections are the
    brief's own scanner-written lines (counts, labels, check names; never trace text).
    """
    ov, ce, scoped = b["overview"], b["cost_estimate"], b.get("scoped_costs") or {}
    return {
        "harness": [
            {key: row.get(key) for key in ("agent", "version", "model", "trials")}
            for row in b.get("agent_rows") or []
        ],
        "jobs": [r.get("job_name") or r.get("job_id") for r in b.get("runs") or []],
        "trials": {key: ov["trials"].get(key) for key in ("present", "planned", "errored")},
        "accuracy": ov.get("accuracy"),
        "tokens": dict(ce.get("tokens") or {}),
        "walltime": {
            "agent_sec": ov["walltime"]["agent"]["seconds"],
            "trial_sec": ov["walltime"]["trial"]["seconds"],
            "recorded_trials": ov["walltime"]["agent"]["recorded_trials"],
        },
        "cost": {
            "recorded_usd": ov["cost"].get("total_usd"),
            "without_cost": ov["cost"].get("missing"),
            "estimate_usd": ce.get("estimate_usd"),
            "estimate_source": ce.get("price_source"),
            "estimate_method": ce.get("method"),
            "rates_per_mtok": ce.get("rates_per_mtok"),
            "observed_usd": scoped.get("estimated_observed_cost_usd"),
            "observed_trials": scoped.get("trials"),
        },
        "selection": {
            key: (b.get("selection") or {}).get(key)
            for key in ("kind", "name", "pending", "not_counted", "as_run_accuracy")
        }
        if b.get("selection")
        else None,
        "sections": sections(b),
    }


# --- review mode: a blind human-review export ----------------------------------------

REVIEW_FORMAT = "atif-scan-review/1"
REVIEW_CHECK = "review.candidates"


def _candidates(question: Question, trace: TraceType) -> Doc | None:
    """The question's review targets as one assessment, so the viewer's receipts jump to
    them with proven highlights. Candidates, never findings: no priority, not scored."""
    targets = question.review_targets(trace) if question.review_targets else []
    if not targets:
        return None
    numbers = trace.step_numbers
    evidence = [
        {
            "step": at.step,
            "step_id": numbers[at.step],
            "channel": at.channel.value,
            "call": at.call,
            "observation": at.observation,
            "field": at.field,
            "span": list(at.span) if at.span else None,
        }
        for at in targets
    ]
    return {
        "id": REVIEW_CHECK,
        "kind": "detector",
        "version": question.version,
        "severity": "info",
        "status": "match",
        "evidence": evidence,
        "unread": [],
        "measure": {"candidates": len(evidence)},
        "expected_by": [],
    }


def blind_item(item: Doc, question: Question, trace: TraceType | None) -> Doc:
    """A report item with the scanner's view removed (findings, priority, score, the
    checks left unknown) and the question's candidates in their place."""
    candidates = _candidates(question, trace) if trace is not None else None
    return {
        **item,
        "assessments": [candidates] if candidates else [],
        "severity": None,
        "score": None,
        "coverage_gaps": None,
    }


def review_block(question: Question) -> Doc:
    """What the reviewer is asked: the question's own text and closed answer sets, and a
    random export ID that binds saved verdicts to this export."""
    return {
        "format": REVIEW_FORMAT,
        "export_id": secrets.token_hex(8),
        "question": question.id,
        "version": question.version,
        "title": question.title,
        "ask": question.ask,
        "answers": dict(question.answers),
        "mechanisms": dict(question.mechanisms),
        # Answers a mechanism describes; the others take "none".
        "positive": [a for a in question.answers if a not in (*UNIVERSAL, "unclear")],
        "candidates": question.evidence_label if question.review_targets else None,
    }


def _review_trial(trial: Doc, question: Question) -> Doc:
    for finding in trial["findings"]:
        if finding["check_id"] == REVIEW_CHECK:
            finding["title"] = question.evidence_label
            finding["category"] = "behaviour"
    return trial


def _trial(
    trial_id: str, item: Doc, trace: TraceType | None, report: Doc, review: Question | None
) -> Doc:
    if review is None:
        return trial_bundle(trial_id, item, trace, report)
    trial = trial_bundle(trial_id, blind_item(item, review, trace), trace, report)
    return _review_trial(trial, review)


def bundle(
    records: list[Record],
    items: list[Doc],
    digests: list[str | None],
    report: Doc,
    run: Doc | None = None,
    review: Question | None = None,
) -> Doc:
    """The whole export document. Traces are re-read only when still matching their pin.
    `review`: a blind human-review export for that question (no findings, scores or
    scanner run sections; the question's candidates to jump between)."""
    if not len(records) == len(items) == len(digests):
        raise ValueError("record_item_count_mismatch")
    trials = []
    for number, ((source, _), item, sha256) in enumerate(zip(records, items, digests, strict=True)):
        if source.label != item.get("input_id"):
            raise ValueError("record_item_mismatch")
        trace = None
        if item.get("input_status") == "available":
            trace = load_bytes(pinned_bytes(source.local, sha256))
        trials.append(_trial(str(number), item, trace, report, review))
    document = {
        "format": FORMAT,
        "scanner_version": report.get("scanner_version"),
        "packs": deepcopy(report.get("packs")),
        "run": deepcopy(run),
        "trials": trials,
    }
    if review is not None:
        document["review"] = review_block(review)
        if document["run"]:
            document["run"]["sections"] = []  # the scanner's own run brief: not blind
    return document


def data_script(document: Doc) -> str:
    """ASCII-only JSON assigned to one global; `<` escaped so no markup survives."""
    text = json.dumps(document, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return "window.ATIF_VIEWER = " + text.replace("<", "\\u003c") + ";\n"


def _writable(directory: Path) -> None:
    if directory.is_symlink() or (
        directory.exists() and (not directory.is_dir() or any(directory.iterdir()))
    ):
        raise ValueError("viewer_directory_not_empty")


def write(directory: Path, document: Doc) -> None:
    """A new or empty directory, private until the reviewer publishes it themselves."""
    _writable(directory)
    directory.mkdir(mode=0o700, exist_ok=True)
    static = resources.files("atif_scan.browser").joinpath("viewer")
    files = {name: static.joinpath(name).read_text(encoding="utf-8") for name in ASSETS}
    files[DATA_FILE] = data_script(document)
    for name, text in files.items():
        fd = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
