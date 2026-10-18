"""Pure per-trace accounting; private evidence enters, allowlisted documents leave.

Recorded tool invocations and statically recovered code-mode call sites are distinct
units. Output records count observations once, never once per linked/derived call.
Shared captures are not missing per-command results, but establish no individual
command outcome. Followup negatives describe recorded accesses to resolvable targets;
unindicated or unresolvable retention stays unknown. No output absence or unmatched
marker proves that the model saw complete output.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from harness_followups import find_followups, find_process_followups
from harness_outputs import parse_output
from harness_trace import canonical_tool
from harness_visibility import Visibility

from atif_scan.data.jsonval import as_str, count
from atif_scan.data.tools import normalize_tool

if TYPE_CHECKING:
    from harness_followups import Followup
    from harness_model_calls import CallAudit
    from harness_outputs import OutputEvidence
    from harness_trace import TraceEvidence

    from atif_scan.data.jsonval import Doc
    from atif_scan.data.model import Step, ToolCall

_TOOL = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}")
_PROCESS = re.compile(r"(?m)^process_id: ([A-Za-z0-9_-]{1,100})\s*$")
_POLL = frozenset({"process", "poll_process", "write_stdin", "bash_input", "awaitToolCall"})


@dataclass(frozen=True)
class Metrics:
    trace: Doc
    tools: tuple[Doc, ...]
    outputs: tuple[Doc, ...]
    followups: tuple[Doc, ...]


def tool_label(name: str, family: str | None) -> str | None:
    value = canonical_tool(name, family)
    return value if _TOOL.fullmatch(value) else None


@dataclass(frozen=True)
class _ToolCounts:
    direct: Counter[str]
    derived: Counter[str]
    recorded_unknown_tool_labels: int
    static_unknown_tool_labels: int


def _tool_counts(trace: TraceEvidence, excluded: frozenset[tuple[int, int]]) -> _ToolCounts:
    direct: Counter[str] = Counter()
    derived: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    for step in trace.steps:
        for call in step.calls:
            if (step.index, call.index) in excluded:
                continue
            is_direct = call.result_id is None
            name = call.name if is_direct else call.name.removeprefix("exec>")
            label = tool_label(name, trace.family)
            if label is None:
                unknown["recorded" if is_direct else "static"] += 1
            else:
                (direct if is_direct else derived)[label] += 1
    return _ToolCounts(direct, derived, unknown["recorded"], unknown["static"])


def _tool_rows(
    trace: TraceEvidence, counts: _ToolCounts, declaration_conflict: bool
) -> tuple[Doc, ...]:
    direct, derived = counts.direct, counts.derived
    declared = set(trace.declarations.tools)
    known = trace.declarations.status == "known"
    usage_known = (
        known
        and not trace.dropped_steps
        and "invalid_steps" not in trace.issues
        and "redacted_input" not in trace.issues
        and not declaration_conflict
        and not counts.recorded_unknown_tool_labels
        and not counts.static_unknown_tool_labels
    )
    return tuple(
        {
            "tool": name,
            "recorded_calls": direct[name],
            "static_call_sites": derived[name],
            "declared": name in declared if known else None,
            "inventory_source": trace.declarations.source if name in declared else "observed",
            "offered_unused": (
                direct[name] == derived[name] == 0 if usage_known and name in declared else None
            ),
        }
        for name in sorted(declared | direct.keys() | derived.keys())
    )


def _shell_call(call: ToolCall, family: str | None) -> bool:
    name = canonical_tool(call.name, family)
    if name == "process" and as_str((call.arguments or {}).get("action")) in {"list", "stop"}:
        return False
    # The loader keeps some provider namespaces as "other"; normalize labels here,
    # without reclassifying an inert code-mode parent as a shell invocation.
    namespaced_shell = call.tool == "other" and normalize_tool(name, call.arguments) == "shell"
    return call.tool == "shell" or namespaced_shell or name in _POLL


def _linked(step: Step, index: int) -> list[ToolCall]:
    return [
        call
        for call in step.calls
        if call.result_id is None and any(j == index for j, _ in step.results_for(call))
    ]


def _shared_context(step: Step, linked: list[ToolCall]) -> tuple[str, str, str] | None:
    parents = {c.id for c in linked if c.id}
    if any(c.result_id in parents and c.tool == "shell" for c in step.calls):
        return "exec", "code_mode_shared", "parent_result"
    return None


def _output_context(
    step: Step, index: int, family: str | None, excluded: frozenset[tuple[int, int]]
) -> tuple[str | None, str, str] | None:
    observation = step.observations[index]
    linked = [c for c in _linked(step, index) if (step.index, c.index) not in excluded]
    if observation.shared_terminal:
        return "bash_command", "shared_terminal", "shared_terminal"
    if any(canonical_tool(c.name, family) == "get_command_or_subagent_output" for c in linked):
        return "get_command_or_subagent_output", "command_or_subagent", "recorded_id"
    shell = [c for c in linked if _shell_call(c, family)]
    if shell:
        pairing = "recorded_id" if observation.source_call_id else "lone_call"
        if observation.pairing_method:
            pairing = observation.pairing_method
        return tool_label(shell[0].name, family), "shell", pairing
    return _shared_context(step, linked)


def _requested_limit(step: Step, index: int) -> int | None:
    limits = {
        value
        for call in _linked(step, index)
        if (value := count((call.arguments or {}).get("max_output_tokens"))) is not None
    }
    return next(iter(limits)) if len(limits) == 1 else None


def _process_handle(text: str, step: Step, index: int) -> str | None:
    match = _PROCESS.search(text)
    if match:
        return as_str(match[1])
    handles = {
        value
        for call in _linked(step, index)
        if (value := as_str((call.arguments or {}).get("process_id"))) is not None
    }
    return next(iter(handles)) if len(handles) == 1 else None


def _followups(
    trace: TraceEvidence, step: Step, index: int, evidence: OutputEvidence
) -> tuple[str, tuple[Followup, ...]]:
    if evidence.spool_paths:
        return "file", find_followups(trace.steps, step.index, evidence.spool_paths)
    if evidence.spool_indicated:
        handle = _process_handle(step.observations[index].content.text, step, index)
        if handle:
            return "process_handle", find_process_followups(trace.steps, step.index, handle)
        return "unresolved_retention", ()
    return "not_indicated", ()


def _evidence_doc(evidence: OutputEvidence, understood: bool) -> Doc:
    doc = evidence.to_doc()
    if not understood:
        # Recognized positive markers survive partial decoding; absence is unknown.
        doc["truncation"] = True if evidence.truncation else None
        doc["returned_chars"] = None
        doc["returned_bytes"] = None
    return doc


def _output_row(
    trace: TraceEvidence,
    step: Step,
    index: int,
    context: tuple[str | None, str, str],
    excluded: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[Doc, list[Doc]]:
    observation = step.observations[index]
    text = observation.content.text
    evidence = parse_output(text, trace.family or "unknown")
    retention, accesses = _followups(trace, step, index, evidence)
    accesses = tuple(a for a in accesses if (a.step, a.call) not in excluded)
    tool, kind, pairing = context
    row: Doc = {
        "step": step.index,
        "observation": index,
        "tool": tool,
        "context": kind,
        "pairing": pairing,
        "text_understood": observation.content.understood,
        "has_media": observation.content.media,
        **_evidence_doc(evidence, observation.content.understood),
        **_visibility_doc(trace, step, index),
        "requested_max_output_tokens": _requested_limit(step, index),
        "retention": retention,
        "followup_accesses": len(accesses),
        "followup_observed": bool(accesses) if retention in {"file", "process_handle"} else None,
        "first_followup_delay_calls": min((a.delay_calls for a in accesses), default=None),
        "content_returned_accesses": sum(a.outcome == "content_returned" for a in accesses),
        "partial_read_accesses": sum(a.partial_requested for a in accesses),
    }
    follows = [
        {"origin_step": step.index, "origin_observation": index, **access.to_doc()}
        for access in accesses
    ]
    return row, follows


def _missing_outputs(trace: TraceEvidence, excluded: frozenset[tuple[int, int]]) -> int:
    return sum(
        not step.results_for(call)
        for step in trace.steps
        if not any(o.shared_terminal for o in step.observations)
        for call in step.calls
        if call.result_id is None
        and (step.index, call.index) not in excluded
        and _shell_call(call, trace.family)
    )


def _visibility_doc(trace: TraceEvidence, step: Step, index: int) -> Doc:
    surfaces = next(
        (
            result.visibility
            for result in trace.result_visibility
            if result.step == step.index and result.index == index
        ),
        Visibility(step.observations[index].content, None, "unknown"),
    )
    selected = surfaces.model_visible if surfaces.model_visible is not None else surfaces.recorded
    measured = Visibility(selected, None, "unknown")
    return {
        "returned_chars": measured.recorded_chars,
        "returned_bytes": measured.recorded_bytes,
        "recorded_chars": surfaces.recorded_chars,
        "recorded_bytes": surfaces.recorded_bytes,
        "model_visible_chars": surfaces.model_visible_chars,
        "model_visible_bytes": surfaces.model_visible_bytes,
        "visibility_basis": surfaces.basis,
        "returned_basis": "model_visible" if surfaces.model_visible is not None else "recorded",
    }


def _excluded_observation(step: Step, index: int, excluded: frozenset[tuple[int, int]]) -> bool:
    # Always pair on the original step: removing calls first enables lone-call
    # fallback and can silently reassign a rejected observation to a valid request.
    linked = _linked(step, index)
    return bool(linked) and all((step.index, c.index) in excluded for c in linked)


def _measure_outputs(
    trace: TraceEvidence, excluded: frozenset[tuple[int, int]]
) -> tuple[tuple[Doc, ...], tuple[Doc, ...], int]:
    outputs: list[Doc] = []
    followups: list[Doc] = []
    unclassified = 0
    for step in trace.steps:
        for index in range(len(step.observations)):
            if _excluded_observation(step, index, excluded):
                continue
            context = _output_context(step, index, trace.family, excluded)
            if context is None:
                unclassified += 1
                continue
            row, accesses = _output_row(trace, step, index, context, excluded)
            outputs.append(row)
            followups.extend(accesses)
    return tuple(outputs), tuple(followups), unclassified


def measure(trace: TraceEvidence, call_audit: CallAudit | None = None) -> Metrics:
    """Measure valid harness requests; the optional audit retains model attempts separately."""
    excluded = (
        frozenset((r.attempt.step, r.attempt.index) for r in call_audit.rejected)
        if call_audit is not None
        else frozenset()
    )
    conflict = call_audit is not None and any(
        r.declaration_conflict is True for r in call_audit.rejected
    )
    counts = _tool_counts(trace, excluded)
    tools = _tool_rows(trace, counts, conflict)
    outputs, followups, unclassified = _measure_outputs(trace, excluded)
    doc: Doc = {
        "family": trace.family or "unknown",
        "version": trace.version,
        "signature": trace.declarations.signature,
        "inventory_source": trace.declarations.source,
        "inventory_status": trace.declarations.status,
        "declared_tools": len(trace.declarations.tools)
        if trace.declarations.status == "known"
        else None,
        "recorded_calls": sum(row["recorded_calls"] for row in tools),
        "static_call_sites": sum(row["static_call_sites"] for row in tools),
        "recorded_unknown_tool_labels": counts.recorded_unknown_tool_labels,
        "static_unknown_tool_labels": counts.static_unknown_tool_labels,
        "agent_steps": len(trace.steps),
        "dropped_steps": trace.dropped_steps,
        "shell_calls_missing_results": _missing_outputs(trace, excluded),
        "unclassified_observations": unclassified,
        "shell_output_records": len(outputs),
        "truncated_output_records": sum(row["truncation"] is True for row in outputs),
        "issues": list(trace.issues),
    }
    return Metrics(doc, tools, outputs, followups)
