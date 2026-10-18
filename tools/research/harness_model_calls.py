"""Offline model-call evidence, separate from harness inventory (stdlib only).

Every nonderived call in the supplied authored steps is an attempt, even without
results or a safe name/model. Rejections require a supported whole error envelope
and an unambiguous loader-established link to that exact call. Missing evidence
is not evidence of availability. Only the first qualifying result per call is
retained, so rejected-call counts share the attempts denominator.

Pass the same raw document used to build TraceEvidence. Raw step model_name is
needed because the loader's label rules are narrower and cannot distinguish an
absent model from an explicitly invalid one. No subagent descent, I/O, execution,
or inventory mutation occurs here; the parent supplies trace IDs and aggregation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from harness_trace import canonical_tool

from atif_scan.data.jsonval import as_list, as_object, as_str, is_object

if TYPE_CHECKING:
    from harness_trace import TraceEvidence

    from atif_scan.data.jsonval import Doc, JsonObject
    from atif_scan.data.model import Step, ToolCall

ModelSource = Literal["step", "root", "unknown"]
RejectionReason = Literal["not_found", "unavailable"]
Pairing = Literal["recorded_id", "position", "unique_remainder", "lone_call"]
_TOOL = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+?=&-]{0,199}")
# Quoted names are labels in the envelope, not quoted mentions in arbitrary output.
_NAME = r"(?:'[^'\r\n]{1,256}'|\"[^\"\r\n]{1,256}\"|[^\s:'\"<>]{1,256})"
_AVAILABLE = r"[A-Za-z][A-Za-z0-9_.-]{0,127}"
_NO_SUCH = re.compile(
    r"<tool_use_error>Error: No such tool available: (?P<name>[^<>\r\n]{1,256})"
    r"</tool_use_error>(?:\n\n\[error\] tool reported failure)?"
)
_UNAVAILABLE = re.compile(
    rf"Tool (?P<name>{_NAME}) is not available\."
    rf"(?: Available tools: {_AVAILABLE}(?:, {_AVAILABLE})*\.)?"
)
_NOT_FOUND = re.compile(
    rf"(?:Error: )?(?:Tool (?P<name>{_NAME}) not found\.?|Unknown tool: "
    rf"(?P<unknown>{_NAME}))"
)
_EXECUTING = re.compile(rf"Error executing tool (?P<name>{_NAME}): (?P<error>.+)")
MAX_ERROR_CHARS = 4096


@dataclass(frozen=True, repr=False)
class Attempt:
    step: int
    index: int
    tool_label: str | None
    model: str | None
    model_source: ModelSource

    def to_doc(self) -> Doc:
        """Allowlisted locator and labels; never call IDs, arguments or text."""
        return {
            "step": self.step,
            "call": self.index,
            "tool": self.tool_label,
            "model": self.model,
            "model_source": self.model_source,
        }


@dataclass(frozen=True, repr=False)
class RejectedCall:
    attempt: Attempt
    observation: int
    reason: RejectionReason
    # True means a declared tool was unavailable at this call (dynamic availability).
    # False is only a known inventory nonmembership, not proof of nonexistence.
    declaration_conflict: bool | None
    pairing: Pairing

    def to_doc(self) -> Doc:
        """Model-level evidence only: no harness field or parent trace ID."""
        return {
            **self.attempt.to_doc(),
            "observation": self.observation,
            "reason": self.reason,
            "declaration_conflict": self.declaration_conflict,
            "pairing": self.pairing,
        }


@dataclass(frozen=True, repr=False)
class CallAudit:
    attempts: tuple[Attempt, ...]
    rejected: tuple[RejectedCall, ...]

    def to_doc(self) -> Doc:
        return {
            "attempts": [attempt.to_doc() for attempt in self.attempts],
            "rejected": [rejected.to_doc() for rejected in self.rejected],
        }


def _model_label(value: object) -> str | None:
    text = as_str(value)
    return text if text and _MODEL.fullmatch(text) and "://" not in text else None


def _model_for(step: Step, raw_steps: list[object], root: object) -> tuple[str | None, ModelSource]:
    if not 0 <= step.index < len(raw_steps) or not is_object(raw_steps[step.index]):
        return None, "unknown"
    raw_step = as_object(raw_steps[step.index])
    source: ModelSource = "step" if "model_name" in raw_step else "root"
    model = _model_label(raw_step.get("model_name") if source == "step" else root)
    return model, source if model is not None else "unknown"


def _tool_label(name: str, family: str | None) -> str | None:
    canonical = canonical_tool(name, family)
    return canonical if _TOOL.fullmatch(canonical) else None


def _unquote(name: str) -> str:
    return name[1:-1] if name[:1] in {"'", '"'} and name[-1:] == name[:1] else name


def _simple_error(text: str) -> tuple[str, RejectionReason] | None:
    for pattern in (_NO_SUCH, _UNAVAILABLE):
        if match := pattern.fullmatch(text):
            return _unquote(match["name"]), "unavailable"
    if match := _NOT_FOUND.fullmatch(text):
        return _unquote(match["name"] or match["unknown"]), "not_found"
    return None


def _error_reason(text: str, name: str, family: str | None) -> RejectionReason | None:
    if len(text) > MAX_ERROR_CHARS or not name:
        return None
    text = text.strip()
    if match := _EXECUTING.fullmatch(text):
        if canonical_tool(_unquote(match["name"]), family) != canonical_tool(name, family):
            return None
        text = as_str(match["error"]) or ""
    error = _simple_error(text)
    if error and canonical_tool(error[0], family) == canonical_tool(name, family):
        return error[1]
    return None


def _owners(step: Step) -> dict[int, list[int]]:
    owners: dict[int, list[int]] = {}
    for call in step.calls:
        if call.result_id is None:
            for index, _ in step.results_for(call):
                owners.setdefault(index, []).append(call.index)
    return owners


def _rejection(
    trace: TraceEvidence,
    step: Step,
    call: ToolCall,
    attempt: Attempt,
    owners: dict[int, list[int]],
) -> RejectedCall | None:
    for index, observation in step.results_for(call):
        if (
            owners.get(index) != [call.index]
            or observation.pairing_unresolved
            or observation.shared_terminal
            or not observation.content.understood
        ):
            continue
        reason = _error_reason(observation.content.text, call.name, trace.family)
        if reason is not None:
            declared = (
                canonical_tool(call.name, trace.family) in trace.declarations.tools
                if trace.declarations.status == "known"
                else None
            )
            pairing: Pairing = observation.pairing_method or (
                "recorded_id" if observation.source_call_id else "lone_call"
            )
            return RejectedCall(attempt, index, reason, declared, pairing)
    return None


def evaluate_calls(trace: TraceEvidence, raw_doc: JsonObject) -> CallAudit:
    """Classify recorded attempts and explicit dispatch rejections, without I/O."""
    raw_steps = as_list(raw_doc.get("steps"))
    root_model = as_object(raw_doc.get("agent")).get("model_name")
    attempts: list[Attempt] = []
    rejected: list[RejectedCall] = []
    for step in trace.steps:
        if not step.authored:
            continue
        model, source = _model_for(step, raw_steps, root_model)
        owners = _owners(step)
        for call in step.calls:
            if call.result_id is not None:
                continue
            attempt = Attempt(
                step.index, call.index, _tool_label(call.name, trace.family), model, source
            )
            attempts.append(attempt)
            rejection = _rejection(trace, step, call, attempt, owners)
            if rejection is not None:
                rejected.append(rejection)
    return CallAudit(tuple(attempts), tuple(rejected))
