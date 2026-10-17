"""How a trace's agent loop ended: with the agent's reply, or some other way.

A run that stops without the agent concluding its turn (a failed model call, an output
limit hit while reasoning, an empty final reply, a safety stop, a time limit while
tools ran) is evidence something unexpected happened. It is not a recording defect
(the steps are there) and not misconduct. Facts here are codes and counts only.

The recorded stop reason (fast-agent `extra.stop_reason`) decides when there is one.
Without it only the shape of the final step can say anything: tool calls left without
a result, or no reply and no call at all. A final reply with no stop reason is
undetermined, never "completed".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Step, Trace


class Ending(StrEnum):
    COMPLETED = "completed"  # the turn ended with the agent's reply
    MODEL_CALL_FAILED = "model_call_failed"  # the last model call errored
    OUTPUT_LIMIT = "output_limit"  # the last response hit its output-token limit
    EMPTY_REPLY = "empty_reply"  # the turn ended with no reply and no tool call
    SAFETY_STOP = "safety_stop"  # the provider stopped the response (safety/refusal)
    INTERRUPTED = "interrupted"  # cancelled, timed out or paused by the client
    STOPPED_MID_LOOP = "stopped_mid_loop"  # the run stopped with tool calls outstanding
    UNDETERMINED = "undetermined"  # nothing recorded says how it ended


# Recorded stop-reason codes (lowercased, `LlmStopReason.` stripped), by meaning.
COMPLETE_CODES = frozenset({"end_turn", "stop", "stop_sequence", "end", "complete", "completed"})
LIMIT_CODES = frozenset({"max_tokens", "length", "max_output_tokens"})
SAFETY_CODES = frozenset({"safety", "content_filter", "refusal"})
ERROR_CODES = frozenset({"error"})
INTERRUPT_CODES = frozenset({"cancelled", "canceled", "timeout", "pause", "pause_turn"})
TOOL_CODES = frozenset({"tool_use", "tool_calls", "function_call"})


@dataclass(frozen=True)
class EndingFacts:
    kind: Ending
    step: int  # index of the last agent step
    # "stop_reason" when a recorded stop reason decided it, else "structure".
    basis: str
    stop_reason: str | None = None
    reply: bool = False
    calls: int = 0
    # Reasoning text but no reply and no call: everything the response held was thinking.
    reasoning_only: bool = False
    # The last agent step recorded its own token usage (a cut-off stream often doesn't).
    usage_recorded: bool = False
    # The HTTP status of the final call's failed attempts, when one was recorded.
    http_status: int | None = None
    # Inline media in the conversation before the last agent step (what its request
    # carried): occurrences and decoded bytes of identified payloads.
    media_items: int = 0
    media_bytes: int = 0

    @property
    def unexpected(self) -> bool:
        return self.kind not in (Ending.COMPLETED, Ending.UNDETERMINED)


def _last_agent_step(trace: Trace) -> Step | None:
    """The step the recorded stop reason belongs to (`loader.final_stop_reason`)."""
    return next((s for s in reversed(trace.steps) if s.source == "agent"), None)


def _calls_unanswered(step: Step) -> bool:
    """The step's calls have no result recorded at all: the run stopped before they
    returned. Any recorded result (even one shared by a batch of terminal keystrokes, as
    Terminus 2 records its `mark_task_complete` turn) leaves it undecided."""
    return bool(step.calls) and not step.observations


# A stop reason that alone says how the loop ended, whatever the final step holds.
ENDED_BY_CODE = (
    (ERROR_CODES, Ending.MODEL_CALL_FAILED),
    (SAFETY_CODES, Ending.SAFETY_STOP),
    (LIMIT_CODES, Ending.OUTPUT_LIMIT),
    (INTERRUPT_CODES, Ending.INTERRUPTED),
    (TOOL_CODES, Ending.STOPPED_MID_LOOP),
)


def _from_stop_reason(trace: Trace, code: str | None, step: Step, reply: bool) -> Ending | None:
    """None when no recorded stop reason (or harness termination error) decides it."""
    if trace.termination_error is not None:
        return Ending.MODEL_CALL_FAILED
    decided = next((kind for codes, kind in ENDED_BY_CODE if code in codes), None)
    if decided is None and code in COMPLETE_CODES:
        # A completed turn that still issued calls stopped before they were answered.
        decided = Ending.STOPPED_MID_LOOP if step.calls else None
        decided = decided or (Ending.COMPLETED if reply else Ending.EMPTY_REPLY)
    return decided


def _from_structure(step: Step, reply: bool) -> Ending:
    """No usable stop reason: only what the final step's shape shows."""
    if step.calls:
        return Ending.STOPPED_MID_LOOP if _calls_unanswered(step) else Ending.UNDETERMINED
    return Ending.UNDETERMINED if reply else Ending.EMPTY_REPLY


def _media_before(trace: Trace, end: int) -> tuple[int, int]:
    items = size = 0
    for step in trace.steps[:end]:
        for content in (step.message, *(o.content for o in step.observations)):
            items += len(content.media_ids)
            size += content.media_bytes
    return items, size


def ending(trace: Trace) -> EndingFacts | None:
    """How the agent loop ended; None when no agent step of this session was recorded."""
    step = _last_agent_step(trace)
    if step is None or step.copied:  # copied context ended in another session
        return None
    reply = bool(step.message.text.strip()) or step.message.media
    code = trace.final_stop_reason
    decided = _from_stop_reason(trace, code, step, reply)
    kind = decided if decided is not None else _from_structure(step, reply)
    failed = kind is Ending.MODEL_CALL_FAILED
    items, size = _media_before(trace, step.index) if failed else (0, 0)
    # The failed call's own status, else its retried attempts' (before HTTP 413s
    # stopped being retried, fast-agent recorded the status only there).
    status = trace.final_error_status or (
        trace.final_retry_statuses[-1] if trace.final_retry_statuses else None
    )
    return EndingFacts(
        kind,
        step.index,
        "stop_reason" if decided is not None else "structure",
        stop_reason=code,
        reply=reply,
        calls=len(step.calls),
        reasoning_only=not reply and not step.calls and bool(step.reasoning.text.strip()),
        usage_recorded=step.completion_tokens is not None,
        http_status=status if failed else None,
        media_items=items,
        media_bytes=size,
    )
