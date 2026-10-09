"""ATIF v1 adapter. Reject invalid structure; preserve unknown text representations.

Parsing and loading; what a call, its content, its pairing and its usage mean lives in
`tools`, `content`, `pairing` and `usage`.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from . import jslit
from .content import content
from .jsonval import JsonObject, as_list, as_object, as_str, compact_json, count, is_object
from .model import Observation, Step, ToolCall, Trace
from .pairing import drop_rejected_unknowns, reconstruct_observation_links
from .tools import (
    RAW_INPUT_KEY,
    call_fields,
    computed_fields,
    normalize_tool,
    program_calls,
    program_fields,
    program_tool,
    raw_input,
)
from .usage import (
    COMPACTED,
    agent_info,
    last_call,
    model_label,
    retry_outcomes,
    root_usage,
    step_completion_tokens,
    step_reasoning,
    step_usage,
    stream_retries,
    usage,
)
from .web_inputs import parse_web_input

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

# TB4 leaderboard traces reach ~192 MiB (legacy-utility-triage: ~900 base64 screenshots).
# That one loads in ~1 s and peaks at ~790 MiB RSS, so 256 MiB covers every TB4 trace with
# headroom; larger files are rejected (reported, never cleared).
MAX_BYTES = 256 * 1024 * 1024


class TraceError(ValueError):
    """Messages contain fixed error codes only, never input snippets or filenames."""


def timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def freeze(value: object) -> object:
    if isinstance(value, dict):
        return freeze_object(value)
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TraceError("invalid_argument_value")


def freeze_object(value: object) -> Mapping[str, object]:
    """A read-only copy of a dict with string keys (a caller's own dict may have others)."""
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise TraceError("invalid_argument_keys")
    return MappingProxyType({k: freeze(v) for k, v in value.items()})


SCHEMA_VERSION = re.compile(r"ATIF-v1\.\d+(?:\.\d+)?")
STEP_SOURCES = ("system", "user", "agent")


def parse_trace(value: object) -> Trace:
    if not is_object(value) or not isinstance(value.get("steps"), list):
        raise TraceError("invalid_steps")
    version = value.get("schema_version")
    if version is not None and (
        not isinstance(version, str) or SCHEMA_VERSION.fullmatch(version) is None
    ):
        raise TraceError("unsupported_schema_version")
    raw_steps = as_list(value["steps"])
    steps = [parse_step(index, raw) for index, raw in enumerate(raw_steps)]
    metrics = value.get("final_metrics")
    tokens = as_object(as_object(metrics).get("extra")).get("total_tool_use_tokens")
    per_step, unmetered, partial_kinds = step_usage(raw_steps)
    retry_steps, retry_attempts, calls_complete = stream_retries(raw_steps, metrics)
    before_output, retry_statuses = retry_outcomes(raw_steps)
    last, metered, rising = last_call(raw_steps)
    return Trace(
        version,
        tuple(steps),
        tokens if type(tokens) is int else None,
        usage(metrics, agent_info(value.get("agent"))),
        compacted=tuple(
            s.index
            for s in steps
            if s.source in ("system", "user") and not s.copied and COMPACTED.search(s.message.text)
        ),
        context_compactions=context_compactions(raw_steps),
        termination_error=termination_error(value),
        final_stop_reason=final_stop_reason(raw_steps),
        llm_calls=_llm_calls(raw_steps),
        agent=agent_info(value.get("agent")),
        step_usage=per_step,
        calls_without_usage=unmetered if per_step is not None else 0,
        step_kinds_partial=partial_kinds,
        last_step_usage=last,
        metered_steps=metered,
        step_usage_rising=rising,
        stream_retry_steps=retry_steps,
        stream_retry_attempts=retry_attempts,
        retry_before_output=before_output,
        retry_statuses=retry_statuses,
        usage_calls_complete=calls_complete,
        root_usage=root_usage(value),
    )


def final_stop_reason(raw_steps: list[object]) -> str | None:
    """The last agent step's `extra.stop_reason` as a code ("LlmStopReason.SAFETY" ->
    "safety", "end_turn" -> "end_turn"); None when absent or not code-like."""
    for raw in reversed(raw_steps):
        step = as_object(raw)
        if step.get("source") != "agent":
            continue
        reason = as_str(as_object(step.get("extra")).get("stop_reason"))
        code = reason.rsplit(".", 1)[-1].lower() if reason else None
        return code if code and re.fullmatch(r"[a-z][a-z0-9_]{0,40}", code) else None
    return None


def termination_error(value: JsonObject) -> str | None:
    """The trajectory's own record of how it ended in error (fast-agent `extra.termination`):
    an error type code only, never its message."""
    termination = as_object(as_object(value.get("extra")).get("termination"))
    kind = as_str(termination.get("error_type"))
    if as_str(termination.get("status")) != "error" or kind is None:
        return None
    return kind if re.fullmatch(r"[A-Za-z_][\w.]{0,80}", kind) else None


def context_compactions(raw_steps: list[object]) -> tuple[int, ...]:
    """Steps recording a harness context compaction (`extra.context_management.type`)."""
    return tuple(
        index
        for index, raw in enumerate(raw_steps)
        if as_object(as_object(as_object(raw).get("extra")).get("context_management")).get("type")
        == "compaction"
    )


def _llm_calls(raw_steps: list[object]) -> int | None:
    """Summed agent-step `llm_call_count`, or None when no step recorded one."""
    counted = [
        n
        for raw in raw_steps
        if is_object(raw) and raw.get("source") == "agent"
        if (n := count(raw.get("llm_call_count"))) is not None
    ]
    return sum(counted) if counted else None


def parse_step(index: int, value: object) -> Step:
    raw = as_object(value)
    source = as_str(raw.get("source"))
    if source is None or source not in STEP_SOURCES:
        raise TraceError("invalid_step")
    copied = raw.get("is_copied_context", False)
    if not isinstance(copied, bool):
        raise TraceError("invalid_copied_context")
    raw_calls = raw.get("tool_calls")
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, list):
        raise TraceError("invalid_calls")
    calls = parse_calls(
        as_list(raw_calls), copied, as_object(raw.get("extra")).get("tool_call_details")
    )
    observations = parse_observations(raw.get("observation"))
    observations = reconstruct_observation_links(calls[: len(raw_calls)], observations)
    calls = drop_rejected_unknowns(calls, observations)
    step_id = raw.get("step_id")
    return Step(
        index,
        source,
        copied,
        content(raw.get("message")),
        content(raw.get("reasoning_content")),
        tuple(calls),
        tuple(observations),
        step_id=step_id if type(step_id) is int else None,
        step_id_recorded="step_id" in raw,
        timestamp=timestamp(raw.get("timestamp")),
        timestamp_recorded=raw.get("timestamp") is not None,
        agent_only_fields=source != "agent"
        and any(raw.get(k) for k in ("tool_calls", "reasoning_content", "metrics")),
        completion_tokens=step_completion_tokens(raw.get("metrics")),
        reasoning_tokens=step_reasoning(raw.get("metrics")),
        model_name=model_label(raw.get("model_name")),
    )


# A call read from a tool program: (program call id, program tool name, name, arguments).
ProgramCall = tuple[str, str, str, object]


def parse_calls(raw_calls: list[object], copied: bool, details: object = None) -> list[ToolCall]:
    """A step's recorded calls, then the calls read from their tool programs (so recorded
    positions stay stable)."""
    # Observations link to calls within their step, so ids must be unique per step.
    # Reuse across steps (Linghun's WriteReport re-dispatching as Write) is reported
    # by integrity.call_id_reused rather than rejecting the trace.
    call_ids: set[str] = set()
    calls: list[ToolCall] = []
    program: list[ProgramCall] = []
    for call_index, raw in enumerate(raw_calls):
        name, call_id, args = _call_parts(raw)
        # An empty id (Codex's hosted web tool) links nothing, so it can't collide.
        if not copied and call_id:
            if call_id in call_ids:
                raise TraceError("duplicate_call_id")
            call_ids.add(call_id)
        inner = program_calls(name, args)
        tool = "inert" if inner is not None else normalize_tool(name, args)
        frozen = freeze_object(args) if is_object(args) else None
        chars, basis = argument_accounting(
            as_object(raw).get("arguments"), args, as_object(details).get(call_id)
        )
        calls.append(
            ToolCall(
                call_index,
                tool,
                program_fields(args) if inner is not None else call_fields(tool, args),
                call_id,
                name,
                frozen,
                status=call_status(as_object(details).get(call_id)),
                raw_argument_chars=chars,
                raw_argument_basis=basis,
            )
        )
        program.extend((call_id, name, inner_name, a) for inner_name, a in inner or ())
    recorded = len(calls)
    calls.extend(_program_call(recorded + k, k, found) for k, found in enumerate(program))
    return calls


def call_status(
    detail: object,
) -> Literal["incomplete", "completed", "in_progress", "failed"] | None:
    """Only documented status codes; never export arbitrary provider metadata."""
    value = as_object(detail).get("status")
    if value in ("incomplete", "completed", "in_progress", "failed"):
        if value == "incomplete":
            return "incomplete"
        if value == "completed":
            return "completed"
        return "in_progress" if value == "in_progress" else "failed"
    return None


def argument_accounting(
    recorded: object, args: object, detail: object
) -> tuple[int | None, Literal["recorded_json", "raw_text", "compact_json"] | None]:
    """Count the authored payload, not its envelope or duplicate exporter metadata."""
    metadata = as_object(detail)
    raw = as_str(metadata.get("raw_arguments"))
    kind = metadata.get("item_type")
    if raw is not None and kind == "function_call" and matching_json(raw, args):
        return len(raw), "recorded_json"
    if raw is not None and kind == "custom_tool_call" and raw_input(args) == raw:
        return len(raw), "raw_text"
    # A recorded JSON argument string retains whitespace and escaping exactly.
    if isinstance(recorded, str) and matching_json(recorded, args):
        return len(recorded), "recorded_json"
    text = raw_input(args)
    compact = compact_json(args) if is_object(args) else None
    fallback = (len(compact), "compact_json") if compact is not None else (None, None)
    return (len(text), "raw_text") if text is not None else fallback


def matching_json(raw: str, args: object) -> bool:
    """Require an object with identical JSON types/values (True is not 1)."""
    try:
        parsed: object = json.loads(raw)
    except (ValueError, RecursionError):
        return False
    normalized = compact_json(args) if is_object(args) else None
    return normalized is not None and is_object(parsed) and compact_json(parsed) == normalized


def _call_parts(value: object) -> tuple[str, str, object]:
    """(function name, call id, decoded arguments) of one recorded call."""
    call = as_object(value)
    name = as_str(call.get("function_name"))
    if name is None:
        raise TraceError("invalid_call")
    call_id = as_str(call.get("tool_call_id"))
    if call_id is None:
        raise TraceError("invalid_call_id")
    return name, call_id, decode_arguments(call.get("arguments"))


def decode_arguments(args: object) -> object:
    """Recorded arguments; a string is decoded when it holds a JSON object."""
    if not isinstance(args, str):
        return args
    try:
        decoded = json.loads(args)
    except ValueError:
        # Text that starts like JSON but doesn't parse is corrupted structured
        # arguments (unknown); other text is a non-JSON tool's raw input.
        decoded = None if args.lstrip().startswith(("{", "[")) else args
    # Objects are structured and strings raw text; lists/numbers are not
    # understood (argv as the whole arguments is not a recorded convention).
    return decoded if isinstance(decoded, (dict, str)) else None


def _program_call(index: int, k: int, found: ProgramCall) -> ToolCall:
    parent, outer, name, args = found
    if isinstance(args, str):
        args = {RAW_INPUT_KEY: args}  # `tools.apply_patch("*** Begin Patch…")`
    tool = program_tool(name, args)
    if args is jslit.UNREAD:
        fields = computed_fields(tool)
    else:
        fields = call_fields(tool, args if is_object(args) else None)
    return ToolCall(
        index,
        tool,
        fields,
        f"{parent}#{k}",
        f"{outer}>{name}",
        None,
        result_id=parent,
        web_input=parse_web_input(name, args),
    )


def parse_observations(value: object) -> list[Observation]:
    observation = {} if value is None else value
    if not is_object(observation) or not isinstance(observation.get("results", []), list):
        raise TraceError("invalid_observation")
    return [_observation(item) for item in as_list(observation.get("results", []))]


def _observation(value: object) -> Observation:
    if not is_object(value):
        raise TraceError("invalid_observation_result")
    source_call_id = value.get("source_call_id")
    if source_call_id == "":
        source_call_id = None
    if source_call_id is not None and not isinstance(source_call_id, str):
        raise TraceError("invalid_observation_call_id")
    return Observation(source_call_id, content(value.get("content")))


_STRING_OR_REDACTED = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|\[REDACTED\]')


def _unredact(text: str) -> tuple[str, int]:
    """Some published trajectories replace values (token counts) with a bare, unquoted
    `[REDACTED]`, which is invalid JSON (TB4 on the Harbor Hub). Read those values as null
    (unknown) and count them; redacted text inside strings is left as it is."""
    found = 0

    def value(m: re.Match[str]) -> str:
        nonlocal found
        if m[0][0] != "[":
            return m[0]
        found += 1
        return "null"

    return _STRING_OR_REDACTED.sub(value, text), found


def load_bytes(data: bytes) -> Trace:
    """Parse raw trace bytes (from any source) under the same size cap."""
    if len(data) > MAX_BYTES:
        raise TraceError("trace_too_large")
    try:
        text = data.decode("utf-8")
        redacted = 0
        try:
            value = json.loads(text)
        except ValueError:
            if "[REDACTED]" not in text:
                raise
            text, redacted = _unredact(text)
            value = json.loads(text)
        trace = parse_trace(value)
        return replace(trace, redacted_values=redacted) if redacted else trace
    except TraceError:
        raise
    except (UnicodeError, ValueError, RecursionError):
        raise TraceError("unreadable_trace") from None


def load_trace(path: Path) -> Trace:
    try:
        if path.stat().st_size > MAX_BYTES:
            raise TraceError("trace_too_large")
        data = path.read_bytes()
    except OSError:
        raise TraceError("unreadable_trace") from None
    return load_bytes(data)
