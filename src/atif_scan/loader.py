"""ATIF v1 adapter. Reject invalid structure; preserve unknown text representations."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from .model import Channel, Content, Observation, Step, ToolCall, Trace

MAX_BYTES = 64 * 1024 * 1024


class TraceError(ValueError):
    """Messages contain fixed error codes only, never input snippets or filenames."""


def content(value: object) -> Content:
    if value is None:
        return Content()
    if isinstance(value, str):
        return Content(value)
    if isinstance(value, list):
        parts = [content(part) for part in value]
        return Content("\n".join(p.text for p in parts), all(p.understood for p in parts))
    if isinstance(value, dict):
        # Only text-bearing blocks, never arbitrary metadata/URL/image payloads.
        if value.get("type") in ("image", "image_url", "audio", "video"):
            return Content()
        if isinstance(value.get("text"), str):
            return Content(value["text"])
    return Content(understood=False)


# Tool categories observed in real agent exports. Extend with a regression test per alias.
TOOLS = {
    "shell": {
        "bash",
        "Bash",
        "shell",
        "Shell",
        "exec",
        "execute",
        "execute_command",
        "run_command",
        "run_script",
        "run_shell_command",
        "terminal",
    },
    "read": {"read_text_file", "read_file", "read", "Read", "view_file"},
    "write": {
        "write_text_file",
        "write_file",
        "write",
        "Write",
        "edit",
        "Edit",
        "edit_text",
        "edit_file",
        "str_replace",
    },
    "search_files": {"grep", "Grep", "glob", "Glob", "find_file_by_name", "list_files"},
    "web_fetch": {"web_fetch", "fetch_url", "webfetch", "WebFetch", "fetch"},
    "web_search": {"web_search", "web_search_preview", "WebSearch"},
    # Orchestration tools whose arguments carry no command/path/URL evidence.
    "inert": {"todo_write", "TodoWrite", "get_output", "kill_shell", "service_status"},
}
ALIASES = {name: category for category, names in TOOLS.items() for name in names}

# category -> (channel, accepted argument keys, required)
FIELDS = {
    "shell": ((Channel.COMMAND, ("command", "cmd", "script"), True),),
    "read": ((Channel.PATH, ("path", "file_path"), True),),
    "write": ((Channel.PATH, ("path", "file_path"), True),),
    "search_files": ((Channel.PATH, ("path",), False),),
    "web_fetch": ((Channel.URL, ("url",), True),),
    "web_search": ((Channel.QUERY, ("query",), True),),
}


def normalize_tool(name: str) -> str:
    return ALIASES.get(name, "other")


def strings(value: object) -> list[str]:
    """String leaves of an argument structure, in order; keys are not evidence."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return []


def argv(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, str) for v in value)


def call_fields(tool: str, args: object) -> tuple[tuple[Channel, Content], ...]:
    if tool == "other":
        # Unrecognized tool: expose raw argument text only on the ARGUMENTS channel.
        if not isinstance(args, dict):
            return ((Channel.ARGUMENTS, Content(understood=False)),)
        return ((Channel.ARGUMENTS, Content("\n".join(strings(args)))),)
    fields = []
    for channel, keys, required in FIELDS.get(tool, ()):
        value = next((args[k] for k in keys if isinstance(args, dict) and k in args), None)
        if channel == Channel.COMMAND and argv(value):
            value = " ".join(value)  # argv form, e.g. ["bash", "-lc", "..."]
        if isinstance(value, str):
            fields.append((channel, content(value)))
        elif required or value is not None or not isinstance(args, dict):
            fields.append((channel, Content(understood=False)))
    return tuple(fields)


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
        if any(not isinstance(k, str) for k in value):
            raise TraceError("invalid_argument_keys")
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TraceError("invalid_argument_value")


def parse_trace(value: object) -> Trace:
    if not isinstance(value, dict) or not isinstance(value.get("steps"), list):
        raise TraceError("invalid_steps")
    version = value.get("schema_version")
    if version is not None and (
        not isinstance(version, str) or re.fullmatch(r"ATIF-v1\.\d+(?:\.\d+)?", version) is None
    ):
        raise TraceError("unsupported_schema_version")
    steps = []
    call_ids: set[str] = set()
    for index, raw in enumerate(value["steps"]):
        if not isinstance(raw, dict) or raw.get("source") not in ("system", "user", "agent"):
            raise TraceError("invalid_step")
        copied = raw.get("is_copied_context", False)
        if not isinstance(copied, bool):
            raise TraceError("invalid_copied_context")
        raw_calls = raw.get("tool_calls")
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            raise TraceError("invalid_calls")
        calls = []
        for call_index, call in enumerate(raw_calls):
            if not isinstance(call, dict) or not isinstance(call.get("function_name"), str):
                raise TraceError("invalid_call")
            call_id = call.get("tool_call_id")
            if not isinstance(call_id, str):
                raise TraceError("invalid_call_id")
            if not copied:
                if call_id in call_ids:
                    raise TraceError("duplicate_call_id")
                call_ids.add(call_id)
            args = call.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = None
            tool = normalize_tool(call["function_name"])
            fields = call_fields(tool, args)
            calls.append(
                ToolCall(
                    call_index,
                    tool,
                    fields,
                    call_id,
                    call["function_name"],
                    freeze(args) if isinstance(args, dict) else None,
                )
            )
        observation = raw.get("observation")
        if observation is None:
            observation = {}
        if not isinstance(observation, dict) or not isinstance(
            observation.get("results", []), list
        ):
            raise TraceError("invalid_observation")
        observations = []
        for item in observation.get("results", []):
            if not isinstance(item, dict):
                raise TraceError("invalid_observation_result")
            source_call_id = item.get("source_call_id")
            if source_call_id is not None and not isinstance(source_call_id, str):
                raise TraceError("invalid_observation_call_id")
            observations.append(Observation(source_call_id, content(item.get("content"))))
        step_id = raw.get("step_id")
        steps.append(
            Step(
                index,
                raw["source"],
                copied,
                content(raw.get("message")),
                content(raw.get("reasoning_content")),
                tuple(calls),
                tuple(observations),
                step_id=step_id if type(step_id) is int else None,
                step_id_recorded="step_id" in raw,
                timestamp=timestamp(raw.get("timestamp")),
                timestamp_recorded=raw.get("timestamp") is not None,
                agent_only_fields=raw["source"] != "agent"
                and any(raw.get(k) for k in ("tool_calls", "reasoning_content", "metrics")),
            )
        )
    metrics = value.get("final_metrics")
    extra = metrics.get("extra") if isinstance(metrics, dict) else None
    tokens = extra.get("total_tool_use_tokens") if isinstance(extra, dict) else None
    return Trace(version, tuple(steps), tokens if type(tokens) is int else None)


def load_bytes(data: bytes) -> Trace:
    """Parse raw trace bytes (from any source) under the same size cap."""
    if len(data) > MAX_BYTES:
        raise TraceError("trace_too_large")
    try:
        return parse_trace(json.loads(data.decode("utf-8")))
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
