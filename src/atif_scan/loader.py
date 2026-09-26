"""ATIF v1 adapter. Reject invalid structure; preserve unknown text representations."""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from .model import Channel, Content, Observation, Step, ToolCall, Trace, Usage

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
    # ToolSearch is Claude Code's deferred-tool registry lookup, not a web search.
    "inert": {
        "todo_write",
        "TodoWrite",
        "get_output",
        "kill_shell",
        "service_status",
        "ToolSearch",
    },
}
ALIASES = {name: category for category, names in TOOLS.items() for name in names}

# Tool categories are *hints*: they name what a tool does (so e.g. only a known web-search
# tool counts as a web search) and which evidence must be present. Evidence extraction
# itself is generic and applies to every call; see `classify`.
REQUIRED = {
    "shell": Channel.COMMAND,
    "read": Channel.PATH,
    "write": Channel.PATH,
    "web_fetch": Channel.URL,
    "web_search": Channel.QUERY,
}


def _keys(*names: str) -> frozenset[str]:
    return frozenset(names)


# Argument key conventions (compared lowercased, without `_`/`-`), not tool names.
COMMAND_KEYS = _keys("command", "cmd", "commands", "script", "shellcommand", "bash", "code")
QUERY_KEYS = _keys("query", "q", "searchquery", "searchterm", "queries")
URL_KEYS = _keys("url", "urls", "uri", "href", "link", "endpoint")
PATH_KEYS = _keys(
    "path",
    "paths",
    "filepath",
    "file",
    "files",
    "filename",
    "dir",
    "directory",
    "cwd",
    "workdir",
    "workingdirectory",
    "source",
    "src",
    "target",
    "destination",
    "dest",
    "output",
    "outputpath",
    "root",
    "notebookpath",
)
PAYLOAD_KEYS = _keys(
    "content",
    "contents",
    "text",
    "body",
    "data",
    "newstring",
    "newstr",
    "newtext",
    "insertline",
    "patch",
    "diff",
    "edits",
    "description",
    "prompt",
    "message",
    "messages",
    "instructions",
    "note",
    "notes",
    "todos",
    "summary",
    "reasoning",
    "thought",
    "explanation",
    "title",
    "pattern",
    "regex",
    "glob",
    "include",
    "exclude",
)
# Existing text an edit quotes (not authored by the agent): not evidence of anything.
QUOTED_KEYS = _keys("oldstring", "oldstr", "oldtext", "original", "search", "find")
URL_VALUE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://\S+")
PATH_VALUE = re.compile(r"(?:/|~/|\./|\.\./|[A-Za-z]:[\\/])[^\n]{0,4095}")


def normalize_tool(name: str) -> str:
    return ALIASES.get(name, "other")


def _norm(key: str | None) -> str:
    return re.sub(r"[_\-\s]", "", key.lower()) if key else ""


def leaves(value: object, key: str | None = None):
    """(nearest key, value) for every non-dict/list leaf; list items inherit their key."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from leaves(v, k)
    elif isinstance(value, list):
        if _norm(key) in COMMAND_KEYS:
            # argv form, e.g. ["bash", "-lc", "..."]; mixed lists are not understood.
            yield key, " ".join(value) if argv(value) else value
            return
        for v in value:
            yield from leaves(v, key)
    else:
        yield key, value


def argv(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, str) for v in value)


def classify(key: str | None, value: str) -> tuple[Channel, str]:
    """Route one argument string by key convention and value shape (see design.md)."""
    k = _norm(key)
    if k in PAYLOAD_KEYS:
        return Channel.PAYLOAD, value
    if value.startswith("file://"):
        return Channel.PATH, value[len("file://") :]
    if URL_VALUE.fullmatch(value):
        return Channel.URL, value
    if k in COMMAND_KEYS:
        return Channel.COMMAND, value
    if k in QUERY_KEYS:
        return Channel.QUERY, value
    if k in URL_KEYS:
        return Channel.URL, value
    if k in PATH_KEYS or PATH_VALUE.fullmatch(value):
        return Channel.PATH, value
    return Channel.ARGUMENTS, value


def call_fields(tool: str, args: object) -> tuple[tuple[Channel, Content], ...]:
    if tool == "inert":
        return ()
    if not isinstance(args, dict):
        # Unparseable arguments could hold anything: every tool-input check is unknown.
        channel = REQUIRED.get(tool, Channel.ARGUMENTS)
        return ((channel, Content(understood=False)),)
    fields = []
    for key, value in leaves(args):
        if _norm(key) in QUOTED_KEYS:
            continue
        if isinstance(value, str):
            channel, text = classify(key, value)
            fields.append((channel, content(text)))
        elif isinstance(value, list) and _norm(key) in COMMAND_KEYS:
            fields.append((Channel.COMMAND, Content(understood=False)))  # e.g. ["bash", 1]
    required = REQUIRED.get(tool)
    if required is not None and all(channel != required for channel, _ in fields):
        fields.append((required, Content(understood=False)))
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
    return Trace(version, tuple(steps), tokens if type(tokens) is int else None, usage(metrics))


def usage(metrics: object) -> Usage | None:
    if not isinstance(metrics, dict):
        return None

    def count(key: str) -> int | None:
        value = metrics.get(key)
        return value if type(value) is int and value >= 0 else None

    cost = metrics.get("total_cost_usd")
    cost = float(cost) if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0 else None
    found = Usage(
        cost,
        count("total_prompt_tokens"),
        count("total_completion_tokens"),
        count("total_cached_tokens"),
    )
    return None if found == Usage() else found


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
