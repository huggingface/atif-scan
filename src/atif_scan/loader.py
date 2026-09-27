"""ATIF v1 adapter. Reject invalid structure; preserve unknown text representations."""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from . import jslit
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
# Harness-specific names, one table per harness (extend with a regression test).
HARNESS_ALIASES = {
    # Cursor CLI (cursor-cli), seen on the TB2.1 leaderboard.
    "shellToolCall": "shell",
    "readToolCall": "read",
    "editToolCall": "write",
    "writeToolCall": "write",
    "deleteToolCall": "write",
    "grepToolCall": "search_files",
    "globToolCall": "search_files",
    "webFetchToolCall": "web_fetch",
    "webSearchToolCall": "web_search",
    "awaitToolCall": "inert",
    "updateTodosToolCall": "inert",
    # Devin CLI, seen on the TB2.1 leaderboard: text typed into an interactive shell.
    "write_to_process": "shell",
    # Ouroboros, seen on the TB2.1 leaderboard.
    "start_service": "shell",
    "search_code": "search_files",
    "query_code": "search_files",
    "view_image": "read",
    "ocr_pdf": "read",
    "stop_service": "inert",
    "service_logs": "inert",
    "wait_tasks": "inert",
    "verify_and_record": "inert",
    "task_acceptance_review": "inert",
    "schedule_subagent": "inert",
    "skill": "inert",  # Devin CLI skill invocation (generic skills, name + subcommand)
    # Codex CLI.
    "exec_command": "shell",
    "write_stdin": "shell",
    "apply_patch": "write",
    "update_plan": "inert",
    "wait": "inert",
    # Terminus 2.
    "bash_command": "shell",
    "mark_task_complete": "inert",
    # fast-agent.
    "process": "inert",
    "attach_media": "read",
    # Linghun.
    "ReadSnippets": "read",
    "MultiEdit": "write",
    "SourcePack": "search_files",
    "Diff": "read",
    "Todo": "inert",
    "GitStatusInspect": "inert",
    "RunVerification": "inert",
}
ALIASES = {name: category for category, names in TOOLS.items() for name in names}
ALIASES.update(HARNESS_ALIASES)

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
COMMAND_KEYS = _keys(
    "command",
    "cmd",
    "commands",
    "script",
    "shellcommand",
    "bash",
    "code",
    # Text typed into a terminal: Devin text_input, Terminus keystrokes, Codex write_stdin.
    "textinput",
    "keystrokes",
    "chars",
)
QUERY_KEYS = _keys("query", "q", "searchquery", "searchterm", "queries")
URL_KEYS = _keys("url", "urls", "uri", "href", "link", "endpoint")
PATH_KEYS = _keys(
    "targetdirectory",
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
    "bytesinput",
    "input",  # Codex apply_patch
    "globpattern",
    "streamcontent",
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


# Codex CLI records its hosted web tool as one name whose `action_type` says what it did.
ACTION_TOOLS = {
    "web_search_call": {
        "search": "web_search",
        "open_page": "web_fetch",
        "find_in_page": "web_fetch",
    }
}


def program_calls(name: str, args: object) -> list[tuple[str, object]] | None:
    """Tool calls inside a Codex code-mode program (`exec` with a JavaScript `input`), or
    None when this isn't one. Read statically; see `jslit`."""
    if name != "exec" or not isinstance(args, dict) or set(args) - {"input", "timeout_ms"}:
        return None
    program = args.get("input")
    if not isinstance(program, str) or not jslit.CALL.search(program):
        return None
    return jslit.tool_calls(program)


def program_tool(name: str, args: object) -> str:
    # Codex's hosted web tool inside code mode: `{search_query: [{q}]}` or `{open: [...]}`.
    if name == "web__run":
        keys = set(args) if isinstance(args, dict) else set()
        if "search_query" in keys:
            return "web_search"
        return "web_fetch" if keys & {"open", "click", "find"} else "other"
    return normalize_tool(name, args if isinstance(args, dict) else None)


def normalize_tool(name: str, args: object = None) -> str:
    actions = ACTION_TOOLS.get(name)
    if actions is not None:
        action = args.get("action_type") if isinstance(args, dict) else None
        return actions.get(action, "other") if isinstance(action, str) else "other"
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
        if value is jslit.UNREAD:
            # Present in a tool program but not a literal (a variable, `${}`): unknown.
            fields.append((classify(key, "")[0], Content(understood=False)))
        elif isinstance(value, str):
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
    for index, raw in enumerate(value["steps"]):
        # Observations link to calls within their step, so ids must be unique per step.
        # Reuse across steps (Linghun's WriteReport re-dispatching as Write) is reported
        # by integrity.call_id_reused rather than rejecting the trace.
        call_ids: set[str] = set()
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
        program: list[tuple[str, str, str, object]] = []
        for call_index, call in enumerate(raw_calls):
            if not isinstance(call, dict) or not isinstance(call.get("function_name"), str):
                raise TraceError("invalid_call")
            call_id = call.get("tool_call_id")
            if not isinstance(call_id, str):
                raise TraceError("invalid_call_id")
            # An empty id (Codex's hosted web tool) links nothing, so it can't collide.
            if not copied and call_id:
                if call_id in call_ids:
                    raise TraceError("duplicate_call_id")
                call_ids.add(call_id)
            args = call.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = None
            inner = program_calls(call["function_name"], args)
            tool = "inert" if inner is not None else normalize_tool(call["function_name"], args)
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
            program.extend((call_id, call["function_name"], name, a) for name, a in inner or ())
        # Calls read from tool programs follow the recorded calls (positions stay stable).
        for k, (parent, outer, name, inner_args) in enumerate(program):
            tool = program_tool(name, inner_args)
            calls.append(
                ToolCall(
                    len(calls),
                    tool,
                    call_fields(tool, inner_args if isinstance(inner_args, dict) else None),
                    f"{parent}#{k}",
                    f"{outer}>{name}",
                    None,
                    result_id=parent,
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
            if source_call_id == "":
                source_call_id = None
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
                completion_tokens=step_completion_tokens(raw.get("metrics")),
            )
        )
    metrics = value.get("final_metrics")
    extra = metrics.get("extra") if isinstance(metrics, dict) else None
    tokens = extra.get("total_tool_use_tokens") if isinstance(extra, dict) else None
    calls = [
        raw.get("llm_call_count")
        for raw in value["steps"]
        if isinstance(raw, dict) and raw.get("source") == "agent"
    ]
    counted = [c for c in calls if type(c) is int and c >= 0]
    return Trace(
        version,
        tuple(steps),
        tokens if type(tokens) is int else None,
        usage(metrics),
        compacted=tuple(
            s.index
            for s in steps
            if s.source in ("system", "user") and not s.copied and COMPACTED.search(s.message.text)
        ),
        llm_calls=sum(counted) if counted else None,
        agent=agent_info(value.get("agent")),
    )


def step_completion_tokens(metrics: object) -> int | None:
    value = metrics.get("completion_tokens") if isinstance(metrics, dict) else None
    return value if type(value) is int and value >= 0 else None


def agent_info(value: object) -> tuple[str | None, str | None, str | None]:
    """(name, version, model_name) from the ATIF root `agent` block, label-safe only."""
    if not isinstance(value, dict):
        return (None, None, None)

    def label(key: str) -> str | None:
        v = value.get(key)
        return v if isinstance(v, str) and re.fullmatch(r"[\w.:@/+-]{1,100}", v) else None

    return (label("name"), label("version"), label("model_name"))


# Harness notices that earlier conversation history was replaced by a summary.
COMPACTED = re.compile(
    r"session is being continued from a previous conversation|\[COMPACTED HISTORY\]|"
    r"conversation history (?:was|has been) (?:compacted|summari[sz]ed)|"
    r"(?:ran|run) out of context\b[^.\n]{0,80}summary",
    re.I,
)


def usage(metrics: object) -> Usage | None:
    if not isinstance(metrics, dict):
        return None

    def count(key: str) -> int | None:
        value = metrics.get(key)
        return value if type(value) is int and value >= 0 else None

    extra = metrics.get("extra") if isinstance(metrics.get("extra"), dict) else {}
    reasoning = extra.get("total_reasoning_tokens")
    cost = metrics.get("total_cost_usd")
    cost = float(cost) if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0 else None
    found = Usage(
        cost,
        count("total_prompt_tokens"),
        count("total_completion_tokens"),
        count("total_cached_tokens"),
        reasoning if type(reasoning) is int and reasoning >= 0 else None,
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
