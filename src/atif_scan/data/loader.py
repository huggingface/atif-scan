"""ATIF v1 adapter. Reject invalid structure; preserve unknown text representations."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeGuard

from . import jslit
from .jsonval import JsonObject, as_list, as_object, as_str, compact_json, count, is_object, number
from .loader_accounting import root_usage
from .model import Channel, Content, Observation, Step, ToolCall, Trace, Usage
from .web_inputs import parse_web_input

if TYPE_CHECKING:
    from pathlib import Path

# TB4 leaderboard traces reach ~192 MiB (legacy-utility-triage: ~900 base64 screenshots).
# That one loads in ~1 s and peaks at ~790 MiB RSS, so 256 MiB covers every TB4 trace with
# headroom; larger files are rejected (reported, never cleared).
MAX_BYTES = 256 * 1024 * 1024


class TraceError(ValueError):
    """Messages contain fixed error codes only, never input snippets or filenames."""


# Harness text standing in for an attached image (Devin CLI: "[Image 1]").
# Padding is same-line whitespace only: `^\s*` would span blank lines and rescan every
# run of them from each line start (quadratic); the matches are the same.
MEDIA_PLACEHOLDER = re.compile(
    r"^[^\S\n]*\[(?:Image|Screenshot|Attachment)\s*#?\d*\][^\S\n]*$", re.I | re.M
)
# Media serialized into a string: Codex `view_image` results (a repr of
# `[{'type': 'input_image', 'image_url': 'data:image/png;base64,…'}]`), a data URI with a
# real payload (the smallest PNG is ~90 base64 characters), Claude Code's image-size
# note for a Read image, and PDF/image blocks written as JSON (TB2.1). A short data URI
# is text, not an image: sanitizer tests and HTML the agent writes (`<img
# src="data:image/png;base64,abc">`, TB4 html-js-filter) would otherwise make the field
# unreadable and every unprompted-recall check unknown from that step on.
MEDIA_INLINE = re.compile(
    r"\bdata:(?:image|audio|video)/[\w.+-]+;base64,[A-Za-z0-9+/]{64}|"
    r"""["']image_url["']\s*:\s*["']data:(?:image|audio|video)/|"""
    r"^\s*\[Image: original \d+x\d+, displayed at \d+x\d+\.|"
    r"""["']media_type["']\s*:\s*["'](?:image/|audio/|video/|application/pdf)""",
    re.I | re.M,
)


def content(value: object) -> Content:
    if value is None:
        return Content()
    if isinstance(value, str):
        media = bool(MEDIA_PLACEHOLDER.search(value) or MEDIA_INLINE.search(value[:4096]))
        return Content(value, media=media)
    if isinstance(value, list):
        parts = [content(part) for part in as_list(value)]
        return Content(
            "\n".join(p.text for p in parts),
            all(p.understood for p in parts),
            any(p.media for p in parts),
        )
    return _block_content(value)


def _block_content(value: object) -> Content:
    """One content block. Only text-bearing blocks, never arbitrary metadata/URL/image
    payloads."""
    block = as_object(value)
    if block.get("type") in ("image", "image_url", "audio", "video"):
        return Content(media=True)
    text = as_str(block.get("text"))
    return Content(text) if text is not None else Content(understood=False)


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
    # Claude Code task tools.
    "TaskCreate": "inert",
    "TaskUpdate": "inert",
    "TaskStop": "inert",
    # LemonCrow (Claude Code MCP tools), seen on the TB2.1 leaderboard.
    "mcp__lc__bash": "shell",
    "mcp__lc__read": "read",
    "mcp__lc__edit": "write",
    "mcp__lc__code_search": "search_files",
    # Surf harness (nano-grok-build), seen on the TB2.1 leaderboard.
    "run_terminal_command": "shell",
    "search_replace": "write",
    "list_dir": "search_files",
    "get_terminal_command_output": "inert",
    "kill_terminal_command": "inert",
    # AiWork.Coder (dtcoder).
    "background_exec": "shell",
    "plan": "inert",
    "read_tool_result_page": "inert",
    # Gemini CLI.
    "replace": "write",
    "grep_search": "search_files",
    "list_directory": "search_files",
    "update_topic": "inert",
    "google_web_search": "web_search",
    "read_background_output": "inert",
    "list_background_processes": "inert",
    # indusagi.
    "ls": "search_files",
    "find": "search_files",
    # Dext.
    "rg": "search_files",
    "fd": "search_files",
    "read_symbol": "read",
    "multi_edit": "write",
    "http": "web_fetch",
    "git_diff": "read",
    "git_log": "read",
    # File inspection with arguments rather than a path (a `read` needs a path).
    "awk": "search_files",
    "jq": "search_files",
    "csvkit": "search_files",
    "todo_read": "inert",
    # Mobile Coder.
    "pdf_parse": "read",
    "todowrite": "inert",
    "invalid": "inert",
    # WorkHarness.
    "tool_search": "inert",
    "image_to_text": "read",
    "task_output": "inert",
    "task_get": "inert",
    "task_stop": "inert",
    "sleep": "inert",
    # OrcaTerm: completion reports (prose summaries, not written files).
    "goal_complete": "inert",
    "goal_blocked": "inert",
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


# Non-JSON ("freeform") tools take raw text rather than an object: Responses custom
# tools, Codex apply_patch, fast-agent's raw-command shell. Exporters record that text as
# the whole `arguments` string or as a lone `input` field. It is routed by the tool's
# kind, not by the field name; kinds without a natural channel keep it as a payload.
RAW_INPUT_KEY = "input"
RAW_INPUT_CHANNEL = {
    "shell": Channel.COMMAND,
    "read": Channel.PATH,
    "write": Channel.PAYLOAD,
    "web_fetch": Channel.URL,
    "web_search": Channel.QUERY,
}


def raw_input(args: object) -> str | None:
    """The raw text of a non-JSON tool call, or None for structured arguments."""
    if isinstance(args, str):
        return args
    if isinstance(args, dict) and set(args) == {RAW_INPUT_KEY}:
        value = args[RAW_INPUT_KEY]
        return value if isinstance(value, str) else None
    return None


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
    if name in ("web__run", "web.run"):
        keys = set(args) if isinstance(args, dict) else set()
        if "search_query" in keys:
            return "web_search"
        return "web_fetch" if keys & {"open", "click", "find"} else "other"
    return normalize_tool(name, args if isinstance(args, dict) else None)


def normalize_tool(name: str, args: object = None) -> str:
    if name in ("web__run", "web.run"):
        return program_tool(name, args)
    if name == "write_stdin" and isinstance(args, Mapping) and not args.get("chars"):
        return "inert"  # Codex polling a running session: nothing was typed
    if name == "background_exec" and isinstance(args, Mapping) and not args.get("command"):
        return "inert"  # dtcoder waiting on/stopping/listing a background task: nothing ran
    return _normalize_action(name, args)


def _normalize_action(name: str, args: object) -> str:
    actions = ACTION_TOOLS.get(name)
    if actions is not None:
        action = args.get("action_type") if isinstance(args, dict) else None
        return actions.get(action, "other") if isinstance(action, str) else "other"
    return ALIASES.get(name, "other")


def _norm(key: str | None) -> str:
    return re.sub(r"[_\-\s]", "", key.lower()) if key else ""


def leaves(value: object, key: str | None = None) -> Iterator[tuple[str | None, object]]:
    """(nearest key, value) for every non-dict/list leaf; list items inherit their key."""
    if isinstance(value, dict):
        for k, v in as_object(value).items():
            yield from leaves(v, k)
    elif isinstance(value, list):
        value = as_list(value)
        if _norm(key) in COMMAND_KEYS:
            # argv form, e.g. ["bash", "-lc", "..."]; mixed lists are not understood.
            yield key, " ".join(value) if argv(value) else value
            return
        for v in value:
            yield from leaves(v, key)
    else:
        yield key, value


def argv(value: object) -> TypeGuard[list[str]]:
    return isinstance(value, list) and bool(value) and all(isinstance(v, str) for v in value)


# Key conventions checked after the value shape, in order (see `classify`).
KEY_CHANNELS = (
    (COMMAND_KEYS, Channel.COMMAND),
    (QUERY_KEYS, Channel.QUERY),
    (URL_KEYS, Channel.URL),
    (PATH_KEYS, Channel.PATH),
)


def classify(key: str | None, value: str) -> tuple[Channel, str]:
    """Route one argument string by key convention and value shape (see design.md)."""
    k = _norm(key)
    if k in PAYLOAD_KEYS:
        return Channel.PAYLOAD, value
    if value.startswith("file://"):
        return Channel.PATH, value[len("file://") :]
    return _value_channel(k, value), value


def _value_channel(k: str, value: str) -> Channel:
    if URL_VALUE.fullmatch(value):
        return Channel.URL
    by_key = next((channel for keys, channel in KEY_CHANNELS if k in keys), None)
    if by_key is not None:
        return by_key
    return Channel.PATH if PATH_VALUE.fullmatch(value) else Channel.ARGUMENTS


# Codex apply_patch envelope: the file paths live inside the patch text.
PATCH_ENVELOPE = re.compile(r"^\*\*\* Begin Patch\b", re.M)
# Greedy to the line end, then rstrip(): a lazy `(\S.*?)\s*$` retries `\s*$` at every
# character (quadratic on long lines).
PATCH_PATHS = re.compile(r"^\*\*\* (?:Add File|Update File|Delete File|Move to): *(\S.*)$", re.M)


Field = tuple[Channel, Content]


def call_fields(tool: str, args: object) -> tuple[Field, ...]:
    if tool == "inert":
        return ()
    text = raw_input(args)
    # Only a recognized kind says what raw text means: an unrecognized tool's raw text
    # could be a command or a URL, so it stays unknown (string) or keyed (lone `input`).
    if text is not None and tool in RAW_INPUT_CHANNEL:
        return tuple(_text_fields(RAW_INPUT_CHANNEL[tool], text))
    if not is_object(args):
        # Unparseable arguments could hold anything: every tool-input check is unknown.
        channel = REQUIRED.get(tool, Channel.ARGUMENTS)
        return ((channel, Content(understood=False)),)
    return _argument_fields(tool, args)


def _argument_fields(tool: str, args: JsonObject) -> tuple[Field, ...]:
    fields = [
        found
        for key, value in leaves(args)
        if _norm(key) not in QUOTED_KEYS
        for found in _leaf_fields(key, value)
    ]
    required = REQUIRED.get(tool)
    if required is not None and all(channel != required for channel, _ in fields):
        fields.append((required, Content(understood=False)))
    return tuple(fields)


def _leaf_fields(key: str | None, value: object) -> list[Field]:
    if value is jslit.UNREAD:
        # Present in a tool program but not a literal (a variable, `${}`): unknown.
        return [(classify(key, "")[0], Content(understood=False))]
    if isinstance(value, str):
        return _text_fields(*classify(key, value))
    if isinstance(value, list) and _norm(key) in COMMAND_KEYS:
        return [(Channel.COMMAND, Content(understood=False))]  # e.g. ["bash", 1]
    return []


def _text_fields(channel: Channel, text: str) -> list[Field]:
    """The text's field, plus the file paths inside an apply_patch payload."""
    fields = [(channel, content(text))]
    if channel == Channel.PAYLOAD and PATCH_ENVELOPE.search(text):
        fields.extend((Channel.PATH, content(m.rstrip())) for m in PATCH_PATHS.findall(text))
    return fields


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


# The harness refused the call's input before running it (arguments that weren't valid
# JSON, required parameters missing), so nothing ran: Claude Code, WorkHarness, OrcaTerm.
INPUT_REJECTED = re.compile(
    r"^\s*(?:<tool_use_error>\s*InputValidationError\b|Invalid input for [\w.-]+:|"
    r"Validation failed for tool \"?[\w.-]+\"?:)"
)


def _drop_rejected_unknowns(
    calls: list[ToolCall], observations: list[Observation]
) -> list[ToolCall]:
    """A rejected call's unreadable input can't hide an action: drop its unreadable fields
    (readable ones, e.g. an unparsed command's text, are still scanned)."""
    rejected = {
        o.source_call_id
        for o in observations
        if o.source_call_id and INPUT_REJECTED.match(o.content.text)
    }
    return [
        replace(c, fields=tuple(f for f in c.fields if f[1].understood))
        if c.result_key in rejected
        else c
        for c in calls
    ]


# A lone call's results need no pairing by order.
MIN_PAIRED_CALLS = 2


def _positional_links_valid(calls: list[ToolCall], observations: list[Observation]) -> bool:
    ids = [c.id for c in calls]
    return (
        len(calls) == len(observations)
        and all(ids)
        and len(set(ids)) == len(ids)
        and all(o.source_call_id in (None, c.id) for c, o in zip(calls, observations, strict=True))
    )


def _unique_remainder(calls: list[ToolCall], observations: list[Observation]) -> ToolCall | None:
    """Only valid, unambiguous explicit IDs can eliminate recorded calls.

    Multipart results may eliminate the same call repeatedly. Foreign IDs or duplicate
    nonempty recorded IDs block inference altogether; empty IDs eliminate nothing.
    """
    ids = [c.id for c in calls if c.id]
    if len(set(ids)) != len(ids):
        return None
    explicit = {o.source_call_id for o in observations if o.source_call_id is not None}
    if not explicit <= set(ids):
        return None
    if sum(o.source_call_id is None for o in observations) != 1:
        return None
    remaining = [c for c in calls if not c.id or c.id not in explicit]
    return remaining[0] if len(remaining) == 1 else None


def _reconstruct_observation_links(
    calls: list[ToolCall], observations: list[Observation]
) -> list[Observation]:
    """Infer links with warnings, without altering explicit links or source data.

    Preserve valid positional reconstruction first. Otherwise a sole unlinked result
    and sole unmatched recorded call can be associated by index, regardless of order.
    This intentionally resolves formerly contradictory positional examples when their
    explicit links prove a unique remainder. Derived code-mode calls are excluded.
    """
    if len(calls) < MIN_PAIRED_CALLS or not any(o.source_call_id is None for o in observations):
        return observations
    if _positional_links_valid(calls, observations):
        return [
            replace(
                o,
                source_call_id=c.id,
                source_call_index=c.index,
                pairing_method="position",
                pairing_reconstructed=True,
            )
            if o.source_call_id is None
            else o
            for c, o in zip(calls, observations, strict=True)
        ]
    remainder = _unique_remainder(calls, observations)
    return [
        replace(
            o,
            source_call_index=remainder.index,
            pairing_method="unique_remainder",
            pairing_reconstructed=True,
        )
        if remainder is not None and o.source_call_id is None
        else replace(o, pairing_unresolved=True)
        if o.source_call_id is None
        else o
        for o in observations
    ]


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
    observations = _reconstruct_observation_links(calls[: len(raw_calls)], observations)
    calls = _drop_rejected_unknowns(calls, observations)
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
                call_fields(tool, args),
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
    return ToolCall(
        index,
        tool,
        call_fields(tool, args if is_object(args) else None),
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


def step_usage(raw_steps: list[object]) -> tuple[Usage | None, int, tuple[str, ...]]:
    """(summed per-step token usage, LLM calls without usage, token kinds only some
    metered steps record). Steps record `metrics.{prompt,completion,cached}_tokens`; the
    sum is the fallback when final_metrics has no totals (e.g. the harness died before
    writing them). A step without metrics leaves all its calls (`llm_call_count`, else 1)
    unmetered; a step with metrics but several calls is read as metering one of them (a
    retry's usage is often lost). A kind is complete only when every metered step records
    it: recorded on some steps it's a lower bound (a partial kind), on none it's unknown
    (None), never 0. None when no agent step recorded usage."""
    totals = [0, 0, 0]
    seen = missing = 0
    recorded = [0, 0, 0]  # metered steps recording each kind
    for raw in raw_steps:
        if not is_object(raw) or raw.get("source") != "agent":
            continue
        metrics = as_object(raw.get("metrics"))
        calls = count(raw.get("llm_call_count"))
        values = [count(metrics.get(k)) for k in STEP_TOKEN_KINDS]
        if values[0] is None and values[1] is None:
            missing += 1 if calls is None else calls
            continue
        seen += 1
        missing += max((calls or 1) - 1, 0)
        totals = [t + (v or 0) for t, v in zip(totals, values, strict=True)]
        recorded = [r + (v is not None) for r, v in zip(recorded, values, strict=True)]
    if not seen:
        return None, missing, ()
    kinds = [t if r else None for t, r in zip(totals, recorded, strict=True)]
    partial = tuple(
        name for name, r in zip(TOKEN_KIND_NAMES, recorded, strict=True) if 0 < r < seen
    )
    return Usage(None, *kinds), missing, partial


def last_call(raw_steps: list[object]) -> tuple[Usage | None, int, bool]:
    """(the last metered agent step's own usage, agent steps with usage, whether every
    metered step's prompt and completion tokens are both at least the previous step's).
    A step is metered as in step_usage: prompt or completion tokens recorded."""
    metered: list[list[int | None]] = []
    for raw in raw_steps:
        if not is_object(raw) or raw.get("source") != "agent":
            continue
        metrics = as_object(raw.get("metrics"))
        values = [count(metrics.get(k)) for k in STEP_TOKEN_KINDS]
        if values[0] is not None or values[1] is not None:
            metered.append(values)
    if not metered:
        return None, 0, False
    rising = all((b[n] or 0) >= (a[n] or 0) for a, b in pairwise(metered) for n in (0, 1))
    return Usage(None, *metered[-1]), len(metered), rising


FAST_AGENT_RETRY_SCHEMA = "fast-agent.retry/v1"


def stream_retries(raw_steps: list[object], final_metrics: object) -> tuple[int, int, bool | None]:
    """Count explicit fast-agent.retry/v1 provider retry markers and read the
    harness's usage-completeness flag. A marker does not establish whether a failure
    was mid-stream, whether usage was received, or whether output entered context.
    Legacy field names retain "stream" for report compatibility."""
    steps = attempts = 0
    for raw in raw_steps:
        if not is_object(raw) or raw.get("source") != "agent":
            continue
        retry = as_object(as_object(raw.get("extra")).get("retry"))
        tries = count(retry.get("provider_attempts"))
        if as_str(retry.get("schema")) == FAST_AGENT_RETRY_SCHEMA and tries and tries > 1:
            steps += 1
            attempts += tries - 1
    complete = as_object(as_object(final_metrics).get("extra")).get("llm_usage_calls_complete")
    return steps, attempts, complete if isinstance(complete, bool) else None


# A provider refusal recorded by the client: "Error code: 503 - {...}" (OpenAI/Anthropic
# SDKs). Only the status number is kept, never the message.
HTTP_STATUS = re.compile(r"\AError code: ([1-5]\d\d)\b")
HTTP_ERROR_MIN = 400


def _retry_status(retry: Mapping[str, object]) -> int | None:
    match = HTTP_STATUS.match(as_str(retry.get("error_message")) or "")
    return int(match.group(1)) if match else None


def _before_output(retry: Mapping[str, object]) -> bool:
    """The failed attempt returned nothing: no stream event arrived, or the provider
    refused the request with an HTTP error status before opening a stream. A count of
    events (even one) or an unknown count with no status is not this."""
    events = retry.get("stream_events_received")
    status = _retry_status(retry)
    return events == 0 or (events is None and status is not None and status >= HTTP_ERROR_MIN)


def retry_outcomes(raw_steps: list[object]) -> tuple[int, tuple[int, ...]]:
    """(fast-agent retried attempts that failed before any output, the HTTP statuses
    of all failed attempts that recorded one, sorted and unique)."""
    before, statuses = 0, set[int]()
    for raw in raw_steps:
        if not is_object(raw) or raw.get("source") != "agent":
            continue
        retry = as_object(as_object(raw.get("extra")).get("retry"))
        if as_str(retry.get("schema")) != FAST_AGENT_RETRY_SCHEMA:
            continue
        for attempt in as_list(retry.get("retries")):
            record = as_object(attempt)
            before += _before_output(record)
            if (status := _retry_status(record)) is not None:
                statuses.add(status)
    return before, tuple(sorted(statuses))


# How the report names each step token kind (usage fields: input, output, cache).
TOKEN_KIND_NAMES = ("input", "output", "cached")
STEP_TOKEN_KINDS = ("prompt_tokens", "completion_tokens", "cached_tokens")


def step_completion_tokens(metrics: object) -> int | None:
    return count(as_object(metrics).get("completion_tokens"))


def step_reasoning(metrics: object) -> int | None:
    """Do not invent zero or subtract an invalid split from completion tokens."""
    raw = as_object(metrics)
    completion = count(raw.get("completion_tokens"))
    reasoning = count(as_object(raw.get("extra")).get("reasoning_tokens"))
    if completion is None or reasoning is None or reasoning > completion:
        return None
    return reasoning


def model_label(v: object) -> str | None:
    """A label-safe agent/model name, else None (e.g. Claude Code's `<synthetic>`
    placeholder, or a name carrying query parameters, stays unknown)."""
    return v if isinstance(v, str) and re.fullmatch(r"[\w.:@/+-]{1,100}", v) else None


def agent_info(value: object) -> tuple[str | None, str | None, str | None]:
    """(name, version, model_name) from the ATIF root `agent` block, label-safe only."""
    agent = as_object(value)
    return (
        model_label(agent.get("name")),
        model_label(agent.get("version")),
        model_label(agent.get("model_name")),
    )


# Harness notices that earlier conversation history was replaced by a summary.
COMPACTED = re.compile(
    r"session is being continued from a previous conversation|\[COMPACTED HISTORY\]|"
    # Devin CLI: "You are continuing work from a previous conversation thread. Below is a
    # summary of the previous conversation thread".
    r"continuing work from a previous conversation thread|"
    r"summary of the previous conversation thread|"
    r"conversation history (?:was|has been) (?:compacted|summari[sz]ed)|"
    r"(?:ran|run) out of context\b[^.\n]{0,80}summary",
    re.I,
)


def _reasoning_tokens(extra: JsonObject) -> int | None:
    """Prefer the canonical field, including zero; Codex uses a historical alias.

    An absent/null canonical value permits fallback. A malformed non-null value
    remains unknown rather than being silently repaired from another field.
    """
    value = extra.get("total_reasoning_tokens")
    return count(extra.get("reasoning_output_tokens") if value is None else value)


def usage(
    metrics: object,
    agent: tuple[str | None, str | None, str | None] = (None, None, None),
) -> Usage | None:
    if not is_object(metrics):
        return None
    extra = as_object(metrics.get("extra"))
    found = Usage(
        number(metrics.get("total_cost_usd"), low=0),
        count(metrics.get("total_prompt_tokens")),
        count(metrics.get("total_completion_tokens")),
        count(metrics.get("total_cached_tokens")),
        _reasoning_tokens(extra),
        count(extra.get("total_cache_creation_input_tokens")),
    )
    if found == Usage():
        return None
    # Grok Build exports visible output and reasoning separately. This is a
    # harness convention, not a model-name or harness-version heuristic.
    if agent[0] == "grok-build":
        return replace(found, completion_basis="separate_visible")
    return found


# A JSON string (skipped as a whole) or a bare `[REDACTED]` token outside any string.
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
