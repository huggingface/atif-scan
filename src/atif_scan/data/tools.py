"""Tool calls: harness tool names normalised to categories (hints, not proof of what ran),
and argument values classified into channels (command, path, query, URL, payload).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import TypeGuard

from . import jslit
from .content import content
from .jsonval import JsonObject, as_list, as_object, is_object
from .model import Channel, Content

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
TERMINAL_INPUT_KEYS = _keys("textinput", "keystrokes", "chars")
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


def types_into_terminal(arguments: Mapping[str, object] | None) -> bool:
    """The call types text into a terminal (a top-level keystrokes-style argument), so its
    output is whatever the terminal shows next, not a result of its own."""
    return arguments is not None and any(_norm(k) in TERMINAL_INPUT_KEYS for k in arguments)


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
