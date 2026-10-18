"""Offline, same-trace spool access correlation; requests are not proof of success.

Only authored steps strictly after ``after_step`` participate: a producing step's
output is available at its end. Pass steps from ONE benchmark trace, never a corpus
or subagent concatenation. Paths are lexical POSIX identities, not filesystem facts;
relative paths require a literal absolute call cwd/workdir. Shell cwd changes are
not reconstructed. Pure writes/edits are intentionally unsupported.

``delay_calls`` is the one-based count of recorded, authored calls in later steps
through the linked call (inclusive). Derived static call sites are not observed
accesses and do not add delay. Each call produces at most one row per access kind.
No cache is added here; shared parsers have bounded 512-entry caches and command
inputs here are capped. No indicated file is ever opened or command executed.
"""

from __future__ import annotations

import ast
import json
import posixpath
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from atif_scan.data import shell
from atif_scan.data.file_requests import command_paths
from atif_scan.data.jsonval import as_list, as_object, as_str
from atif_scan.data.model import Channel
from atif_scan.data.pysource import parse_module
from atif_scan.data.python_file_requests import PYTHON, python_paths

if TYPE_CHECKING:
    from collections.abc import Iterator

    from atif_scan.data.jsonval import Doc
    from atif_scan.data.model import Observation, Step, ToolCall

AccessKind = Literal["content_read", "metadata_or_other_access"]
Outcome = Literal["content_returned", "explicit_failure", "unknown"]
MAX_COMMAND_CHARS = 256 * 1024
MAX_RESULT_CHARS = 1024 * 1024
MAX_ERROR_DEPTH = 8
TEXT_READERS = frozenset({"cat", "head", "tail", "sed", "grep", "rg"})
PARTIAL_READERS = TEXT_READERS - {"cat"}
METADATA = frozenset({"ls", "stat", "find", "cd", "cp", "wc", "md5sum", "sha256sum", "cmp"})
READ_TOOLS = frozenset({"read_text_file", "read_file", "read", "Read", "view_file", "readToolCall"})
CONTENT_SEARCH_TOOLS = frozenset({"grep", "Grep", "grepToolCall", "search_code"})
METADATA_TOOLS = frozenset({"glob", "Glob", "globToolCall", "list_files", "find_file_by_name"})
PATH_KEYS = frozenset({"path", "file", "file_path", "filepath", "filename", "paths", "files"})
RANGE_KEYS = frozenset(
    {"line", "limit", "offset", "start_line", "end_line", "line_start", "line_end"}
)
SEARCH_VALUES = frozenset(
    {
        "-A",
        "-B",
        "-C",
        "-m",
        "-g",
        "-t",
        "-T",
        "--glob",
        "--type",
        "--type-not",
        "--max-count",
        "--max-columns",
        "--encoding",
        "--context",
        "--after-context",
        "--before-context",
        "--ignore-file",
        "--threads",
        "--color",
        "--colors",
        "--include",
        "--exclude",
        "--exclude-dir",
        "--iglob",
        "--directories",
        "--devices",
        "--binary-files",
        "--label",
        "--type-add",
        "--type-clear",
        "--pre",
        "--pre-glob",
        "--sort",
        "--sortr",
        "--path-separator",
        "--replace",
        "-r",
        "-d",
        "-D",
    }
)
RESULT_HEADERS = re.compile(
    r"(?m)^(?:Chunk ID: [^\n]*|Wall time[^\n]*|Final output:|Output:|"
    r"process_id: [^\n]*)\n?"
)
EXIT = re.compile(r"(?m)^(?:Process exited with code|process exit code was)\s+(-?\d+)\s*$")
ERROR = re.compile(
    r"(?im)^(?:error\b|tool_use_error\b|<tool_use_error>|"
    r"(?:cat|head|tail|sed|grep|rg|python\d*|read_text_file):[^\n]*"
    r"(?:no such file|permission denied|cannot open|can't open))"
)
STATUS_TEXT = frozenset({"ok", "success", "done", "completed", "null", "none", "true", "false"})


@dataclass(frozen=True)
class Followup:
    """Allowlisted locator and evidence classification, never paths or excerpts."""

    step: int
    call: int
    tool: str
    access_kind: AccessKind
    partial_requested: bool
    outcome: Outcome
    delay_calls: int

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 0 for v in (self.step, self.call, self.delay_calls)):
            raise ValueError("invalid_followup_index")
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", self.tool) is None:
            raise ValueError("invalid_followup_tool")
        if self.access_kind not in {"content_read", "metadata_or_other_access"}:
            raise ValueError("invalid_access_kind")
        if self.outcome not in {"content_returned", "explicit_failure", "unknown"}:
            raise ValueError("invalid_followup_outcome")
        if type(self.partial_requested) is not bool:
            raise ValueError("invalid_partial_requested")

    def to_doc(self) -> Doc:
        return {
            "step": self.step,
            "call": self.call,
            "tool": self.tool,
            "access_kind": self.access_kind,
            "partial_requested": self.partial_requested,
            "outcome": self.outcome,
            "delay_calls": self.delay_calls,
        }


@dataclass(frozen=True)
class _Request:
    path: str
    kind: AccessKind = "content_read"
    partial: bool = False


def _tool(call: ToolCall) -> str:
    name = call.name
    while name.startswith(("functions.", "tools.")):
        name = name.split(".", 1)[1]
    return name if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name) else "other"


def _path(path: str, cwd: str | None = None) -> str | None:
    if not path or any(c in path for c in ("\0", "$", "`", "*", "?", "{", "~")):
        return None
    if not path.startswith("/"):
        if cwd is None:
            return None
        path = posixpath.join(cwd, path)
    # POSIX permits a special meaning for //; retain it rather than guessing.
    return posixpath.normpath(path)


def _cwd(call: ToolCall) -> str | None:
    args = call.arguments or {}
    values = [as_str(args[k]) for k in ("cwd", "workdir") if k in args]
    paths = {_path(v) if v is not None else None for v in values}
    return paths.pop() if len(paths) == 1 and None not in paths else None


def _later_calls(steps: tuple[Step, ...], after_step: int) -> Iterator[tuple[Step, ToolCall, int]]:
    delay = 0
    for step in steps:
        if step.index <= after_step or not step.authored:
            continue
        for call in step.calls:
            if call.result_id is not None:
                # Static code-mode references do not establish an executed request.
                continue
            delay += 1
            yield step, call, delay


def _search_paths(command: shell.Command) -> list[str]:
    """Do not mistake patterns or option values for requested files."""
    argv = command.argv()
    values = shell.operands(argv, SEARCH_VALUES | {"-e", "--regexp", "-f", "--file"})
    scripted = any(
        w.split("=", 1)[0] in {"-e", "--regexp", "-f", "--file"}
        or (w.startswith(("-e", "-f")) and not w.startswith("--"))
        for w, _ in argv[1:]
    )
    if command.name == "sed":
        scripted = any(
            w.split("=", 1)[0] in shell.SED_SCRIPT
            or (w.startswith(("-e", "-f")) and not w.startswith("--"))
            for w, _ in argv[1:]
        )
        values = shell.operands(argv, SEARCH_VALUES | shell.SED_SCRIPT)
    files = [w for w, _ in (values if scripted else values[1:])]
    for i, (word, _) in enumerate(argv):
        if word in {"-f", "--file"} and i + 1 < len(argv):
            files.append(argv[i + 1][0])
        elif word.startswith("--file="):
            files.append(word.split("=", 1)[1])
        elif word.startswith("-f") and word != "-f":
            files.append(word[2:])
    return files


def _extra_paths(command: shell.Command) -> list[str]:
    if command.name in {"grep", "rg", "sed"}:
        if command.name == "rg" and any(w == "--files" for w, _ in command.argv()):
            return [w for w, _ in shell.operands(command.argv(), SEARCH_VALUES)]
        return _search_paths(command)
    if command.name == "stat":
        return [w for w, _ in shell.operands(command.argv(), frozenset({"-c", "--format"}))]
    return []


def _pathlib_target(node: ast.Call, constructors: set[str]) -> str | None:
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in {
        "read_text",
        "read_bytes",
    }:
        return None
    owner = node.func.value
    if not isinstance(owner, ast.Call) or ast.unparse(owner.func) not in constructors:
        return None
    if len(owner.args) != 1 or owner.keywords:
        return None
    arg = owner.args[0]
    return arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None


def _pathlib_paths(code: str) -> list[str]:
    try:
        tree = parse_module(code)
    except (SyntaxError, ValueError, RecursionError):
        return []
    constructors: set[str] = set()
    paths: list[str] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "pathlib":
            constructors.update(a.asname or a.name for a in stmt.names if a.name == "Path")
        elif isinstance(stmt, ast.Import):
            constructors.update(
                (a.asname or a.name) + ".Path" for a in stmt.names if a.name == "pathlib"
            )
        elif isinstance(stmt, ast.Expr):
            deferred = (
                ast.Lambda,
                ast.GeneratorExp,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.BoolOp,
                ast.IfExp,
                ast.NamedExpr,
            )
            if any(isinstance(n, deferred) for n in ast.walk(stmt.value)):
                constructors.clear()
                continue
            paths.extend(
                p
                for n in ast.walk(stmt.value)
                if isinstance(n, ast.Call) and (p := _pathlib_target(n, constructors)) is not None
            )
        else:
            constructors.clear()  # unsupported rebindings/control flow stay unknown
    return paths


def _python_extra(command: shell.Command) -> list[str]:
    if not PYTHON.fullmatch(command.name or ""):
        return []
    argv = [w for w, _ in command.argv()]
    if "-c" not in argv or argv.index("-c") + 1 >= len(argv):
        return []
    return _pathlib_paths(argv[argv.index("-c") + 1])


def _python_bodies(text: str, script: shell.Script, command: shell.Command) -> list[str]:
    if not PYTHON.fullmatch(command.name or ""):
        return []
    paths: list[str] = []
    for (start, end), owner in zip(script.bodies, script.body_commands, strict=True):
        if owner is command and (start, end) not in script.file_bodies:
            code = text[start:end]
            paths.extend(python_paths(code))
            paths.extend(_pathlib_paths(code))
    return paths


def _shell_requests(text: str) -> tuple[_Request, ...]:
    script = shell.parse(text)
    base = command_paths(text).requests
    requests: list[_Request] = []
    for command in script.commands:
        start, end = command.span
        nested_spans = [c.span for c in script.commands if start <= c.span[0] and c.span[1] < end]
        input_spans = {span for op, _, span in command.redirects if op == "<"}
        paths = [
            r.path
            for r in base
            if start <= r.span[0] < end
            and r.span not in input_spans
            and not any(a <= r.span[0] < b for a, b in nested_spans)
        ]
        paths.extend(_extra_paths(command))
        paths.extend(_python_extra(command))
        paths.extend(_python_bodies(text, script, command))
        metadata = command.name in METADATA or (
            command.name == "rg" and any(w == "--files" for w, _ in command.argv())
        )
        kind: AccessKind = "metadata_or_other_access" if metadata else "content_read"
        requests.extend(_Request(p, kind, command.name in PARTIAL_READERS) for p in paths)
        requests.extend(_Request(p) for op, p, _ in command.redirects if op == "<")
    return tuple(requests)


def _field_paths(call: ToolCall) -> list[str]:
    args = call.arguments
    allowed: set[str] | None = None
    if args is not None:
        allowed = set()
        for key in PATH_KEYS:
            value = args.get(key)
            if isinstance(value, str):
                allowed.add(value)
            elif isinstance(value, (tuple, list)):
                allowed.update(v for v in value if isinstance(v, str))
    return [
        c.text
        for channel, c in call.fields
        if channel == Channel.PATH
        and c.understood
        and not c.computed
        and (allowed is None or c.text in allowed)
    ]


def _requests(call: ToolCall) -> tuple[_Request, ...]:
    name = _tool(call)
    if call.tool == "shell":
        return tuple(
            r
            for channel, c in call.fields
            if channel == Channel.COMMAND
            and c.understood
            and not c.computed
            and len(c.text) <= MAX_COMMAND_CHARS
            for r in _shell_requests(c.text)
        )
    if name not in READ_TOOLS | CONTENT_SEARCH_TOOLS | METADATA_TOOLS:
        return ()
    kind: AccessKind = "metadata_or_other_access" if name in METADATA_TOOLS else "content_read"
    partial = name in CONTENT_SEARCH_TOOLS or any(k in (call.arguments or {}) for k in RANGE_KEYS)
    return tuple(_Request(p, kind, partial) for p in _field_paths(call))


def _independent_results(step: Step, call: ToolCall) -> list[Observation]:
    """Only independently paired observations establish a recorded call's outcome."""
    return [
        o
        for o in step.observations
        if not o.pairing_unresolved
        and not o.shared_terminal
        and (
            (o.source_call_index is not None and o.source_call_index == call.index)
            or (
                o.source_call_index is None
                and bool(call.id)
                and call.id != call.result_id
                and sum(c.id == call.id for c in step.calls) == 1
                and o.source_call_id == call.id
            )
        )
    ]


def _json_result(text: str) -> object:
    if len(text) > MAX_RESULT_CHARS:
        return None
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


def _json_error(value: object, depth: int = 0) -> bool:
    if depth > MAX_ERROR_DEPTH:
        return False
    obj = as_object(value)
    explicit = (
        bool(obj.get("error"))
        or obj.get("is_error") is True
        or obj.get("isError") is True
        or obj.get("type") in ("tool_use_error", "error")
        or obj.get("status") in ("failed", "error")
    )
    children = as_list(value) + as_list(obj.get("content")) + as_list(obj.get("results"))
    return explicit or any(_json_error(v, depth + 1) for v in children)


def _result_parts(text: str) -> tuple[str, int | None, bool]:
    value = _json_result(text)
    obj = as_object(value)
    failure = bool(ERROR.search(text)) or _json_error(value)
    exit_value = obj.get("exit_code", obj.get("returncode"))
    code = exit_value if type(exit_value) is int else None
    payload = text
    if "output" in obj or "stdout" in obj:
        payload = as_str(obj.get("output", obj.get("stdout"))) or ""
    if not obj and (match := EXIT.search(text)) is not None:
        code = int(match[1])
        payload = RESULT_HEADERS.sub("", EXIT.sub("", text)).strip()
    failure |= bool(ERROR.search(payload)) or bool(ERROR.search(as_str(obj.get("stderr")) or ""))
    return payload, code, failure


def _single_reader(call: ToolCall) -> bool:
    commands = [c.text for channel, c in call.fields if channel == Channel.COMMAND]
    if len(commands) != 1:
        return False
    script = shell.parse(commands[0])
    if not script.complete or len(script.commands) != 1:
        return False
    command = script.commands[0]
    flags = {w for w, _ in command.argv()[1:]}
    unsafe = {
        "-q",
        "--quiet",
        "-l",
        "--files-with-matches",
        "-L",
        "--files-without-match",
        "-c",
        "--count",
        "--count-matches",
        "--files",
    }
    return (
        command.name in TEXT_READERS | METADATA
        and command.words[0].rsplit("/", 1)[-1] == command.name
        and not shell.writes(command)
        and len({r.path for r in _shell_requests(commands[0])}) == 1
        and not (command.name in {"grep", "rg"} and flags & unsafe)
    )


def _outcome(step: Step, call: ToolCall, kind: AccessKind) -> Outcome:
    results = _independent_results(step, call)
    # Compound commands' exit/output cannot identify the particular spool access.
    if call.tool == "shell" and not _single_reader(call):
        return "unknown"
    outcomes: list[Outcome] = []
    for observation in results:
        content = observation.content
        if not content.understood or content.computed:
            outcomes.append("unknown")
            continue
        payload, code, failure = _result_parts(content.text)
        search_no_match = code == 1 and any(
            c.name in {"grep", "rg"}
            for channel, content in call.fields
            if channel == Channel.COMMAND
            for c in shell.parse(content.text).commands
        )
        if failure or (code is not None and code != 0 and not search_no_match):
            outcomes.append("explicit_failure")
        elif (
            kind == "content_read"
            and payload.strip()
            and payload.strip().lower() not in STATUS_TEXT
            and (call.tool != "shell" or code == 0)
        ):
            outcomes.append("content_returned")
        else:
            outcomes.append("unknown")
    return outcomes[0] if outcomes and len(set(outcomes)) == 1 else "unknown"


def find_followups(
    steps: tuple[Step, ...], after_step: int, paths: tuple[str, ...]
) -> tuple[Followup, ...]:
    """Correlate exact normalized literal paths in later authored access requests."""
    targets = {p for raw in paths if (p := _path(raw)) is not None}
    found: list[Followup] = []
    for step, call, delay in _later_calls(steps, after_step):
        cwd = _cwd(call)
        # A shell call that changes cwd invalidates relative-path anchoring.
        if call.tool == "shell" and any(
            command.name in {"cd", "pushd", "popd"}
            for channel, c in call.fields
            if channel == Channel.COMMAND and len(c.text) <= MAX_COMMAND_CHARS
            for command in shell.parse(c.text).commands
        ):
            cwd = None
        matches = [r for r in _requests(call) if _path(r.path, cwd) in targets]
        for kind in ("content_read", "metadata_or_other_access"):
            selected = [r for r in matches if r.kind == kind]
            if selected:
                found.append(
                    Followup(
                        step.index,
                        call.index,
                        _tool(call),
                        kind,
                        any(r.partial for r in selected),
                        _outcome(step, call, kind),
                        delay,
                    )
                )
    return tuple(found)


def find_process_followups(
    steps: tuple[Step, ...], after_step: int, handle: str
) -> tuple[Followup, ...]:
    """Only process.read_output for this exact handle, never status/wait/list."""
    if not handle:
        return ()
    found: list[Followup] = []
    for step, call, delay in _later_calls(steps, after_step):
        args = call.arguments or {}
        if (
            _tool(call).lower() == "process"
            and args.get("action") == "read_output"
            and args.get("process_id") == handle
        ):
            partial = any(k in args for k in ("query", "offset", "limit"))
            found.append(
                Followup(
                    step.index,
                    call.index,
                    "process",
                    "content_read",
                    partial,
                    _outcome(step, call, "content_read"),
                    delay,
                )
            )
    return tuple(found)
