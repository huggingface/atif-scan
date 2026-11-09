"""Bounded shell source-read facts. Parse strings only; never execute commands.

Only literal operands and preceding simple assignments/cd are reconstructed.
Nested shells have separate state; pipelines/background mutations cannot establish
the following shell's directory. Facts describe requests, not successful reads.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from collections import Counter
from dataclasses import dataclass, field

from . import shell
from .python_units import PYTHON, PYTHON_FLAGS

READERS = frozenset({"cat", "head", "tail", "less", "more", "bat", "nl", "tac"})
SEARCHERS = frozenset({"grep", "egrep", "fgrep", "rg"})
COMPARERS = frozenset({"diff", "cmp"})
COPY = frozenset({"cp", "rsync"})
SINGLE_ARGUMENT = 2
SEARCH_VALUES = frozenset(
    {
        "-e",
        "-f",
        "--regexp",
        "--file",
        "-A",
        "-B",
        "-C",
        "-m",
        "--max-count",
        "--include",
        "--exclude",
        "--exclude-dir",
        "-g",
        "--glob",
        "-t",
        "--type",
        "--type-not",
        "--encoding",
    }
)
VARIABLE = re.compile(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))")
ASSIGNMENT = re.compile(r"([A-Za-z_]\w*)=(.*)", re.S)


@dataclass(frozen=True)
class PathRequest:
    path: str
    span: shell.Span
    kind: str


@dataclass(frozen=True)
class PythonUnit:
    code: str
    interpreter: str
    span: shell.Span
    search_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceCommands:
    paths: tuple[PathRequest, ...] = ()
    python: tuple[PythonUnit, ...] = ()
    complete: bool = True


@dataclass
class _State:
    bindings: dict[str, str] = field(default_factory=dict)
    exported: set[str] = field(default_factory=set)
    cwd: str | None = None
    unknown_pythonpath: bool = False

    def copy(self) -> _State:
        return _State(dict(self.bindings), set(self.exported), self.cwd, self.unknown_pythonpath)


def _variable(raw: str, index: int, state: _State) -> tuple[str, int] | None:
    match = VARIABLE.match(raw, index)
    if match is None or (value := state.bindings.get(match[1] or match[2])) is None:
        return None
    return value, match.end() - 1


def _escaped(raw: str, index: int, quote: str) -> str:
    char = raw[index]
    if char == "\n":
        return ""
    return "\\" + char if quote == '"' and char not in '$`"\\' else char


def _expand(raw: str, state: _State) -> str | None:
    """Expand only simple variables with shell quoting; never recursively expand values."""
    pieces: list[str] = []
    quote = ""
    index = 0
    while index < len(raw):
        char = raw[index]
        if char in {"'", '"'} and (not quote or quote == char):
            quote = char if not quote else ""
        elif char == "\\" and quote != "'":
            index += 1
            if index >= len(raw):
                return None
            pieces.append(_escaped(raw, index, quote))
        elif char in {"$", "`"} and quote != "'":
            variable = _variable(raw, index, state)
            if variable is None:
                return None
            value, index = variable
            pieces.append(value)
        else:
            pieces.append(char)
        index += 1
    return None if quote else "".join(pieces)


def _path(raw: str, state: _State) -> str | None:
    resolved = _expand(raw, state)
    if resolved is None:
        return None
    if state.cwd and not resolved.startswith(("/", "~")):
        resolved = posixpath.join(state.cwd, resolved)
    return posixpath.normpath(resolved)


def _assign(command: shell.Command, text: str, state: _State, *, export: bool = False) -> None:
    words = list(zip(command.words, command.spans, strict=True))
    if command.name in {"export", "readonly"}:
        words = command.argv()[1:]
    for word, (a, b) in words:
        match = ASSIGNMENT.fullmatch(word)
        if match is None:
            break
        key = match[1]
        raw = text[a:b]
        value = _expand(raw[raw.find("=") + 1 :], state)
        if key == "PYTHONPATH":
            state.unknown_pythonpath = value is None
        if value is None:
            state.bindings.pop(key, None)
        else:
            state.bindings[key] = value
        if export:
            state.exported.add(key)


def _declare(command: shell.Command, text: str, state: _State) -> None:
    _assign(command, text, state, export=command.name == "export")
    argv = command.argv()
    if command.name == "export" and len(argv) > 1 and argv[1][0] == "-n":
        state.exported.difference_update(word for word, _ in argv[2:])


def _mutation(command: shell.Command, text: str, state: _State) -> bool:
    argv = command.argv()
    name = command.name
    if not argv:
        _assign(command, text, state)
    elif name in {"export", "readonly"}:
        _declare(command, text, state)
    elif name == "unset":
        for word, _ in argv[1:]:
            state.bindings.pop(word, None)
            state.exported.discard(word)
    elif name == "cd":
        state.cwd = _path(text[slice(*argv[1][1])], state) if len(argv) == SINGLE_ARGUMENT else None
        if state.cwd and not state.cwd.startswith(("/", "~")):
            state.cwd = None
    else:
        return False
    return True


def _takes_value(word: str, values: frozenset[str]) -> bool:
    if word.startswith("--"):
        return word in values
    for index, letter in enumerate(word[1:], 1):
        if "-" + letter in values:
            return index == len(word) - 1
    return False


def _file_operands(
    argv: list[tuple[str, shell.Span]], values: frozenset[str]
) -> list[tuple[str, shell.Span]]:
    result: list[tuple[str, shell.Span]] = []
    options = True
    skip = False
    for word, span in argv[1:]:
        if skip:
            skip = False
        elif options and word == "--":
            options = False
        elif options and word.startswith("-") and word != "-":
            skip = _takes_value(word, values)
        else:
            result.append((word, span))
    return result


def _sed_files(command: shell.Command) -> list[tuple[str, shell.Span]]:
    argv = command.argv()
    if any(word == "--in-place" or word.startswith("-i") for word, _ in argv[1:]):
        return []
    explicit = any(word in shell.SED_SCRIPT or word.startswith("--expression=") for word, _ in argv)
    operands = _file_operands(argv, shell.SED_SCRIPT)
    return operands if explicit else operands[1:]


def _search_files(command: shell.Command) -> list[tuple[str, shell.Span]]:
    if any(word in {"--files", "--files-without-match"} for word in command.words):
        return []
    patterns = shell.search_patterns(command)
    if patterns is None:
        return []
    _, short_values, long_values = shell.SEARCHERS[command.name or ""]
    values = frozenset("-" + letter for letter in short_values | {"e", "f"})
    return [
        (word, span)
        for word, span in _file_operands(
            command.argv(), values | long_values | {"--regexp", "--file"}
        )
        if span not in patterns
    ]


def _operands(command: shell.Command) -> list[tuple[str, shell.Span]]:
    name = command.name
    if name in SEARCHERS:
        return _search_files(command)
    if name == "sed":
        return _sed_files(command)
    if name in COPY:
        destinations = {span for _, span in shell.writes(command)}
        return [
            (word, span)
            for word, span in _file_operands(command.argv(), shell.COPY_VALUES)
            if span not in destinations
        ]
    return (
        _file_operands(command.argv(), _reader_options(name)) if name in READERS | COMPARERS else []
    )


def _reader_options(name: str | None) -> frozenset[str]:
    options: dict[str, frozenset[str]] = {
        "head": frozenset({"-n", "-c", "--lines", "--bytes"}),
        "tail": frozenset({"-n", "-c", "--lines", "--bytes", "--pid", "-s"}),
        "bat": frozenset({"--theme", "--language", "-l", "--style", "--line-range", "-r"}),
        "diff": frozenset({"-I", "-F", "-L", "--label", "--ignore-matching-lines"}),
        "nl": frozenset({"-b", "-d", "-f", "-h", "-i", "-l", "-n", "-s", "-v", "-w"}),
    }
    return options.get(name or "", frozenset[str]())


def _input_redirects(command: shell.Command, text: str) -> list[tuple[str, shell.Span]]:
    result: list[tuple[str, shell.Span]] = []
    for op, target, (a, b) in command.redirects:
        if op == "<" and (prefix := re.match(r"\d*<\s*", text[a:b])):
            result.append((target, (a + prefix.end(), b)))
    return result


def _known_short(word: str, flags: str, values: frozenset[str], numeric: bool) -> bool:
    if numeric and word[1:].isdigit():
        return True
    for char in word[1:]:
        if "-" + char in values:
            return True
        if char not in flags:
            return False
    return True


def _reader_flags_known(command: shell.Command) -> bool:
    flags = {
        "cat": "AbeEnstTuv",
        "head": "cqvn",
        "tail": "cfFqnsv",
        "less": "EFIKMNQRSVXacdefgimnqrsuwxz",
        "more": "dflpcsu",
        "bat": "Apn",
        "nl": "bp",
        "tac": "brs",
        "diff": "abBcdeEfHiNnpPqrsStuUvwy",
        "cmp": "bls",
        "cp": "abdfHilLnPpRrsTtuvx",
        "rsync": "avzrcn",
    }
    known = flags.get(command.name or "")
    if known is None:
        return True
    values = _reader_options(command.name)
    long_flags = {
        "--number",
        "--number-nonblank",
        "--show-all",
        "--squeeze-blank",
        "--verbose",
        "--quiet",
        "--silent",
        "--recursive",
        "--unified",
    }
    for word, _ in command.argv()[1:]:
        if word == "--":
            break
        if word.startswith("--") and word.split("=", 1)[0] not in values | long_flags:
            return False
        if (
            word.startswith("-")
            and not word.startswith("--")
            and not _known_short(word, known, values, command.name in {"head", "tail"})
        ):
            return False
    return True


def _no_read(command: shell.Command) -> bool:
    for word, _ in command.argv()[1:]:
        if word == "--":
            break
        if word in {"--help", "--version"}:
            return True
    return False


def _unresolved_operands(command: shell.Command) -> list[tuple[str, shell.Span]]:
    inert = {
        "echo",
        "printf",
        "ls",
        "find",
        "stat",
        "file",
        "pwd",
        "which",
        "pip",
        "pip3",
        "readlink",
        "test",
        "[",
        "true",
        "false",
    }
    if command.name in inert or command.name in READERS | SEARCHERS | COMPARERS | COPY | {"sed"}:
        return []
    return _file_operands(command.argv(), frozenset())


def _requests(command: shell.Command, text: str, state: _State) -> tuple[list[PathRequest], bool]:
    name = command.name
    if _no_read(command):
        return [], True
    if not _reader_flags_known(command) or (
        name in SEARCHERS and shell.search_patterns(command) is None
    ):
        return [], False
    unresolved = _unresolved_operands(command)
    operands = unresolved or _operands(command)
    if name in READERS | SEARCHERS | COMPARERS:
        operands += _input_redirects(command, text)
    kind = (
        "unresolved"
        if unresolved
        else "search"
        if name in SEARCHERS
        else "copy"
        if name in COPY
        else "read"
    )
    paths: list[PathRequest] = []
    complete = True
    for _word, span in operands:
        path = _path(text[slice(*span)], state)
        if path is None:
            complete = False
        elif path != "-":
            paths.append(PathRequest(path, span, kind))
    return paths, complete


def _python_paths(command: shell.Command, state: _State) -> tuple[str, ...]:
    explicit = any(word.startswith("PYTHONPATH=") for word in _prefix_words(command))
    if explicit or "PYTHONPATH" in state.exported:
        return tuple(
            posixpath.normpath(posixpath.join(state.cwd, path))
            if state.cwd and path and not path.startswith(("/", "~"))
            else path
            for path in state.bindings.get("PYTHONPATH", "").split(":")
        )
    return ()


def _python_invocation(
    command: shell.Command, text: str, state: _State
) -> tuple[str, list[tuple[str, shell.Span]]] | None:
    argv = command.argv()
    if not argv:
        return None
    interpreter = _expand(text[slice(*argv[0][1])], state)
    if interpreter and "/" in interpreter and not interpreter.startswith(("/", "~")) and state.cwd:
        interpreter = posixpath.normpath(posixpath.join(state.cwd, interpreter))
    if interpreter is None or PYTHON.fullmatch(interpreter.rsplit("/", 1)[-1]) is None:
        return None
    args = argv[1:]
    while args and args[0][0] in PYTHON_FLAGS:
        args = args[1:]
    return interpreter, args


def _python_units(
    text: str, script: shell.Script, command: shell.Command, state: _State, environment: _State
) -> list[PythonUnit]:
    invocation = _python_invocation(command, text, state)
    if invocation is None:
        return []
    interpreter, args = invocation
    roots = _python_paths(command, environment)
    if any(word in {"-E", "-I"} for word, _ in command.argv()[1:]):
        roots = ()
    if len(args) >= SINGLE_ARGUMENT and args[0][0] == "-c":
        code = _expand(text[slice(*args[1][1])], state)
        return [PythonUnit(code, interpreter, args[1][1], roots)] if code is not None else []
    # External script files are never opened or assumed to expose their imports.
    spans = [
        span
        for span, owner in zip(script.bodies, script.body_commands, strict=True)
        if owner is command and span not in script.file_bodies and (not args or args[0][0] == "-")
    ]
    return [PythonUnit(text[a:b], interpreter, (a, b), roots) for a, b in spans]


def _parents(script: shell.Script) -> dict[int, int]:
    return {
        id(child): id(command)
        for command in script.commands
        for child in (*command.nested, *command.substituted)
    }


def _simple_scope(text: str, script: shell.Script) -> bool:
    masked = list(text)
    for a, b in script.bodies:
        masked[a:b] = " " * (b - a)
    lexer = shlex.shlex("".join(masked), posix=True, punctuation_chars="();&|")
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    # These require control-flow/group state beyond this bounded reconstruction.
    return not any(
        token in {"(", ")", "if", "for", "while", "until", "case", "alias"} for token in tokens
    )


def _prefix_words(command: shell.Command) -> list[str]:
    argv = command.argv()
    end = command.spans.index(argv[0][1]) if argv else len(command.words)
    return command.words[:end]


def _env_assignments(
    command: shell.Command, text: str, state: _State, local: _State, start: int
) -> None:
    for word, (a, b) in zip(
        _prefix_words(command)[start + 1 :],
        command.spans[start + 1 : len(_prefix_words(command))],
        strict=True,
    ):
        if match := ASSIGNMENT.fullmatch(word):
            value = _expand(text[a:b].split("=", 1)[1], state)
            key = match[1]
            if key == "PYTHONPATH":
                local.unknown_pythonpath = value is None
            if value is not None:
                local.bindings[key] = value
                local.exported.add(key)
            else:
                local.bindings.pop(key, None)


def _environment(command: shell.Command, text: str, state: _State) -> _State:
    local = state.copy()
    _assign(command, text, local)
    words = command.words
    if "env" in words or "/usr/bin/env" in words:
        start = next(i for i, word in enumerate(words) if word.rsplit("/", 1)[-1] == "env")
        args = words[start + 1 :]
        if "-i" in args or "--ignore-environment" in args:
            local.bindings.clear()
            local.exported.clear()
            local.unknown_pythonpath = False
        for i, word in enumerate(args):
            if word in {"-u", "--unset"} and i + 1 < len(args):
                local.bindings.pop(args[i + 1], None)
                local.exported.discard(args[i + 1])
            elif word.startswith(("-u", "--unset=")):
                key = word[2:] if word.startswith("-u") else word.split("=", 1)[1]
                local.bindings.pop(key, None)
                local.exported.discard(key)
        _env_assignments(command, text, state, local, start)
    return local


def _child_state(command: shell.Command, state: _State, environment: _State) -> _State:
    # A new sh process gets exported variables and command-local assignments, not
    # arbitrary parent shell locals. Substitutions are handled separately below.
    names = set(environment.exported)
    names.update(m[1] for word in _prefix_words(command) if (m := ASSIGNMENT.fullmatch(word)))
    return _State({k: v for k, v in environment.bindings.items() if k in names}, names, state.cwd)


def _stdin_unresolved(script: shell.Script) -> bool:
    for command in script.commands:
        inputs = [op for op, _, _ in command.redirects if op in {"<", "<<", "<<-", "<<<"}]
        if "<<<" in inputs or len(inputs) > 1:
            return True
    return False


def _missing_inline(command: shell.Command, units: list[PythonUnit]) -> bool:
    return (
        PYTHON.fullmatch(command.name or "") is not None
        and "-c" in command.words
        and not _no_read(command)
        and not units
    )


def _unsupported_mutation(command: shell.Command) -> bool:
    if command.name in {
        "read",
        "readarray",
        "mapfile",
        "declare",
        "typeset",
        "local",
        "pushd",
        "popd",
        "let",
        "set",
        "trap",
        "unalias",
        "enable",
        "builtin",
    }:
        return True
    args = [word for word, _ in command.argv()[1:]]
    if command.name in {"export", "readonly", "unset"} and any(
        word.startswith("-") for word in args
    ):
        return not (
            command.name == "export"
            and args[:1] == ["-n"]
            and all(not word.startswith("-") and "=" not in word for word in args[1:])
        )
    return False


def source_commands(text: str) -> SourceCommands:
    """Extract lexical source-read operands and executed inline Python units."""
    script = shell.parse(text)
    if (
        not script.complete
        or not _simple_scope(text, script)
        or _stdin_unresolved(script)
        or any(_unsupported_mutation(command) for command in script.commands)
    ):
        return SourceCommands(complete=False)
    parents = _parents(script)
    pipelines = Counter(c.pipeline for c in script.commands)
    states: dict[int, _State] = {0: _State()}
    paths: list[PathRequest] = []
    python: list[PythonUnit] = []
    complete = True
    for command in sorted(script.commands, key=lambda c: c.span[0]):
        parent = parents.get(id(command), 0)
        shared = states.get(parent, _State())
        state = shared.copy()
        environment = _environment(command, text, shared)
        mutated = _mutation(command, text, state)
        if not mutated:
            found, understood = _requests(command, text, shared)
            paths.extend(found)
            complete &= understood
            units = _python_units(text, script, command, shared, environment)
            python.extend(units)
            complete &= not _missing_inline(command, units)
            ignores_environment = any(word in {"-E", "-I"} for word, _ in command.argv()[1:])
            complete &= not (units and environment.unknown_pythonpath and not ignores_environment)
            state = shared.copy()  # Command-local environment assignments do not persist.
            if command.name in {"source", ".", "eval"}:
                state = _State()
                complete = False
        if mutated and re.search(r"(?:&&|\|\|)\s*$", text[: command.span[0]]):
            state = _State()
            complete = False
        states[id(command)] = (
            _child_state(command, shared, environment) if command.nested else state.copy()
        )
        if pipelines[command.pipeline] == 1 and not command.background and not command.repeated:
            states[parent] = state
    return SourceCommands(tuple(paths), tuple(python), complete)
