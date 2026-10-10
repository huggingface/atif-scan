"""Static reader for shell command text (stdlib only; nothing is ever executed).

`parse(text)` splits a shell tool's command into simple commands: their words (quotes
removed), redirections, which pipeline they belong to, and the commands nested in them
(`$(…)`, `<(…)`, backticks, subshells, `sh -c '…'`, a heredoc fed to a shell). Heredoc
bodies given to any other program are file contents, returned as `bodies` spans: they are
not commands (a Python heredoc's `if a > b:` isn't a redirect).

It reads the common POSIX/bash forms agents write, not the whole grammar. Input it can't
structure (an unterminated quote or substitution, nesting too deep) sets
`complete=False`; callers then fall back to text patterns rather than trusting a partial
structure.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

Span = tuple[int, int]

SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "ash"})
# Words before the command name that don't name it.
KEYWORDS = frozenset(
    {"!", "{", "}", "then", "do", "else", "elif", "if", "while", "until", "time", "exec"}
)
# Wrappers that run the next word; value-taking options are skipped with their value.
WRAPPERS = {
    "sudo": frozenset({"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U"}),
    "env": frozenset({"-u", "-C", "-S"}),
    "nohup": frozenset(),
    "command": frozenset(),
    "builtin": frozenset(),
    "nice": frozenset({"-n"}),
    "stdbuf": frozenset({"-i", "-o", "-e"}),
    "timeout": frozenset({"-s", "-k", "--signal", "--kill-after"}),
}
WRITE_OPS = frozenset({">", ">>", ">|", "&>", "&>>", "<>", ">&"})
ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=")
REDIRECT = re.compile(r"\d*(?:&>>|&>|>>|>\||>&|<<-|<<<|<<|<&|<>|>|<)")
OPERATOR = re.compile(r"&&|\|\||;;|\|&|[|;&]")
WORD_END = frozenset(" \t\n;&|<>()")
# Runs read as they are (fast paths): blanks, and word/double-quoted text without quotes,
# escapes, expansions or (for a word) its end.
BLANKS = re.compile(r"[ \t]+")
PLAIN = re.compile(r"[^ \t\n;&|<>()\\'\"$`]+")
QUOTED_PLAIN = re.compile(r'[^"\\$`]+')
SHELL_C = re.compile(r"-[a-z]*c[a-z]*")  # -c, -lc, -ec, …
MAX_DEPTH = 32


@dataclass(slots=True)
class Command:
    words: list[str] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    redirects: list[tuple[str, str, Span]] = field(default_factory=list)  # (op, target, span)
    # Commands this one runs: an `sh -c` string, a heredoc fed to a shell (`nested`), and
    # command/process substitutions in its words (`substituted`: `$(…)`, `<(…)`, backticks).
    nested: list[Command] = field(default_factory=list)
    substituted: list[Command] = field(default_factory=list)
    pipeline: int = 0
    span: Span = (0, 0)

    def argv(self) -> list[tuple[str, Span]]:
        """(word, span) from the command name on: assignments, keywords and wrappers
        (`sudo -u x`, `env A=1`, `timeout 5`) skipped."""
        words = list(zip(self.words, self.spans, strict=True))
        i = 0
        while i < len(words):
            word = words[i][0]
            base = word.rsplit("/", 1)[-1]
            if word in KEYWORDS or ASSIGNMENT.match(word):
                i += 1
            elif base in WRAPPERS:
                i += 1
                takes_value = WRAPPERS[base]
                while i < len(words) and (
                    words[i][0].startswith("-") or ASSIGNMENT.match(words[i][0])
                ):
                    i += 2 if words[i][0] in takes_value else 1
                if base == "timeout" and i < len(words):
                    i += 1  # the duration
            else:
                break
        return words[i:]

    @property
    def name(self) -> str | None:
        """The command's basename (`/usr/bin/curl` → `curl`); a substitution run as the
        command (`$(curl …)`) keeps its text."""
        argv = self.argv()
        if not argv:
            return None
        word = argv[0][0]
        return word if word.startswith(("$(", "`")) else word.rsplit("/", 1)[-1]


@dataclass(slots=True)
class Script:
    commands: list[Command] = field(default_factory=list)  # every command, nested included
    bodies: list[Span] = field(default_factory=list)  # heredoc contents (not commands)
    # The bodies `cat`/`tee` write to a file (`cat > f <<EOF`): file content, not input
    # to a program. A body fed to an interpreter (`python - <<EOF`) isn't one.
    file_bodies: list[Span] = field(default_factory=list)
    # The file each of `file_bodies` is written to (same order).
    file_targets: list[str] = field(default_factory=list)
    complete: bool = True


@functools.lru_cache(maxsize=512)
def parse(text: str) -> Script:
    """Never raises: anything the reader trips over makes the script incomplete. Cached
    (several checks read the same command): treat the result as read-only."""
    script = Script()
    try:
        _Reader(text, script, count(1)).commands(0, None)
    except (IndexError, ValueError, RecursionError):
        script.complete = False
    return script


class _Reader:
    """One pass over `text[pos:]`; nested substitutions reuse the same reader (and
    offsets), backticks and `sh -c` strings get a reader of their own."""

    def __init__(self, text: str, script: Script, pipelines: count[int], base: int = 0) -> None:
        self.text, self.script, self.pipelines, self.base = text, script, pipelines, base
        self.pos = 0
        self.literal = False  # inside an `sh -c` string: substitutions are its own
        self.heredocs: list[tuple[str, bool, Command]] = []

    def span(self, start: int, end: int) -> Span:
        return (self.base + start, self.base + end)

    def fail(self) -> None:
        self.script.complete = False
        self.pos = len(self.text)

    def commands(self, depth: int, stop: str | None) -> None:
        """Read commands up to `stop` (")" for a substitution/subshell) or the end."""
        if depth > MAX_DEPTH:
            return self.fail()
        pending = _Pending(next(self.pipelines), self.pos)
        while self.pos < len(self.text):
            if not self.command_part(pending, depth, stop):
                return None
        self.finish(pending, True)
        if stop is not None:
            self.fail()
        return None

    def finish(self, pending: _Pending, new_pipeline: bool) -> None:
        """Record the command being read (if it has anything) and start the next one."""
        command = pending.command
        if command.words or command.redirects:
            command.span = self.span(pending.start, self.pos)
            self.script.commands.append(command)
        if new_pipeline:
            pending.pipeline = next(self.pipelines)
        pending.command = Command(pipeline=pending.pipeline)
        pending.start = self.pos

    def command_part(self, pending: _Pending, depth: int, stop: str | None) -> bool:
        """Read one blank, separator or token at `pos`; False when `commands` is done."""
        if self.blank(pending) or self.separator(pending, depth):
            return True
        if stop is not None and self.text[self.pos] == stop:
            self.finish(pending, True)
            self.pos += 1
            return False
        return self.token(pending.command, depth)

    def blank(self, pending: _Pending) -> bool:
        """Skip a space/tab, line continuation or comment at `pos`."""
        text = self.text
        if blanks := BLANKS.match(text, self.pos):
            self.pos = blanks.end()
            if not pending.command.words and not pending.command.redirects:
                pending.start = self.pos
            return True
        if text.startswith("\\\n", self.pos):
            self.pos += 2
            return True
        if text[self.pos] == "#":
            end = text.find("\n", self.pos)
            self.pos = len(text) if end == -1 else end
            return True
        return False

    def separator(self, pending: _Pending, depth: int) -> bool:
        """A newline (then the heredoc bodies it opens) or a control operator at `pos`."""
        text = self.text
        if text[self.pos] == "\n":
            self.finish(pending, True)
            self.pos += 1
            self.read_heredocs(depth)
            pending.start = self.pos
            return True
        op = OPERATOR.match(text, self.pos)
        if op is None or text.startswith("&>", self.pos):
            return False
        self.pos = op.end()
        self.finish(pending, op[0] not in ("|", "|&"))
        return True

    def token(self, command: Command, depth: int) -> bool:
        """A process substitution, redirection, subshell or word of `command`."""
        text = self.text
        c = text[self.pos]
        if c in "<>" and text.startswith("(", self.pos + 1):
            self.substitution(command, depth, self.pos, self.pos + 2)
            return True
        # At a token start, so a leading fd number (`2>`) belongs to the redirection.
        redirect = REDIRECT.match(text, self.pos)
        if redirect:
            self.pos = redirect.end()
            self.redirect(command, redirect[0].lstrip("0123456789"), redirect.start(), depth)
            return True
        if c in "()":
            return self.subshell(command, depth)
        word = self.word(depth, command)
        if word is not None:
            command.words.append(word[0])
            command.spans.append(word[1])
        return word is not None

    def subshell(self, command: Command, depth: int) -> bool:
        if self.text[self.pos] == ")":  # a `case` pattern or unbalanced input
            self.fail()
            return False
        self.pos += 1
        if command.words:  # `f() {…}` or a stray paren: not something we structure
            self.fail()
            return False
        self.commands(depth + 1, ")")
        return True

    def redirect(self, command: Command, op: str, at: int, depth: int) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t":
            self.pos += 1
        target = self.word(depth, command, operand=False)
        if target is None:
            return
        word, _ = target
        command.redirects.append((op, word, self.span(at, self.pos)))
        if op in ("<<", "<<-"):
            self.heredocs.append((word, op == "<<-", command))

    def read_heredocs(self, depth: int) -> None:
        """Bodies of the heredocs opened on the line just ended, in order."""
        text = self.text
        for delimiter, strip_tabs, command in self.heredocs:
            body = self.pos
            end = close = len(text)
            line = self.pos
            while line < len(text):
                nl = text.find("\n", line)
                stop = len(text) if nl == -1 else nl
                current = text[line:stop]
                if (current.lstrip("\t") if strip_tabs else current) == delimiter:
                    end, close = line, stop
                    break
                line = stop + 1
            self.pos = close
            if command.name in SHELLS and not any(SHELL_C.fullmatch(w) for w in command.words):
                self.nested(command, text[body:end], body, depth)
            else:
                self.script.bodies.append(self.span(body, end))
                if command.name in FILE_WRITERS and (targets := writes(command)):
                    self.script.file_bodies.append(self.span(body, end))
                    self.script.file_targets.append(targets[0][0])
        self.heredocs = []

    def nested(
        self, command: Command, text: str, offset: int, depth: int, substituted: bool = False
    ) -> None:
        """Commands in `text` (a backtick body, heredoc or `sh -c` string) run by
        `command`: parsed with their own reader, spans kept relative to this text."""
        if depth > MAX_DEPTH:
            return self.fail()
        inner = Script()
        reader = _Reader(text, inner, self.pipelines, self.base + offset)
        reader.commands(depth + 1, None)
        self.script.complete &= inner.complete
        self.script.bodies += inner.bodies
        (command.substituted if substituted else command.nested).extend(inner.commands)
        self.script.commands += inner.commands
        return None

    def substitution(self, command: Command, depth: int, start: int, inner: int) -> None:
        """`$(…)`, `<(…)` or `>(…)` starting at `start`: its commands belong to `command`."""
        before = len(self.script.commands)
        self.pos = inner
        self.commands(depth + 1, ")")
        command.substituted += self.script.commands[before:]
        command.words.append(self.text[start : self.pos])
        command.spans.append(self.span(start, self.pos))

    def word(self, depth: int, command: Command, operand: bool = True) -> tuple[str, Span] | None:
        """One word, quotes removed; substitutions inside it are read as commands (their
        raw text stays in the word)."""
        text = self.text
        start = self.pos
        out: list[str] = []
        # `sh -c '…'` / `bash -lc "…"`: the string is a script the shell runs. Read it
        # literally here and parse it once, as a script.
        script = operand and start < len(text) and runs_script(command)
        self.literal, outer = script, self.literal
        while self.pos < len(text) and text[self.pos] not in WORD_END:
            if plain := PLAIN.match(text, self.pos):
                out.append(plain[0])
                self.pos = plain.end()
                continue
            step = self.word_part(out, depth, command)
            if step is not _Next.CONTINUE:
                return self.fail() if step is _Next.FAIL else None
        word = "".join(out)
        self.literal = outer
        if script:
            self.nested(command, word, start + (1 if text[start] in "'\"" else 0), depth)
        return word, self.span(start, self.pos)

    def word_part(self, out: list[str], depth: int, command: Command) -> _Next:
        """Read one quoted string, escape, expansion or character of a word into `out`."""
        c = self.text[self.pos]
        if self.literal and c in "$`":
            return self.keep(out)
        expansion = self.expansion(out, depth, command)
        if expansion is not None:
            return expansion
        return WORD_PARTS.get(c, _Reader.keep)(self, out, depth, command)

    def keep(self, out: list[str], _depth: int = 0, _command: Command | None = None) -> _Next:
        """The character at `pos`, as it is."""
        out.append(self.text[self.pos])
        self.pos += 1
        return _Next.CONTINUE

    def escaped(self, out: list[str], _depth: int, _command: Command) -> _Next:
        if not self.text.startswith("\n", self.pos + 1):  # else a line continuation
            out.append(self.text[self.pos + 1 : self.pos + 2])
        self.pos += 2
        return _Next.CONTINUE

    def single_quoted(self, out: list[str], _depth: int, _command: Command) -> _Next:
        end = self.text.find("'", self.pos + 1)
        if end == -1:
            return _Next.FAIL
        out.append(self.text[self.pos + 1 : end])
        self.pos = end + 1
        return _Next.CONTINUE

    def quoted(self, out: list[str], depth: int, command: Command) -> _Next:
        return _Next.CONTINUE if self.double_quoted(out, depth, command) else _Next.FAIL

    def parameter(self, out: list[str], _depth: int, _command: Command) -> _Next:
        """`${…}` kept as text; any other `$` is a plain character."""
        if not self.text.startswith("${", self.pos):
            return self.keep(out)
        end = self.text.find("}", self.pos)
        if end == -1:
            return _Next.FAIL
        out.append(self.text[self.pos : end + 1])
        self.pos = end + 1
        return _Next.CONTINUE

    def expansion(self, out: list[str], depth: int, command: Command) -> _Next | None:
        """`$((…))`, `$(…)` or a backtick substitution at `pos`, else None."""
        text = self.text
        if text.startswith("$((", self.pos):
            return _Next.CONTINUE if self.balanced(out, 1) else _Next.FAIL
        if text.startswith("$(", self.pos):
            read = self.command_substitution(out, depth, command)
            return _Next.CONTINUE if read else _Next.STOP
        if text.startswith("`", self.pos):
            return _Next.CONTINUE if self.backticks(out, depth, command) else _Next.FAIL
        return None

    def command_substitution(self, out: list[str], depth: int, command: Command) -> bool:
        start = self.pos
        before = len(self.script.commands)
        self.pos += 2
        self.commands(depth + 1, ")")
        command.substituted += self.script.commands[before:]
        out.append(self.text[start : self.pos])
        return self.script.complete

    def backticks(self, out: list[str], depth: int, command: Command) -> bool:
        end = self.pos + 1
        while end < len(self.text) and self.text[end] != "`":
            end += 2 if self.text[end] == "\\" else 1
        if end >= len(self.text):
            return False
        self.nested(command, self.text[self.pos + 1 : end], self.pos + 1, depth, True)
        out.append(self.text[self.pos : end + 1])
        self.pos = end + 1
        return True

    def balanced(self, out: list[str], skip: int) -> bool:
        """`$((…))` arithmetic: kept as text."""
        level, i = 0, self.pos + skip
        while i < len(self.text):
            level += {"(": 1, ")": -1}.get(self.text[i], 0)
            i += 1
            if level == 0:
                break
        if level:
            return False
        out.append(self.text[self.pos : i])
        self.pos = i
        return True

    def double_quoted(self, out: list[str], depth: int, command: Command) -> bool:
        text = self.text
        self.pos += 1
        while self.pos < len(text):
            if text[self.pos] == '"':
                self.pos += 1
                return True
            if plain := QUOTED_PLAIN.match(text, self.pos):
                out.append(plain[0])
                self.pos = plain.end()
                continue
            if not self.quoted_part(out, depth, command):
                return False
        return False

    def quoted_part(self, out: list[str], depth: int, command: Command) -> bool:
        """Read one escape, expansion or character inside double quotes into `out`."""
        text = self.text
        c = text[self.pos]
        if c == "\\" and self.pos + 1 < len(text):
            nxt = text[self.pos + 1]
            if nxt != "\n":
                out.append(nxt if nxt in '\\$`"' else c + nxt)
            self.pos += 2
            return True
        # In an `sh -c` string, `$` and backticks are the script's own.
        expansion = None if self.literal else self.expansion(out, depth, command)
        if expansion is None:
            self.keep(out)
            return True
        return expansion is _Next.CONTINUE


class _Next(Enum):
    """What `word` does after reading one part."""

    CONTINUE = auto()
    FAIL = auto()  # unreadable: the script is incomplete
    STOP = auto()  # a substitution already failed: stop where it did


# Word parts by their first character (anything else is a plain character).
WORD_PARTS: dict[str, Callable[[_Reader, list[str], int, Command], _Next]] = {
    "\\": _Reader.escaped,
    "'": _Reader.single_quoted,
    '"': _Reader.quoted,
    "$": _Reader.parameter,
}


@dataclass(slots=True)
class _Pending:
    """The command `commands` is reading, where it started, and its pipeline."""

    pipeline: int
    start: int
    command: Command = field(init=False)

    def __post_init__(self) -> None:
        self.command = Command(pipeline=self.pipeline)


# Commands whose heredoc body is written out verbatim (to their redirect or tee target).
FILE_WRITERS = frozenset({"cat", "tee"})


def unused_file_bodies(script: Script) -> list[Span]:
    """The written heredoc bodies (`file_bodies`) whose file no other command in the script
    names afterwards: file content, not something the script then runs or uses."""
    out = []
    for span, target in zip(script.file_bodies, script.file_targets, strict=True):
        used = any(
            any(_names_file(w, target) for w in c.words)
            for c in script.commands
            if not (c.name in FILE_WRITERS and any(t == target for t, _ in writes(c)))
        )
        if not used:
            out.append(span)
    return out


def _names_file(word: str, target: str) -> bool:
    """`word` is the written file `target`: the same path, or its file name."""
    base = target.rsplit("/", 1)[-1]
    return word in {target, base} or word.endswith("/" + base)


def runs_script(command: Command) -> bool:
    """`command` is a shell whose last word is `-c` (the next word is a script)."""
    return bool(command.words and SHELL_C.fullmatch(command.words[-1]) and command.name in SHELLS)


# --- facts callers use -----------------------------------------------------------------


def _operands(
    argv: list[tuple[str, Span]], values: frozenset[str] = frozenset()
) -> list[tuple[str, Span]]:
    """Non-option arguments after the command name, skipping the values of `values`
    options (everything after `--` is an operand)."""
    out: list[tuple[str, Span]] = []
    options, skip = True, False
    for word, span in argv[1:]:
        if skip:
            skip = False
        elif options and word == "--":
            options = False
        elif options and word.startswith("-") and word != "-":
            skip = word in values
        else:
            out.append((word, span))
    return out


# Options whose value is a separate word (so it isn't taken for a file).
COPY_VALUES = frozenset(
    {
        "-t",
        "--target-directory",
        "-S",
        "--suffix",
        "-m",
        "--mode",
        "-o",
        "--owner",
        "-g",
        "--group",
        "-e",
        "--rsh",
    }
)
SED_SCRIPT = frozenset({"-e", "--expression", "-f", "--file"})


def writes(command: Command) -> list[tuple[str, Span]]:
    """Files `command` writes: redirection targets, and the destinations of tee, touch,
    cp, mv, install, rsync, ln, dd `of=` and `sed -i`."""
    out = [
        (target, span)
        for op, target, span in command.redirects
        if op in WRITE_OPS and not (op == ">&" and (target.isdigit() or target == "-"))
    ]
    written = ARGUMENT_WRITES.get(command.name or "")
    return out + written(command.argv()) if written else out


# A source and a destination.
MIN_COPY_OPERANDS = 2


def _copy_destination(argv: list[tuple[str, Span]]) -> list[tuple[str, Span]]:
    words = argv[1:]
    for i, (word, span) in enumerate(words):
        if word in ("-t", "--target-directory") and i + 1 < len(words):
            return [words[i + 1]]
        if word.startswith("--target-directory="):
            return [(word.split("=", 1)[1], span)]
    operands = _operands(argv, COPY_VALUES)
    return operands[-1:] if len(operands) >= MIN_COPY_OPERANDS else []


def _dd_output(argv: list[tuple[str, Span]]) -> list[tuple[str, Span]]:
    return [(w[3:], s) for w, s in argv[1:] if w.startswith("of=")]


def _sed_in_place(argv: list[tuple[str, Span]]) -> list[tuple[str, Span]]:
    if not any(w.startswith(("-i", "--in-place")) for w, _ in argv[1:]):
        return []
    operands = _operands(argv, SED_SCRIPT)
    scripted = any(w.split("=", 1)[0] in SED_SCRIPT for w, _ in argv[1:])
    return operands if scripted else operands[1:]


# Files a command names as its output, by command name.
ARGUMENT_WRITES: dict[str, Callable[[list[tuple[str, Span]]], list[tuple[str, Span]]]] = {
    "tee": _operands,
    "touch": _operands,
    **dict.fromkeys(("cp", "mv", "install", "rsync", "ln"), _copy_destination),
    "dd": _dd_output,
    "sed": _sed_in_place,
}


def later_in_pipeline(script: Script) -> dict[int, list[Command]]:
    """For each command (by id), the commands its output is piped into."""
    by_pipeline: dict[int, list[Command]] = {}
    for command in sorted(script.commands, key=lambda c: c.span[0]):
        by_pipeline.setdefault(command.pipeline, []).append(command)
    return {
        id(command): members[i + 1 :]
        for members in by_pipeline.values()
        for i, command in enumerate(members)
    }
