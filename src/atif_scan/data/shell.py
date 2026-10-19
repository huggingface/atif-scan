"""Static reader for shell command text (stdlib only; nothing is ever executed).

`parse(text)` splits a shell tool's command into simple commands: their words (quotes
removed), redirections, pipeline membership, background/repeated scopes, and nested commands
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
    # Syntactic scopes, not proof of runtime duration/iterations. False means no
    # recognized scope, not proof that execution is foreground or bounded.
    background: bool = False
    repeated: bool = False

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
    body_commands: list[Command] = field(default_factory=list)  # owners, parallel to bodies
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
    _propagate_scope(script.commands)
    return script


def _mark_scope(
    commands: list[Command], *, background: bool = False, repeated: bool = False
) -> None:
    for command in commands:
        command.background |= background
        command.repeated |= repeated


def _propagate_scope(commands: list[Command]) -> None:
    """Children can precede their parent, or be read later (shell heredocs)."""
    pending = list(commands)
    while pending:
        parent = pending.pop()
        for child in (*parent.nested, *parent.substituted):
            background = parent.background and not child.background
            repeated = parent.repeated and not child.repeated
            if background or repeated:
                child.background |= background
                child.repeated |= repeated
                pending.append(child)


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

    def commands(self, depth: int, stop: str | None, pipeline: int | None = None) -> bool:
        """Read a list, returning whether its closing token (or EOF) was reached."""
        if depth > MAX_DEPTH:
            self.fail()
            return False
        pending = _Pending(next(self.pipelines) if pipeline is None else pipeline, self.pos)
        while self.pos < len(self.text):
            if not self.command_part(pending, depth, stop):
                return self.script.complete
        self.finish(pending, True)
        if stop is not None or pending.needs_command:
            self.fail()
        return self.script.complete

    def finish(self, pending: _Pending, new_pipeline: bool) -> None:
        """Record the command being read (if it has anything) and start the next one."""
        command = pending.command
        if command.words or command.redirects:
            command.span = self.span(pending.start, self.pos)
            self.script.commands.append(command)
            pending.members.append(command)
        if new_pipeline:
            pending.pipeline = next(self.pipelines)
            pending.members = []
        pending.grouped = False
        pending.command = Command(pipeline=pending.pipeline)
        pending.start = self.pos

    def command_part(self, pending: _Pending, depth: int, stop: str | None) -> bool:
        """Read one blank, separator or token at `pos`; False when `commands` is done."""
        if self.blank(pending) or self.separator(pending, depth):
            return True
        if self.at_stop(pending, stop):
            if pending.needs_command:
                self.fail()
            else:
                self.finish(pending, True)
                self.pos += len(stop or "")
            return False
        read = self.compound_part(pending, depth)
        pending.needs_command = False
        return read

    def keyword(self, word: str) -> bool:
        """An unquoted, whole reserved word at the current token boundary."""
        end = self.pos + len(word)
        return self.text.startswith(word, self.pos) and (
            end == len(self.text) or self.text[end] in WORD_END
        )

    def at_stop(self, pending: _Pending, stop: str | None) -> bool:
        if stop is None:
            return False
        if stop == ")":
            return self.text[self.pos] == stop
        return pending.empty and self.keyword(stop)

    def compound_part(self, pending: _Pending, depth: int) -> bool:
        """Keep compound commands together for a following pipeline or `&`."""
        if self.compound_start(pending):
            if self.text[self.pos] == "(" or self.keyword("{"):
                return self.group(pending, depth)
            for kind in ("while", "until", "for"):
                if self.keyword(kind):
                    return self.loop(pending, depth, kind)
        if self.misplaced_token(pending):
            self.fail()
            return False
        return self.token(pending.command, depth)

    def compound_start(self, pending: _Pending) -> bool:
        """Reserved prefixes can introduce compounds, but quoted words cannot."""
        command = pending.command
        return not (pending.grouped or command.redirects) and all(
            word in {"if", "then", "else", "elif", "!", "time"}
            and self.text[start - self.base : end - self.base] == word
            for word, (start, end) in zip(command.words, command.spans, strict=True)
        )

    def misplaced_token(self, pending: _Pending) -> bool:
        if pending.empty and any(self.keyword(word) for word in ("do", "done", "}")):
            return True
        return pending.grouped and REDIRECT.match(self.text, self.pos) is None

    def group(self, pending: _Pending, depth: int) -> bool:
        stop = ")" if self.text[self.pos] == "(" else "}"
        before = len(self.script.commands)
        self.pos += 1
        # The group's first pipeline shares its surrounding pipe, not substitutions.
        closed = self.commands(depth + 1, stop, pending.pipeline)
        pending.command = Command(pipeline=pending.pipeline)
        pending.members.extend(self.script.commands[before:])
        pending.grouped = True
        return closed

    def loop(self, pending: _Pending, depth: int, kind: str) -> bool:
        """Structure loop headers/bodies; only exact constant headers are unbounded."""
        before = len(self.script.commands)
        self.pos += len(kind)
        repeated = self.loop_header(depth, kind)
        if repeated is None:
            self.fail()
            return False
        body = len(self.script.commands)
        closed = self.commands(depth + 1, "done")
        members = self.script.commands[before:]
        # A following redirect belongs to the loop, even though it has no command name.
        pending.command = Command(pipeline=pending.pipeline, repeated=closed and repeated)
        if closed:
            _mark_scope(self.script.commands[body:], repeated=repeated)
        pending.members.extend(members)
        pending.grouped = True
        return closed

    def loop_header(self, depth: int, kind: str) -> bool | None:
        """Whether the header is unbounded; None means unsupported/incomplete."""
        if kind == "for":
            start = self.pos - len(kind)
            self.header_space()
            if self.text.startswith("((", self.pos):
                return True if self.arithmetic_loop_header() else None
            return False if self.word_loop_header(depth, start) else None
        before = len(self.script.commands)
        if not self.commands(depth + 1, "do") or len(self.script.commands) == before:
            return None
        return self.constant_condition(self.script.commands[before:], kind)

    def word_loop_header(self, depth: int, start: int) -> bool:
        """Keep the existing `for name [in words]` header and its nested expansions."""
        pending = _Pending(next(self.pipelines), start)
        command = pending.command
        command.words.append("for")
        command.spans.append(self.span(start, start + len("for")))
        if not self.word_loop_words(pending, depth):
            return False
        if self.word_loop_continuation(command) and not self.word_loop_words(pending, depth):
            return False
        if not self.valid_word_loop(command):
            return False
        self.finish(pending, True)
        self.pos += 1  # the required semicolon/newline before `do`
        return self.loop_do()

    def word_loop_continuation(self, command: Command) -> bool:
        """Only a newline after the variable may precede the optional `in` clause."""
        if len(command.words[1:]) != 1 or not self.text.startswith("\n", self.pos):
            return False
        start = self.pos
        self.header_space()
        if self.keyword("in"):
            return True
        self.pos = start
        return False

    def word_loop_words(self, pending: _Pending, depth: int) -> bool:
        command = pending.command
        while self.pos < len(self.text) and self.text[self.pos] not in ";\n":
            if self.blank(pending):
                continue
            if self.text[self.pos] in WORD_END:
                return False
            word = self.word(depth, command, operand=False)
            if word is None:
                return False
            command.words.append(word[0])
            command.spans.append(word[1])
        return True

    def valid_word_loop(self, command: Command) -> bool:
        words = command.words[1:]
        if not words or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", words[0]) is None:
            return False
        if words[1:] and words[1] != "in":
            return False
        return all(
            self.text[start - self.base : end - self.base] == word
            for word, (start, end) in zip(words[:2], command.spans[1:3], strict=True)
        )

    def constant_condition(self, commands: list[Command], kind: str) -> bool:
        if len(commands) != 1:
            return False
        command = commands[0]
        expected = {("false",)} if kind == "until" else {("true",), (":",), ("[", "1", "]")}
        literal = all(
            self.text[start - self.base : end - self.base] == word
            for word, (start, end) in zip(command.words, command.spans, strict=True)
        )
        return literal and not command.redirects and tuple(command.words) in expected

    def header_space(self, *, comments: bool = True) -> None:
        """Layout in the deliberately narrow `for ((;;))` header."""
        pending = _Pending(next(self.pipelines), self.pos)
        while self.pos < len(self.text):
            if self.text[self.pos] == "\n":
                self.pos += 1
            elif (not comments and self.text[self.pos] == "#") or not self.blank(pending):
                break

    def arithmetic_loop_header(self) -> bool:
        for token in ("((", ";", ";", "))"):
            self.header_space(comments=False)
            if not self.text.startswith(token, self.pos):
                return False
            self.pos += len(token)
        self.header_space()
        if self.text.startswith(";", self.pos):
            self.pos += 1
        return self.loop_do()

    def loop_do(self) -> bool:
        self.header_space()
        if not self.keyword("do"):
            return False
        self.pos += len("do")
        return True

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
            self.newline(pending, depth)
            return True
        op = OPERATOR.match(text, self.pos)
        if op is None or text.startswith("&>", self.pos):
            return False
        if pending.needs_command or op[0] == ";;":
            self.fail()
            return True
        self.pos = op.end()
        self.finish(pending, False)
        self.end_operator(pending, op[0])
        return True

    def newline(self, pending: _Pending, depth: int) -> None:
        if not pending.needs_command:
            self.finish(pending, True)
        self.pos += 1
        self.read_heredocs(depth)
        pending.start = self.pos

    def end_operator(self, pending: _Pending, op: str) -> None:
        if op == "&":
            _mark_scope(pending.members, background=True)
        pending.needs_command = op in ("|", "|&", "&&", "||")
        if op not in ("|", "|&"):
            # An asynchronous AND/OR list also belongs to the same `&`.
            if op not in ("&&", "||"):
                pending.members = []
            pending.pipeline = next(self.pipelines)
            pending.command.pipeline = pending.pipeline

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
            self.fail()  # a function, case pattern, or misplaced parenthesis
            return False
        word = self.word(depth, command)
        if word is not None:
            command.words.append(word[0])
            command.spans.append(word[1])
        return word is not None

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
                self.script.body_commands.append(command)
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
        # Spans are already absolute (`base`). The four lists are parallel pairs: merge
        # all of them, or a heredoc nested in `bash <<EOF` misaligns bodies and owners.
        self.script.bodies += inner.bodies
        self.script.body_commands += inner.body_commands
        self.script.file_bodies += inner.file_bodies
        self.script.file_targets += inner.file_targets
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
    members: list[Command] = field(default_factory=list)  # current AND/OR list, groups included
    grouped: bool = False
    needs_command: bool = False  # after a pipeline or AND/OR operator

    @property
    def empty(self) -> bool:
        return not (self.command.words or self.command.redirects or self.grouped)

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


def operands(
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
    paths = operands(argv, COPY_VALUES)
    return paths[-1:] if len(paths) >= MIN_COPY_OPERANDS else []


def _dd_output(argv: list[tuple[str, Span]]) -> list[tuple[str, Span]]:
    return [(w[3:], s) for w, s in argv[1:] if w.startswith("of=")]


def _sed_in_place(argv: list[tuple[str, Span]]) -> list[tuple[str, Span]]:
    if not any(w.startswith(("-i", "--in-place")) for w, _ in argv[1:]):
        return []
    paths = operands(argv, SED_SCRIPT)
    scripted = any(w.split("=", 1)[0] in SED_SCRIPT for w, _ in argv[1:])
    return paths if scripted else paths[1:]


# Files a command names as its output, by command name.
ARGUMENT_WRITES: dict[str, Callable[[list[tuple[str, Span]]], list[tuple[str, Span]]]] = {
    "tee": operands,
    "touch": operands,
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
