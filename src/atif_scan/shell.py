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
from itertools import count

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
SHELL_C = re.compile(r"-[a-z]*c[a-z]*")  # -c, -lc, -ec, …
MAX_DEPTH = 32


@dataclass
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


@dataclass
class Script:
    commands: list[Command] = field(default_factory=list)  # every command, nested included
    bodies: list[Span] = field(default_factory=list)  # heredoc contents (not commands)
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

    def __init__(self, text: str, script: Script, pipelines: count, base: int = 0) -> None:
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
        text = self.text
        pipeline = next(self.pipelines)
        command = Command(pipeline=pipeline)
        start = self.pos

        def finish(new_pipeline: bool) -> None:
            nonlocal command, pipeline, start
            if command.words or command.redirects:
                command.span = self.span(start, self.pos)
                self.script.commands.append(command)
            if new_pipeline:
                pipeline = next(self.pipelines)
            command = Command(pipeline=pipeline)
            start = self.pos

        while self.pos < len(text):
            c = text[self.pos]
            if c in " \t":
                self.pos += 1
                if not command.words and not command.redirects:
                    start = self.pos
                continue
            if stop is not None and c == stop:
                finish(True)
                self.pos += 1
                return None
            if text.startswith("\\\n", self.pos):
                self.pos += 2
                continue
            if c == "\n":
                finish(True)
                self.pos += 1
                self.read_heredocs(depth)
                start = self.pos
                continue
            if c == "#":
                end = text.find("\n", self.pos)
                self.pos = len(text) if end == -1 else end
                continue
            if c in "<>" and text.startswith("(", self.pos + 1):
                self.substitution(command, depth, self.pos, self.pos + 2)
                continue
            op = OPERATOR.match(text, self.pos)
            if op and not text.startswith("&>", self.pos):
                self.pos = op.end()
                finish(op[0] not in ("|", "|&"))
                continue
            # At a token start, so a leading fd number (`2>`) belongs to the redirection.
            redirect = REDIRECT.match(text, self.pos)
            if redirect:
                self.pos = redirect.end()
                self.redirect(command, redirect[0].lstrip("0123456789"), redirect.start(), depth)
                continue
            if c == "(":
                self.pos += 1
                if command.words:  # `f() {…}` or a stray paren: not something we structure
                    return self.fail()
                self.commands(depth + 1, ")")
                continue
            if c == ")":  # a `case` pattern or unbalanced input
                return self.fail()
            word = self.word(depth, command)
            if word is None:
                return None
            command.words.append(word[0])
            command.spans.append(word[1])
        finish(True)
        if stop is not None:
            self.fail()
        return None

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
        script = bool(
            operand
            and start < len(text)
            and command.words
            and SHELL_C.fullmatch(command.words[-1])
            and command.name in SHELLS
        )
        self.literal, outer = script, self.literal
        while self.pos < len(text) and text[self.pos] not in WORD_END:
            c = text[self.pos]
            if c == "\\":
                if text.startswith("\n", self.pos + 1):
                    self.pos += 2
                    continue
                out.append(text[self.pos + 1 : self.pos + 2])
                self.pos += 2
            elif c == "'":
                end = text.find("'", self.pos + 1)
                if end == -1:
                    return self.fail()
                out.append(text[self.pos + 1 : end])
                self.pos = end + 1
            elif c == '"':
                if not self.double_quoted(out, depth, command):
                    return self.fail()
            elif self.literal and c in "$`":
                out.append(c)
                self.pos += 1
            elif c == "$" and text.startswith("$((", self.pos):
                if not self.balanced(out, 1):
                    return self.fail()
            elif c == "$" and text.startswith("$(", self.pos):
                if not self.command_substitution(out, depth, command):
                    return None
            elif c == "$" and text.startswith("${", self.pos):
                end = text.find("}", self.pos)
                if end == -1:
                    return self.fail()
                out.append(text[self.pos : end + 1])
                self.pos = end + 1
            elif c == "`":
                if not self.backticks(out, depth, command):
                    return self.fail()
            else:
                out.append(c)
                self.pos += 1
        word = "".join(out)
        self.literal = outer
        if script:
            self.nested(command, word, start + (1 if text[start] in "'\"" else 0), depth)
        return word, self.span(start, self.pos)

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
            c = text[self.pos]
            if c == '"':
                self.pos += 1
                return True
            if c == "\\" and self.pos + 1 < len(text):
                nxt = text[self.pos + 1]
                if nxt != "\n":
                    out.append(nxt if nxt in '\\$`"' else c + nxt)
                self.pos += 2
            elif self.literal and c in "$`":
                out.append(c)
                self.pos += 1
            elif text.startswith("$((", self.pos):
                if not self.balanced(out, 1):
                    return False
            elif text.startswith("$(", self.pos):
                if not self.command_substitution(out, depth, command):
                    return False
            elif c == "`":
                if not self.backticks(out, depth, command):
                    return False
            else:
                out.append(c)
                self.pos += 1
        return False


# --- facts callers use -----------------------------------------------------------------


def _operands(argv: list[tuple[str, Span]], values: frozenset[str] = frozenset()):
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
    argv = command.argv()
    name = command.name
    if name in ("tee", "touch"):
        out += _operands(argv)
    elif name in ("cp", "mv", "install", "rsync", "ln"):
        words = argv[1:]
        for i, (word, span) in enumerate(words):
            if word in ("-t", "--target-directory") and i + 1 < len(words):
                return out + [words[i + 1]]
            if word.startswith("--target-directory="):
                return out + [(word.split("=", 1)[1], span)]
        operands = _operands(argv, COPY_VALUES)
        if len(operands) >= 2:
            out.append(operands[-1])
    elif name == "dd":
        out += [(w[3:], s) for w, s in argv[1:] if w.startswith("of=")]
    elif name == "sed" and any(w.startswith(("-i", "--in-place")) for w, _ in argv[1:]):
        operands = _operands(argv, SED_SCRIPT)
        scripted = any(w.split("=", 1)[0] in SED_SCRIPT for w, _ in argv[1:])
        out += operands if scripted else operands[1:]
    return out


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
