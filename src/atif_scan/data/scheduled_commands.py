"""Static schedule installation inputs; private shell source, never executed or exported."""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import shell

_CRON_COLUMNS = 5
_CRON_ATOM = r"(?:\d+|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|sun|mon|tue|wed|thu|fri|sat)"
_CRON_FIELD = re.compile(rf"(?:\*|{_CRON_ATOM})(?:-{_CRON_ATOM})?(?:/\d+)?", re.IGNORECASE)
_MACROS = frozenset(
    {"@reboot", "@yearly", "@annually", "@monthly", "@weekly", "@daily", "@midnight", "@hourly"}
)
_LITERAL = re.compile(r"'[^'\\]*'|\"[^\"$`\\]*\"|[\w./:=,+@%-]+")


@dataclass(frozen=True)
class ScheduledCommands:
    """Recovered installed source; completeness is coverage, not execution success."""

    scripts: tuple[str, ...]
    complete: bool


def scheduled_commands(
    command: shell.Command, script: shell.Script, text: str
) -> ScheduledCommands | None:
    """Read an at/crontab installation from the same parse/text; inspections return None."""
    if command.name not in {"at", "crontab"}:
        return None
    options = _options(command)
    inspection = "lrdc" if command.name == "at" else "lreTV"
    if any(any(flag in option[1:] for flag in inspection) for option in options):
        return None
    external = (
        any("f" in option[1:] for option in options)
        if command.name == "at"
        else any(word != "-" for word, _ in shell.operands(command.argv(), frozenset({"-u"})))
    )
    source = None if external else _source(command, script, text)
    if source is None:
        return ScheduledCommands((), False)
    result = _cron(source) if command.name == "crontab" else ScheduledCommands((source,), True)
    return ScheduledCommands(result.scripts, result.complete and script.complete)


def _options(command: shell.Command) -> list[str]:
    options: list[str] = []
    skip = False
    for word, _ in command.argv()[1:]:
        if skip:
            skip = False
        elif word == "--":
            break
        elif word.startswith("-") and word != "-":
            options.append(word[:2] if word[1:2] in {"u", "q", "t", "f"} else word)
            skip = word in {"-u", "-q", "-t", "-f"}
    return options


def _source(command: shell.Command, script: shell.Script, text: str) -> str | None:
    bodies = [
        span
        for span, owner in zip(script.bodies, script.body_commands, strict=True)
        if owner is command
    ]
    if command.redirects:
        return _heredoc(command, bodies, text)
    prior = [
        c for c in script.commands if c.pipeline == command.pipeline and c.span[0] < command.span[0]
    ]
    producer = max(prior, key=lambda c: c.span[0], default=None)
    if producer is None or not producer.spans:
        return None
    if not re.fullmatch(r"\s*\|\s*", text[producer.spans[-1][1] : command.span[0]]):
        return None
    return _emitted(producer, text)


def _heredoc(command: shell.Command, bodies: list[shell.Span], text: str) -> str | None:
    if len(bodies) != 1 or len(command.redirects) != 1:
        return None
    op, delimiter, (start, end) = command.redirects[0]
    header = re.fullmatch(r"0?<<-?\s*(['\"]?)(\w+)\1", text[start:end])
    begin, finish = bodies[0]
    closing = text[finish:].splitlines()
    if not header or not closing or closing[0].lstrip("\t") != delimiter:
        return None
    body = text[begin:finish]
    if not header[1] and any(char in body for char in "$`\\"):
        return None
    return (
        "".join(line.lstrip("\t") for line in body.splitlines(keepends=True))
        if op == "<<-"
        else body
    )


def _emitted(command: shell.Command, text: str) -> str | None:
    if command.name not in {"echo", "printf"} or command.redirects:
        return None
    args = command.argv()[1:]
    separator = _printf_separator(args, text) if command.name == "printf" else " "
    if separator is None or (command.name == "echo" and args and args[0][0].startswith("-")):
        return None
    args = args[1:] if command.name == "printf" else args
    if any(not _LITERAL.fullmatch(text[start:end]) for _, (start, end) in args):
        return None
    source = separator.join(word for word, _ in args)
    return source + ("\n" if command.name == "echo" or separator == "\n" else "")


def _printf_separator(args: list[tuple[str, shell.Span]], text: str) -> str | None:
    if not args or args[0][0] not in {"%s", "%s\\n"}:
        return None
    fmt, (start, end) = args[0]
    if text[start:end] not in {fmt, f"'{fmt}'", f'"{fmt}"'}:
        return None
    return "\n" if fmt == "%s\\n" else ""


def _cron(source: str) -> ScheduledCommands:
    scripts: list[str] = []
    complete = True
    for line in map(str.strip, source.splitlines()):
        if not line or line.startswith("#") or re.match(r"[A-Za-z_]\w*\s*=", line):
            continue
        parts = line.split(None, 1 if line.startswith("@") else _CRON_COLUMNS)
        if not _cron_entry(parts) or "%" in parts[-1] or "\\" in parts[-1]:
            complete = False
        else:
            scripts.append(parts[-1])
    return ScheduledCommands(tuple(scripts), complete)


def _cron_entry(parts: list[str]) -> bool:
    if parts[0].startswith("@"):
        return bool(parts[1:]) and parts[0] in _MACROS
    return len(parts) == _CRON_COLUMNS + 1 and all(
        _CRON_FIELD.fullmatch(piece) for field in parts[:-1] for piece in field.split(",")
    )
