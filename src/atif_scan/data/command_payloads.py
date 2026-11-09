"""Bounded proof of inert local command payloads, without executing any text.

Everything not proved inert remains visible to callers' original predicates.
Uncertain regions are separate: a parser gap is not a clean command.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from . import shell
from .python_payloads import python_payloads
from .python_units import PYTHON, PYTHON_FLAGS


@dataclass(frozen=True)
class CommandPayloads:
    inert: tuple[shell.Span, ...] = ()
    uncertain: tuple[shell.Span, ...] = ()
    complete: bool = True


SED_PARTS = 2


def _sed_substitution(expression: str) -> bool:
    """Only a single substitution with non-executing flags; never sed's e command."""
    expression = re.sub(r"^\d+", "", expression)
    if len(expression) < SED_PARTS or expression[0] != "s":
        return False
    delimiter = expression[1]
    if delimiter.isalnum() or delimiter in {"\\", "\n"}:
        return False
    index, parts = 2, 0
    while index < len(expression) and parts < SED_PARTS:
        if expression[index] == "\\":
            index += 2
            continue
        parts += expression[index] == delimiter
        index += 1
    return parts == SED_PARTS and re.fullmatch(r"[gIpP0-9]*", expression[index:]) is not None


def _sed_expressions(command: shell.Command) -> list[tuple[str, shell.Span]]:
    args = command.argv()[1:]
    expressions = []
    take_expression = False
    for word, span in args:
        if take_expression:
            expressions.append((word, span))
            take_expression = False
        elif word in {"-e", "--expression"}:
            take_expression = True
        elif word in {"-f", "--file"} or word.startswith(("-f", "--file=")):
            return []  # External scripts can run commands.
        elif not word.startswith("-") and not expressions:
            expressions.append((word, span))
    return [] if take_expression else expressions


def _sed_payloads(command: shell.Command) -> list[shell.Span]:
    expressions = _sed_expressions(command)
    if (
        command.substituted
        or not expressions
        or not all(_sed_substitution(w) for w, _ in expressions)
    ):
        return []
    return [span for _, span in expressions]


def _local_reader(command: shell.Command) -> bool:
    if command.name in {"cat", "head", "tail", "grep", "egrep", "fgrep"}:
        return True
    if command.name == "rg":
        return not any(word == "--pre" or word.startswith("--pre=") for word in command.words)
    return False


def _benign_reader(script: shell.Script, command: shell.Command) -> bool:
    return _local_reader(command) and all(
        _local_reader(other) for other in script.commands if other.pipeline == command.pipeline
    )


def _dynamic_executor(command: shell.Command) -> bool:
    python = PYTHON.fullmatch(command.name or "") is not None
    executor = command.name in shell.SHELLS | {"source", ".", "eval", "exec"} or python
    computed = command.name == "eval" or (python and "-c" in command.words)
    computed |= any("$" in word or "`" in word for word in command.words)
    return executor and computed


def _later_uses(script: shell.Script, command: shell.Command, words: set[str]) -> bool:
    bases = {word.rsplit("/", 1)[-1] for word in words}
    return any(
        other.span[0] > command.span[0]
        and not _benign_reader(script, other)
        and (
            any(word in words or word.rsplit("/", 1)[-1] in bases for word in other.words)
            or _dynamic_executor(other)
        )
        for other in script.commands
    )


def _python_unit(command: shell.Command) -> tuple[str, shell.Span] | None:
    args = command.argv()[1:]
    while args and args[0][0] in PYTHON_FLAGS:
        args = args[1:]
    if len(args) > 1 and args[0][0] == "-c":
        return args[1]
    return None


def _python_stdin(command: shell.Command) -> bool:
    args = command.argv()[1:]
    return all(word in PYTHON_FLAGS | {"-"} for word, _ in args)


def _unit_payloads(source: str, span: shell.Span, exact: bool) -> CommandPayloads:
    read = python_payloads(source)
    if not exact:
        # Quoting/escapes in -c may change offsets. Whole unit is a safe locator;
        # only exclude it when every literal is proved inert.
        if read.uncertain or not read.complete:
            return CommandPayloads(uncertain=(span,), complete=read.complete)
        return CommandPayloads() if read.active else CommandPayloads(inert=(span,))
    start = span[0]
    return CommandPayloads(
        tuple((start + a, start + b) for a, b in read.inert),
        tuple((start + a, start + b) for a, b in read.uncertain),
        read.complete,
    )


def _body_input_uncertain(text: str, owner: shell.Command, source: str) -> bool:
    inputs = [op for op, _, _ in owner.redirects if op in {"<", "<<", "<<-", "<<<"}]
    quoted = any(
        op in {"<<", "<<-"} and any(c in text[a:b] for c in "'\\\"\\\\")
        for op, _, (a, b) in owner.redirects
    )
    return len(inputs) != 1 or (not quoted and ("$(" in source or "`" in source))


def _python_payloads(
    text: str, script: shell.Script, owner: shell.Command, span: shell.Span, exact: bool
) -> CommandPayloads:
    literals = set(re.findall(r"""['"]([^'"]+)['"]""", text))
    read = _unit_payloads(text, span, exact=exact)
    if _later_uses(script, owner, literals):
        return CommandPayloads(uncertain=(*read.inert, *read.uncertain), complete=read.complete)
    return read


def _body_payloads(text: str, script: shell.Script) -> list[CommandPayloads]:
    reads = []
    for span, owner in zip(script.bodies, script.body_commands, strict=True):
        source = text[span[0] : span[1]]
        if _body_input_uncertain(text, owner, source):
            reads.append(CommandPayloads(uncertain=(span,)))
        elif span in script.file_bodies:
            targets = {target for target, _ in shell.writes(owner)}
            if not _later_uses(script, owner, targets):
                reads.append(CommandPayloads(inert=(span,)))
        elif PYTHON.fullmatch(owner.name or "") and _python_stdin(owner):
            reads.append(_python_payloads(source, script, owner, span, exact=True))
        else:
            reads.append(CommandPayloads(uncertain=(span,)))
    return reads


def _inline_payloads(text: str, script: shell.Script, command: shell.Command) -> CommandPayloads:
    unit = _python_unit(command)
    if unit is None:
        return CommandPayloads()
    source, span = unit
    raw = text[span[0] : span[1]]
    exact = raw == source or (len(raw) > 1 and raw[0] in "'\\\"" and raw[1:-1] == source)
    inner = (span[0] + 1, span[1] - 1) if exact and raw != source else span
    return _python_payloads(source, script, command, inner, exact=exact)


def _with_following(script: shell.Script, following: str, offset: int) -> shell.Script:
    """Later surfaces are use evidence only, never payloads of the current surface."""
    later = shell.parse(following)
    pipeline = max((command.pipeline for command in script.commands), default=0) + 1
    commands = [
        replace(
            command,
            span=(offset + command.span[0], offset + command.span[1]),
            pipeline=pipeline + command.pipeline,
        )
        for command in later.commands
    ]
    return replace(
        script, commands=[*script.commands, *commands], complete=script.complete and later.complete
    )


def command_payloads(text: str, following: str = "") -> CommandPayloads:
    """Return only proven inert spans, plus explicitly unresolved command regions."""
    script = _with_following(shell.parse(text), following, len(text) + 1)
    reads = _body_payloads(text, script)
    for command in script.commands:
        if command.span[0] >= len(text):
            continue
        if command.name == "sed":
            writes = {target for target, _ in shell.writes(command)}
            spans = tuple(_sed_payloads(command))
            read = (
                CommandPayloads(uncertain=spans)
                if _later_uses(script, command, writes)
                else CommandPayloads(inert=spans)
            )
            reads.append(read)
        elif PYTHON.fullmatch(command.name or ""):
            reads.append(_inline_payloads(text, script, command))
    return CommandPayloads(
        tuple(span for read in reads for span in read.inert),
        tuple(span for read in reads for span in read.uncertain),
        script.complete and all(read.complete for read in reads),
    )
