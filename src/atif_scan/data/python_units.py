"""Statically extract Python source units from supported shell forms; never execute them."""

from __future__ import annotations

import re

from . import shell

PYTHON = re.compile(r"python(?:[23](?:\.\d+)?)?")
PYTHON_FLAGS = frozenset({"-u", "-B", "-E", "-I", "-s", "-S", "-q"})


def _python_args(command: shell.Command) -> list[tuple[str, shell.Span]] | None:
    if PYTHON.fullmatch(command.name or "") is None:
        return None
    args = command.argv()[1:]
    while args and args[0][0] in PYTHON_FLAGS:
        args = args[1:]
    return args


def _stdin_python(command: shell.Command) -> bool:
    args = _python_args(command)
    return args is not None and (not args or args[0][0] == "-")


def shell_python_sources(text: str) -> tuple[list[str], bool]:
    """Literal Python -c, stdin heredocs and .py file heredocs, kept as separate units."""
    script = shell.parse(text)
    sources: list[str] = []
    complete = script.complete
    for command in script.commands:
        args = _python_args(command)
        if args and args[0][0] == "-c":
            if len(args) > 1:
                sources.append(args[1][0])
            else:
                complete = False
    files = dict(zip(script.file_bodies, script.file_targets, strict=True))
    for span, owner in zip(script.bodies, script.body_commands, strict=True):
        if _stdin_python(owner) or files.get(span, "").endswith(".py"):
            sources.append(text[span[0] : span[1]])
            complete &= span[1] < len(text)
    return sources, complete
