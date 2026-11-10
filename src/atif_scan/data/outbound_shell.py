"""Bounded curl/wget and inline Python requests, not a general shell interpreter."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from . import shell

if TYPE_CHECKING:
    from .model import Locator
from .outbound_python import python_requests
from .outbound_types import MAX_CHARS, MAX_ITEMS, Carrier, Requests, Role, body, request, scalar

_INLINE_ARGS = 2
_PYTHON = frozenset({"python", "python3", "python2"})
_INERT = frozenset({"echo", "printf", "cat", "tee", "true", ":"})
_HTTP = frozenset({"curl", "wget"})
_BINDING = frozenset({"alias", "unalias", "eval", "source", ".", "function", "exec"})
_PYTHON_HTTP = re.compile(r"\b(?:requests|httpx|urllib\.request)\b")
_NO_VALUE = frozenset({"-s", "-S", "-sS", "-q", "-f", "-i", "-v", "--silent", "--show-error"})
_VALUE = {
    "-X": "method",
    "--request": "method",
    "--method": "method",
    "-H": "header",
    "--header": "header",
    "-d": "body",
    "--data": "body",
    "--data-raw": "body",
    "--data-binary": "body",
    "--data-urlencode": "body",
    "--json": "body",
    "--post-data": "body",
    "--body-data": "body",
    "--url": "target",
    "-o": "ignore",
    "--output": "ignore",
    "-O": "ignore",
    "--output-document": "ignore",
}
_CURL_OPTIONS = frozenset(
    {
        "-X",
        "--request",
        "-H",
        "--header",
        "-d",
        "--data",
        "--data-raw",
        "--data-binary",
        "--data-urlencode",
        "--json",
        "--url",
        "-o",
        "--output",
    }
)
_WGET_OPTIONS = frozenset(
    {
        "--method",
        "--header",
        "--post-data",
        "--body-data",
        "-O",
        "--output-document",
    }
)


def _scope_supported(script: shell.Script, text: str) -> bool:
    if not script.complete or len(script.commands) > MAX_ITEMS:
        return False
    for command in script.commands:
        if (
            not command.words
            or command.words[0] in _BINDING | shell.KEYWORDS
            or shell.ASSIGNMENT.match(command.words[0]) is not None
            or command.background
            or command.repeated
            or command.nested
            or command.substituted
        ):
            return False
    # Anything outside parsed words, redirections and heredocs must be a plain
    # separator/comment. This refuses pipelines, conditionals, grouping and scopes.
    spans = [span for c in script.commands for span in c.spans]
    spans += [span for c in script.commands for _, _, span in c.redirects]
    for start, end in script.bodies:
        closing_end = text.find("\n", end)
        spans.append((start, len(text) if closing_end < 0 else closing_end + 1))
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    residue = re.sub(r"#[^\n]*", "", "".join(chars))
    return not residue.strip(" \t\r\n;")


def _option(words: list[str], index: int) -> tuple[str, str, int]:
    word = words[index]
    name, sep, value = word.partition("=") if word.startswith("--") else (word, "", "")
    if name in _NO_VALUE and (words[0] == "curl" or name in {"-q", "-S", "-v"}):
        return "ignore", "", index + 1
    if words[0] == "curl" and name in {"-I", "--head", "-O"}:
        return ("ignore", "", index + 1) if name == "-O" else ("method", "HEAD", index + 1)
    allowed = _CURL_OPTIONS if words[0] == "curl" else _WGET_OPTIONS
    if name not in allowed:
        raise ValueError("unsupported_option")
    if not sep:
        if index + 1 >= len(words):
            raise ValueError("missing_option_value")
        value = words[index + 1]
    if name == "--data-urlencode" and "@" in value and "=" not in value.split("@", 1)[0]:
        raise ValueError("file_body")
    return _VALUE[name], scalar(value), index + (1 if sep else 2)


def _header(value: str) -> Carrier:
    name, sep, content = value.partition(":")
    if not sep or not name.strip() or value.startswith("@"):
        raise ValueError("unsupported_header")
    return Carrier(Role.HEADER, name.strip(), content.strip())


def _http_options(words: list[str]) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {
        "target": [],
        "method": [],
        "header": [],
        "body": [],
        "ignore": [],
    }
    index = 1
    while index < len(words):
        if words[index].startswith("-"):
            kind, value, index = _option(words, index)
        else:
            kind, value, index = "target", scalar(words[index]), index + 1
        options[kind].append(value)
    return options


def _http_command(command: shell.Command, at: Locator) -> Requests:
    if command.redirects or len(command.words) > MAX_ITEMS:
        return Requests(complete=False)
    options = _http_options(command.words)
    targets = options["target"]
    methods = options["method"]
    if len(targets) != 1 or len(methods) > 1:
        raise ValueError("ambiguous_target_or_method")
    method = methods[0].upper() if methods else ("POST" if options["body"] else "GET")
    carriers = [_header(value) for value in options["header"]]
    carriers.extend(carrier for value in options["body"] for carrier in body(value))
    return Requests((request(at, targets[0], method, carriers, "shell_literal"),))


def _python_command(
    command: shell.Command, script: shell.Script, text: str, at: Locator
) -> Requests:
    args = command.words[1:]
    if len(args) == _INLINE_ARGS and args[0] == "-c" and not command.redirects:
        return python_requests(scalar(args[1]), at)
    bodies = [
        span
        for span, owner in zip(script.bodies, script.body_commands, strict=True)
        if owner is command
    ]
    # Only a quoted heredoc delimiter prevents shell expansion of Python text.
    if args in ([], ["-"]) and len(bodies) == 1 and len(command.redirects) == 1:
        op, _, (start, end) = command.redirects[0]
        if op == "<<" and re.fullmatch(r"<<\s*(['\"]).+\1", text[start:end]):
            a, b = bodies[0]
            return python_requests(text[a:b], at)
    return Requests(complete=False)


def _lookup(command: shell.Command) -> bool:
    return command.words[:2] in (["command", "-v"], ["command", "-V"])


def _python_candidate(command: shell.Command, script: shell.Script, text: str) -> bool:
    if command.name not in _PYTHON:
        return False
    words = " ".join(command.words[1:])
    if _PYTHON_HTTP.search(words) or "$" in words or "`" in words:
        return True
    return any(
        owner is command and _PYTHON_HTTP.search(text[a:b])
        for (a, b), owner in zip(script.bodies, script.body_commands, strict=True)
    )


def _candidate(script: shell.Script, text: str) -> bool:
    return any(
        not _lookup(command)
        and (
            command.name in _HTTP
            or (command.words and "$" in command.words[0])
            or _python_candidate(command, script, text)
        )
        for command in script.commands
    )


def shell_requests(text: str, at: Locator) -> Requests:
    """Complete only for the supported request vocabulary, not all shell networking.

    Unrelated ordinary commands do not create request gaps. A relevant request in
    unsupported shell scope, or an unparseable command, does remain unknown.
    """
    if len(text) > MAX_CHARS:
        return Requests(complete=False)
    script = shell.parse(text)
    if script.complete and not _candidate(script, text):
        return Requests()
    if not _scope_supported(script, text):
        return Requests(complete=False)
    return _script_requests(script, text, at)


def _script_requests(script: shell.Script, text: str, at: Locator) -> Requests:
    facts = []
    complete = True
    for command in script.commands:
        place = replace(at, span=command.span)
        try:
            if command.name in _HTTP and not _lookup(command):
                parsed = _http_command(command, place)
            elif _python_candidate(command, script, text):
                parsed = _python_command(command, script, text, place)
            else:
                parsed = Requests()
            facts.extend(parsed.requests)
            complete &= parsed.complete
        except (ValueError, UnicodeError, RecursionError):
            complete = False
    return Requests(tuple(facts), complete)
