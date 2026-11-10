"""Offline code-search actions and prompt anchors. Query text is memory-only.

Supported current searchcode requests are literal JSON POST bodies in curl; MCP,
dynamic bodies, aliases and undocumented endpoints are deliberately not inferred.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlsplit

from . import jsonval, shell
from .model import Channel, Locator, Surface, Trace
from .pysource import parse_module
from .search_requests import (
    VALUE_OPTIONS,
    RequestRead,
    command_requests,
    literal_http_target,
    url_request,
)

MAX_TEXT = 512_000
MIN_NUMERIC_DIGITS = 6
MIN_IDENTIFIER = 8
MIN_COMPOUND = 16
MIN_SEPARATORS = 2
ENDPOINTS = {
    ("grep.app", "/api/search"): "grep",
    ("grep.app", "/search"): "grep",
    ("sourcegraph.com", "/search"): "sourcegraph",
    ("sourcegraph.com", "/search/stream"): "sourcegraph",
    ("sourcegraph.com", "/.api/search/stream"): "sourcegraph",
    ("searchcode.com", "/api/codesearch_I"): "searchcode_legacy",
    ("api.searchcode.com", "/api/v1/code_search"): "searchcode",
    ("api.github.com", "/search/code"): "github",
    ("github.com", "/search"): "github",
}


@dataclass(frozen=True)
class CodeSearchRequest:
    service: str
    at: Locator
    queries: tuple[str, ...] = field(repr=False)
    repository: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class CodeSearchRead:
    requests: tuple[CodeSearchRequest, ...] = ()
    complete: bool = True


def service_for(url: str) -> str | None:
    """Exact hosts and paths only, not URL-looking strings inside query/payloads."""
    if not literal_http_target(url):
        return None
    try:
        parts = urlsplit(url)
        if parts.username or parts.password or parts.port not in {None, 80, 443}:
            return None
        service = ENDPOINTS.get((parts.hostname, parts.path.rstrip("/")))
        params = dict(parse_qsl(parts.query, max_num_fields=4096))
        is_code = parts.hostname != "github.com" or params.get("type", "").lower() == "code"
        return service if is_code else None
    except ValueError:
        return None


def _option_value(option: str, value: str, targets: list[str], bodies: list[str]) -> None:
    if option == "--url":
        targets.append(value)
    elif option in {"-d", "--data", "--data-raw", "--data-binary", "--json"}:
        bodies.append(value)


def _curl_words(command: shell.Command) -> tuple[list[str], list[str], bool]:
    """Only documented literal options; unknown option values are not URL targets."""
    words = [word for word, _ in command.argv()[1:]]
    targets: list[str] = []
    bodies: list[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        option, sep, attached = word.partition("=")
        if option in VALUE_OPTIONS | {"--url", "--json", "-O", "--output-document"}:
            value = attached if sep else (words[i + 1] if i + 1 < len(words) else "")
            i += 1 if sep else 2
            if not value:
                return targets, bodies, False
            _option_value(option, value, targets, bodies)
        elif _switch(word, command.name):
            i += 1
        elif word.startswith("-"):
            return targets, bodies, False
        else:
            targets.append(word)
            i += 1
    return targets, bodies, all(literal_http_target(url) for url in targets)


def _switch(word: str, name: str | None) -> bool:
    return bool(
        word in {"--silent", "--show-error", "--fail", "--get", "--location", "--compressed"}
        or re.fullmatch(r"-[sfSLGq]+", word)
        or (name == "wget" and word in {"-qO-", "-O-"})
    )


def _post_body(body: str, at: Locator) -> CodeSearchRead:
    try:
        doc = jsonval.as_object(json.loads(body))
    except (ValueError, RecursionError):
        return CodeSearchRead(complete=False)
    query = jsonval.as_str(doc.get("query"))
    repo = jsonval.as_str(doc.get("repository"))
    if not query or not query.strip() or not repo or not repo.strip():
        return CodeSearchRead(complete=False)
    return CodeSearchRead((CodeSearchRequest("searchcode", at, (query,), repo),))


def _post_method(words: list[str]) -> bool:
    methods = [words[i + 1] for i, word in enumerate(words[:-1]) if word in {"-X", "--request"}] + [
        word.partition("=")[2] for word in words if word.startswith("--request=")
    ]
    return all(method.upper() == "POST" for method in methods) and not any(
        word in {"-G", "--get", "-I", "--head"} for word in words
    )


def _post_request(command: shell.Command, at: Locator) -> CodeSearchRead:
    """Read only a bounded literal curl JSON body, never file-backed data."""
    if command.name != "curl":
        return CodeSearchRead()
    targets, bodies, complete = _curl_words(command)
    if not any(service_for(url) == "searchcode" for url in targets):
        return CodeSearchRead(complete=complete)
    valid = (
        complete
        and len(targets) == 1
        and len(bodies) == 1
        and len(bodies[0]) <= MAX_TEXT
        and _post_method([word for word, _ in command.argv()[1:]])
    )
    return _post_body(bodies[0], at) if valid else CodeSearchRead(complete=False)


def _literal_argument(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, int, float, bool)) or node.value is None
    return isinstance(node, ast.Dict) and all(
        isinstance(k, ast.Constant) and isinstance(k.value, str) and isinstance(v, ast.Constant)
        for k, v in zip(node.keys, node.values, strict=True)
    )


def _http_expression(node: ast.expr, modules: set[str]) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    name = ast.unparse(node.func)
    values = [*node.args, *(kw.value for kw in node.keywords)]
    if not all(_literal_argument(value) for value in values):
        return False
    return any(
        name == f"{module}.{method}" for module in modules for method in ("get", "post", "urlopen")
    )


def _python_import(node: ast.Import, modules: set[str]) -> bool:
    valid = all(
        a.name in {"requests", "httpx", "urllib.request"} and not a.asname for a in node.names
    )
    if valid:
        modules.update(a.name for a in node.names)
    return valid


def _python_assignment(node: ast.Assign, modules: set[str], responses: set[str]) -> bool:
    if len(node.targets) != 1:
        return False
    target = node.targets[0]
    if not isinstance(target, ast.Name) or target.id in responses | {
        "requests",
        "httpx",
        "urllib",
        "print",
    }:
        return False
    responses.add(target.id)
    return _http_expression(node.value, modules)


def _python_print(node: ast.expr, modules: set[str], responses: set[str]) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return False
    if node.func.id != "print" or node.keywords or len(node.args) != 1:
        return False
    value = node.args[0]
    if not isinstance(value, ast.Attribute) or value.attr != "text":
        return False
    return _http_expression(value.value, modules) or (
        isinstance(value.value, ast.Name) and value.value.id in responses
    )


def _python_statement(node: ast.stmt, modules: set[str], responses: set[str]) -> bool:
    if isinstance(node, ast.Import):
        return _python_import(node, modules)
    if isinstance(node, ast.Assign):
        return _python_assignment(node, modules, responses)
    return isinstance(node, ast.Expr) and (
        _http_expression(node.value, modules) or _python_print(node.value, modules, responses)
    )


def _safe_python(code: str) -> bool:
    try:
        tree = parse_module(code)
    except (SyntaxError, ValueError, RecursionError):
        return False
    modules: set[str] = set()
    responses: set[str] = set()
    return all(_python_statement(node, modules, responses) for node in tree.body)


def _introspection(command: shell.Command) -> bool:
    """The shell reader unwraps command -v/-V; neither invokes the named program."""
    prefix = command.words[: len(command.words) - len(command.argv())]
    return any(
        word.rsplit("/", 1)[-1] == "command"
        and any(
            option in {"--help", "--version"} or bool(re.fullmatch(r"-[p]*[vV][p]*", option))
            for option in prefix[i + 1 :]
        )
        for i, word in enumerate(prefix)
    )


def _executable_requests(text: str, script: shell.Script) -> RequestRead:
    if not any(_introspection(command) for command in script.commands):
        return command_requests(text)
    if script.bodies:
        return RequestRead(complete=False)
    reads = [
        command_requests(shlex.join(command.words))
        for command in script.commands
        if not _introspection(command) and command.name in {"curl", "wget", "python", "python3"}
    ]
    return RequestRead(
        tuple(request for read in reads for request in read.requests),
        all(read.complete for read in reads),
    )


def _safe_commands(text: str) -> bool:
    script = shell.parse(text)
    if not script.complete or re.search(
        r"(?:\bfunction\s+|\b(?:curl|wget)\s*\(\s*\)|\balias\s+(?:curl|wget)=)",
        text,
    ):
        return False
    return all(_safe_command(command, script, text) for command in script.commands)


def _safe_command(command: shell.Command, script: shell.Script, text: str) -> bool:
    if _introspection(command) or command.name not in {"curl", "wget", "python", "python3"}:
        return True
    if command.name in {"curl", "wget"}:
        return _curl_words(command)[2]
    words = [word for word, _ in command.argv()[1:]]
    if "-c" in words:
        index = words.index("-c") + 1
        return index < len(words) and _safe_python(words[index])
    bodies = [
        text[start:end]
        for (start, end), owner in zip(script.bodies, script.body_commands, strict=True)
        if owner is command
    ]
    return len(bodies) == 1 and _safe_python(bodies[0])


def _command_code_search(surface: Surface) -> CodeSearchRead:
    text = surface.content.text
    script = shell.parse(text)
    parsed = _executable_requests(text, script)
    if not _safe_commands(text):
        return CodeSearchRead(complete=False)
    posts = [_post_request(c, surface.at) for c in script.commands if not _introspection(c)]
    if not parsed.complete and not any(p.requests for p in posts):
        return CodeSearchRead(complete=False)
    found = [
        CodeSearchRequest(service, surface.at, r.queries)
        for r in parsed.requests
        if (service := service_for(r.url)) is not None and service != "searchcode"
    ]
    found += [r for p in posts for r in p.requests]
    return CodeSearchRead(tuple(found), all(p.complete for p in posts))


def surface_code_search(surface: Surface) -> CodeSearchRead:
    if surface.at.channel not in {Channel.COMMAND, Channel.ARGUMENTS, Channel.URL}:
        return CodeSearchRead()
    expected_tool = "web_fetch" if surface.at.channel == Channel.URL else "shell"
    if (
        surface.tool != expected_tool
        or not surface.content.understood
        or len(surface.content.text) > MAX_TEXT
    ):
        return CodeSearchRead(complete=False)
    if surface.at.channel != Channel.URL:
        return _command_code_search(surface)
    text = surface.content.text
    request = url_request(text)
    service = service_for(text)
    found = (
        (CodeSearchRequest(service, surface.at, request.queries),)
        if request and service and service != "searchcode" and surface.tool == "web_fetch"
        else ()
    )
    return CodeSearchRead(found, literal_http_target(text))


def _possibly_relevant(text: str) -> bool:
    """Coverage hint only, never positive evidence or a query fingerprint."""
    for match in re.finditer(r"https?://[^\s\"\'<>]+", text):
        try:
            parts = urlsplit(match.group())
            if parts.hostname in {host for host, _ in ENDPOINTS}:
                return True
        except ValueError:
            continue
    return False


def code_search_requests(trace: Trace) -> CodeSearchRead:
    requests = []
    complete = True
    for surface in trace.agent_surfaces():
        read = surface_code_search(surface)
        requests.extend(read.requests)
        if not read.complete and _possibly_relevant(surface.content.text):
            complete = False
    return CodeSearchRead(tuple(requests), complete)


@dataclass(frozen=True)
class PromptAnchors:
    """Pre-action user instruction, not assistant claims or observations."""

    text: str = field(default="", repr=False)
    evidence: tuple[Locator, ...] = ()
    complete: bool = False


def prompt_anchors(trace: Trace) -> PromptAnchors:
    prompts = []
    evidence = []
    complete = not trace.compacted
    for step in trace.steps:
        if step.authored:
            break
        if step.source == "user" and not step.copied:
            prompts.append(step.message.text)
            evidence.append(Locator(step.index, Channel.MESSAGE))
            complete = complete and step.message.understood
    return PromptAnchors("\n".join(prompts), tuple(evidence), complete and bool(prompts))


TOKEN = re.compile(r"[A-Za-z_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?|(?<![\w.])-?\d+(?:\.\d+)?(?!\w)")
COMMON = frozenset(
    {
        "json.loads",
        "json.dumps",
        "main.py",
        "app.py",
        "index.js",
        "test.py",
        "test_outputs.py",
        "solve.sh",
        "__init__.py",
        "package.json",
        "readme.md",
        "requirements.txt",
        "import",
        "imports",
        "python",
        "requests.get",
        "httpx.get",
        "batch_size",
        "max_length",
        "learning_rate",
        "input_ids",
        "test_function",
        "hidden_size",
        "num_epochs",
        "attention_mask",
        "train_test_split",
        "grep.app",
        "sourcegraph.com",
        "searchcode.com",
        "valueerror",
        "typeerror",
        "keyerror",
        "runtimeerror",
        "exception",
        "dataframe",
        "basemodel",
        "baseexception",
        "filenotfounderror",
        "notimplementederror",
        "jsondecodeerror",
    }
)


def _distinctive(token: str) -> bool:
    if token.lower() in COMMON:
        return False
    # Tiny integers, years, and version numbers are not task fingerprints.
    if re.fullmatch(r"-?\d+(?:\.\d+)?", token):
        digits = token.replace("-", "").replace(".", "")
        return (
            len(digits) >= MIN_NUMERIC_DIGITS
            and len(set(digits)) >= MIN_NUMERIC_DIGITS - 2
            and digits not in {"123456", "1234567", "12345678", "123456789"}
            and not digits.endswith("000")
        )
    if len(token) < MIN_IDENTIFIER:
        return False
    return (
        (
            "_" in token
            and (
                token.count("_") >= MIN_SEPARATORS
                or any(c.isdigit() for c in token)
                or len(token) >= MIN_COMPOUND
            )
        )
        or bool(re.fullmatch(r"[A-Za-z0-9_-]{8,}\.(?:py|js|ts|json|csv|txt|cpp|rs|go)", token))
        or ("-" in token and len(token) >= MIN_COMPOUND)
        or (any(c.isupper() for c in token[1:]) and any(c.islower() for c in token))
    )


def fingerprint_request(request: CodeSearchRequest, anchors: PromptAnchors) -> bool:
    """A short distinctive exact token shared with explicit task instructions."""
    tokens = {m.group() for m in TOKEN.finditer(anchors.text) if _distinctive(m.group())}
    return any(m.group() in tokens for q in request.queries for m in TOKEN.finditer(q))


def _stdout_options(words: list[str], name: str | None) -> bool:
    options = {w.partition("=")[0] for w in words}
    risky = {
        "-w",
        "--write-out",
        "--location",
        "-K",
        "--config",
        "-i",
        "--include",
        "-o",
        "--output",
        "--output-document",
        "--resolve",
        "--connect-to",
        "-x",
        "--proxy",
        "-e",
        "--referer",
        "--request-target",
    }
    if options.intersection(risky) or any(re.fullmatch(r"-[A-Za-z]*L[A-Za-z]*", w) for w in words):
        return False
    return name != "wget" or any(w in {"-qO-", "-O-"} for w in words)


def single_response_producer(surface: Surface) -> bool:
    """Conservative output provenance: one plain curl/wget, no pipeline or local output.

    Python extraction supports request review, but Python result printing/transforming
    is intentionally not claimed as a uniquely attributable service response.
    """
    if surface.at.channel == Channel.URL:
        return surface.tool == "web_fetch"
    if surface.tool != "shell":
        return False
    script = shell.parse(surface.content.text)
    if not script.complete or len(script.commands) != 1:
        return False
    command = script.commands[0]
    words = [w for w, _ in command.argv()[1:]]
    targets, _, complete = _curl_words(command)
    return (
        complete
        and not _introspection(command)
        and command.name in {"curl", "wget"}
        and not command.redirects
        and not command.substituted
        and not command.background
        and not command.repeated
        and _stdout_options(words, command.name)
        and len(targets) == 1
    )
