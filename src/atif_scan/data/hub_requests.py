"""Offline, syntax-only Hub request categories; never execute or fetch source data.

Results are private evidence, not report documents. Only a validated namespace can
survive extraction (hidden from repr); URL/path/query/credential text never does.
``complete`` covers this reader's vocabulary, not arbitrary program effects.
Branches are lexical requests, not proof of execution or success. Bindings and
user functions are not evaluated. Unknown targets must not be treated as clean.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal
from urllib.parse import parse_qsl, unquote, urlsplit

from . import shell
from .pysource import parse_module

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@dataclass(frozen=True)
class HubTarget:
    kind: Literal["identity", "inventory", "object"]
    owner: str | None = field(default=None, repr=False)
    # A literal source banner encloses this actual identity request, not just its URL.
    labelled: bool = False


@dataclass(frozen=True)
class RequestRead:
    targets: tuple[HubTarget, ...] = ()
    complete: bool = True
    other_requests: int = 0
    # Independent of candidate completeness: unsafe stdout never binds a response.
    attributable: bool = True


_MAX_SOURCE = 131_072
_MAX_URL = 16_384
_MAX_NODES = 20_000
_MAX_QUERY_FIELDS = 64
_MAX_NAMESPACE = 96
_MAX_OPEN_ARGS = 2
_FILE_FIELDS = 4
_HEAD_ARGS = 2
_MAX_LIMIT_DIGITS = 9
_MAX_BANNER = 1024
_HOSTS = frozenset({"huggingface.co", "hf.co"})
_REPO_TYPES = frozenset({"datasets", "models", "spaces"})
_NAMESPACE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_-]{0,94}[A-Za-z0-9])?\Z")
_BAD_ESCAPE = re.compile(r"%(?![0-9a-fA-F]{2})")
_DYNAMIC = re.compile(r"[$`{}*\[\]\x00]")
_PYTHON = re.compile(r"python(?:[23](?:\.\d+)?)?\Z")
_UNKNOWN = RequestRead(complete=False)


def _merge(reads: Iterable[RequestRead]) -> RequestRead:
    targets: list[HubTarget] = []
    complete, others, attributable = True, 0, True
    for read in reads:
        targets.extend(read.targets)
        complete &= read.complete
        others += read.other_requests
        attributable &= read.attributable
    return RequestRead(tuple(targets), complete, others, attributable)


def namespace(value: str) -> str | None:
    """Validate a literal Hub namespace; never infer a computed owner."""
    valid = _NAMESPACE.fullmatch(value) and "--" not in value
    return value if valid else None


def _repo(value: str) -> bool:
    return bool(
        value
        and len(value) <= _MAX_NAMESPACE
        and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value)
        and not value.endswith(("-", "."))
        and ".." not in value
        and "--" not in value
    )


def _owned(kind: Literal["inventory", "object"], value: str) -> RequestRead:
    owner = namespace(value)
    return RequestRead((HubTarget(kind, owner),), complete=owner is not None)


def _author(query: str) -> RequestRead:
    pairs = parse_qsl(
        query, keep_blank_values=True, errors="strict", max_num_fields=_MAX_QUERY_FIELDS
    )
    authors = [value for key, value in pairs if key == "author"]
    if not authors:
        return _UNKNOWN if _DYNAMIC.search(query) else RequestRead(other_requests=1)
    if len(authors) != 1:
        return RequestRead((HubTarget("inventory"),), complete=False)
    return _owned("inventory", authors[0])


def _bucket_inventory_suffix(parts: list[str]) -> bool:
    match parts:
        case [] | ["tree", *_] | [_, "tree", *_]:
            return True
        case _:
            return False


def _bucket_route(parts: list[str]) -> RequestRead | None:
    match parts:
        case ["api", "buckets", owner, *suffix] if _bucket_inventory_suffix(suffix):
            return _owned("inventory", owner)
        case ["buckets", owner, bucket, "resolve", *_]:
            return _owned("object", owner) if _repo(bucket) else _UNKNOWN
        case _:
            return None


def _repo_route(parts: list[str], query: str) -> RequestRead | None:
    match parts:
        case ["api", repo_type] if repo_type in _REPO_TYPES:
            return _author(query)
        case ["api", repo_type, owner, repo, "tree", *_] if repo_type in _REPO_TYPES:
            return _owned("inventory", owner) if _repo(repo) else _UNKNOWN
        case [("datasets" | "spaces"), owner, repo, "resolve", *_]:
            return _owned("object", owner) if _repo(repo) else _UNKNOWN
    return _model_route(parts)


def _model_route(parts: list[str]) -> RequestRead | None:
    match parts:
        case [owner, repo, "resolve", *_] if owner not in {"api", "buckets", *_REPO_TYPES}:
            return _owned("object", owner) if _repo(repo) else _UNKNOWN
        case _:
            return None


def _hub_route(path: str, query: str) -> RequestRead:
    # Decode segments separately: an escaped slash is never a namespace separator.
    parts = [
        unquote(part, errors="strict")
        for part in path.removeprefix("/").removesuffix("/").split("/")
    ]
    if any(segment in {".", ".."} for part in parts for segment in part.split("/")):
        return _UNKNOWN
    if parts in (["api", "whoami"], ["api", "whoami-v2"]):
        return RequestRead((HubTarget("identity"),))
    result = _bucket_route(parts)
    if result is None:
        result = _repo_route(parts, query)
    return (
        result
        if result is not None
        else RequestRead(complete=not _DYNAMIC.search(path + query), other_requests=1)
    )


def _read_url(text: str) -> RequestRead:
    if not text or len(text) > _MAX_URL or re.search(r"[\x01-\x20\x7f]", text):
        return _UNKNOWN
    url = urlsplit(text)
    host = url.hostname
    # Reading .port validates it; never preserve authority, userinfo or port.
    invalid = url.scheme not in {"http", "https"} or not host or url.port == 0
    dynamic = _DYNAMIC.search(host or "") or "\\\\" in url.netloc
    if invalid or dynamic or _BAD_ESCAPE.search(text):
        return _UNKNOWN
    if host not in _HOSTS:
        return RequestRead(other_requests=1)
    return _hub_route(url.path, url.query)


def url_targets(text: str) -> RequestRead:
    """Read one absolute HTTP(S) URL, not a prose search for embedded URLs.

    Dynamic suffixes can preserve a proven route/owner; dynamic authorities and
    namespaces never become literal bindings. Unrecognized literal Hub routes are
    outside the target categories; known requests outside these categories are counted separately.
    Input/decoding/query limits fail unknown, with no source retained in results.
    """
    try:
        return _read_url(text)
    except (ValueError, UnicodeError, RecursionError):
        return _UNKNOWN


# Explicit option tables avoid misreading headers, referers, outputs, credentials
# and request bodies as invoked URLs. Unknown option grammars stay incomplete.
_CURL_VALUES = frozenset(
    {
        "--header",
        "--output",
        "--referer",
        "--user-agent",
        "--request",
        "--data",
        "--data-raw",
        "--data-binary",
        "--data-urlencode",
        "--json",
        "--form",
        "--form-string",
        "--user",
        "--proxy",
        "--proxy-user",
        "--cookie",
        "--cookie-jar",
        "--write-out",
        "--connect-timeout",
        "--max-time",
        "--retry",
        "--retry-delay",
        "--cacert",
        "--cert",
        "--key",
        "--resolve",
        "--connect-to",
        "--range",
        "--dump-header",
        "--oauth2-bearer",
        "--upload-file",
        "--url-query",
    }
)
_CURL_FLAGS = frozenset(
    {
        "--silent",
        "--show-error",
        "--fail",
        "--fail-with-body",
        "--location",
        "--compressed",
        "--insecure",
        "--head",
        "--include",
        "--remote-name",
        "--remote-header-name",
        "--no-progress-meter",
        "--globoff",
        "--next",
        "--verbose",
        "--http1.1",
        "--http2",
        "--get",
    }
)
_WGET_VALUES = frozenset(
    {
        "--header",
        "--output-document",
        "--output-file",
        "--append-output",
        "--referer",
        "--user-agent",
        "--post-data",
        "--post-file",
        "--body-data",
        "--body-file",
        "--method",
        "--user",
        "--password",
        "--http-user",
        "--http-password",
        "--load-cookies",
        "--save-cookies",
        "--directory-prefix",
        "--timeout",
        "--tries",
        "--wait",
        "--execute",
        "--ca-certificate",
        "--certificate",
        "--private-key",
    }
)
_WGET_FLAGS = frozenset(
    {
        "--quiet",
        "--no-verbose",
        "--verbose",
        "--continue",
        "--no-check-certificate",
        "--server-response",
        "--spider",
        "--content-disposition",
        "--trust-server-names",
        "--no-cookies",
        "--no-proxy",
        "--recursive",
        "--timestamping",
    }
)


@dataclass
class _Options:
    reads: list[RequestRead] = field(default_factory=list)
    complete: bool = True
    operands: bool = False
    attributable: bool = True
    wget_stdout: bool = False


def _long_option(state: _Options, word: str, words: Iterator[str], curl: bool) -> None:
    name, equals, attached = word.partition("=")
    value = None
    value_options = _CURL_VALUES if curl else _WGET_VALUES
    flags = _CURL_FLAGS if curl else _WGET_FLAGS
    hidden = {"--config"} if curl else {"--input-file", "--execute"}
    url_options = {"--url"} if curl else set()
    if name in value_options | hidden | url_options:
        value = attached if equals else next(words, None)
        state.complete &= value is not None and name not in hidden | {"--url-query"}
        if name == "--url" and value is not None and curl:
            state.reads.append(url_targets(value))
    elif name not in flags:
        state.complete = False
        # Its arity is unknown. Stop rather than positively reading its value.
        list(words)
    if name in {"--output", "--remote-name", "--remote-header-name", "--write-out"}:
        state.attributable = False
    if not curl and name == "--output-document":
        state.wget_stdout = value == "-"
        state.attributable &= state.wget_stdout
    if name == "--get":
        state.complete = False  # Body/query options may supply an unresolved author.


def _short_option(state: _Options, word: str, words: Iterator[str], curl: bool) -> None:
    values = "oHeAXdFubcwUTDxmKr" if curl else "OoaUPetTwil"
    flags = "fsSLkIvNROJgq#v" if curl else "qvcSNnrpk"
    for index, letter in enumerate(word[1:], start=2):
        if letter in values:
            value = word[index:] or next(words, None)
            state.complete &= value is not None and letter not in ("K" if curl else "ie")
            if letter in ("ow" if curl else "O"):
                state.wget_stdout = not curl and value == "-"
                state.attributable &= state.wget_stdout
            break
        if curl and letter in "ROJ":
            state.attributable = False
        if letter not in flags:
            state.complete = False
            list(words)
            break


def _download(command: shell.Command) -> RequestRead:
    state = _Options()
    words = iter(word for word, _ in command.argv()[1:])
    curl = command.name == "curl"
    for word in words:
        if word == "--":
            state.operands = True
        elif not state.operands and word.startswith("--"):
            _long_option(state, word, words, curl)
        elif not state.operands and word.startswith("-") and word != "-":
            _short_option(state, word, words, curl)
        else:
            state.reads.append(url_targets(word))
    return _merge(
        [
            RequestRead(
                complete=state.complete,
                attributable=state.attributable and (curl or state.wget_stdout),
            ),
            *state.reads,
        ]
    )


_NETWORK_ROOTS = frozenset({"requests", "httpx", "urllib", "huggingface_hub"})
# Opaque calls may hide SDK/network dispatch. Only simple data-processing calls
# are exempt; an unknown callable cannot establish absence of requests.
_SAFE_CALLS = frozenset(
    {
        "print",
        "len",
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "list",
        "dict",
        "tuple",
        "set",
        "range",
        "enumerate",
        "zip",
        "sorted",
        "sum",
        "min",
        "max",
        "json.load",
        "json.loads",
        "json.dump",
        "json.dumps",
        "os.getenv",
        "os.environ.get",
        "urllib.parse.quote",
        "urllib.parse.quote_plus",
        "urllib.parse.urlencode",
    }
)
_SAFE_ROOTS = frozenset(name.split(".")[0] for name in _SAFE_CALLS)
_DIRECT_CALLS = frozenset(
    {
        "requests.get",
        "requests.post",
        "httpx.get",
        "httpx.post",
        "urllib.request.urlopen",
    }
)
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.GeneratorExp)


def _name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _url_argument(call: ast.Call) -> ast.expr | None:
    if any(isinstance(arg, ast.Starred) for arg in call.args):
        return None
    values = [kw.value for kw in call.keywords if kw.arg in {"url", "fullurl"}]
    if call.args:
        values.append(call.args[0])
    return values[0] if len(values) == 1 and all(kw.arg for kw in call.keywords) else None


def _url_expression(node: ast.expr | None) -> RequestRead:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return url_targets(node.value)
    if isinstance(node, ast.JoinedStr):
        prefix = ""
        for value in node.values:
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                return url_targets(prefix + "\x00")
            prefix += value.value
        return url_targets(prefix)
    return _UNKNOWN


def _definition_headers(node: ast.AST) -> list[ast.expr]:
    if isinstance(node, ast.ClassDef):
        return [*node.decorator_list, *node.bases, *(kw.value for kw in node.keywords)]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        defaults = [*node.args.defaults, *(v for v in node.args.kw_defaults if v is not None)]
        return defaults if isinstance(node, ast.Lambda) else [*node.decorator_list, *defaults]
    if isinstance(node, ast.GeneratorExp):
        return [node.generators[0].iter]
    return []


def _write_open(node: ast.Call) -> bool:
    mode = (
        node.args[1]
        if len(node.args) > 1
        else next((kw.value for kw in node.keywords if kw.arg == "mode"), None)
    )
    known_mode = isinstance(mode, ast.Constant) and mode.value in {"w", "a", "x", "wb", "ab", "xb"}
    keywords = {"mode", "encoding", "errors", "newline", "buffering"}
    return (
        _name(node.func) == "open"
        and known_mode
        and len(node.args) <= _MAX_OPEN_ARGS
        and all(kw.arg in keywords for kw in node.keywords)
    )


def _file_write(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "write"
        and isinstance(func.value, ast.Call)
        and _write_open(func.value)
    )


def _binding_root(node: ast.AST) -> str:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def _captures(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.MatchAs, ast.MatchStar, ast.ExceptHandler)):
        return [node.name] if node.name else []
    if isinstance(node, ast.MatchMapping):
        return [node.rest] if node.rest else []
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        return [_binding_root(node)] if isinstance(node.ctx, (ast.Store, ast.Del)) else []
    return []


def _all_bindings(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.alias):
        return [node.asname or node.name.split(".")[0]]
    return _captures(node)


@dataclass
class _PythonReader:
    reads: list[RequestRead] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
    complete: bool = True
    remaining: int = _MAX_NODES
    http_calls: set[int] = field(default_factory=set)
    attributable: bool = True

    def name(self, node: ast.AST) -> str:
        name = _name(node)
        root, dot, suffix = name.partition(".")
        return self.aliases.get(root, root) + dot + suffix

    def potential(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Import):
            return any(alias.name.split(".")[0] in _NETWORK_ROOTS for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            return (node.module or "").split(".")[0] in _NETWORK_ROOTS
        return isinstance(node, ast.Call) and self.name(node.func) not in _SAFE_CALLS

    def shadow(self, original: str) -> None:
        # Never treat reassigned API/import names as proven request functions.
        root = self.aliases.get(original, original).split(".")[0]
        if root in _NETWORK_ROOTS | _SAFE_ROOTS | {"open"}:
            self.aliases[original] = "_opaque"
            self.complete = False
            self.attributable = False

    def imports(self, node: ast.Import | ast.ImportFrom) -> None:
        for alias in node.names:
            if isinstance(node, ast.ImportFrom):
                self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                if alias.name == "*":
                    self.complete = False
            else:
                self.aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )

    def deferred(self, node: ast.AST) -> None:
        headers = _definition_headers(node)
        skipped = {id(header) for header in headers}
        pending = list(ast.iter_child_nodes(node))
        while pending:
            child = pending.pop()
            self.remaining -= 1
            if self.remaining < 0:
                self.complete = False
                break
            if id(child) not in skipped:
                self.complete &= not self.potential(child)
                pending.extend(ast.iter_child_nodes(child))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.shadow(node.name)
        for header in headers:
            self.visit(header)

    def active(self, node: ast.AST) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self.imports(node)
        for name in _captures(node):
            self.shadow(name)
        if isinstance(node, ast.Call):
            self.call(node)

    def visit(self, node: ast.AST) -> None:
        self.remaining -= 1
        if self.remaining < 0:
            self.complete = False
            return
        if isinstance(node, _DEFINITIONS):
            self.deferred(node)
        else:
            self.active(node)
            for child in ast.iter_child_nodes(node):
                self.visit(child)

    def call(self, node: ast.Call) -> None:
        if self.name(node.func) in _DIRECT_CALLS:
            self.reads.append(_url_expression(_url_argument(node)))
            self.http_calls.add(id(node))
            if any(kw.arg == "params" for kw in node.keywords):
                self.complete = False  # Separately supplied query data is not evaluated.
        elif not (
            (_write_open(node) or _file_write(node))
            and self.name(node.func).split(".")[0] != "_opaque"
        ) and self.potential(node):
            self.complete = False
            self.attributable = False


_SECTION = re.compile(r"[=-]{3,}[^\r\n]*")
_WHOAMI = re.compile(
    r"[=-]{3,}\s*(?:https?://(?:huggingface\.co|hf\.co)/api/)?whoami(?:-v2)?\s*[=-]{3,}",
    re.I,
)


def _banner(value: str) -> bool:
    return not value.strip() or bool(_SECTION.fullmatch(value))


def _result_assignment(node: ast.AST, calls: set[int]) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target, value = node.targets[0], node.value
    elif isinstance(node, ast.AnnAssign):
        target, value = node.target, node.value
    else:
        return None
    return target.id if isinstance(target, ast.Name) and id(value) in calls else None


def _response_variables(nodes: list[ast.AST], calls: set[int]) -> dict[str, tuple[int, int]]:
    writes: dict[str, int] = {}
    for node in nodes:
        for name in _all_bindings(node):
            writes[name] = writes.get(name, 0) + 1
    assignments = {
        name: (node.end_lineno or node.lineno, node.end_col_offset or node.col_offset)
        for node in nodes
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (name := _result_assignment(node, calls)) is not None
    }
    return {name: line for name, line in assignments.items() if writes.get(name) == 1}


def _response_attr(
    node: ast.AST, attribute: str, variables: dict[str, tuple[int, int]]
) -> str | None:
    if isinstance(node, ast.Attribute) and node.attr == attribute:
        value = node.value
        if isinstance(value, ast.Name) and variables.get(value.id, (_MAX_NODES, 0)) < (
            node.lineno,
            node.col_offset,
        ):
            return value.id
    return None


def _preview(node: ast.AST, variables: dict[str, tuple[int, int]]) -> bool:
    if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Slice):
        return False
    bound = node.slice.upper
    bounded = isinstance(bound, ast.Constant) and type(bound.value) is int and bound.value > 0
    return bool(
        bounded
        and node.slice.lower is None
        and node.slice.step is None
        and _response_attr(node.value, "text", variables)
    )


def _file_record(args: list[ast.expr], variables: dict[str, tuple[int, int]]) -> bool:
    if len(args) != _FILE_FIELDS:
        return False
    marker, path, status, size = args
    if not (
        isinstance(marker, ast.Constant)
        and marker.value == "FILE:"
        and isinstance(path, (ast.Name, ast.Constant))
    ):
        return False
    if not (
        isinstance(size, ast.Call)
        and _name(size.func) == "len"
        and len(size.args) == 1
        and not size.keywords
    ):
        return False
    result = _response_attr(status, "status_code", variables)
    return result is not None and result == _response_attr(size.args[0], "content", variables)


def _literal_banner(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _banner(node.value)
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
        return False
    left, right = node.left, node.right
    if not (isinstance(left, ast.Constant) and isinstance(right, ast.Constant)):
        return False
    marker, count = left.value, right.value
    return (
        isinstance(marker, str)
        and type(count) is int
        and 0 < count <= _MAX_BANNER
        and len(marker) * count <= _MAX_BANNER
        and bool(marker)
        and not set(marker) - {"=", "-"}
    )


def _safe_print(node: ast.Call, variables: dict[str, tuple[int, int]]) -> bool:
    if node.keywords:
        return False
    if len(node.args) == 1:
        arg = node.args[0]
        marker = _literal_banner(arg)
        return bool(marker or _preview(arg, variables))
    return not node.args or _file_record(node.args, variables)


def _reads_file(node: ast.Call) -> bool:
    name = _name(node.func)
    return (name == "open" and not _write_open(node)) or (
        name in {"json.load", "json.dump"}
        or name.rsplit(".", 1)[-1] in {"read", "read_text", "read_bytes"}
    )


def _python_stdout(tree: ast.AST, calls: set[int]) -> bool:
    nodes = list(ast.walk(tree))
    variables = _response_variables(nodes, calls)
    allowed: set[int] = set()
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        if _reads_file(node) or (_name(node.func) == "print" and not _safe_print(node, variables)):
            return False
        if _name(node.func) == "print" or _file_write(node):
            allowed.update(id(child) for child in ast.walk(node))
    return not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in variables
        and id(node) not in allowed
        for node in nodes
    )


def _python_source(source: str) -> RequestRead:
    reader = _PythonReader()
    try:
        tree = parse_module(source)
        reader.visit(tree)
        reader.attributable &= _python_stdout(tree, reader.http_calls)
    except (SyntaxError, ValueError, RecursionError):
        reader.complete = False
        reader.attributable = False
    return _merge(
        [RequestRead(complete=reader.complete, attributable=reader.attributable), *reader.reads]
    )


def _python_command(command: shell.Command, script: shell.Script, text: str) -> RequestRead:
    words = iter(word for word, _ in command.argv()[1:])
    for word in words:
        if word == "-c" or word.startswith("-c"):
            source = word[2:] or next(words, None)
            return _python_source(source) if source is not None else _UNKNOWN
        if word == "-" or not word.startswith("-"):
            break
        if word.startswith("-m") or word in {"-W", "-X", "--"}:
            return _UNKNOWN
    else:
        word = "-"
    if word != "-":
        return _UNKNOWN  # Never open a named script or infer its contents.
    return _python_stdin(command, script, text)


def _stdin_redirect(text: str, span: shell.Span) -> bool:
    descriptor = re.match(r"\d+", text[span[0] : span[1]])
    return descriptor is None or not descriptor[0].strip("0")


def _python_stdin(command: shell.Command, script: shell.Script, text: str) -> RequestRead:
    redirects = [
        (op, span)
        for op, _, span in command.redirects
        if op.startswith("<") and _stdin_redirect(text, span)
    ]
    if not redirects or redirects[-1][0] not in {"<<", "<<-"}:
        return _UNKNOWN
    # All heredocs have body entries, even ones redirected to a different fd.
    bodies = [
        text[start:end]
        for (start, end), owner in zip(script.bodies, script.body_commands, strict=True)
        if owner is command
    ]
    heredocs = [span for op, _, span in command.redirects if op in {"<<", "<<-"}]
    index = heredocs.index(redirects[-1][1])
    if index >= len(bodies):
        return _UNKNOWN
    source = bodies[index]
    if redirects[-1][0] == "<<-":
        source = "\n".join(line.lstrip("\t") for line in source.split("\n"))
    return _python_source(source)


_UNSUPPORTED = frozenset({"node", "nodejs", "deno", "bun", "ruby", "perl", "php"})
_HEAD_COUNT = re.compile(r"-[cn][1-9]\d*\Z")


def _static_print(command: shell.Command) -> str | None:
    words = [word for word, _ in command.argv()[1:]]
    if command.name == "echo":
        safe = not any(_DYNAMIC.search(word) or "\\" in word for word in words)
        safe &= not words or not words[0].startswith("-")
        return " ".join(words) if safe else None
    if command.name != "printf" or not words or any(_DYNAMIC.search(word) for word in words):
        return None
    fmt, *args = words
    if fmt in {"%s\\n", "\\n%s\\n", "%s\n", "\n%s\n"} and len(args) == 1:
        return args[0] if "\\" not in args[0] else None
    literal = fmt.replace("\\n", "\n")
    safe = not args and "%" not in literal and "\\" not in literal and literal.endswith("\n")
    return literal.strip("\r\n") if safe else None


def _head_filter(command: shell.Command) -> bool:
    args = [word for word, _ in command.argv()[1:]]
    count = len(args) == 1 and bool(_HEAD_COUNT.fullmatch(args[0]))
    separate = len(args) == _HEAD_ARGS and args[0] in {"-c", "-n"} and args[1].isdigit()
    return command.name == "head" and (count or separate) and not command.redirects


def _download_pipeline(command: shell.Command, script: shell.Script) -> bool:
    return any(
        other.pipeline == command.pipeline
        and other.span[0] < command.span[0]
        and other.name in {"curl", "wget"}
        for other in script.commands
    )


def _quiet_builtin(command: shell.Command) -> bool:
    words = [word for word, _ in command.argv()[1:]]
    if command.name in {"true", "false", ":"}:
        return True
    if command.name == "export":
        return bool(words) and all(re.match(r"[A-Za-z_][A-Za-z0-9_]*=", word) for word in words)
    return command.name == "cd" and all(
        word != "-" and not re.search(r"[\\x00-\\x1f]", word) for word in words
    )


def _command_read(command: shell.Command, script: shell.Script, text: str) -> RequestRead:
    name = command.name or ""
    if name in {"curl", "wget"}:
        read = _download(command)
    elif _PYTHON.fullmatch(name):
        read = _python_command(command, script, text)
    elif (
        name in _UNSUPPORTED | {"hf", "huggingface-cli"}
        or _DYNAMIC.search(name)
        or (name in shell.SHELLS and not command.nested)
    ):
        read = RequestRead(complete=False, attributable=False)
    else:
        printed = _static_print(command)
        safe = (
            (printed is not None and _banner(printed))
            or (_head_filter(command) and _download_pipeline(command, script))
            or bool(command.nested)
            or _quiet_builtin(command)
            or not name
        )
        read = RequestRead(attributable=safe)
    unsafe = command.background or command.repeated or bool(command.substituted)
    # Output redirection/capture is never evidence that a returned body is this request's stdout.
    unsafe |= any(op in shell.WRITE_OPS for op, _, _ in command.redirects)
    return replace(read, attributable=read.attributable and not unsafe)


def _section_command(command: shell.Command) -> str | None:
    if command.redirects or command.background or command.repeated or command.substituted:
        return None
    value = _static_print(command)
    return value if value is not None and _banner(value) else None


def _bounded_identity(commands: list[shell.Command], index: int, *, source_end: bool) -> bool:
    current = commands[index]
    for following in commands[index + 1 :]:
        if following.pipeline == current.pipeline and _head_filter(following):
            continue
        printed = _section_command(following)
        if printed is None:
            return False
        if printed.strip():
            return True
    return source_end


def _blank_tail(commands: list[shell.Command], index: int) -> bool:
    return all(_section_command(command) == "" for command in commands[index + 1 :])


def _scope_labels(
    commands: list[shell.Command], reads: dict[int, RequestRead], *, source_end: bool = True
) -> None:
    children = {
        id(child) for command in commands for child in (*command.nested, *command.substituted)
    }
    scope = [command for command in commands if id(command) not in children]
    previous: str | None = None
    for index, command in enumerate(scope):
        read = reads[id(command)]
        identity = read.targets == (HubTarget("identity"),) and not read.other_requests
        if (
            identity
            and read.complete
            and read.attributable
            and previous is not None
            and _WHOAMI.fullmatch(previous.strip())
            and _bounded_identity(scope, index, source_end=source_end)
        ):
            reads[id(command)] = replace(read, targets=(HubTarget("identity", labelled=True),))
        printed = _section_command(command)
        if printed is None or printed.strip():
            previous = printed
        for nested in (command.nested, command.substituted):
            if nested:
                _scope_labels(nested, reads, source_end=source_end and _blank_tail(scope, index))


def command_targets(text: str) -> RequestRead:
    """Read syntax only; retain candidates independently of stdout attribution.

    Only literal printed sections immediately enclosing an actual identity command
    label it. Opaque stdout, file output and unsafe Python response bindings cannot
    authorize response attribution. Named scripts are never opened.
    """
    if len(text) > _MAX_SOURCE:
        return _UNKNOWN
    script = shell.parse(text)
    reads = {id(command): _command_read(command, script, text) for command in script.commands}
    _scope_labels(script.commands, reads)
    return _merge(
        [RequestRead(complete=script.complete, attributable=script.complete), *reads.values()]
    )


def _byte_limit(command: shell.Command) -> int | None:
    words = [word for word, _ in command.argv()[1:]]
    value = words[0][2:] if len(words) == 1 and words[0].startswith("-c") else None
    if len(words) == _HEAD_ARGS and words[0] == "-c":
        value = words[1]
    valid = value is not None and value.isdecimal() and len(value) <= _MAX_LIMIT_DIGITS
    limit = int(value) if valid else 0
    return limit if limit > 0 else None


def identity_preview_limit(text: str) -> int | None:
    """A single literal byte-prefix filter applied to every identity request.

    This count can distinguish framing newlines printed after ``head -c`` from
    invalid control characters inside the recorded JSON. No repair without a
    statically known byte boundary, and never a local file read.
    """
    script = shell.parse(text)
    limits: list[int | None] = []
    for command in script.commands:
        if command.name not in {"curl", "wget"}:
            continue
        if not any(target.kind == "identity" for target in _download(command).targets):
            continue
        filters = [
            other
            for other in script.commands
            if other.pipeline == command.pipeline and other.span[0] > command.span[0]
        ]
        limit = _byte_limit(filters[0]) if len(filters) == 1 and _head_filter(filters[0]) else None
        limits.append(limit)
    return limits[0] if limits and limits[0] is not None and len(set(limits)) == 1 else None
