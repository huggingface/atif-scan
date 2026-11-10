"""Offline recognition of bearer webhook URLs and bounded Python literal assembly.

Only explicit HTTPS authorities and complete request-target grammars qualify. This is
shape recognition, never proof of issuance, liveness, invocation or authorization.
Masking includes dummy shapes; exposure excludes unmistakable repeated-character
placeholders. Genuine-shaped capabilities remain sensitive even in documentation.

Source recovery reads only string constants, uniquely bound top-level names and ``+``.
It never evaluates calls, imports, interpolations or runtime state. Source spans are
retained so redaction removes contributing literals, not merely a reconstructed URL.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from .jsonval import as_str
from .pysource import parse_module

if TYPE_CHECKING:
    from collections.abc import Iterator

_SLACK_HOSTS = frozenset({"hooks.slack.com", "hooks.slack-gov.com"})
_DISCORD_HOSTS = frozenset(
    {
        "discord.com",
        "discordapp.com",
        "canary.discord.com",
        "ptb.discord.com",
        "canary.discordapp.com",
        "ptb.discordapp.com",
    }
)
_SLACK_PATH = re.compile(
    r"/services/(?P<first>T[A-Z0-9]{8,12})/(?P<second>B[A-Z0-9]{8,12})/"
    r"(?P<token>[A-Za-z0-9]{24,64})"
)
_DISCORD_PATH = re.compile(
    r"/api(?:/v[0-9]{1,2})?/webhooks/(?P<first>[0-9]{17,20})/"
    r"(?P<token>[A-Za-z0-9_-]{60,100})(?:/(?:slack|github))?"
)
_URL = re.compile(r"""(?<![\w:/])https?://[^\s'"<>`\\]+""", re.I)
# Decode individual, bounded JSON string literals only. No eval, recursive
# program unescaping, or source execution. Whole-literal redaction is conservative
# but avoids inaccurate offsets after JSON's slash/Unicode escapes.
_JSON_STRING = re.compile(r'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})){1,4096}"')
_MAX_SOURCE = 65_536
_MAX_NODES = 1024
_MAX_DEPTH = 20
_MAX_PARTS = 64
_MAX_VALUE = 8192
_MIN_REPEAT = 8
_ASCII_SPACE = 32
_DELETE = 127
Span = tuple[int, int]


@dataclass(frozen=True)
class WebhookCapability:
    """A structurally valid capability; all bearer components are private."""

    service: str
    token: str = field(repr=False)
    components: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True)
class _Target:
    service: str
    token: str = field(repr=False)
    components: tuple[tuple[str, Span], ...] = field(repr=False)


def _path(path: str) -> tuple[str, list[Span]] | None:
    """Decode only escaped unreserved bytes, preserving raw-character provenance."""
    decoded: list[str] = []
    spans: list[Span] = []
    pos = 0
    while pos < len(path):
        end = pos + 1
        char = path[pos]
        if char == "%":
            escape = path[pos + 1 : pos + 3]
            if not re.fullmatch("[0-9a-fA-F]{2}", escape):
                return None
            char = chr(int(escape, 16))
            if not (char.isascii() and (char.isalnum() or char in "-._~")):
                return None
            end = pos + 3
        decoded.append(char)
        spans.append((pos, end))
        pos = end
    return "".join(decoded), spans


def _target(url: str) -> _Target | None:
    if any(c.isspace() or ord(c) < _ASCII_SPACE or ord(c) == _DELETE or c == "\\" for c in url):
        return None
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    # Exact authority comparison also rejects userinfo, suffixes, local copies,
    # trailing dots, escaped hostnames and non-HTTPS/non-default ports.
    host = parsed.netloc.lower()
    if host.endswith(":443"):
        host = host[:-4]
    if parsed.scheme != "https" or host not in _SLACK_HOSTS | _DISCORD_HOSTS:
        return None
    return _target_path(url, parsed.path, host)


def _target_path(url: str, path: str, host: str) -> _Target | None:
    decoded = _path(path)
    if decoded is None:
        return None
    value, spans = decoded
    service = "slack" if host in _SLACK_HOSTS else "discord"
    pattern = _SLACK_PATH if service == "slack" else _DISCORD_PATH
    match = pattern.fullmatch(value)
    if match is None:
        return None
    offset = url.index(path, url.index("://") + 3)
    components = []
    for name, component in match.groupdict().items():
        start, end = match.span(name)
        components.append((component, (offset + spans[start][0], offset + spans[end - 1][1])))
    return _Target(service, match["token"], tuple(components))


def webhook_capability(url: str) -> WebhookCapability | None:
    """Recognize one complete URL, not a URL substring or an arbitrary source string.

    Query/fragment text is not part of the bearer path. IDs and tokens may be
    synthetic/all-zero: this predicate intentionally does not claim issuance.
    Unreserved percent escapes are decoded; escaped separators are rejected.
    """
    target = _target(url)
    if target is None:
        return None
    return WebhookCapability(
        target.service, target.token, tuple(value for value, _ in target.components)
    )


def _exposable(token: str) -> bool:
    # Documentation's XXXXXX/000000 and punctuation-only stand-ins are masked,
    # but not findings. Do not blanket-exempt "test", filenames or prose roles.
    return len(set(token.lower())) > 1 and any(c.isalnum() for c in token)


def _direct(text: str, *, exposure: bool) -> Iterator[tuple[Span, Span, str]]:
    for candidate in _URL.finditer(text):
        url = candidate.group().rstrip(".,;:!)]}")
        target = _target(url)
        if target is None:
            continue
        if exposure and not _exposable(target.token):
            continue
        whole = (candidate.start(), candidate.start() + len(url))
        for value, (start, end) in target.components:
            yield whole, (candidate.start() + start, candidate.start() + end), value


@dataclass(frozen=True)
class _Piece:
    value: str = field(repr=False)
    source: Span


def _source_span(text: str, node: ast.expr, lines: list[str]) -> Span:
    # AST columns count UTF-8 bytes, while masking spans count Python characters.
    start = sum(map(len, lines[: node.lineno - 1]))
    end_line = node.end_lineno or node.lineno
    end = sum(map(len, lines[: end_line - 1]))
    start += len(lines[node.lineno - 1].encode()[: node.col_offset].decode())
    end += len(lines[end_line - 1].encode()[: node.end_col_offset].decode())
    return min(start, len(text)), min(end, len(text))


def _read(
    node: ast.AST,
    env: dict[str, tuple[_Piece, ...]],
    source: str,
    lines: list[str],
    depth: int = 0,
) -> tuple[_Piece, ...]:
    if depth > _MAX_DEPTH:
        return ()
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and len(node.value) <= _MAX_VALUE
    ):
        return (_Piece(node.value, _source_span(source, node, lines)),)
    if isinstance(node, ast.Name):
        return env.get(node.id, ())
    return _read_add(node, env, source, lines, depth)


def _read_add(
    node: ast.AST,
    env: dict[str, tuple[_Piece, ...]],
    source: str,
    lines: list[str],
    depth: int,
) -> tuple[_Piece, ...]:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
        return ()
    left = _read(node.left, env, source, lines, depth + 1)
    right = _read(node.right, env, source, lines, depth + 1)
    parts = left + right if left and right else ()
    if len(parts) > _MAX_PARTS or sum(len(p.value) for p in parts) > _MAX_VALUE:
        return ()
    return parts


def _tree(text: str) -> ast.Module | None:
    if len(text) > _MAX_SOURCE or "+" not in text:
        return None
    try:
        tree = parse_module(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        return None
    return tree


def _bound_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        return node.id
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.arg)):
        return node.arg if isinstance(node, ast.arg) else node.name
    if isinstance(node, ast.alias):
        return node.asname or node.name.split(".")[0]
    return None


def _binding(statement: ast.stmt, stores: Counter[str]) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    name = statement.targets[0]
    return name.id if isinstance(name, ast.Name) and stores[name.id] == 1 else None


def _assemblies(text: str) -> Iterator[tuple[_Piece, ...]]:
    tree = _tree(text)
    if tree is None:
        return
    lines = text.splitlines(keepends=True)
    stores = Counter(name for n in ast.walk(tree) if (name := _bound_name(n)) is not None)
    env: dict[str, tuple[_Piece, ...]] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
            continue
        # Only prior, uniquely bound top-level assignments supply names. No calls,
        # attributes, imports, nested scopes, forward refs or mutation are read.
        for node in ast.walk(statement):
            if isinstance(node, ast.BinOp) and (parts := _read(node, env, text, lines)):
                yield parts
        if isinstance(statement, ast.Assign) and (name := _binding(statement, stores)):
            env[name] = _read(statement.value, env, text, lines)


def _contributions(text: str, *, exposure: bool) -> Iterator[tuple[_Piece, str]]:
    for parts in _assemblies(text):
        joined = "".join(p.value for p in parts)
        for _, (start, end), component in _direct(joined, exposure=exposure):
            offset = 0
            for part in parts:
                next_offset = offset + len(part.value)
                if offset < end and start < next_offset:
                    yield part, component
                offset = next_offset


def _json_capabilities(text: str, *, exposure: bool) -> Iterator[tuple[Span, WebhookCapability]]:
    if r"\/" not in text:
        return
    for match in _JSON_STRING.finditer(text):
        raw = match.group()
        if r"\/" not in raw:
            continue
        try:
            decoded = as_str(json.loads(raw))
        except (ValueError, RecursionError):
            continue
        capability = webhook_capability(decoded) if decoded is not None else None
        if capability is not None and (not exposure or _exposable(capability.token)):
            yield match.span(), capability


def find(text: str, *, exposure: bool = False) -> Iterator[tuple[Span, Span]]:
    """Private (whole locator, sensitive value) spans, including literal provenance."""
    for whole, sensitive, _ in _direct(text, exposure=exposure):
        yield whole, sensitive
    for part, _ in _contributions(text, exposure=exposure):
        yield part.source, part.source
    for span, _ in _json_capabilities(text, exposure=exposure):
        yield span, span


def values(text: str) -> frozenset[str]:
    """Bearer components and contributing literal fragments for trace-wide masking."""
    out: set[str] = set()
    for _, (start, end), component in _direct(text, exposure=False):
        out.update((component, text[start:end]))
    for part, component in _contributions(text, exposure=False):
        out.update((component, part.value, text[part.source[0] : part.source[1]]))
    for span, capability in _json_capabilities(text, exposure=False):
        out.update(capability.components)
        out.add(text[span[0] : span[1]])
    return frozenset(value for value in out if len(value) >= _MIN_REPEAT)
