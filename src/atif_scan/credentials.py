"""Credential shapes shared by the secret detectors and citation masking (stdlib only).

Two kinds of evidence:

- **Token shapes** that are credentials wherever they appear (`sk-…`, `hf_…`, `ghp_…`,
  AWS key IDs, JWTs, private-key blocks, `LLM|<id>|<token>` harness keys, …).
- **Named values**: an assignment whose *name* says it holds a secret
  (`OPENAI_API_KEY=…`, `"api_key": "…"`, `os.environ["X_TOKEN"] = "…"`,
  `--api-key …`, `x-api-key: …`). Names are split into components, so
  `TIKTOKEN_CACHE_DIR` or `MAX_TOKENS` are not secrets, and path/URL/placeholder values
  (`/root/.ssh/id_rsa`, `${KEY}`, `<your-key>`, `***`) are ignored.

Findings carry spans only; values never leave memory except as masking input.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

TOKEN_SHAPES = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)|"
    r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}|"
    r"\bhf_[A-Za-z0-9]{20,}|"
    r"\bgh[pousr]_[A-Za-z0-9]{20,}|"
    r"\bgithub_pat_[A-Za-z0-9_]{20,}|"
    r"\bxox[abprs]-[A-Za-z0-9-]{10,}|"
    r"\bAKIA[0-9A-Z]{16}\b|"
    r"\bAIza[0-9A-Za-z_-]{35}|"
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}|"
    # Harness-issued model keys (`LLM|<numeric id>|<token>`), seen in leaderboard sandboxes.
    r"\bLLM\|\d{6,}\|[A-Za-z0-9_-]{12,}",
    re.S,
)

# NAME=value (env output, .env files, `export`), with the value optionally quoted.
_ENV = re.compile(r"(?<![\w$])(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<q>['\"]?)(?P<value>[^\s'\"]+)")
# 'name': 'value' / "name" = "value" / os.environ["NAME"] = "value" / name="value":
# the value must be quoted, so code like `token = tok.encode(x)` isn't a secret.
_KEYED = re.compile(
    r"(?<![\w-])(?P<nq>['\"]?)(?P<name>[A-Za-z_][\w-]*)(?P=nq)\]?\s*[:=]\s*(?P<q>['\"])(?P<value>[^'\"\s]+)(?P=q)"
)
# --api-key VALUE / --token=VALUE / x-api-key: VALUE (headers, CLI flags).
_FLAG = re.compile(
    r"(?i)(?:--?(?P<flag>api[_-]?key|token|password|secret)[= ]\s*|"
    r"\b(?P<header>x-api-key|api-key)\s*:\s*)(?P<q>['\"]?)(?P<value>[^\s'\"]+)"
)
_SECRET_PART = re.compile(
    r"(?:^|_)(?:api_?key|apikey|secret|token|password|passwd|pwd|credentials?|"
    r"private_key|access_key|secret_key|session_key|auth_key|key)(?:_|$)"
)
_NOT_SECRET_PART = re.compile(
    r"(?:^|_)(?:path|file|dir|url|uri|host|port|name|type|cache|len|length|size|count|"
    r"max|min|limit|env|header|field|prefix|id|ids|format|mode|var|vars|fn|func)(?:_|$)"
)
_PLACEHOLDER = re.compile(
    r"(?i)^(?:\*+|x+|\.+|<[^>]*>|\$\{?\w+\}?|%\w+%|your[\w-]*|changeme|none|null|nil|"
    r"true|false|dummy|test\w*|example\w*|placeholder|redacted|masked|secret|password|"
    r"undefined|sk-\.\.\.|\.\.\.)$"
)


def secret_name(name: str) -> bool:
    """Does an assignment name say it holds a secret? Components split on _ - and camelCase."""
    parts = re.split(r"[_\-]+|(?<=[a-z0-9])(?=[A-Z])", name)
    joined = "_".join(p.lower() for p in parts if p)
    return bool(_SECRET_PART.search(joined)) and not _NOT_SECRET_PART.search(joined)


def plausible_value(value: str) -> bool:
    """A value that could be a real credential: not a placeholder, path, URL or number."""
    if len(value) < 8 or _PLACEHOLDER.match(value) or value.isdigit():
        return False
    if value[0] in "/~." or "://" in value or value.startswith(("$(", "`")):
        return False
    return any(c.isalpha() for c in value)


@dataclass(frozen=True)
class Found:
    kind: str  # "token" (a known credential shape) or "named" (a secret-named assignment)
    span: tuple[int, int]  # where to point a citation: the name for named values
    value: tuple[int, int]  # the secret itself (masked in citations)


def find(text: str) -> Iterator[Found]:
    """Credential occurrences in `text`, token shapes first; spans may overlap."""
    if not text:
        return
    for m in TOKEN_SHAPES.finditer(text):
        yield Found("token", m.span(), m.span())
    for pattern in (_ENV, _KEYED):
        for m in pattern.finditer(text):
            if secret_name(m.group("name")) and plausible_value(m.group("value")):
                yield Found("named", m.span("name"), m.span("value"))
    for m in _FLAG.finditer(text):
        if plausible_value(m.group("value")):
            yield Found("named", m.span(), m.span("value"))


def values(texts: Iterable[str]) -> frozenset[str]:
    """Distinct credential values in `texts`: used to mask the same secret wherever it
    reappears (a bare value printed on its own line, copied into code)."""
    out = set()
    for text in texts:
        for f in find(text):
            out.add(text[f.value[0] : f.value[1]])
    return frozenset(v for v in out if len(v) >= 8)


def mask(text: str, known: Iterable[str] = ()) -> str:
    """Replace every credential value (and any `known` value) with `***`."""
    spans = sorted(f.value for f in find(text))
    for value in sorted(set(known) - {""}, key=len, reverse=True):
        start = text.find(value)
        while start != -1:
            spans.append((start, start + len(value)))
            start = text.find(value, start + 1)
    if not spans:
        return text
    spans.sort()
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out, last = [], 0
    for start, end in merged:
        out.append(text[last:start])
        block = text[start:end]
        out.append("[private key]" if block.startswith("-----BEGIN") else "***")
        last = end
    out.append(text[last:])
    return "".join(out)
