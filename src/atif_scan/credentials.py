"""Credential shapes shared by the secret detectors and citation masking (stdlib only).

Two kinds of evidence:

- **Token shapes** that are credentials wherever they appear (`sk-…`, `hf_…`, `ghp_…`,
  AWS key IDs, JWTs, private-key blocks, `LLM|<id>|<token>` harness keys, …).
- **Named values**: an assignment whose *name* says it holds a secret
  (`OPENAI_API_KEY=…`, `"api_key": "…"`, `os.environ["X_TOKEN"] = "…"`,
  `--api-key …`, `x-api-key: …`). Names are split into components, so
  `TIKTOKEN_CACHE_DIR` or `MAX_TOKENS` are not secrets, and path/URL/placeholder values
  (`/root/.ssh/id_rsa`, `${KEY}`, `<your-key>`, `***`) are ignored.

`find` is the broad masking contract, not the exposure predicate. `find_exposures`
adds contextual exclusions for code identifiers and explicit dummy values. Never
use exposure findings to decide what is safe to publish.

Findings carry spans only; values never leave memory except as masking input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

# The leading lookahead lists every alternative's first character (`-----BEGIN`, sk/pk/rk,
# hf_, gh*_/github_pat_, xox*, AKIA/AIza, eyJ, LLM|): it adds nothing to the match, but lets
# the regex engine skip to candidate positions (3-4x faster). Update it with any new
# alternative (tests/test_detectors_review.py checks it against the alternatives).
_PRIVATE_KEY = r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"
PRIVATE_KEY = re.compile(_PRIVATE_KEY, re.S)
TOKEN_SHAPES = re.compile(
    r"(?=[-sprhgxAeL])(?:" + _PRIVATE_KEY + "|"
    # Real keys carry a digit (`sk-proj-…`, `sk-ant-api03-…`); `rk-free-variables-list`
    # in Lisp code doesn't.
    r"\b(?:sk|pk|rk)-(?=[\w-]*\d)[A-Za-z0-9_-]{16,}|"
    r"\bhf_[A-Za-z0-9]{20,}|"
    r"\bgh[pousr]_[A-Za-z0-9]{20,}|"
    r"\bgithub_pat_[A-Za-z0-9_]{20,}|"
    r"\bxox[abprs]-[A-Za-z0-9-]{10,}|"
    r"\bAKIA[0-9A-Z]{16}\b|"
    r"\bAIza[0-9A-Za-z_-]{35}|"
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}|"
    # Harness-issued model keys (`LLM|<numeric id>|<token>`), seen in leaderboard sandboxes.
    r"\bLLM\|\d{6,}\|[A-Za-z0-9_-]{12,})",
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
    # Not inside a hyphenated word: "per-token probabilities", "multi-token approach".
    # The lookahead (first characters of both alternatives) only speeds up scanning.
    r"(?i)(?=[-xa])(?:(?<![\w-])--?(?P<flag>api[_-]?key|token|password|secret)[= ]\s*|"
    r"\b(?P<header>x-api-key|api-key)\s*:\s*)(?P<q>['\"]?)(?P<value>[^\s'\"]+)"
)
_SECRET_PART = re.compile(
    r"(?:^|_)(?:api_?key|apikey|secret|token|password|passwd|pwd|credentials?|"
    r"private_key|access_key|secret_key|session_key|auth_key|key)(?:_|$)"
)
_NOT_SECRET_PART = re.compile(
    r"(?:^|_)(?:path|file|dir|url|uri|host|port|name|type|cache|len|length|size|count|"
    r"max|min|limit|env|header|field|prefix|id|ids|format|mode|var|vars|fn|func|"
    # `KeyError: 'x'`, `"apiKeySource": "…"`, public site/captcha/Stripe keys in page HTML.
    r"error|source|public|publishable|site|captcha|"
    # keyUsage, past_key_values, src_key_padding_mask, token_owner, secret_oid, token_chars
    r"usage|values?|mask|padding|owner|oid|chars?|"
    # Public signing keys (`GPG_KEY=<fingerprint>` in every official python image's env)
    # and database/dict keys: primary_key, foreign_key, sort_key, partition_key, displayKey.
    r"gpg|pgp|fingerprint|primary|sort|partition|display|lookup)(?:_|$)"
)
# Identifiers, not secrets: `rate_key = "embedding_lr"`, `primaryKey="customerId"`, Redis
# key names (`STATE_KEY = "cutover:state"`), dunder markers, parameter paths
# (`layers.0.mlp.weight`). Letters only (bar numeric path segments), so
# `opaqueCredential98765` or `svc_pass_2026` still count.
_IDENTIFIER = re.compile(
    r"[a-z]+(?:_[a-z]+)+|[a-z]+(?:[A-Z][a-z]+)+|[a-z_]+(?::[a-z_]+)+|__[a-z_]+__|"
    r"[a-z_]+(?:\.(?:[a-z_]+|\d+))*\.\d+(?:\.(?:[a-z_]+|\d+))+"
)
_PLACEHOLDER = re.compile(
    r"(?i)^(?:\*+|x+|\.+|<[^>]*>|\$\{?\w+\}?|%\w+%|your[\w-]*|changeme|none|null|nil|"
    r"true|false|dummy|test\w*|example\w*|placeholder|redacted|masked|secret|password|"
    r"undefined|sk-\.\.\.|\.\.\.)$"
)


# Code, not secrets: attribute access (`scores.get`, `cv2.contourArea`) and references to
# an environment variable by name (`-password=KEY_PASSWORD`).
_CODE_VALUE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+(?:=\S*)?")


def secret_name(name: str) -> bool:
    """Does an assignment name say it holds a secret? Components split on _ - and camelCase."""
    parts = re.split(r"[_\-]+|(?<=[a-z0-9])(?=[A-Z])", name)
    joined = "_".join(p.lower() for p in parts if p)
    if joined == "key":  # `sorted(x, key=len)`, `const key = "OP_"`: code, not secrets
        return False
    return bool(_SECRET_PART.search(joined)) and not _NOT_SECRET_PART.search(joined)


# Shorter values (and all-digit ones) are never treated as secrets.
MIN_SECRET_LENGTH = 8


def plausible_value(value: str) -> bool:
    """Historical masking heuristic, retained independently of exposure filtering."""
    if len(value) < MIN_SECRET_LENGTH or _PLACEHOLDER.match(value) or value.isdigit():
        return False
    return not _code_like(value) and any(c.isalpha() for c in value)


def _code_like(value: str) -> bool:
    """A path, URL, substitution, code expression or bare identifier (non-empty value)."""
    return bool(
        value[0] in "/~."
        or "://" in value
        or value.startswith(("$(", "`"))
        or any(c in value for c in "()[]{}<>\\")
        or value.endswith(",")
        or _CODE_VALUE.fullmatch(value)
        or _IDENTIFIER.fullmatch(value)
    )


# These exclusions apply only to exposure findings, never to masking. A generic
# "key" can be a lookup/schema key; explicit authentication components take priority.
_AUTH_PARTS = frozenset(
    {
        "api",
        "apikey",
        "secret",
        "token",
        "password",
        "passwd",
        "pwd",
        "credential",
        "credentials",
        "private",
        "access",
        "session",
        "auth",
    }
)
_LOOKUP_PARTS = frozenset(
    {
        # Configuration and descriptive metadata keys.
        "setting",
        "settings",
        "enabled",
        "analysis",
        "description",
        "options",
        # Message routing, merge selectors and relational schema keys.
        "routing",
        "merge",
        "foreign",
        # Scoped registries, plugin registries, object handles and map selectors.
        # Match components, not particular task identifiers or adjacent word pairs.
        "scope",
        "plugin",
        "handle",
        "map",
    }
)

# Whole-value semantics, not prefixes: "testament..." and arbitrary credentials
# beginning with "test" are still possible secrets. A test directory is not evidence.
_LITERAL_PLACEHOLDER = re.compile(
    r"(?i)^(?:\*+|x+|\.+|<[^>]*>|\$\{?\w+\}?|%\w+%|your[\w-]*|changeme|"
    r"none|null|nil|true|false|dummy|test|example|placeholder|redacted|masked|"
    r"secret|password|undefined|sk-\.\.\.)$"
)
_DUMMY_CREDENTIAL = re.compile(
    r"(?i)^(?:sk[-_])?(?:dummy|test|example|fake|placeholder)[-_]"
    r"(?:(?:api|access|auth|secret)[-_])?(?:key|token|secret|password)"
    r"(?:[-_](?:\d+|dummy|test|example|fake|placeholder))?$"
)
_PEM_PLACEHOLDER = re.compile(
    r"-----BEGIN (?P<label>[A-Z ]*PRIVATE KEY)-----\s*"
    r"(?:\.{3}|…|<[^<>\r\n]*>)\s*"
    r"-----END (?P=label)-----"
)


def _exposure_name(name: str) -> bool:
    if not secret_name(name):
        return False
    # Also split acronym-to-word boundaries (e.g. APIKey).
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    parts = set(re.split(r"[_\-]+|(?<=[a-z0-9])(?=[A-Z])", name))
    parts = {p.lower() for p in parts}
    if parts & _AUTH_PARTS:
        return True
    return not parts & _LOOKUP_PARTS


def _exposure_value(value: str) -> bool:
    # Keep the historical length/numeric boundary: collected values are replaced
    # globally in prompts and citations, not just at the assignment site.
    if len(value) < MIN_SECRET_LENGTH or value.isdigit():
        return False
    if _LITERAL_PLACEHOLDER.fullmatch(value) or _DUMMY_CREDENTIAL.fullmatch(value):
        return False
    return not _code_like(value) and any(c.isalpha() for c in value)


def _masked_value(value: str) -> bool:
    return plausible_value(value) or _exposure_value(value)


@dataclass(frozen=True)
class Found:
    kind: str  # "token" (a known credential shape) or "named" (a secret-named assignment)
    span: tuple[int, int]  # where to point a citation: the name for named values
    value: tuple[int, int]  # the secret itself (masked in citations)


def find(text: str) -> Iterator[Found]:
    """Broad masking candidates, token shapes first; spans may overlap.

    Preserves the historical candidates even when exposure context excludes them.
    Also includes opaque test-prefixed values, with the same length/numeric limits.
    Use `find_exposures` for detector findings, not this privacy-oriented API.
    """
    if not text:
        return
    for m in TOKEN_SHAPES.finditer(text):
        yield Found("token", m.span(), m.span())
    yield from _named(text, secret_name, _masked_value)


def find_exposures(text: str) -> Iterator[Found]:
    """Context-aware review findings; deliberately narrower than masking candidates.

    Known token shapes override non-secret assignment names. Only self-contained,
    explicit dummy tokens and placeholder-only closed PEM blocks are excluded.
    Incomplete or otherwise suspicious PEM blocks remain review candidates.
    """
    for m in TOKEN_SHAPES.finditer(text):
        value = m.group()
        if not _DUMMY_CREDENTIAL.fullmatch(value) and not _PEM_PLACEHOLDER.fullmatch(value):
            yield Found("token", m.span(), m.span())
    yield from _named(text, _exposure_name, _exposure_value)


def _named(
    text: str, secret: Callable[[str], bool], plausible: Callable[[str], bool]
) -> Iterator[Found]:
    """Secret-named assignments (`secret` names, `plausible` values) and secret flags."""
    for pattern in (_ENV, _KEYED):
        for m in pattern.finditer(text):
            if secret(m.group("name")) and plausible(m.group("value")):
                yield Found("named", m.span("name"), m.span("value"))
    for m in _FLAG.finditer(text):
        if plausible(m.group("value")):
            yield Found("named", m.span(), m.span("value"))


def values(texts: Iterable[str]) -> frozenset[str]:
    """Distinct credential values in `texts`: used to mask the same secret wherever it
    reappears (a bare value printed on its own line, copied into code)."""
    out: set[str] = set()
    for text in texts:
        for f in find(text):
            out.add(text[f.value[0] : f.value[1]])
    return frozenset(v for v in out if len(v) >= MIN_SECRET_LENGTH)


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
    out: list[str] = []
    last = 0
    for start, end in merged:
        out.append(text[last:start])
        block = text[start:end]
        out.append("[private key]" if block.startswith("-----BEGIN") else "***")
        last = end
    out.append(text[last:])
    return "".join(out)
