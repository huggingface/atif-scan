"""OpenAI/ChatGPT account identifiers in trace text (stdlib only).

They're not credentials, but they identify the account a run used. They reach traces via
request headers (`ChatGPT-Account-Id: …`), decoded OAuth JWT claims (the
`"https://api.openai.com/auth"` claim object) and `OpenAI-Organization`/`OpenAI-Project`
headers. Scrub them before publishing a trace.

Three kinds of evidence, most specific first:

- **bound**: an identifier tied to an account key, header or claim:
  `chatgpt_account_id`/`chatgpt-account-id`/`ChatGPT-Account-Id` (any case; JSON key,
  header line, query or `KEY=value` form) with a UUID value; `chatgpt_user_id`;
  `OpenAI-Organization`/`OpenAI-Project` header values; and the id fields of the auth
  claim object (`chatgpt_account_id`, `chatgpt_user_id`, `user_id`, `organization_id`,
  `organizations[].id`).
- **shape**: an OpenAI org/user/project id on its own (`org-…`, `user-…`, `proj_…` with
  20+ alphanumerics, at least one digit or capital), not already part of a bound match.
- **context**: the auth claim object or a `chatgpt_plan_type` value with no id in it.

A bare UUID is never a finding: UUIDs are everywhere (request ids, tool-call ids). Values
already redacted (`<redacted>`, `[REDACTED]`, `***`) have no id shape and aren't found.

Findings carry spans only; values never leave memory except as masking input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

UUID = r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
# Random ids carry a digit or a capital; prose and hyphenated words ("user-friendly",
# "org-babel-execute") don't, and the whole hyphen-free token must be 20+ characters.
_SHAPE_BODY = r"(?=[a-z]*[A-Z0-9])[A-Za-z0-9]{20,}"
_SHAPE = rf"(?:org|user)-{_SHAPE_BODY}|proj_{_SHAPE_BODY}"
SHAPES = re.compile(rf"(?<![\w-])(?:{_SHAPE})(?![\w-])")
# Between a key and its value: optional (possibly JSON-escaped) quotes around `:` or `=`,
# or a comma for header tuples (`("ChatGPT-Account-Id", "…")`).
_SEP = r"""\\*["']?\]?\s*[:=,]\s*\\*["']?\s*"""
_VALUE_END = r"(?![\w-])"
# `chatgpt_account_id`, `chatgpt-account-id`, `ChatGPT-Account-Id`, `chatgptAccountId`,
# `CHATGPT_ACCOUNT_ID`, also after a prefix (`x-chatgpt-account-id`, `CODEX_…`): the
# lookbehind only rejects a letter or digit, so the name span starts at `chatgpt`.
ACCOUNT_KEY = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?P<name>chatgpt[_-]?(?:account|user)[_-]?id)"
    + _SEP
    + rf"(?P<value>{UUID}|user-[A-Za-z0-9]{{8,}}){_VALUE_END}"
)
# `OpenAI-Organization: org-…`, `OPENAI_ORG_ID=…`, `"openai-project": "proj_…"`.
# Any id-like value with a digit counts here: organization headers can carry a slug.
ORG_KEY = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?P<name>openai[_-](?:organization|org|project)(?:[_-]id)?)"
    + _SEP
    + rf"(?P<value>(?=[\w-]*\d)[A-Za-z0-9][\w-]{{7,}}){_VALUE_END}"
)
# The namespaced claim object in OpenAI OAuth JWTs, as JSON or a Python repr.
NAMESPACE = "https://api.openai.com/auth"
CLAIM_NAMESPACE = re.compile(re.escape(NAMESPACE) + _SEP.replace("[:=,]", ":") + r"\{")
CLAIM_ID = re.compile(
    r"""(?<![A-Za-z0-9_-])\\*["']?(?P<name>chatgpt_account_id|chatgpt_user_id|user_id|"""
    r"organization_id|account_id|id)"
    + _SEP.replace("[:=,]", "[:=]")
    + rf"(?P<value>{UUID}|(?=[\w-]*\d)[A-Za-z0-9][\w-]{{7,}}){_VALUE_END}"
)
PLAN_TYPE = re.compile(
    r"""(?i)(?<![A-Za-z0-9])(?P<name>chatgpt[_-]?plan[_-]?type)"""
    + _SEP.replace("[:=,]", "[:=]")
    + r"(?P<value>[a-z][a-z_-]{1,31})(?![\w-])"
)
CLAIM_LIMIT = 4096  # characters of a claim object scanned for its id fields
_PLACEHOLDER = re.compile(r"(?i)redacted|masked|placeholder|x{4,}|\*")


@dataclass(frozen=True)
class Found:
    kind: Literal["bound", "shape", "context"]
    span: tuple[int, int]  # where to point a citation: the key for bound values
    value: tuple[int, int]  # the identifier itself (masked in citations)


def find(text: str) -> Iterator[Found]:
    """Account identifiers in `text`: bound values first, then shape-only ids, then
    context (a claim object or plan type with no id found in it). Spans may overlap
    only across kinds that point at the same key."""
    if not text:
        return
    bound = list(_bound(text))
    yield from bound
    covered = [f.value for f in bound]
    for m in SHAPES.finditer(text):
        if not any(start <= m.start() < end for start, end in covered):
            yield Found("shape", m.span(), m.span())
    yield from _context(text, covered)


def _bound(text: str) -> Iterator[Found]:
    """One finding per value: a claim field can also be an account key."""
    found: dict[tuple[int, int], Found] = {}
    matches = [*ACCOUNT_KEY.finditer(text), *ORG_KEY.finditer(text)]
    for start, end in _claim_objects(text):
        matches += CLAIM_ID.finditer(text, start, end)
    for m in matches:
        if not _PLACEHOLDER.search(m["value"]):
            found.setdefault(m.span("value"), Found("bound", m.span("name"), m.span("value")))
    yield from sorted(found.values(), key=lambda f: f.value)


def _claim_objects(text: str) -> Iterator[tuple[int, int]]:
    """(start, end) of each auth claim object, by brace depth, capped at CLAIM_LIMIT."""
    for m in CLAIM_NAMESPACE.finditer(text):
        depth = 0
        end = min(len(text), m.end() + CLAIM_LIMIT)
        for i in range(m.end() - 1, end):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        yield m.start(), end


def _context(text: str, covered: list[tuple[int, int]]) -> Iterator[Found]:
    """Claim objects and plan types not already holding a bound id."""
    for start, end in _claim_objects(text):
        if not any(start <= s < end for s, _ in covered):
            name = (start, start + len(NAMESPACE))
            yield Found("context", name, name)
    for m in PLAN_TYPE.finditer(text):
        yield Found("context", m.span("name"), m.span("value"))


def values(texts: Iterable[str]) -> frozenset[str]:
    """Distinct identifier values (bound and shape, never context) in `texts`: used to
    mask the same id wherever it reappears, e.g. a bare UUID copied elsewhere."""
    out: set[str] = set()
    for text in texts:
        for f in find(text):
            if f.kind != "context":
                out.add(text[f.value[0] : f.value[1]])
    return frozenset(out)


def spans(text: str) -> list[tuple[int, int]]:
    """Identifier value spans to mask (bound and shape)."""
    return [f.value for f in find(text) if f.kind != "context"]
