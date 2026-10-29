"""Bounded Hub response syntax, never retrieval or report documents.

Only first-level identity fields and explicit HTTP/body frames are projected.
A valid identity prefix can survive a clipped tail, with incomplete coverage;
nested names, organizations and arbitrary model prose never become identities.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .hub_requests import namespace
from .jsonval import JsonObject, as_str

_MAX_RESPONSE = 262_144
_MAX_OBJECTS = 32
_MAX_FIELDS = 128
_NO_BODY_STATUS = 204
_FAILURE_STATUS = 400
_MAX_LENGTH_DIGITS = 12
_INVALID_LENGTH = -1
_ROOT = re.compile(r"(?m)^[ \t]*(?=[{\[])")
_SECTION = re.compile(r"(?m)^[=-]{3,}[^\r\n]*$")
_HTTP = re.compile(r"(?m)^HTTP/\d(?:\.\d)?[ \t]+(?P<status>[1-5]\d\d)(?P<reason>[^\r\n]*)\r?$")
_FILE = re.compile(
    r"(?m)^FILE[ \t]*:[ \t]+[^\r\n]+[ \t]+(?P<status>[1-5]\d\d)"
    r"[ \t]+(?P<size>\d{1,12})[ \t]*\r?$"
)
_WHOAMI = re.compile(
    r"[=-]{3,}\s*(?:https?://(?:huggingface\.co|hf\.co)/api/)?whoami(?:-v2)?\s*[=-]{3,}", re.I
)

_CONTENT_LENGTH = re.compile(r"(?im)^content-length:[ \t]*(\d+)[ \t]*\r?$")
_TRANSFER = re.compile(r"(?im)^transfer-encoding:")
_RUNNER = re.compile(
    r"(?m)^(?:Process exited with code \d+|Wall time: [^\r\n]+|"
    r"Chunk ID: [^\r\n]+|Final output:)[ \t]*\r?$"
)


@dataclass(frozen=True)
class HubIdentity:
    owner: str = field(repr=False)
    span: tuple[int, int]
    complete: bool
    scoped: bool = False


@dataclass(frozen=True)
class IdentityRead:
    identities: tuple[HubIdentity, ...] = ()
    complete: bool = True


@dataclass(frozen=True)
class HttpBody:
    status: int
    span: tuple[int, int]
    body_chars: int
    complete: bool = True


def _space(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _member(text: str, position: int, fields: JsonObject) -> tuple[str, object, int]:
    decoder = json.JSONDecoder()
    try:
        key, end = decoder.raw_decode(text, _space(text, position))
    except json.JSONDecodeError:
        raise ValueError("incomplete_object_key") from None
    name = as_str(key)
    colon = _space(text, end)
    if name is None or name in fields or text[colon : colon + 1] != ":":
        raise ValueError("invalid_or_duplicate_object_member")
    value: object
    value, end = decoder.raw_decode(text, _space(text, colon + 1))
    return name, value, _space(text, end)


def _field(text: str, position: int, fields: JsonObject) -> tuple[int, bool]:
    try:
        key, value, end = _member(text, position, fields)
    except json.JSONDecodeError as error:
        clipped = error.pos >= len(text) or error.msg.startswith("Unterminated string")
        if not clipped:
            fields.clear()
        return len(text), False
    except (ValueError, RecursionError):
        fields.clear()
        return len(text), False
    fields[key] = value
    return end, True


def _prefix(text: str, start: int) -> tuple[JsonObject, int, bool]:
    fields: JsonObject = {}
    position = _space(text, start + 1)
    if text[position : position + 1] == "}":
        return fields, position + 1, True
    end = len(text)
    for _ in range(_MAX_FIELDS):
        end, read = _field(text, position, fields)
        if not read:
            break
        delimiter = text[end : end + 1]
        if delimiter == "}":
            return fields, end + 1, True
        if delimiter != ",":
            if end < len(text):
                fields.clear()
            break
        position = end + 1
    return fields, end, False


def _preview_end(text: str, start: int, end: int, limit: int | None) -> int:
    if limit is None or not 0 < limit <= _MAX_RESPONSE:
        return end
    try:
        body = text[start:end].encode("utf-8")
        if len(body) < limit or body[limit:].strip():
            return end
        prefix = body[:limit].decode("utf-8")
    except UnicodeError:
        return end
    return start + len(prefix)


def _identity(
    text: str, start: int, preview_bytes: int | None
) -> tuple[HubIdentity | None, int, bool]:
    section = _SECTION.search(text, start + 1)
    boundary = section.start() if section else len(text)
    boundary = _preview_end(text, start, boundary, preview_bytes)
    fields, end, complete = _prefix(text[:boundary], start)
    name = as_str(fields.get("name"))
    owner = namespace(name) if name is not None else None
    profile = fields.get("type") == "user" and owner is not None
    headers = list(_SECTION.finditer(text, 0, start))
    scoped = bool(headers and _WHOAMI.fullmatch(headers[-1][0]))
    identity = HubIdentity(owner, (start, end), complete, scoped) if profile and owner else None
    return identity, end, complete


def _root_profile(
    text: str, start: int, preview_bytes: int | None
) -> tuple[HubIdentity | None, int, bool]:
    if text[start] == "{":
        return _identity(text, start, preview_bytes)
    try:
        _, end = json.JSONDecoder().raw_decode(text, start)
    except (ValueError, RecursionError):
        return None, len(text), False
    return None, end, True


def identity_profiles(text: str, *, preview_bytes: int | None = None) -> IdentityRead:
    """Read root user profiles only; a clipped JSON prefix stays explicitly partial.

    Stop at a malformed root rather than rediscovering nested profile-shaped data.
    No recovery of names from regex, prose, nested records or organizations.
    """
    if len(text) > _MAX_RESPONSE:
        return IdentityRead(complete=False)
    identities: list[HubIdentity] = []
    cursor = 0
    for _ in range(_MAX_OBJECTS):
        match = _ROOT.search(text, cursor)
        if match is None:
            return IdentityRead(tuple(identities))
        start = match.end()
        identity, cursor, complete = _root_profile(text, start, preview_bytes)
        if identity is not None:
            identities.append(identity)
        if not complete:
            return IdentityRead(tuple(identities), complete=False)
    return IdentityRead(tuple(identities), complete=False)


def _body_end(text: str, start: int, end: int) -> int:
    boundaries = [pattern.search(text, start, end) for pattern in (_SECTION, _RUNNER)]
    return min((match.start() for match in boundaries if match), default=end)


def _declared_length(headers: str) -> int | None:
    values = _CONTENT_LENGTH.findall(headers)
    names = len(re.findall(r"(?im)^content-length:", headers))
    if not names:
        return None
    if names != len(values) or len(values) != 1 or len(values[0]) > _MAX_LENGTH_DIGITS:
        return _INVALID_LENGTH
    return int(values[0])


def _body_complete(text: str, start: int, end: int, size: int | None, status: int) -> bool:
    if status >= _FAILURE_STATUS or status == _NO_BODY_STATUS or size == 0:
        return True
    try:
        actual = len(text[start:end].encode("utf-8"))
    except UnicodeError:
        return False
    return actual >= size if size is not None else actual > 0


def _http_body(text: str, match: re.Match[str], end: int) -> HttpBody | None:
    separator = re.search(r"\r?\n\r?\n", text[match.start() : end])
    start = match.start() + separator.end() if separator else end
    headers = text[match.end() : start]
    declared = _declared_length(headers)
    if declared == _INVALID_LENGTH or _TRANSFER.search(headers):
        return None
    end = _body_end(text, start, end)
    # A zero-length origin response or a CONNECT acknowledgement has no file body.
    no_body = declared == 0 or int(match["status"]) == _NO_BODY_STATUS
    no_body |= "connection established" in match["reason"].casefold()
    if declared is not None:
        end = min(end, start + declared)
    length = len(text[start:end].strip()) if not no_body else 0
    complete = _body_complete(text, start, end, declared, int(match["status"]))
    return HttpBody(int(match["status"]), (start, end), length, complete)


def _file_body(text: str, match: re.Match[str], end: int) -> HttpBody:
    start = match.end()
    end = _body_end(text, start, end)
    length = min(int(match["size"]), len(text[start:end].strip()))
    complete = int(match["size"]) == 0 or length > 0 or int(match["status"]) >= _FAILURE_STATUS
    return HttpBody(int(match["status"]), (start, end), length, complete)


def response_bodies(text: str) -> tuple[HttpBody, ...]:
    """Explicit HTTP headers or FILE: <target> <status> <size> preview records.

    Bare text, a status alone and shell exit codes do not establish receipt. These
    are syntax facts: endpoint and result attribution belong to the detector.
    """
    if len(text) > _MAX_RESPONSE:
        return ()
    frames = sorted(
        [(m, True) for m in _HTTP.finditer(text)] + [(m, False) for m in _FILE.finditer(text)],
        key=lambda pair: pair[0].start(),
    )
    bodies: list[HttpBody] = []
    for index, (match, http) in enumerate(frames):
        end = frames[index + 1][0].start() if index + 1 < len(frames) else len(text)
        body = _http_body(text, match, end) if http else _file_body(text, match, end)
        if body is not None:
            bodies.append(body)
    return tuple(bodies)
