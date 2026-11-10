"""Private, bounded request facts. Values are never report/export fields."""

from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlsplit

if TYPE_CHECKING:
    from .model import Locator

MAX_CHARS = 65_536
MAX_ITEMS = 128
MAX_DEPTH = 8
METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})


class Destination(StrEnum):
    EXTERNAL = "external"
    LOCAL = "local"
    UNKNOWN = "unknown"


class Role(StrEnum):
    HEADER = "header"
    BODY = "body"
    QUERY = "query"


@dataclass(frozen=True)
class Carrier:
    role: Role
    name: str = field(repr=False)
    value: str = field(repr=False)


@dataclass(frozen=True)
class OutboundRequest:
    at: Locator
    target: str = field(repr=False)
    method: str
    carriers: tuple[Carrier, ...] = field(repr=False)
    destination: Destination
    provenance: str
    complete: bool = True


@dataclass(frozen=True)
class Requests:
    requests: tuple[OutboundRequest, ...] = ()
    complete: bool = True


def _address_destination(host: str) -> Destination:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if host in {"localhost", "localhost.localdomain"} or "." not in host:
            return Destination.LOCAL
        if host.endswith((".localhost", ".local", ".internal", ".lan", ".home")):
            return Destination.LOCAL
        # Ambiguous legacy numeric IPv4 spellings are not external proof.
        numeric = re.fullmatch(r"[\d.xa-f]+", host)
        return Destination.UNKNOWN if numeric else Destination.EXTERNAL
    return Destination.EXTERNAL if address.is_global else Destination.LOCAL


def destination(target: str) -> Destination:
    """No resolution: private literals and local names are not external transfers."""
    try:
        url = urlsplit(target)
        host = (url.hostname or "").lower().rstrip(".")
        valid = (
            url.scheme in {"http", "https"}
            and bool(host)
            and url.username is None
            and url.password is None
            and (url.port is None or 0 < url.port < MAX_CHARS)
            and not re.search(r"[\s\\$`{}]", target)
        )
    except ValueError:
        return Destination.UNKNOWN
    return _address_destination(host) if valid else Destination.UNKNOWN


def scalar(value: object) -> str:
    if isinstance(value, str):
        if len(value) > MAX_CHARS or re.search(r"[$`]", value):
            raise ValueError("dynamic_or_large")
        return value
    if isinstance(value, bytes):
        return scalar(value.decode("utf-8"))
    raise ValueError("not_literal_text")


def _mapping_pairs(value: Mapping[object, object], role: Role, depth: int) -> list[Carrier]:
    if len(value) > MAX_ITEMS:
        raise ValueError("too_many")
    out = []
    for name, item in value.items():
        if not isinstance(name, str):
            raise ValueError("non_string_key")
        if isinstance(item, Mapping | list | tuple):
            out.extend(pairs(item, role, depth + 1))
        elif isinstance(item, str | bytes):
            out.append(Carrier(role, name, scalar(item)))
        elif item is not None and not isinstance(item, bool | int | float):
            raise ValueError("non_literal")
    return out


def pairs(value: object, role: Role, depth: int = 0) -> list[Carrier]:
    if depth > MAX_DEPTH:
        raise ValueError("too_deep")
    if isinstance(value, Mapping):
        return _mapping_pairs(value, role, depth)
    if isinstance(value, list | tuple) and len(value) <= MAX_ITEMS:
        return [carrier for item in value for carrier in pairs(item, role, depth + 1)]
    raise ValueError("not_pairs")


def body(value: object) -> list[Carrier]:
    if isinstance(value, Mapping | list | tuple):
        return pairs(value, Role.BODY)
    text = scalar(value)
    if text.startswith("@"):
        raise ValueError("file_body")
    try:
        decoded = json.loads(text)
    except (ValueError, RecursionError):
        decoded = None
    raw = Carrier(Role.BODY, "", text)
    if isinstance(decoded, dict | list):
        return [raw, *pairs(decoded, Role.BODY)]
    if "=" in text and not any(c in text for c in "\n{}<>"):
        return [
            raw,
            *(Carrier(Role.BODY, k, v) for k, v in parse_qsl(text, max_num_fields=MAX_ITEMS)),
        ]
    return [raw]


def request(
    at: Locator,
    target: str,
    method: str,
    carriers: list[Carrier],
    provenance: str,
    *,
    complete: bool = True,
) -> OutboundRequest:
    target = scalar(target)
    if method not in METHODS or len(carriers) > MAX_ITEMS:
        raise ValueError("unsupported_request")
    category = destination(target)
    if category == Destination.UNKNOWN:
        raise ValueError("unknown_target")
    query = urlsplit(target).query
    carriers = [
        *carriers,
        *(Carrier(Role.QUERY, k, v) for k, v in parse_qsl(query, max_num_fields=MAX_ITEMS)),
    ]
    return OutboundRequest(at, target, method, tuple(carriers), category, provenance, complete)
