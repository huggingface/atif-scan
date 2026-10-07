"""Inline media payloads: base64 images recorded inside a trajectory.

Only two things leave this module: a payload's sha256 (its identity, so an image check
can be matched to it) and, for the CLI's opt-in `--image-model`, the decoded bytes. The
bytes are never reported, cached in results or shown as text.

Recognised shapes: a data URI (in a content string, an image block's `source.path` or
`source.url` as fast-agent writes it, or OpenAI `image_url.url`) and an Anthropic-style
`source: {media_type, data}` block. File references and remote URLs are not read.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .jsonval import as_list, as_object, as_str

if TYPE_CHECKING:
    from collections.abc import Iterator

# The payload must be plain base64 (no whitespace): what every recorded shape uses.
DATA_URI = re.compile(r"data:((?:image|audio|video)/[\w.+-]+);base64,([A-Za-z0-9+/]+={0,2})")
MEDIA_TYPE = re.compile(r"(?:image|audio|video)/[\w.+-]+")


@dataclass(frozen=True)
class Media:
    digest: str  # sha256 of the decoded bytes, hex
    mime: str
    data: bytes = field(repr=False)


def decode(mime: str, payload: str) -> Media | None:
    """One base64 payload, or None when it isn't valid base64."""
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None
    return Media(hashlib.sha256(data).hexdigest(), mime.lower(), data) if data else None


def text_media(text: str) -> list[Media]:
    """Every data-URI payload in a string."""
    found = (decode(m[1], m[2]) for m in DATA_URI.finditer(text))
    return [m for m in found if m is not None]


def block_media(block: object) -> list[Media]:
    """The payloads of one image/audio/video content block (empty when it only refers to
    a file or URL, or its payload isn't valid base64)."""
    value = as_object(block)
    source = as_object(value.get("source"))
    image_url = value.get("image_url")
    uris = (source.get("path"), source.get("url"), as_object(image_url).get("url"), image_url)
    found = [m for uri in uris if (text := as_str(uri)) for m in _uri_media(text)]
    mime = as_str(source.get("media_type"))
    data = as_str(source.get("data"))
    if data is not None and mime is not None and MEDIA_TYPE.fullmatch(mime):
        media = decode(mime, data)
        found.extend([media] if media is not None else [])
    return found


def _uri_media(text: str) -> list[Media]:
    m = DATA_URI.fullmatch(text)
    media = decode(m[1], m[2]) if m else None
    return [media] if media is not None else []


MEDIA_BLOCKS = ("image", "image_url", "audio", "video")


def find_media(value: object) -> Iterator[Media]:
    """Every recognised payload anywhere in a raw (JSON) value, in document order."""
    if isinstance(value, str):
        yield from text_media(value) if "data:" in value else ()
    elif isinstance(value, dict):
        yield from _object_media(as_object(value))
    elif isinstance(value, list):
        for item in as_list(value):
            yield from find_media(item)


def _object_media(value: dict[str, object]) -> Iterator[Media]:
    if value.get("type") in MEDIA_BLOCKS:
        yield from block_media(value)
        return
    for item in value.values():
        yield from find_media(item)
