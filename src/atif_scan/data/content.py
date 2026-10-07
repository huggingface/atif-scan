"""Text content of a raw ATIF value: its text, whether it was understood, and whether it
is (or stands in for) media. Never arbitrary metadata, URLs or image payloads: an
inline payload is kept only as its sha256 (`Content.media_ids`, see `media`).
"""

from __future__ import annotations

import re

from .jsonval import as_list, as_object, as_str
from .media import MEDIA_BLOCKS, block_media, text_media
from .model import Content

# Harness text standing in for an attached image (Devin CLI: "[Image 1]").
# Padding is same-line whitespace only: `^\s*` would span blank lines and rescan every
# run of them from each line start (quadratic); the matches are the same.
MEDIA_PLACEHOLDER = re.compile(
    r"^[^\S\n]*\[(?:Image|Screenshot|Attachment)\s*#?\d*\][^\S\n]*$", re.I | re.M
)
# Media serialized into a string: Codex `view_image` results (a repr of
# `[{'type': 'input_image', 'image_url': 'data:image/png;base64,…'}]`), a data URI with a
# real payload (the smallest PNG is ~90 base64 characters), Claude Code's image-size
# note for a Read image, and PDF/image blocks written as JSON (TB2.1). A short data URI
# is text, not an image: sanitizer tests and HTML the agent writes (`<img
# src="data:image/png;base64,abc">`, TB4 html-js-filter) would otherwise make the field
# unreadable and every unprompted-recall check unknown from that step on.
MEDIA_INLINE = re.compile(
    r"\bdata:(?:image|audio|video)/[\w.+-]+;base64,[A-Za-z0-9+/]{64}|"
    r"""["']image_url["']\s*:\s*["']data:(?:image|audio|video)/|"""
    r"^[^\S\n]*\[Image: original \d+x\d+, displayed at \d+x\d+\.|"
    r"""["']media_type["']\s*:\s*["'](?:image/|audio/|video/|application/pdf)""",
    re.I | re.M,
)
# Media in a string that isn't a data URI: Claude Code's note for an image it read, or a
# block written as JSON. Its payload (if any) isn't identified, so the field stays unknown.
MEDIA_UNIDENTIFIED = re.compile(
    r"^[^\S\n]*\[Image: original \d+x\d+, displayed at \d+x\d+\.|"
    r"""["']media_type["']\s*:\s*["'](?:image/|audio/|video/|application/pdf)""",
    re.I | re.M,
)


def content(value: object) -> Content:
    if value is None:
        return Content()
    if isinstance(value, str):
        return _text_content(value)
    if isinstance(value, list):
        parts = [content(part) for part in as_list(value)]
        media = [p for p in parts if p.media]
        # Identified only when every media part is: one placeholder leaves it unknown.
        ids = (
            tuple(i for p in media for i in p.media_ids) if all(p.media_ids for p in media) else ()
        )
        return Content(
            "\n".join(p.text for p in parts),
            all(p.understood for p in parts),
            bool(media),
            ids,
        )
    return _block_content(value)


def _text_content(value: str) -> Content:
    if MEDIA_PLACEHOLDER.search(value):
        return Content(value, media=True)  # "[Image 1]": nothing to identify
    if not MEDIA_INLINE.search(value[:4096]):
        return Content(value)
    # Identified when the media it holds are data URIs (a Codex `view_image` repr) only.
    unidentified = MEDIA_UNIDENTIFIED.search(value) is not None
    ids = () if unidentified else tuple(m.digest for m in text_media(value))
    return Content(value, media=True, media_ids=ids)


def _block_content(value: object) -> Content:
    """One content block. Only text-bearing blocks, never arbitrary metadata/URL/image
    payloads (an inline payload is kept as its digest only)."""
    block = as_object(value)
    if block.get("type") in MEDIA_BLOCKS:
        return Content(media=True, media_ids=tuple(m.digest for m in block_media(block)))
    text = as_str(block.get("text"))
    return Content(text) if text is not None else Content(understood=False)
