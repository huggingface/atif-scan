"""Text content of a raw ATIF value: its text, whether it was understood, and whether it
is (or stands in for) media. Never arbitrary metadata, URLs or image payloads.
"""

from __future__ import annotations

import re

from .jsonval import as_list, as_object, as_str
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
    r"^\s*\[Image: original \d+x\d+, displayed at \d+x\d+\.|"
    r"""["']media_type["']\s*:\s*["'](?:image/|audio/|video/|application/pdf)""",
    re.I | re.M,
)


def content(value: object) -> Content:
    if value is None:
        return Content()
    if isinstance(value, str):
        media = bool(MEDIA_PLACEHOLDER.search(value) or MEDIA_INLINE.search(value[:4096]))
        return Content(value, media=media)
    if isinstance(value, list):
        parts = [content(part) for part in as_list(value)]
        return Content(
            "\n".join(p.text for p in parts),
            all(p.understood for p in parts),
            any(p.media for p in parts),
        )
    return _block_content(value)


def _block_content(value: object) -> Content:
    """One content block. Only text-bearing blocks, never arbitrary metadata/URL/image
    payloads."""
    block = as_object(value)
    if block.get("type") in ("image", "image_url", "audio", "video"):
        return Content(media=True)
    text = as_str(block.get("text"))
    return Content(text) if text is not None else Content(understood=False)
