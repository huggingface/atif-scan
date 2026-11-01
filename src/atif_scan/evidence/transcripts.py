"""Stored image transcripts: what an image model read in an image, by image hash.

`--image-model` (scan) and `hunt --image-model` ask an image model for the readable text
of images in a trace and keep each answer privately under <atif-scan home>/images/, one
file per image sha256 and model (`cli.images`). This module is the read side: the
answer's shape, its validation, and where a stored answer lives. The inspection server
reads transcripts from here and never asks a model.

A transcript is complete when the model read all of the image's text, or says it has
none; `partial` and `unreadable` are incomplete, and an image without a stored answer is
unread. Only a complete transcript stands in for the image: an incomplete one would read
as "the image didn't show it".
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..checks import MAX_IMAGE_TEXT, Legible
from ..data.jsonval import as_object, as_str

if TYPE_CHECKING:
    from pathlib import Path

SENSITIVE = frozenset({"none", "credential", "personal", "unclear"})
# What the model answers, and how each counts: all of an image's text, or the certainty
# that it has none, is a complete reading; anything less leaves the image unread.
TEXT_READ: dict[str, Legible] = {
    "all": "complete",
    "no_text": "complete",
    "partial": "partial",
    "unreadable": "none",
}
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
MAX_STORED = 256 * 1024
# Part of every stored answer's name. "s2": asked with atif-scan's own system prompt
# (`cli.fast_agent.INSTRUCTION`); earlier answers may have had the asking workspace's
# AGENTS.md as their system prompt, so they are not read (they are asked again).
STORE_VERSION = "s2"  # a stored answer: a capped transcript plus a few fields


@dataclass(frozen=True)
class Transcript:
    text: str
    text_read: str  # a TEXT_READ code
    instructions: bool
    sensitive: str

    @property
    def legible(self) -> Legible:
        return TEXT_READ[self.text_read]

    @property
    def complete(self) -> bool:
        return self.legible == "complete"

    def doc(self) -> dict[str, object]:
        return {
            "text": self.text,
            "text_read": self.text_read,
            "instructions": self.instructions,
            "sensitive": self.sensitive,
        }


def transcript_from(value: object) -> Transcript | None:
    """A valid answer object (a model's reply or a stored one), or None."""
    reply = as_object(value)
    text, read = as_str(reply.get("text")), as_str(reply.get("text_read"))
    sensitive, instructions = as_str(reply.get("sensitive")), reply.get("instructions")
    if text is None or read not in TEXT_READ or not isinstance(instructions, bool):
        return None
    if sensitive not in SENSITIVE:
        return None
    if read == "no_text" and text.strip():
        read = "all"  # it transcribed something after all: judge the text
    if len(text) > MAX_IMAGE_TEXT:
        read = "partial"  # cut, so not all of it
    return Transcript(text[:MAX_IMAGE_TEXT], read, instructions, sensitive)


def model_key(model: str) -> str:
    """A filename-safe key for a model string (which may hold any characters)."""
    return hashlib.sha256(model.encode()).hexdigest()[:16]


def stored_path(store: Path, digest: str, model: str) -> Path:
    """Where the answer for this image (sha256 hex) and model is kept."""
    return store / f"{digest}.{model_key(model)}.{STORE_VERSION}.json"


def read_stored(path: Path) -> Transcript | None:
    """A stored answer, or None when it is missing, too large or invalid."""
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_STORED + 1)
        if len(data) > MAX_STORED:
            return None
        return transcript_from(json.loads(data))
    except (OSError, ValueError, RecursionError):
        return None


@dataclass(frozen=True)
class TranscriptStore:
    """The stored transcripts of one image model: read only."""

    store: Path
    model: str

    def get(self, digest: str) -> Transcript | None:
        if not SHA256_HEX.fullmatch(digest):
            return None  # a sha256 hex digest, never a path
        return read_stored(stored_path(self.store, digest, self.model))
