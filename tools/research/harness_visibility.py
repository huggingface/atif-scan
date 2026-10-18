"""Pure, conservative tool-result visibility evidence for private research.

The caller supplies its decoded recorded Content; this module neither imports the
trace adapter nor re-decodes that projection. Recorded lengths are a proxy, NOT
proof of provider-visible delivery. Unknown visibility never discards a result.

Verified source: Harbor's claude_code.py _format_tool_result at revision a59ebaac
appends tool_use_result stdout/stderr/metadata to native tool_result.content. The
exporter preserves that native block in extra.tool_result_metadata.raw_tool_result.
Only a matching native type/call ID establishes the model-visible content surface;
this is evidence in the supplied trace, not authentication of an untrusted export.
Block text uses Content's newline-joined projection, not provider wire byte counts.

Indusagi content/details JSON and Muse output/command JSON have no verified local
transport proof here. They stay unknown, as do plain ATIF strings: even an exact
persisted-output envelope or '[stdout]' delimiter could be command output. No
splitting, file/path access, recursive metadata search or serialized JSON guessing.

Lengths measure fully understood textual projections only. Media, computed text,
partial decoding and invalid UTF-8 bytes remain unknown (not zero). Only to_doc()
is an export; its fixed allowlist contains no content, paths, IDs or metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from atif_scan.data.content import content
from atif_scan.data.jsonval import as_list, as_object, as_str, is_object
from atif_scan.data.media import MEDIA_BLOCKS

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc, JsonObject
    from atif_scan.data.model import Content

Basis = Literal["claude_native_tool_result", "unknown"]


@dataclass(frozen=True)
class Visibility:
    """Private content surfaces plus snippet-free visibility measurements."""

    recorded: Content = field(repr=False)
    model_visible: Content | None = field(repr=False)
    basis: Basis

    @property
    def recorded_chars(self) -> int | None:
        return _chars(self.recorded)

    @property
    def recorded_bytes(self) -> int | None:
        return _bytes(self.recorded)

    @property
    def model_visible_chars(self) -> int | None:
        return _chars(self.model_visible)

    @property
    def model_visible_bytes(self) -> int | None:
        return _bytes(self.model_visible)

    def to_doc(self) -> Doc:
        return {
            "basis": self.basis,
            "recorded_chars": self.recorded_chars,
            "recorded_bytes": self.recorded_bytes,
            "model_visible_chars": self.model_visible_chars,
            "model_visible_bytes": self.model_visible_bytes,
        }


def _chars(value: Content | None) -> int | None:
    if value is None or not value.understood or value.media or value.computed:
        return None
    return len(value.text)


def _bytes(value: Content | None) -> int | None:
    if _chars(value) is None or value is None:
        return None
    try:
        return len(value.text.encode("utf-8"))
    except UnicodeEncodeError:
        return None


def _native_content(value: object) -> Content | None:
    """Only native strings or typed block arrays, never metadata text fields."""
    if isinstance(value, str):
        return content(value)
    if not isinstance(value, list):
        return None
    blocks = as_list(value)
    if not all(_content_block(block) for block in blocks):
        return None
    return content(blocks)


def _content_block(value: object) -> bool:
    if not is_object(value):
        return False
    kind = as_str(value.get("type"))
    if kind == "text":
        return as_str(value.get("text")) is not None
    # Payloads/URLs remain opaque; Content retains only media flags/digests.
    return kind in MEDIA_BLOCKS


def _claude_native(raw_result: JsonObject) -> Content | None:
    extra = as_object(raw_result.get("extra"))
    metadata = as_object(extra.get("tool_result_metadata"))
    native = as_object(metadata.get("raw_tool_result"))
    call_id = as_str(raw_result.get("source_call_id"))
    if (
        not call_id
        or native.get("type") != "tool_result"
        or as_str(native.get("tool_use_id")) != call_id
    ):
        return None
    if "is_error" in native and not isinstance(native["is_error"], bool):
        return None
    return _native_content(native.get("content"))


def visible_result(raw_result: JsonObject, family: str | None, recorded: Content) -> Visibility:
    """Select verified native content, retaining the caller's recorded projection.

    ``family`` is the parent's canonical family (``claude-code``). Other families
    and malformed/mismatched native evidence return model_visible=None. An empty
    native string/array is valid zero-length content; missing/null is unknown.
    """
    visible = _claude_native(raw_result) if family == "claude-code" else None
    basis: Basis = "claude_native_tool_result" if visible is not None else "unknown"
    return Visibility(recorded, visible, basis)
