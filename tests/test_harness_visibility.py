"""Synthetic-only visibility tests; no traces, providers or trajectory execution."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from atif_scan.data.content import content
from atif_scan.data.model import Content

if TYPE_CHECKING:
    from atif_scan.data.jsonval import JsonObject

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools" / "research"))
import harness_visibility as visibility


def _result(native_content: object = "native synthetic text") -> JsonObject:
    return {
        "source_call_id": "synthetic-call",
        "content": "recorded synthetic text",
        "extra": {
            "tool_result_metadata": {
                "raw_tool_result": {
                    "type": "tool_result",
                    "tool_use_id": "synthetic-call",
                    "content": native_content,
                    "is_error": False,
                },
                "tool_use_result": {"stdout": "synthetic appended stdout"},
            },
        },
    }


def _native(value: object) -> JsonObject:
    return {
        "source_call_id": "synthetic-call",
        "extra": {"tool_result_metadata": {"raw_tool_result": value}},
    }


def test_preview_is_native_not_exporter_extras() -> None:
    preview = (
        "<persisted-output>\nOutput too large (30KB). Full output saved to: "
        "/synthetic/output.txt\n\nPreview (first 2KB):\n" + "p" * 2048 + "\n</persisted-output>"
    )
    raw = _result(preview)
    exported = preview + "\n\n[stdout]\n" + "s" * 30000 + "\n[metadata] synthetic"
    raw["content"] = exported
    recorded = Content(exported)
    before = copy.deepcopy(raw)
    result = visibility.visible_result(raw, "claude-code", recorded)
    assert raw == before
    assert result.recorded is recorded
    assert result.model_visible == Content(preview)
    assert result.basis == "claude_native_tool_result"
    assert result.recorded_chars == len(exported)
    assert result.recorded_bytes == len(exported.encode())
    assert result.model_visible_chars == len(preview)
    assert result.model_visible_bytes == len(preview.encode())


def test_native_blocks_not_appended_error_or_metadata() -> None:
    raw = _native(
        {
            "type": "tool_result",
            "tool_use_id": "synthetic-call",
            "is_error": True,
            "content": [
                {"type": "text", "text": " é\n[stdout]\ninside command output "},
                {"type": "text", "text": "second block", "metadata": "not content"},
            ],
        }
    )
    recorded = Content("exporter [error] tool reported failure\n[stdout]\nappended")
    result = visibility.visible_result(raw, "claude-code", recorded)
    expected = " é\n[stdout]\ninside command output \nsecond block"
    assert result.model_visible == Content(expected)
    assert result.model_visible_chars == len(expected)
    assert result.model_visible_bytes == len(expected.encode())
    assert result.recorded is recorded


@pytest.mark.parametrize("value", ["", [], [{"type": "text", "text": ""}]])
def test_explicit_empty_native_is_known_zero(value: object) -> None:
    result = visibility.visible_result(_result(value), "claude-code", Content("exporter extras"))
    assert result.model_visible == Content("")
    assert result.basis == "claude_native_tool_result"
    assert result.model_visible_chars == result.model_visible_bytes == 0
    assert result.recorded_chars == 15


@pytest.mark.parametrize(
    "native",
    [
        None,
        [],
        "spoofed",
        {"type": "text", "tool_use_id": "synthetic-call", "content": "spoofed"},
        {"type": ["tool_result"], "tool_use_id": "synthetic-call", "content": "spoofed"},
        {"type": "tool_result", "tool_use_id": "wrong-call", "content": "spoofed"},
        {"type": "tool_result", "tool_use_id": 1, "content": "spoofed"},
        {"type": "tool_result", "content": "spoofed"},
        {"type": "tool_result", "tool_use_id": "synthetic-call"},
        {"type": "tool_result", "tool_use_id": "synthetic-call", "content": None},
        {
            "type": "tool_result",
            "tool_use_id": "synthetic-call",
            "content": "spoofed",
            "is_error": "false",
        },
    ],
)
def test_invalid_native_identity_or_shape_is_unknown(native: object) -> None:
    recorded = Content("retained synthetic observation")
    result = visibility.visible_result(_native(native), "claude-code", recorded)
    assert result.model_visible is None
    assert result.model_visible_chars is result.model_visible_bytes is None
    assert result.basis == "unknown"
    assert result.recorded is recorded
    assert result.recorded_chars == len(recorded.text)


@pytest.mark.parametrize("call_id", [None, "", 1, False, [], {}])
def test_missing_or_invalid_source_call_id_never_matches(call_id: object) -> None:
    raw = _native({"type": "tool_result", "tool_use_id": call_id, "content": "spoofed"})
    raw["source_call_id"] = call_id
    assert visibility.visible_result(raw, "claude-code", Content()).model_visible is None


@pytest.mark.parametrize(
    "value",
    [
        {"text": "not a native string or block array"},
        0,
        False,
        ["not a native block"],
        [None],
        [[{"type": "text", "text": "nested"}]],
        [{"text": "missing type"}],
        [{"type": "text", "text": None}],
        [{"type": "metadata", "text": "not model content"}],
        [{"type": ["text"], "text": "invalid type"}],
        [{"type": "text", "text": "known"}, {"type": "unknown", "text": "unknown"}],
        {"content": [{"type": "text", "text": "nested wrapper"}]},
    ],
)
def test_malformed_native_content_never_becomes_empty_or_partial_known(value: object) -> None:
    result = visibility.visible_result(_result(value), "claude-code", Content("retained"))
    assert result.model_visible is None
    assert result.model_visible_chars is None


@pytest.mark.parametrize("kind", ["image", "image_url", "audio", "video"])
def test_native_media_is_retained_but_lengths_are_unknown(kind: str) -> None:
    blocks = [
        {"type": "text", "text": "synthetic caption"},
        {"type": kind, "source": {"url": "https://synthetic.invalid/media"}},
    ]
    result = visibility.visible_result(_result(blocks), "claude-code", Content("exporter text"))
    assert result.model_visible is not None
    assert result.model_visible.media
    assert result.model_visible.understood
    assert "synthetic.invalid" not in result.model_visible.text
    assert result.model_visible_chars is result.model_visible_bytes is None
    assert result.basis == "claude_native_tool_result"
    assert result.recorded_bytes == 13


def test_native_serialized_json_string_is_not_decoded() -> None:
    text = '[{"type":"text","text":"synthetic inner text"}]'
    result = visibility.visible_result(_result(text), "claude-code", Content(text))
    assert result.model_visible == Content(text)
    assert result.model_visible_chars == len(text)


@pytest.mark.parametrize("family", [None, "fast-agent", "indusagi", "muse-code", "codex"])
def test_native_metadata_from_other_family_is_not_proof(family: str | None) -> None:
    result = visibility.visible_result(_result(), family, Content("recorded"))
    assert result.model_visible is None
    assert result.basis == "unknown"
    assert result.recorded_chars == 8


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"extra": None},
        {"extra": "malformed"},
        {"extra": {"tool_result_metadata": []}},
        {"extra": {"tool_result_metadata": {"tool_use_result": {"content": "not native"}}}},
        {"raw_tool_result": {"type": "tool_result", "content": "wrong location"}},
    ],
)
def test_missing_or_malformed_metadata_preserves_recorded(raw: JsonObject) -> None:
    recorded = Content("output contains\n[stdout]\nthis delimiter is not evidence")
    result = visibility.visible_result(raw, "claude-code", recorded)
    assert result.recorded is recorded
    assert result.model_visible is None


@pytest.mark.parametrize(
    ("family", "payload"),
    [
        (
            "indusagi",
            {
                "content": [{"type": "text", "text": "synthetic preview"}],
                "details": {
                    "truncation": {"truncated": True},
                    "fullOutputPath": "/synthetic/output.txt",
                },
            },
        ),
        ("indusagi", {"content": [{"type": "text", "text": "preview"}], "details": [None]}),
        ("indusagi", {"content": [{"text": "untyped preview"}], "details": {}}),
        ("muse-code", {"output": "synthetic preview", "command": "not executed", "metadata": {}}),
    ],
)
def test_unverified_serialized_transport_stays_recorded_proxy(
    family: str, payload: JsonObject
) -> None:
    text = json.dumps(payload)
    recorded = Content(text)
    result = visibility.visible_result({"content": text}, family, recorded)
    assert result.model_visible is None
    assert result.basis == "unknown"
    assert result.recorded_chars == len(text)
    assert result.recorded_bytes == len(text.encode())


def test_persisted_envelope_without_native_proof_is_not_split() -> None:
    text = (
        "<persisted-output>\nOutput too large (30KB). Full output saved to: "
        "/synthetic/output.txt\n\nPreview (first 2KB):\nsynthetic\n</persisted-output>"
        "\n\n[stdout]\nsynthetic command output"
    )
    recorded = content(text)
    result = visibility.visible_result({"content": text}, "claude-code", recorded)
    assert result.model_visible is None
    assert result.recorded_chars == len(text)
    assert result.recorded is recorded


@pytest.mark.parametrize(
    "recorded",
    [
        Content(understood=False),
        Content("partial", understood=False),
        Content(media=True),
        Content("runtime", computed=True),
    ],
)
def test_unknown_recorded_lengths_do_not_erase_known_native(recorded: Content) -> None:
    result = visibility.visible_result(_result("native"), "claude-code", recorded)
    assert result.recorded_chars is result.recorded_bytes is None
    assert result.model_visible_chars == result.model_visible_bytes == 6


def test_invalid_unicode_bytes_are_unknown_not_guessed() -> None:
    text = "synthetic\ud800"
    result = visibility.visible_result(_result(text), "claude-code", Content(text))
    assert result.recorded_chars == result.model_visible_chars == len(text)
    assert result.recorded_bytes is result.model_visible_bytes is None


def test_to_doc_is_exact_allowlist_and_repr_has_no_private_content() -> None:
    private = "synthetic-private-payload"
    result = visibility.visible_result(_result(private), "claude-code", Content(private))
    expected = {
        "basis": "claude_native_tool_result",
        "recorded_chars": len(private),
        "recorded_bytes": len(private),
        "model_visible_chars": len(private),
        "model_visible_bytes": len(private),
    }
    assert result.to_doc() == expected
    assert private not in json.dumps(result.to_doc())
    assert private not in repr(result)
    exported = result.to_doc()
    exported["basis"] = "modified"
    assert result.to_doc() == expected
