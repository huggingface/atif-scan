"""Synthetic-only harness output signature and export regressions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools" / "research"))
import harness_outputs as outputs


def _fast(guidance: str, *, total: int = 30) -> str:
    return (
        "synthetic head\n"
        "[Output truncated: showing first 4 bytes and last 6 bytes of "
        f"{total} bytes (~2 of ~7 tokens); omitted 20 middle bytes. {guidance}]\n"
        "synthetic tail"
    )


def _artifact(*, partial: bool = False) -> str:
    availability = (
        "The first 12 bytes of the output are available during this session at "
        "/tmp/synthetic/output.txt; the temporary-file quota was reached."
        if partial
        else "The complete output is available during this session at /tmp/synthetic/output.txt."
    )
    return (
        availability + " Use read_text_file for selected line ranges or run a targeted search "
        "against that file; "
        "avoid reading the entire file unless necessary."
    )


def _mini(*, elided: object = 15, head: str | None = None) -> str:
    return json.dumps(
        {
            "returncode": 0,
            "output_head": head if head is not None else "é" * 5000,
            "output_tail": "z" * 5000,
            "elided_chars": elided,
            "warning": "Output too long.",
        }
    )


def _muse(*, truncated: object = True) -> str:
    return json.dumps(
        {
            "chunk_id": "synthetic",
            "command": "NOT EXECUTED",
            "description": "synthetic only",
            "exit_code": 0,
            "terminal_status": "finished",
            "output": "é🙂",
            "original_output_bytes": 50,
            "original_output_tokens": 20,
            "truncated": truncated,
        }
    )


@pytest.mark.parametrize("partial", [False, True])
def test_fast_head_tail_spool_is_private(partial: bool):
    text = _fast(_artifact(partial=partial))
    evidence = outputs.parse_output(text, "fast-agent")
    assert evidence.truncation
    assert evidence.marker == "fast_agent.head_tail_bytes"
    assert evidence.reported_original_bytes == 30
    assert evidence.omitted_bytes == 20
    assert evidence.limit == 10
    assert evidence.limit_unit == "bytes"
    assert evidence.estimated_original_tokens == 7
    assert evidence.reported_original_tokens is None
    assert evidence.payload_chars is None  # added boundary newlines are not raw payload
    assert evidence.spool_paths == ("/tmp/synthetic/output.txt",)
    assert evidence.spool_complete is not partial
    assert evidence.returned_chars == len(text)
    assert evidence.returned_bytes == len(text.encode("utf-8"))
    report = json.dumps(evidence.to_doc())
    assert "synthetic" not in report
    assert "/tmp/" not in report
    assert "spool_paths" not in report
    assert "/tmp/" not in repr(evidence)


@pytest.mark.parametrize(
    ("guidance", "indicated", "complete"),
    [
        ("Increase shell_execution.output_byte_limit to retain more.", False, None),
        ("Use action='read_output' for retained output.", True, None),
        (
            "The complete output can be read through the managed process handle. "
            "Use the process tool with action='read_output' and the process_id from this result.",
            True,
            True,
        ),
        (
            "The first 12 bytes retained before the temporary-output quota was reached can be "
            "read through the managed process handle. Use the process tool with "
            "action='read_output' and the process_id from this result.",
            True,
            False,
        ),
    ],
)
def test_fast_nonpath_retention(guidance: str, indicated: bool, complete: bool | None):
    evidence = outputs.parse_output(_fast(guidance), "fast-agent")
    assert evidence.truncation
    assert evidence.spool_indicated is indicated
    assert evidence.spool_complete is complete
    assert evidence.spool_paths == ()


def test_fast_extended_guidance_and_preview():
    guidance = _artifact() + (
        " Also, before drawing conclusions from truncated output, "
        "inspect the relevant retained content."
    )
    assert outputs.parse_output(_fast(guidance), "fast-agent").truncation
    text = (
        "é\n[Output truncated: showing 2 of 8 bytes. "
        "Use action='read_output' for retained output.]\n"
    )
    evidence = outputs.parse_output(text, "fast-agent")
    assert evidence.marker == "fast_agent.preview_bytes"
    assert evidence.limit == 2
    assert evidence.reported_original_bytes == 8
    assert evidence.omitted_bytes == 6
    assert evidence.spool_indicated
    assert evidence.returned_bytes == len(text) + 1


@pytest.mark.parametrize(
    "text",
    [
        _fast("Read more at /tmp/arbitrary."),
        _fast(_artifact(), total=40),
        _fast(_artifact()).replace("/tmp/synthetic/output.txt", "https://example.invalid/private"),
        "[Output truncated: something from an unknown harness version]",
    ],
)
def test_fast_damaged_or_unknown_envelope_is_not_negative(text: str):
    evidence = outputs.parse_output(text, "fast-agent")
    assert not evidence.truncation
    assert evidence.kind == "possible_signal"
    assert evidence.spool_paths == ()


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "Script completed\nWall time 0.2 seconds\nOutput:\n\n",
        "Chunk ID: abc123\nWall time: 0.2 seconds\nProcess exited with code 0\nFinal output:\n",
        "Chunk ID: abc123\nWall time: 0.2 seconds\nProcess running with session ID 12\nOutput:\n",
    ],
)
def test_codex_warning_reports_tokens_not_bytes(prefix: str):
    text = (
        prefix
        + "Warning: truncated output (original token count: 12345)\nTotal output lines: 700\nx\n"
    )
    evidence = outputs.parse_output(text, "codex")
    assert evidence.truncation
    assert evidence.marker == "codex.token_warning"
    assert evidence.reported_original_tokens == 12345
    assert evidence.reported_original_lines == 700
    assert evidence.reported_original_bytes is None
    assert evidence.limit is None
    assert evidence.payload_chars is None
    assert evidence.spool_paths == ()


def test_codex_warning_in_program_payload_is_not_harness_header():
    text = "program output\nWarning: truncated output (original token count: 12345)\n"
    assert not outputs.parse_output(text, "codex").truncation


def test_claude_persisted_output_not_arbitrary_saved_path():
    text = (
        "<persisted-output>\n"
        "Output too large (12.5KB). Full output saved to: /tmp/synthetic/output.txt\n\n"
        "Preview (first 2000 bytes):\nsynthetic preview\n</persisted-output>"
    )
    evidence = outputs.parse_output(text, "claude-code")
    assert evidence.truncation
    assert evidence.marker == "claude.persisted_output"
    assert evidence.spool_paths == ("/tmp/synthetic/output.txt",)
    assert evidence.limit == 2000
    assert evidence.limit_unit == "bytes"
    assert evidence.reported_original_bytes is None  # rounded KB is not an exact count
    assert evidence.spool_complete is None
    assert not outputs.parse_output(
        text.removeprefix("<persisted-output>\n"), "claude-code"
    ).truncation
    assert not outputs.parse_output("... (truncated)\n", "claude-code").truncation


def test_mini_json_counts_decoded_characters_not_serialization():
    text = _mini()
    evidence = outputs.parse_output(text, "mini-swe-agent")
    assert evidence.truncation
    assert evidence.marker == "mini_swe.json_elision"
    assert evidence.reported_original_chars == 10015
    assert evidence.reported_original_bytes is None
    assert evidence.payload_chars == 10000
    assert evidence.payload_bytes == 15000
    assert evidence.limit == 10000
    assert evidence.limit_unit == "chars"
    assert evidence.returned_chars == len(text)


def test_mini_exact_boundary_is_elision_envelope_without_loss():
    evidence = outputs.parse_output(_mini(elided=0), "mini-swe-agent")
    assert not evidence.truncation
    assert evidence.marker == "mini_swe.json_elision"
    assert evidence.omitted_chars == 0
    assert evidence.reported_original_chars == 10000


@pytest.mark.parametrize("elided", [True, -1, 3.5, "15", None])
def test_mini_invalid_count_is_unknown(elided: object):
    evidence = outputs.parse_output(_mini(elided=elided), "mini-swe-agent")
    assert not evidence.truncation
    assert evidence.kind == "possible_signal"
    assert evidence.reported_original_chars is None


def test_mini_false_positive_json_and_changed_lengths():
    assert not outputs.parse_output('{"truncated": true}', "mini-swe-agent").truncation
    evidence = outputs.parse_output(_mini(head="short"), "mini-swe-agent")
    assert evidence.truncation
    assert evidence.omitted_chars == 15
    assert evidence.reported_original_chars is None
    assert evidence.limit is None
    assert evidence.payload_chars == 5005


def test_mini_xml_uses_the_exact_warning_and_wrapper():
    text = (
        "<returncode>0</returncode>\n"
        + outputs._MINI_WARNING
        + "<output_head>\n"
        + "🙂" * 5000
        + "\n</output_head><elided_chars>\n8 characters elided\n</elided_chars>"
        + "<output_tail>\n"
        + "é" * 5000
        + "\n</output_tail>"
    )
    evidence = outputs.parse_output(text, "mini-swe-agent")
    assert evidence.marker == "mini_swe.xml_elision"
    assert evidence.payload_chars == 10000
    assert evidence.payload_bytes == 30000
    assert evidence.reported_original_chars == 10008
    assert not outputs.parse_output(
        text.replace("was too long.", "was different."), "mini-swe-agent"
    ).truncation


def test_terminus_byte_omission_and_screen_are_separate():
    text = "é\n[... output limited to 4 bytes; 8 interior bytes omitted ...]\né"
    evidence = outputs.parse_output(text, "terminus-2")
    assert evidence.truncation
    assert evidence.marker == "terminus.byte_omission"
    assert evidence.payload_chars == 2
    assert evidence.payload_bytes == 4
    assert evidence.reported_original_bytes == 12
    assert evidence.limit_unit == "bytes"
    screen = outputs.parse_output("Current Terminal Screen:\nsynthetic screen", "terminus-2")
    assert not screen.truncation
    assert screen.kind == "terminal_screen"
    assert screen.marker == "terminus.current_screen"
    assert outputs.parse_output("New Terminal Output:\nsynthetic", "terminus-2").kind == "no_marker"


def test_grok_exact_middle_line_not_generic_mentions():
    evidence = outputs.parse_output("head\n... (output truncated) ...\ntail", "grok-build")
    assert evidence.truncation
    assert evidence.marker == "grok.middle_omission"
    assert evidence.limit is None
    assert evidence.payload_chars is None
    assert not outputs.parse_output(
        "the phrase ... (output truncated) ... is documented", "grok-build"
    ).truncation


def test_muse_json_metadata_and_no_inferred_boundary():
    evidence = outputs.parse_output(_muse(), "muse-code")
    assert evidence.truncation
    assert evidence.marker == "muse.json_truncated"
    assert evidence.reported_original_bytes == 50
    assert evidence.reported_original_tokens == 20
    assert evidence.payload_chars == 2
    assert evidence.payload_bytes == 6
    assert evidence.limit is None
    assert not outputs.parse_output(_muse(truncated=False), "muse-code").truncation
    assert outputs.parse_output(_muse(truncated="true"), "muse-code").kind == "possible_signal"
    assert not outputs.parse_output('{"truncated": true, "output": "x"}', "muse-code").truncation


@pytest.mark.parametrize("family", ["cursor-cli", "unknown", "codex-v1"])
def test_unknown_family_does_not_infer_behavior_from_version(family: str):
    evidence = outputs.parse_output(_fast(_artifact()), family)
    assert not evidence.truncation
    assert evidence.kind == "unsupported"
    assert evidence.spool_paths == ()
    assert family not in json.dumps(evidence.to_doc())


@pytest.mark.parametrize(
    "text",
    [
        "The word truncated is an arbitrary program diagnostic.",
        '{"truncated": true, "message": "program metadata"}',
        "x /tmp/synthetic/output.txt\nFull output saved to: /tmp/synthetic/output.txt",
        "[stderr] truncated archive: unexpected EOF",
        "",
    ],
)
@pytest.mark.parametrize("family", sorted(outputs._SUPPORTED))
def test_arbitrary_command_output_never_becomes_explicit_truncation(text: str, family: str):
    evidence = outputs.parse_output(text, family)
    assert not evidence.truncation
    assert evidence.spool_paths == ()
    assert evidence.returned_chars == len(text)


def test_json_duplicate_keys_and_nonfinite_are_rejected():
    duplicate = _mini().replace('"elided_chars": 15', '"elided_chars": 15, "elided_chars": 16')
    nonfinite = _mini().replace('"elided_chars": 15', '"elided_chars": NaN')
    for text in (duplicate, nonfinite, "{" * 2000):
        assert not outputs.parse_output(text, "mini-swe-agent").truncation


def test_serialized_content_blocks_are_parent_responsibility():
    raw = json.dumps([{"type": "text", "text": _fast(_artifact())}])
    assert not outputs.parse_output(raw, "fast-agent").truncation
    assert outputs.parse_output(json.loads(raw)[0]["text"], "fast-agent").truncation


def test_invalid_unicode_does_not_invent_utf8_bytes():
    evidence = outputs.parse_output("\ud800", "fast-agent")
    assert evidence.returned_chars == 1
    assert evidence.returned_bytes is None


def test_marker_can_be_early_middle_or_final_line():
    notice = (
        "[Output truncated: showing 2 of 8 bytes. Use action='read_output' for retained output.]"
    )
    for text in (notice + "\ntail", "head\n" + notice + "\ntail", "head\n" + notice):
        assert outputs.parse_output(text, "fast-agent").truncation


@pytest.mark.parametrize(
    ("preview", "limit", "unit"),
    [("2KB", 2, "KB"), ("2000 bytes", 2000, "bytes"), ("2000 characters", 2000, "chars")],
)
def test_claude_preview_unit_preserved_without_guessing_kb_scale(
    preview: str, limit: int, unit: str
):
    text = (
        "<persisted-output>\n"
        "Output too large (5.2MB). Full output saved to: /tmp/synthetic/output.txt\n\n"
        f"Preview (first {preview}):\nsynthetic\n</persisted-output>\n"
    )
    evidence = outputs.parse_output(text, "claude-code")
    assert evidence.truncation
    assert evidence.limit == limit
    assert evidence.limit_unit == unit
    assert evidence.reported_original_bytes is None


def test_unicode_marker_does_not_convert_character_limit_to_byte_limit():
    text = _mini(head="🙂" * 5000)
    evidence = outputs.parse_output(text, "mini-swe-agent")
    assert evidence.limit == 10000
    assert evidence.limit_unit == "chars"
    assert evidence.payload_bytes == 25000
    assert evidence.returned_chars is not None
    assert evidence.returned_chars > 10000  # includes escaped serialization and envelope


def test_export_allowlist_never_includes_unknown_program_keys():
    text = _muse().replace('"exit_code": 0', '"exit_code": 0, "private": "DO_NOT_EXPORT"')
    evidence = outputs.parse_output(text, "muse-code")
    assert "DO_NOT_EXPORT" not in json.dumps(evidence.to_doc())
    assert "command" not in evidence.to_doc()


def test_fast_legacy_partial_quota_notice():
    guidance = (
        "The first 12 bytes retained before the temporary-file quota was reached "
        "is available during this session at /tmp/synthetic/output.txt."
        " Use read_text_file for selected line ranges or run a targeted search against that file; "
        "avoid reading the entire file unless necessary."
    )
    evidence = outputs.parse_output(_fast(guidance), "fast-agent")
    assert evidence.truncation
    assert evidence.spool_complete is False
    assert evidence.spool_paths == ("/tmp/synthetic/output.txt",)
    assert "/tmp/" not in json.dumps(evidence.to_doc())


@pytest.mark.parametrize(
    "status", ["Process exited with code 0", "Process running with session ID 12"]
)
def test_codex_original_token_header_does_not_hide_warning(status: str):
    header = (
        "Chunk ID: synthetic\nWall time: 0.0010 seconds\n"
        + status
        + "\nOriginal token count: 12345\nOutput:\n"
    )
    warning = "Warning: truncated output (original token count: 12345)\nTotal output lines: 90\n"
    evidence = outputs.parse_output(header + warning + "synthetic", "codex")
    assert evidence.truncation
    assert evidence.reported_original_tokens == 12345
    assert evidence.reported_original_lines == 90
    assert not outputs.parse_output(header + "synthetic", "codex").truncation
    assert not outputs.parse_output(header + "program output\n" + warning, "codex").truncation


@pytest.mark.parametrize("head", ["x" * 4998, "x" * 5018])
def test_mini_changed_boundaries_preserve_notice_not_original_size(head: str):
    evidence = outputs.parse_output(_mini(head=head), "mini-swe-agent")
    assert evidence.truncation
    assert evidence.omitted_chars == 15
    assert evidence.reported_original_chars is None
    assert evidence.limit is None
    assert evidence.payload_chars == len(head) + 5000


def test_mini_xml_changed_boundaries_preserve_notice():
    text = (
        "<returncode>0</returncode>\n"
        + outputs._MINI_WARNING
        + "<output_head>\nshort\n</output_head>"
        + "<elided_chars>\n8 characters elided\n</elided_chars>"
        + "<output_tail>\nchanged\n</output_tail>"
    )
    evidence = outputs.parse_output(text, "mini-swe-agent")
    assert evidence.truncation
    assert evidence.omitted_chars == 8
    assert evidence.reported_original_chars is None
    assert evidence.limit is None
    assert evidence.payload_chars == 12


def _dext(total: int = 30) -> str:
    return (
        "synthetic head\n"
        f"…[truncated {total} bytes total; kept first 4 and last 6. Narrow/paginate; "
        "prefer symbol-scoped reads or targeted diff/stat before retrying.]\nsynthetic tail"
    )


def test_dext_fixed_notice_reports_bytes_without_guessing_payload():
    evidence = outputs.parse_output(_dext(), "dext")
    assert evidence.truncation
    assert evidence.marker == "dext.byte_notice"
    assert evidence.reported_original_bytes == 30
    assert evidence.omitted_bytes == 20
    assert evidence.limit == 10
    assert evidence.payload_chars is None
    assert outputs.parse_output(_dext(total=10), "dext").kind == "possible_signal"
    assert not outputs.parse_output(
        _dext().replace("Narrow/paginate", "Other advice"), "dext"
    ).truncation


def _indusagi(*, truncated: object = True, by: str = "bytes") -> str:
    return json.dumps(
        {
            "content": [{"type": "text", "text": "synthetic notice"}],
            "details": {
                "fullOutputPath": "/tmp/synthetic/output.txt",
                "truncation": {
                    "content": "é🙂",
                    "firstLineExceedsLimit": False,
                    "lastLinePartial": False,
                    "maxBytes": 10,
                    "maxLines": 5,
                    "outputBytes": 6,
                    "outputLines": 1,
                    "totalBytes": 30,
                    "totalLines": 8,
                    "truncated": truncated,
                    "truncatedBy": by,
                },
            },
        }
    )


@pytest.mark.parametrize(("by", "limit", "unit"), [("bytes", 10, "bytes"), ("lines", 5, "lines")])
def test_indusagi_metadata_preserves_units_and_private_spool(by: str, limit: int, unit: str):
    evidence = outputs.parse_output(_indusagi(by=by), "indusagi")
    assert evidence.truncation
    assert evidence.marker == "indusagi.truncation_metadata"
    assert evidence.reported_original_bytes == 30
    assert evidence.reported_original_lines == 8
    assert evidence.omitted_bytes == 24
    assert evidence.limit == limit
    assert evidence.limit_unit == unit
    assert evidence.payload_chars == 2
    assert evidence.payload_bytes == 6
    assert evidence.spool_paths == ("/tmp/synthetic/output.txt",)
    assert evidence.spool_complete is None
    assert "synthetic" not in json.dumps(evidence.to_doc())


def test_indusagi_malformed_and_nontruncated_metadata():
    assert not outputs.parse_output(_indusagi(truncated=False), "indusagi").truncation
    assert outputs.parse_output(_indusagi(truncated="true"), "indusagi").kind == "possible_signal"
    for text in (
        _indusagi().replace('"totalBytes": 30', '"totalBytes": true'),
        _indusagi().replace('"maxBytes": 10', '"maxBytes": -1'),
        _indusagi().replace('"truncatedBy": "bytes"', '"truncatedBy": "unknown"'),
    ):
        assert outputs.parse_output(text, "indusagi").kind == "possible_signal"
    assert not outputs.parse_output('{"truncated": true}', "indusagi").truncation


def test_cursor_output_location_is_not_a_truncation_notice():
    text = json.dumps(
        {
            "success": {
                "stdout": "synthetic",
                "stderr": "",
                "interleavedOutput": "synthetic",
                "outputLocation": {
                    "filePath": "/tmp/synthetic/output.txt",
                    "sizeBytes": "9000",
                    "lineCount": "10",
                },
            },
            "isBackground": False,
        }
    )
    evidence = outputs.parse_output(text, "cursor-cli")
    assert not evidence.truncation
    assert evidence.kind == "unsupported"
    assert evidence.spool_paths == ()


@pytest.mark.parametrize("kind", ["unsupported", "possible_signal"])
def test_unknown_output_assessment_exports_null_not_negative(kind: outputs.Kind) -> None:
    evidence = outputs.OutputEvidence(kind=kind)
    assert evidence.truncation is False  # No recognized positive, not an assessment.
    assert evidence.to_doc()["truncation"] is None
