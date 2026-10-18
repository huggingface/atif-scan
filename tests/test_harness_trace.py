"""Synthetic-only tests for the offline research ATIF adapter."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from atif_scan.data.model import Content

if TYPE_CHECKING:
    from atif_scan.data.jsonval import JsonObject

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools" / "research"))
import harness_outputs as outputs
import harness_trace as trace


def _definition(name: str = "tools.Shell", schema: object = None) -> JsonObject:
    return {"name": name, "parameters": {} if schema is None else schema}


def _doc(definitions: object) -> JsonObject:
    return {
        "agent": {
            "name": "fast-agent",
            "version": "1.2.3+synthetic",
            "tool_definitions": definitions,
        },
        "steps": [],
    }


def _step(name: str = "tools.Shell", result: object = "synthetic output") -> JsonObject:
    return {
        "source": "agent",
        "tool_calls": [
            {
                "function_name": name,
                "tool_call_id": "synthetic-call",
                "arguments": {"command": "echo synthetic"},
            }
        ],
        "observation": {"results": [{"content": result}]},
    }


def test_inventory_and_no_mutation() -> None:
    doc = _doc([_definition("tools.Shell"), _definition("functions.Process")])
    doc["steps"] = [_step()]
    before = copy.deepcopy(doc)
    evidence = trace.analyze_trace(doc)
    assert doc == before
    assert evidence.family == "fast-agent"
    assert evidence.version == "1.2.3+synthetic"
    assert evidence.declarations.tools == ("process", "shell")
    assert evidence.declarations.source == "atif"
    assert evidence.declarations.status == "known"
    assert len(evidence.declarations.signature or "") == 64
    assert evidence.steps[0].calls[0].name == "tools.Shell"
    assert trace.canonical_tool("functions.tools.Bash", "fast-agent") == "bash"
    assert trace.canonical_tool("tools.Bash", "claude-code") == "Bash"


def test_absence_empty_null_and_native_precedence() -> None:
    native = [{"type": "function", "function": _definition("functions.Bash")}]
    missing = trace.analyze_trace({"steps": []})
    assert missing.declarations == trace.Inventory((), "unknown", "missing", None)
    assert "declarations_missing" in missing.issues
    fallback = trace.analyze_trace({"agent": {"name": "fast-agent"}, "steps": []}, native)
    assert fallback.declarations.tools == ("bash",)
    assert fallback.declarations.source == "native"
    empty = trace.analyze_trace(_doc([]), native)
    assert empty.declarations.status == "known"
    assert empty.declarations.tools == ()
    assert empty.declarations.source == "atif"
    assert empty.declarations.signature is not None
    null = trace.analyze_trace(_doc(None), native)
    assert null.declarations.status == "malformed"
    assert null.declarations.source == "atif"
    assert null.declarations.signature is None


@pytest.mark.parametrize(
    "definitions",
    [
        {},
        "synthetic",
        [None],
        [{"name": "Shell"}],
        [_definition(schema=[])],
        [_definition(), _definition()],
    ],
)
def test_malformed_is_never_known(definitions: object) -> None:
    evidence = trace.analyze_trace(_doc(definitions))
    assert evidence.declarations.status == "malformed"
    assert evidence.declarations.signature is None
    assert "declarations_malformed" in evidence.issues


def test_partial_inventory_stays_malformed() -> None:
    inventory = trace.analyze_trace(_doc([_definition(), None])).declarations
    assert inventory.tools == ("shell",)
    assert inventory.status == "malformed"


def test_signature_annotations_order_and_schema_structure() -> None:
    schema: JsonObject = {
        "type": "object",
        "description": "synthetic annotation",
        "properties": {
            "description": {"type": "string", "maxLength": 10, "title": "synthetic title"},
            "title": {"enum": [{"description": "actual value"}], "examples": ["synthetic"]},
        },
        "required": ["description"],
        "additionalProperties": False,
        "allOf": [{"properties": {"examples": {"minimum": 2, "description": "annotation"}}}],
        "dependentRequired": {"description": ["title"]},
    }
    annotated = copy.deepcopy(schema)
    annotated["description"] = "different annotation"
    annotated["examples"] = ["another annotation"]
    a = trace.analyze_trace(_doc([_definition(schema=schema), _definition("functions.Process")]))
    b = trace.analyze_trace(_doc([_definition("Process"), _definition("Shell", annotated)]))
    assert a.declarations.signature == b.declarations.signature
    for key, value in (("required", []), ("additionalProperties", True), ("dependentRequired", {})):
        changed = copy.deepcopy(schema)
        changed[key] = value
        c = trace.analyze_trace(_doc([_definition(schema=changed), _definition("Process")]))
        assert c.declarations.signature != a.declarations.signature
    changed = copy.deepcopy(schema)
    changed["properties"] = {"description": {"type": "string", "maxLength": 11}}
    assert (
        trace.analyze_trace(_doc([_definition(schema=changed)])).declarations.signature
        != trace.analyze_trace(_doc([_definition(schema=schema)])).declarations.signature
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Claude Code", "claude-code"),
        ("claude-code", "claude-code"),
        ("FAST-AGENT", "fast-agent"),
        ("synthetic-harness", "synthetic-harness"),
        ("https://synthetic.invalid", None),
        ("secret\nlabel", None),
    ],
)
def test_family_labels(name: str, expected: str | None) -> None:
    evidence = trace.analyze_trace(
        {"agent": {"name": name, "version": "unsafe\nlabel"}, "steps": []}
    )
    assert evidence.family == expected
    assert evidence.version is None
    assert "invalid_agent_version" in evidence.issues


def test_agent_only_original_indexes_and_dropped_counts() -> None:
    doc = _doc([])
    doc["steps"] = [
        {"source": "user", "tool_calls": "ignored"},
        {"source": "agent", "is_copied_context": True, "tool_calls": "ignored"},
        _step(),
        {"source": "agent", "tool_calls": "private exception material"},
        None,
        _step(),
    ]
    doc["subagent_trajectories"] = [{"steps": [_step()]}]
    evidence = trace.analyze_trace(doc)
    assert [s.index for s in evidence.steps] == [2, 5]
    assert all(s.authored for s in evidence.steps)
    assert evidence.dropped_steps == 2
    assert evidence.issues == ("step_parse_failed", "subagent_trajectories_present")
    assert trace.analyze_trace({"steps": None}).issues == ("invalid_steps", "declarations_missing")


def test_code_mode_and_pairing_reuse_loader() -> None:
    step = _step("exec", json.dumps([{"type": "text", "text": "synthetic output"}]))
    step["tool_calls"] = [
        {
            "function_name": "exec",
            "tool_call_id": "parent",
            "arguments": {"input": 'const r = await tools.exec_command({cmd: "echo synthetic"});'},
        }
    ]
    step["observation"] = {
        "results": [
            {
                "source_call_id": "parent",
                "content": json.dumps([{"type": "text", "text": "synthetic output"}]),
            }
        ]
    }
    evidence = trace.analyze_trace({"steps": [step]})
    parsed = evidence.steps[0]
    assert len(parsed.calls) == 2
    outer, derived = parsed.calls
    assert outer.result_id is None
    assert derived.result_id == "parent"
    assert parsed.results_for(outer) == parsed.results_for(derived)
    assert parsed.results_for(derived)[0][1].content.text == "synthetic output"
    assert not parsed.results_for(derived)[0][1].pairing_reconstructed


@pytest.mark.parametrize(
    "serialized",
    [
        json.dumps([{"type": "text", "text": "synthetic\noutput"}]),
        repr([{"type": "text", "text": "synthetic\noutput"}]),
    ],
)
def test_serialized_blocks(serialized: str) -> None:
    decoded = trace.result_text(serialized)
    assert decoded.text == "synthetic\noutput"
    assert decoded.understood
    assert (
        trace.result_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]).text
        == "a\nb"
    )


def test_flags_and_media_are_preserved() -> None:
    original = Content(
        json.dumps([{"type": "text", "text": "synthetic"}]),
        understood=False,
        media=True,
        media_ids=("synthetic-digest",),
        computed=True,
    )
    decoded = trace.unwrap_result(original)
    assert decoded.text == "synthetic"
    assert not decoded.understood
    assert decoded.media and decoded.computed
    assert decoded.media_ids == original.media_ids
    image = trace.result_text(
        json.dumps(
            [
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://synthetic.invalid/image"},
                },
                {"type": "text", "text": "caption"},
            ]
        )
    )
    assert image.media
    assert image.text == "\ncaption"
    assert image.media_ids == ()


@pytest.mark.parametrize(
    "raw",
    [
        "[not JSON",
        "[{'type':'text','text':run_untrusted()}]",
        '["not blocks"]',
        '{"returncode":0,"output":"synthetic"}',
        "[]",
        "[{'type':'text','text':'" + "x" * trace.MAX_REPR_CHARS + "'}]",
    ],
)
def test_nonblocks_malformed_and_bounded_repr_stay_raw(raw: str) -> None:
    assert trace.result_text(raw).text == raw


def test_structured_shell_output_stays_json_for_output_parser() -> None:
    raw: JsonObject = {
        "returncode": 0,
        "output_head": "h" * 5000,
        "output_tail": "t" * 5000,
        "elided_chars": 20,
        "warning": "Output too long.",
    }
    decoded = trace.result_text(raw)
    assert json.loads(decoded.text) == raw
    assert outputs.parse_output(decoded.text, "mini-swe-agent").truncation
    evidence = trace.analyze_trace({"steps": [_step(result=raw)]})
    assert evidence.steps[0].observations[0].content == decoded
    shell_with_text = {"returncode": 0, "text": "synthetic output"}
    assert json.loads(trace.result_text(shell_with_text).text) == shell_with_text


def test_unknown_blocks_stay_unknown_and_bad_schema_values_fail_closed() -> None:
    assert not trace.result_text([{"type": "unknown"}]).understood
    assert not trace.result_text({"output": object()}).understood
    inventory = trace.analyze_trace(
        _doc([_definition(schema={"minimum": float("nan")})])
    ).declarations
    assert inventory.status == "malformed"
    assert inventory.signature is None


def test_reconstructed_pairing_and_unresolved_code_mode_remain_loader_owned() -> None:
    step: JsonObject = {
        "source": "agent",
        "tool_calls": [
            {"function_name": "shell", "tool_call_id": "a", "arguments": {}},
            {"function_name": "shell", "tool_call_id": "b", "arguments": {}},
        ],
        "observation": {"results": [{"content": "first"}, {"content": "second"}]},
    }
    parsed = trace.analyze_trace({"steps": [step]}).steps[0]
    assert [parsed.results_for(c)[0][1].content.text for c in parsed.calls] == ["first", "second"]
    assert all(o.pairing_reconstructed for o in parsed.observations)
    unlinked = _step("exec")
    unlinked["tool_calls"] = [
        {
            "function_name": "exec",
            "tool_call_id": "parent",
            "arguments": {"input": 'await tools.exec_command({cmd: "echo synthetic"});'},
        }
    ]
    code_step = trace.analyze_trace({"steps": [unlinked]}).steps[0]
    # Do not invent an association the loader does not establish.
    assert len(code_step.calls) == 2
    assert code_step.results_for(code_step.calls[1]) == []


def test_json_size_and_repr_node_limits() -> None:
    oversized = json.dumps([{"type": "text", "text": "x" * trace.MAX_SERIALIZED_CHARS}])
    assert trace.result_text(oversized).text == oversized
    many_nodes = repr([{"type": "text", "text": ["x"] * trace.MAX_REPR_NODES}])
    assert len(many_nodes) < trace.MAX_REPR_CHARS
    assert trace.result_text(many_nodes).text == many_nodes


def test_anthropic_native_and_declaration_locations() -> None:
    native = [{"name": "Bash", "input_schema": {"type": "object"}}]
    evidence = trace.analyze_trace({"steps": []}, native)
    assert evidence.declarations.status == "known"
    assert evidence.declarations.tools == ("Bash",)
    docs: tuple[JsonObject, ...] = (
        {"agent": {"tools": [_definition("synthetic_tool")]}, "steps": []},
        {"tool_definitions": [_definition("synthetic_tool")], "steps": []},
    )
    for doc in docs:
        assert trace.analyze_trace(doc).declarations.source == "atif"


def _native_result(recorded: object, native: object, call_id: str = "synthetic-call") -> JsonObject:
    return {
        "source_call_id": call_id,
        "content": recorded,
        "extra": {
            "tool_result_metadata": {
                "raw_tool_result": {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": native,
                }
            }
        },
    }


def test_native_visibility_keeps_original_locators_and_recorded_projection() -> None:
    step = _step("Bash")
    step["tool_calls"] = [
        {"function_name": "Bash", "tool_call_id": "synthetic-call", "arguments": {}}
    ]
    step["observation"] = {
        "results": [
            _native_result("preview\n[stdout]\n" + "x" * 30_000, "preview"),
            _native_result("recorded fallback", None),
        ]
    }
    evidence = trace.analyze_trace(
        {"agent": {"name": "Claude Code"}, "steps": [{"source": "user"}, step]}
    )
    assert [o.content.text for o in evidence.steps[0].observations] == [
        "preview",
        "recorded fallback",
    ]
    first, second = evidence.result_visibility
    assert (first.step, first.index, second.step, second.index) == (1, 0, 1, 1)
    assert first.visibility.recorded.text.endswith("x" * 30_000)
    assert first.visibility.model_visible == evidence.steps[0].observations[0].content
    assert first.visibility.basis == "claude_native_tool_result"
    assert second.visibility.model_visible is None
    assert second.visibility.basis == "unknown"


def test_empty_native_content_and_legacy_trace_constructor() -> None:
    step = _step("Bash")
    step["observation"] = {"results": [_native_result("exporter extras", "")]}
    evidence = trace.analyze_trace({"agent": {"name": "claude-code"}, "steps": [step]})
    assert evidence.steps[0].observations[0].content.text == ""
    assert evidence.result_visibility[0].visibility.model_visible_chars == 0
    legacy = trace.TraceEvidence(
        evidence.family, evidence.version, evidence.declarations, evidence.steps, evidence.issues
    )
    assert legacy.result_visibility == ()


@pytest.mark.parametrize(
    ("offered", "requested"),
    [
        ("edit_file", "Edit_file"),
        ("read_text_file", "Read_Text_File"),
        ("write_text_file", "WRITE_TEXT_FILE"),
        ("attach_media", "Attach_Media"),
        ("apply_patch", "Apply_Patch"),
        ("poll_process", "Poll_Process"),
    ],
)
def test_fast_agent_builtin_case_alias_is_one_declared_tool(offered: str, requested: str) -> None:
    from harness_metrics import measure

    doc = _doc([_definition(offered)])
    doc["steps"] = [_step(requested)]
    evidence = trace.analyze_trace(doc)
    metrics = measure(evidence)
    assert len(metrics.tools) == 1
    assert metrics.tools[0]["tool"] == offered
    assert metrics.tools[0]["recorded_calls"] == 1
    assert metrics.tools[0]["offered_unused"] is False


def test_fast_agent_case_alias_does_not_normalize_unknown_plugin_or_other_family() -> None:
    assert trace.canonical_tool("Plugin_Foo", "fast-agent") == "Plugin_Foo"
    assert trace.canonical_tool("Edit_file", "claude-code") == "Edit_file"
