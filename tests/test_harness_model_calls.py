"""Synthetic-only regression tests for model-level tool-dispatch evidence."""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from atif_scan.data.model import Content

if TYPE_CHECKING:
    from atif_scan.data.jsonval import JsonObject

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools" / "research"))
import harness_model_calls as calls
import harness_trace as trace

_MISSING = object()


def _call(name: str = "Read_file", key: str = "a") -> JsonObject:
    return {
        "function_name": name,
        "tool_call_id": key,
        "arguments": {"path": "/private/synthetic-secret", "content": "synthetic payload"},
    }


def _result(text: object, key: object = "a") -> JsonObject:
    return {"source_call_id": key, "content": text}


def _step(name: str = "Read_file", output: object = _MISSING) -> JsonObject:
    return {
        "source": "agent",
        "tool_calls": [_call(name)],
        "observation": {"results": [] if output is _MISSING else [_result(output)]},
    }


def _doc(steps: list[object], definitions: object = _MISSING) -> JsonObject:
    agent: JsonObject = {"name": "fast-agent", "model_name": "provider/root-model"}
    if definitions is not _MISSING:
        agent["tool_definitions"] = definitions
    return {"agent": agent, "steps": steps}


def _audit(doc: JsonObject) -> calls.CallAudit:
    return calls.evaluate_calls(trace.analyze_trace(doc), doc)


@pytest.mark.parametrize(
    "output,reason",
    [
        (
            "<tool_use_error>Error: No such tool available: Read_file</tool_use_error>",
            "unavailable",
        ),
        (
            "<tool_use_error>Error: No such tool available: functions.Read_file</tool_use_error>"
            "\n\n[error] tool reported failure",
            "unavailable",
        ),
        ("Tool 'Read_file' is not available.", "unavailable"),
        (
            "Tool 'Read_file' is not available. Available tools: Shell, tools.edit_file.",
            "unavailable",
        ),
        ('Tool "tools.Read_file" is not available.', "unavailable"),
        ("Unknown tool: Read_file", "not_found"),
        ("Error: Unknown tool: 'Read_file'", "not_found"),
        ("Error: Tool Read_file not found", "not_found"),
        ("Error executing tool Read_file: Tool 'Read_file' not found", "not_found"),
        ("Error executing tool 'Read_file': Tool 'Read_file' not found.", "not_found"),
        ('Error executing tool "Read_file": Unknown tool: functions.Read_file', "not_found"),
        (
            "Error executing tool Read_file: Tool 'Read_file' is not available.",
            "unavailable",
        ),
    ],
)
def test_supported_whole_envelopes(output: str, reason: str) -> None:
    audit = _audit(_doc([_step("functions.tools.Read_file", output)]))
    assert len(audit.attempts) == len(audit.rejected) == 1
    attempt = audit.attempts[0]
    assert (attempt.step, attempt.index, attempt.tool_label) == (0, 0, "Read_file")
    assert (attempt.model, attempt.model_source) == ("provider/root-model", "root")
    rejection = audit.rejected[0]
    assert rejection.attempt is attempt
    assert rejection.reason == reason
    assert rejection.declaration_conflict is None
    assert rejection.pairing == "recorded_id"


@pytest.mark.parametrize(
    "output",
    [
        "InputValidationError: required path missing",
        "<tool_use_error>InputValidationError: tool Read_file not found</tool_use_error>",
        "Error executing tool Read_file: Invalid arguments: path not found",
        "Error executing tool Read_file: command not found",
        "Error executing tool Read_file: File 'Read_file' not found",
        "Error executing tool Read_file: API returned 404 Not Found",
        "bash: Read_file: command not found",
        "FileNotFoundError: Read_file",
        "HTTP 404: unknown tool Read_file",
        "Printed example: \"Tool 'Read_file' is not available.\"",
        "\"Tool 'Read_file' is not available.\"",
        "Example\nTool 'Read_file' is not available.",
        "Tool 'Read_file' is not available.\nThis is quoted documentation.",
        "<tool_use_error>Error: No such tool available: Read_file</tool_use_error>\nextra prose",
        "Tool 'Write_file' is not available.",
        "Error executing tool Write_file: Tool 'Read_file' not found",
        "Error executing tool Read_file: Tool 'Write_file' not found",
        "Tool 'Read_file' is not available. Available tools: /private/synthetic-path.",
        {"returncode": 1, "output": "Tool 'Read_file' is not available."},
        "Tool 'Read_file' is not available." + " " * calls.MAX_ERROR_CHARS,
        "Error: tool unavailable",
        "Error executing tool Read_file: not found",
    ],
)
def test_non_dispatch_errors_mentions_and_wrong_names(output: object) -> None:
    audit = _audit(_doc([_step(output=output)]))
    assert len(audit.attempts) == 1
    assert audit.rejected == ()


@pytest.mark.parametrize("model", [None, "", 42, {}, "unsafe\nmodel", "https://synthetic.invalid"])
def test_explicit_invalid_step_model_never_falls_back(model: object) -> None:
    step = _step()
    step["model_name"] = model
    audit = _audit(_doc([step]))
    assert audit.attempts[0].model is None
    assert audit.attempts[0].model_source == "unknown"


@pytest.mark.parametrize(
    "model",
    [
        "provider/model:v2",
        "codexresponses.gpt-6-luna?reasoning=medium",
        "codexresponses.gpt-6.1-sol?reasoning=high",
        "provider/model?reasoning=high&limit=20",
    ],
)
def test_explicit_step_model_uses_extended_safe_labels(model: str) -> None:
    step = _step()
    step["model_name"] = model
    # The loader can reject query parameters; raw per-step evidence is authoritative.
    audit = _audit(_doc([step]))
    assert audit.attempts[0].model == model
    assert audit.attempts[0].model_source == "step"


@pytest.mark.parametrize("root_model", [None, 1, "", "unsafe label"])
def test_invalid_root_model_is_unknown(root_model: object) -> None:
    doc = _doc([_step()])
    doc["agent"] = {"name": "fast-agent", "model_name": root_model}
    assert _audit(doc).attempts[0].model_source == "unknown"
    assert _audit(doc).attempts[0].model is None


def test_original_locators_and_model_denominators() -> None:
    step = _step(output="Tool 'Read_file' is not available.")
    step["model_name"] = "provider/step-model"
    step["tool_calls"] = [_call(), _call("Shell", "b")]
    doc = _doc(
        [
            {"source": "user", "tool_calls": [_call()]},
            {"source": "agent", "is_copied_context": True, "tool_calls": [_call()]},
            {"source": "agent", "tool_calls": "malformed"},
            step,
            _step("Shell"),
        ]
    )
    audit = _audit(doc)
    assert [(a.step, a.index) for a in audit.attempts] == [(3, 0), (3, 1), (4, 0)]
    assert Counter(a.model for a in audit.attempts) == {
        "provider/step-model": 2,
        "provider/root-model": 1,
    }
    assert Counter(r.attempt.model for r in audit.rejected) == {"provider/step-model": 1}
    assert audit.attempts[1].tool_label == "shell"


@pytest.mark.parametrize(
    "definitions,expected",
    [
        (_MISSING, None),
        (None, None),
        ([], False),
        ([{"name": "functions.Read_file", "parameters": {}}], True),
        ([{"name": "Write_file", "parameters": {}}], False),
    ],
)
def test_declaration_conflict_is_not_blanket_nonexistence(
    definitions: object, expected: object
) -> None:
    doc = _doc([_step(output="Tool 'Read_file' is not available.")], definitions)
    evidence = trace.analyze_trace(doc)
    audit = calls.evaluate_calls(evidence, doc)
    assert audit.rejected[0].declaration_conflict is expected
    assert audit.rejected[0].reason == "unavailable"
    assert trace.analyze_trace(doc) == evidence  # No inventory modification.


@pytest.mark.parametrize("name", ["Read_file", "synthetic unknown", "Họp", "/private/tool"])
def test_unsafe_tool_names_remain_unknown_without_invented_sentinel(name: str) -> None:
    output = f"<tool_use_error>Error: No such tool available: {name}</tool_use_error>"
    audit = _audit(_doc([_step(name, output)]))
    expected = "Read_file" if name == "Read_file" else None
    assert audit.attempts[0].tool_label == expected
    assert audit.rejected[0].attempt.tool_label == expected
    assert "unknown_tool_label" not in json.dumps(audit.to_doc())


def test_code_mode_references_are_not_invocations() -> None:
    step = _step("exec", "Error: Tool exec_command not found")
    step["tool_calls"] = [
        {
            "function_name": "exec",
            "tool_call_id": "a",
            "arguments": {"input": 'await tools.exec_command({cmd: "echo synthetic"});'},
        }
    ]
    doc = _doc([step])
    evidence = trace.analyze_trace(doc)
    assert len(evidence.steps[0].calls) == 2
    audit = calls.evaluate_calls(evidence, doc)
    assert len(audit.attempts) == 1
    assert audit.attempts[0].tool_label == "exec"
    assert audit.rejected == ()
    step["observation"] = {"results": [_result("Error: Tool exec not found")]}
    assert len(_audit(doc).rejected) == 1


def test_only_exact_linked_call_rejected_and_repeated_results_not_double_counted() -> None:
    step = _step()
    step["tool_calls"] = [_call(key="a"), _call(key="b")]
    step["observation"] = {
        "results": [
            _result("Tool 'Read_file' is not available.", "b"),
            _result("Tool 'Read_file' is not available.", "b"),
        ]
    }
    audit = _audit(_doc([step]))
    assert len(audit.attempts) == 2
    assert len(audit.rejected) == 1
    assert audit.rejected[0].attempt.index == 1
    assert audit.rejected[0].observation == 0


@pytest.mark.parametrize("key", [None, "foreign"])
def test_ambiguous_or_foreign_result_does_not_reject_calls(key: object) -> None:
    step = _step()
    step["tool_calls"] = [_call(key="a"), _call(key="b")]
    step["observation"] = {"results": [_result("Tool 'Read_file' is not available.", key)]}
    assert _audit(_doc([step])).rejected == ()


def test_reconstructed_pairing_is_explicit_in_export() -> None:
    step = _step()
    step["tool_calls"] = [_call(key="a"), _call("Shell", "b")]
    step["observation"] = {
        "results": [
            _result("Tool 'Read_file' is not available.", None),
            _result("synthetic success", None),
        ]
    }
    audit = _audit(_doc([step]))
    assert len(audit.rejected) == 1
    assert audit.rejected[0].pairing == "position"
    assert audit.rejected[0].attempt.index == 0


def test_unknown_and_shared_content_not_rejections() -> None:
    doc = _doc([_step(output="Tool 'Read_file' is not available.")])
    evidence = trace.analyze_trace(doc)
    step = evidence.steps[0]
    obs = step.observations[0]
    for replacement in (
        replace(obs, content=Content(obs.content.text, understood=False)),
        replace(obs, pairing_unresolved=True),
        replace(obs, shared_terminal=True),
    ):
        changed = replace(evidence, steps=(replace(step, observations=(replacement,)),))
        assert calls.evaluate_calls(changed, doc).rejected == ()


@pytest.mark.parametrize("serialized", [False, True])
def test_block_envelopes_are_decoded_by_adapter(serialized: bool) -> None:
    blocks = [{"type": "text", "text": "Tool 'Read_file' is not available."}]
    output = json.dumps(blocks) if serialized else blocks
    assert len(_audit(_doc([_step(output=output)])).rejected) == 1


def test_exports_and_reprs_have_no_private_material_or_harness() -> None:
    doc = _doc([_step(output="Tool 'Read_file' is not available.")])
    doc["session_id"] = "synthetic-private-session"
    audit = _audit(doc)
    exported = audit.to_doc()
    attempt_keys = {"step", "call", "tool", "model", "model_source"}
    assert set(exported) == {"attempts", "rejected"}
    assert set(audit.attempts[0].to_doc()) == attempt_keys
    assert set(audit.rejected[0].to_doc()) == attempt_keys | {
        "observation",
        "reason",
        "declaration_conflict",
        "pairing",
    }
    serialized = json.dumps(exported)
    for private in (
        "harness",
        "trace_id",
        "session_id",
        "synthetic-private-session",
        "/private/",
        "synthetic payload",
        "is not available",
        "arguments",
    ):
        assert private not in serialized
        if private != "harness":  # The module name is harmless in object reprs.
            assert private not in repr(audit)
            assert private not in repr(audit.attempts[0])
            assert private not in repr(audit.rejected[0])


def test_empty_audit_and_unavailable_raw_step_do_not_invent_models() -> None:
    assert _audit({"steps": []}).to_doc() == {"attempts": [], "rejected": []}
    doc = _doc([_step()])
    evidence = trace.analyze_trace(doc)
    audit = calls.evaluate_calls(evidence, {"steps": [], "agent": {"model_name": "other-model"}})
    assert audit.attempts[0].model is None
    assert audit.attempts[0].model_source == "unknown"
