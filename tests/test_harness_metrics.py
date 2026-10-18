"""Synthetic-only accounting regressions; no trace command or spool is executed."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc, JsonObject

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools" / "research"))
import harness_metrics as metrics
import harness_model_calls as model_calls
import harness_trace as trace

SPOOL = "/tmp/synthetic-private/output.txt"
HANDLE = "process-synthetic"
_MISSING = object()


def _call(
    name: str = "shell", args: JsonObject | None = None, call_id: str = "synthetic-call"
) -> JsonObject:
    return {
        "function_name": name,
        "tool_call_id": call_id,
        "arguments": {"command": "echo synthetic"} if args is None else args,
    }


def _result(value: object, call_id: str | None = "synthetic-call") -> JsonObject:
    result: JsonObject = {"content": value}
    if call_id is not None:
        result["source_call_id"] = call_id
    return result


def _step(calls: list[JsonObject], results: list[JsonObject] | None = None) -> JsonObject:
    step: JsonObject = {"source": "agent", "tool_calls": calls}
    if results is not None:
        step["observation"] = {"results": results}
    return step


def _doc(
    steps: list[JsonObject], *, family: str = "fast-agent", declarations: object = _MISSING
) -> JsonObject:
    agent: JsonObject = {"name": family}
    if declarations is not _MISSING:
        agent["tool_definitions"] = declarations
    return {"agent": agent, "steps": steps}


def _definitions(*names: str) -> list[JsonObject]:
    return [{"name": name, "parameters": {}} for name in names]


def _tool(result: metrics.Metrics, name: str) -> Doc:
    return next(row for row in result.tools if row["tool"] == name)


def _preview(guidance: str) -> str:
    return f"ab\n[Output truncated: showing 2 of 8 bytes. {guidance}]"


def _spooled() -> str:
    return _preview(
        f"The complete output is available during this session at {SPOOL}. "
        "Use read_text_file for selected line ranges or run a targeted search against that file; "
        "avoid reading the entire file unless necessary."
    )


def _retained() -> str:
    return _preview(
        "The complete output can be read through the managed process handle. "
        "Use the process tool with action='read_output' and the process_id from this result."
    )


def test_metadata_unavailable_is_not_known_empty() -> None:
    result = metrics.measure(trace.analyze_trace({"steps": []}))
    assert result.trace["family"] == "unknown"
    assert result.trace["version"] is None
    assert result.trace["signature"] is None
    assert result.trace["declared_tools"] is None
    assert result.trace["inventory_source"] == "unknown"
    assert result.trace["inventory_status"] == "missing"
    assert result.trace["recorded_calls"] == result.trace["static_call_sites"] == 0
    assert result.tools == result.outputs == result.followups == ()
    assert result.trace["issues"] == ["declarations_missing"]
    empty = metrics.measure(trace.analyze_trace(_doc([], declarations=[])))
    assert empty.trace["declared_tools"] == 0
    assert empty.trace["inventory_status"] == "known"
    assert empty.trace["signature"] is not None


def test_offered_unused_only_for_known_inventory_without_dropped_steps() -> None:
    steps: list[JsonObject] = [_step([_call()], [_result("synthetic")])]
    doc = _doc(steps, declarations=_definitions("tools.Shell", "functions.Process"))
    result = metrics.measure(trace.analyze_trace(doc))
    assert result.trace["declared_tools"] == 2
    assert _tool(result, "shell")["declared"] is True
    assert _tool(result, "shell")["offered_unused"] is False
    unused = _tool(result, "process")
    assert unused["recorded_calls"] == unused["static_call_sites"] == 0
    assert unused["offered_unused"] is True
    assert unused["inventory_source"] == "atif"
    steps.append({"source": "agent", "tool_calls": "malformed synthetic"})
    dropped = metrics.measure(trace.analyze_trace(doc))
    assert dropped.trace["dropped_steps"] == 1
    assert all(row["offered_unused"] is None for row in dropped.tools)


@pytest.mark.parametrize("declarations", [_MISSING, None, [*_definitions("shell"), None]])
def test_unknown_or_partial_inventory_never_offered_unused(declarations: object) -> None:
    result = metrics.measure(
        trace.analyze_trace(_doc([_step([_call()])], declarations=declarations))
    )
    assert result.trace["declared_tools"] is None
    assert _tool(result, "shell")["declared"] is None
    assert _tool(result, "shell")["offered_unused"] is None
    assert result.trace["shell_calls_missing_results"] == 1


def test_observed_undeclared_tool_not_offered_unused() -> None:
    result = metrics.measure(trace.analyze_trace(_doc([_step([_call()])], declarations=[])))
    row = _tool(result, "shell")
    assert row["declared"] is False
    assert row["offered_unused"] is None
    assert row["inventory_source"] == "observed"


@pytest.mark.parametrize(
    ("name", "family", "label"),
    [
        ("functions.tools.Shell", "fast-agent", "shell"),
        ("tools.functions.Process", "fast-agent", "process"),
        ("functions.Bash", "claude-code", "Bash"),
        ("functions.exec_command", "codex", "exec_command"),
    ],
)
def test_namespace_labels(name: str, family: str, label: str) -> None:
    result = metrics.measure(
        trace.analyze_trace(
            _doc(
                [_step([_call(name)], [_result("synthetic")])],
                family=family,
                declarations=_definitions(name),
            )
        )
    )
    assert result.tools[0]["tool"] == label
    assert result.tools[0]["declared"] is True
    assert result.outputs[0]["tool"] == label
    assert metrics.tool_label("unsafe\nprivate label", family) is None


def test_code_mode_shared_capture_and_static_sites_have_distinct_units() -> None:
    program = (
        'const a = await tools.exec_command({cmd: "echo synthetic-one"});'
        'const b = await tools.exec_command({cmd: "echo synthetic-two"});'
    )
    evidence = trace.analyze_trace(
        _doc(
            [_step([_call("exec", {"input": program})], [_result("synthetic shared capture")])],
            family="codex",
        )
    )
    assert evidence.steps[0].calls[0].tool == "inert"
    assert len(evidence.steps[0].calls) == 3
    result = metrics.measure(evidence)
    assert result.trace["recorded_calls"] == 1
    assert result.trace["static_call_sites"] == 2
    assert _tool(result, "exec")["recorded_calls"] == 1
    derived = _tool(result, "exec_command")
    assert derived["recorded_calls"] == 0
    assert derived["static_call_sites"] == 2
    assert result.trace["shell_output_records"] == 1
    assert result.trace["shell_calls_missing_results"] == 0
    assert result.outputs[0]["context"] == "code_mode_shared"
    assert result.outputs[0]["pairing"] == "parent_result"


def test_terminus_shared_capture_is_not_missing_per_command() -> None:
    calls = [
        _call("bash_command", {"keystrokes": "echo synthetic-one\n", "duration": 0.1}, "one"),
        _call("bash_command", {"keystrokes": "echo synthetic-two\n", "duration": 0.1}, "two"),
    ]
    evidence = trace.analyze_trace(
        _doc(
            [_step(calls, [_result("Current Terminal Screen:\nsynthetic", None)])],
            family="terminus-2",
        )
    )
    assert evidence.steps[0].observations[0].shared_terminal
    result = metrics.measure(evidence)
    assert result.trace["recorded_calls"] == 2
    assert result.trace["static_call_sites"] == 0
    assert result.trace["shell_output_records"] == 1
    assert result.trace["shell_calls_missing_results"] == 0
    assert result.outputs[0]["context"] == "shared_terminal"
    assert result.outputs[0]["pairing"] == "shared_terminal"
    missing = metrics.measure(trace.analyze_trace(_doc([_step(calls)], family="terminus-2")))
    assert missing.trace["shell_calls_missing_results"] == 2


def test_each_observation_counted_once_even_with_multiple_linked_calls() -> None:
    program = (
        'await tools.exec_command({cmd: "echo synthetic-one"});'
        'await tools.exec_command({cmd: "echo synthetic-two"});'
    )
    result = metrics.measure(
        trace.analyze_trace(
            _doc(
                [
                    _step(
                        [_call("exec", {"input": program})],
                        [_result("synthetic-one"), _result("synthetic-two")],
                    )
                ],
                family="codex",
            )
        )
    )
    assert result.trace["recorded_calls"] == 1
    assert result.trace["static_call_sites"] == 2
    assert result.trace["shell_output_records"] == 2
    assert [row["observation"] for row in result.outputs] == [0, 1]
    assert all(row["context"] == "code_mode_shared" for row in result.outputs)


def test_malformed_output_is_a_record_not_known_empty_or_complete() -> None:
    result = metrics.measure(
        trace.analyze_trace(_doc([_step([_call()], [_result(123)]), _step([_call()])]))
    )
    assert result.trace["shell_output_records"] == 1
    assert result.trace["shell_calls_missing_results"] == 1
    assert result.trace["truncated_output_records"] == 0
    row = result.outputs[0]
    assert row["text_understood"] is False
    assert row["returned_chars"] is None
    assert row["returned_bytes"] is None
    assert row["truncation"] is None
    assert row["followup_observed"] is None


def test_partial_decoding_preserves_positive_truncation_evidence() -> None:
    value = [{"type": "text", "text": _spooled()}, {"type": "unknown-synthetic"}]
    result = metrics.measure(trace.analyze_trace(_doc([_step([_call()], [_result(value)])])))
    row = result.outputs[0]
    assert row["text_understood"] is False
    assert row["truncation"] is True
    assert result.trace["truncated_output_records"] == 1
    assert row["returned_chars"] is None


def test_foreign_unlinked_result_and_non_shell_results_not_promoted() -> None:
    result = metrics.measure(
        trace.analyze_trace(
            _doc(
                [
                    _step([_call()], [_result(_spooled(), "foreign-call")]),
                    _step([_call("read_text_file", {"path": SPOOL})], [_result(_spooled())]),
                    _step([], [_result(_spooled(), None)]),
                ]
            )
        )
    )
    assert result.outputs == ()
    assert result.trace["unclassified_observations"] == 3
    assert result.trace["shell_calls_missing_results"] == 1


@pytest.mark.parametrize(
    "name", ["process", "poll_process", "write_stdin", "bash_input", "awaitToolCall"]
)
def test_poll_output_records_included(name: str) -> None:
    result = metrics.measure(
        trace.analyze_trace(
            _doc([_step([_call(name, {"action": "status"})], [_result("synthetic poll output")])])
        )
    )
    assert result.trace["shell_output_records"] == 1
    assert result.outputs[0]["tool"] == name
    assert result.outputs[0]["context"] == "shell"


@pytest.mark.parametrize("action", ["list", "stop"])
def test_process_management_outputs_excluded(action: str) -> None:
    result = metrics.measure(
        trace.analyze_trace(
            _doc(
                [
                    _step([_call("functions.process", {"action": action})], [_result(_spooled())]),
                    _step([_call("functions.process", {"action": action})]),
                ]
            )
        )
    )
    assert result.trace["recorded_calls"] == 2
    assert result.outputs == ()
    assert result.trace["unclassified_observations"] == 1
    assert result.trace["shell_calls_missing_results"] == 0


def test_command_or_subagent_output_keeps_ambiguous_context() -> None:
    result = metrics.measure(
        trace.analyze_trace(
            _doc([_step([_call("tools.get_command_or_subagent_output")], [_result("synthetic")])])
        )
    )
    assert result.outputs[0]["context"] == "command_or_subagent"
    assert result.outputs[0]["tool"] == "get_command_or_subagent_output"


@pytest.mark.parametrize("limit", [None, True, "20", [], {}, -1, 1.5, 0, 20])
def test_requested_limit_narrowed_from_mapping_values(limit: object) -> None:
    result = metrics.measure(
        trace.analyze_trace(
            _doc(
                [
                    _step(
                        [_call(args={"command": "echo synthetic", "max_output_tokens": limit})],
                        [_result("synthetic")],
                    )
                ]
            )
        )
    )
    expected = limit if type(limit) is int and limit >= 0 else None
    assert result.outputs[0]["requested_max_output_tokens"] == expected


@pytest.mark.parametrize("handle", [None, True, 12, [], {}, ""])
def test_unresolvable_process_retention_is_unknown(handle: object) -> None:
    result = metrics.measure(
        trace.analyze_trace(
            _doc(
                [
                    _step(
                        [_call("process", {"process_id": handle, "action": []})],
                        [_result(_retained())],
                    )
                ]
            )
        )
    )
    row = result.outputs[0]
    assert row["spool_indicated"] is True
    assert row["retention"] == "unresolved_retention"
    assert row["followup_observed"] is None
    assert row["followup_accesses"] == 0


@pytest.mark.parametrize(
    "text", ["synthetic", _preview("Increase shell_execution.output_byte_limit to retain more.")]
)
def test_no_spool_indication_is_not_known_no_followup(text: str) -> None:
    result = metrics.measure(trace.analyze_trace(_doc([_step([_call()], [_result(text)])])))
    assert result.outputs[0]["retention"] == "not_indicated"
    assert result.outputs[0]["followup_observed"] is None


def test_spool_followups_any_later_access_not_echo_or_same_step() -> None:
    calls = [_call(call_id="origin"), _call(args={"command": f"cat {SPOOL}"}, call_id="same-step")]
    steps = [
        _step(calls, [_result(_spooled(), "origin"), _result("synthetic", "same-step")]),
        _step([_call(args={"command": f"echo {SPOOL}"})]),
        _step(
            [_call("read_text_file", {"path": SPOOL, "line": 2})], [_result("synthetic content")]
        ),
        _step([_call(args={"command": f"stat {SPOOL}"})]),
        _step(
            [_call(args={"command": f"cat {SPOOL}"})],
            [_result("process exit code was 0\nsynthetic retained text")],
        ),
    ]
    doc = _doc(steps)
    before = copy.deepcopy(doc)
    result = metrics.measure(trace.analyze_trace(doc))
    assert doc == before
    origin = result.outputs[0]
    assert origin["retention"] == "file"
    assert origin["followup_observed"] is True
    assert origin["followup_accesses"] == 3
    assert origin["first_followup_delay_calls"] == 2
    assert origin["partial_read_accesses"] == 1
    assert origin["content_returned_accesses"] == 2
    assert [row["step"] for row in result.followups] == [2, 3, 4]
    assert [row["access_kind"] for row in result.followups] == [
        "content_read",
        "metadata_or_other_access",
        "content_read",
    ]
    serialized = json.dumps([result.trace, result.tools, result.outputs, result.followups])
    assert SPOOL not in serialized
    assert HANDLE not in serialized
    assert "synthetic retained text" not in serialized
    assert "spool_paths" not in serialized
    assert "source_path" not in serialized


def test_resolved_spool_without_later_access_is_observed_false() -> None:
    result = metrics.measure(trace.analyze_trace(_doc([_step([_call()], [_result(_spooled())])])))
    assert result.outputs[0]["followup_observed"] is False


@pytest.mark.parametrize("handle_source", ["result", "arguments"])
def test_process_read_output_followup_exact_handle_not_wait_or_status(handle_source: str) -> None:
    args: JsonObject = {"command": "echo synthetic"}
    text = _retained()
    if handle_source == "result":
        text += f"\nprocess_id: {HANDLE}"
    else:
        args["process_id"] = HANDLE
    steps = [_step([_call(args=args)], [_result(text)])]
    requests: list[JsonObject] = [
        {"process_id": HANDLE, "action": "status"},
        {"process_id": HANDLE, "action": "wait"},
        {"process_id": "process-other", "action": "read_output"},
        {"process_id": HANDLE, "action": "read_output", "limit": 40},
        {"process_id": HANDLE, "action": "read_output"},
    ]
    steps.extend(
        _step([_call("functions.process", request)], [_result("synthetic retained content")])
        for request in requests
    )
    result = metrics.measure(trace.analyze_trace(_doc(steps)))
    row = result.outputs[0]
    assert row["retention"] == "process_handle"
    assert row["followup_observed"] is True
    assert row["followup_accesses"] == row["content_returned_accesses"] == 2
    assert row["partial_read_accesses"] == 1
    assert row["first_followup_delay_calls"] == 4
    assert [follow["step"] for follow in result.followups] == [4, 5]
    assert result.trace["shell_output_records"] == 6
    assert HANDLE not in json.dumps(result.outputs)


def test_resolved_process_without_read_output_is_observed_false() -> None:
    result = metrics.measure(
        trace.analyze_trace(
            _doc([_step([_call()], [_result(_retained() + f"\nprocess_id: {HANDLE}")])])
        )
    )
    assert result.outputs[0]["retention"] == "process_handle"
    assert result.outputs[0]["followup_observed"] is False


def test_subagents_preserve_issue_and_are_not_merged() -> None:
    doc = _doc([_step([_call()])])
    doc["subagent_trajectories"] = [_doc([_step([_call()], [_result(_spooled())])])]
    result = metrics.measure(trace.analyze_trace(doc))
    assert "subagent_trajectories_present" in result.trace["issues"]
    assert result.trace["recorded_calls"] == 1
    assert result.trace["agent_steps"] == 1
    assert result.outputs == ()


def test_invalid_step_collection_cannot_establish_offered_unused() -> None:
    doc = _doc([], declarations=_definitions("shell"))
    doc["steps"] = None
    result = metrics.measure(trace.analyze_trace(doc))
    assert result.trace["inventory_status"] == "known"
    assert result.trace["declared_tools"] == 1
    assert "invalid_steps" in result.trace["issues"]
    assert result.tools[0]["declared"] is True
    assert result.tools[0]["offered_unused"] is None


def _audited(doc: JsonObject) -> tuple[metrics.Metrics, model_calls.CallAudit]:
    evidence = trace.analyze_trace(doc)
    call_audit = model_calls.evaluate_calls(evidence, doc)
    return metrics.measure(evidence, call_audit), call_audit


@pytest.mark.parametrize(
    "text,rejected",
    [
        ("Tool 'shell' not found.", True),
        ("Unknown tool: tools.shell", True),
        ("Tool 'shell' is not available.", True),
        ("Error executing tool shell: Invalid arguments: missing command", False),
        ("<tool_use_error>Invalid arguments for shell</tool_use_error>", False),
        ("command output quotes: Tool 'shell' not found.", False),
    ],
)
def test_only_dispatch_rejections_leave_harness_usage(text: str, rejected: bool) -> None:
    doc = _doc(
        [_step([_call()], [_result(text)])],
        declarations=_definitions("process"),
    )
    result, audit = _audited(doc)
    assert len(audit.attempts) == 1
    assert len(audit.rejected) == int(rejected)
    assert result.trace["recorded_calls"] == int(not rejected)
    assert result.trace["shell_output_records"] == int(not rejected)
    assert result.trace["unclassified_observations"] == 0
    assert result.trace["recorded_calls"] == sum(row["recorded_calls"] for row in result.tools)
    assert not any("reject" in key or "model" in key for key in result.trace)
    assert ("shell" in {row["tool"] for row in result.tools}) is not rejected
    assert _tool(result, "process")["offered_unused"] is True


def test_declaration_conflict_preserves_inventory_but_unused_is_unknown() -> None:
    result, audit = _audited(
        _doc(
            [_step([_call()], [_result("Unknown tool: shell")])],
            declarations=_definitions("shell", "process"),
        )
    )
    assert audit.rejected[0].declaration_conflict is True
    assert result.trace["inventory_status"] == "known"
    assert result.trace["declared_tools"] == 2
    assert {row["tool"] for row in result.tools} == {"shell", "process"}
    assert all(row["declared"] is True and row["offered_unused"] is None for row in result.tools)
    assert result.trace["recorded_calls"] == result.trace["shell_output_records"] == 0


def test_rejected_result_is_not_reassigned_after_excluding_one_of_two_calls() -> None:
    doc = _doc(
        [
            _step(
                [_call("shell", call_id="bad"), _call("shell", call_id="good")],
                [_result("Unknown tool: shell", None), _result("valid output", None)],
            )
        ]
    )
    result, audit = _audited(doc)
    assert len(audit.attempts) == 2 and audit.rejected[0].attempt.index == 0
    assert result.trace["recorded_calls"] == 1
    assert len(result.outputs) == 1
    assert result.outputs[0]["observation"] == 1
    assert result.outputs[0]["pairing"] == "position"
    assert result.outputs[0]["returned_chars"] == len("valid output")
    assert result.trace["shell_calls_missing_results"] == 0


def test_unknown_names_are_coverage_not_an_inventory_sentinel() -> None:
    doc = _doc(
        [_step([_call("private\nunsafe")])],
        declarations=_definitions("shell"),
    )
    result, audit = _audited(doc)
    assert audit.attempts[0].tool_label is None
    assert [row["tool"] for row in result.tools] == ["shell"]
    assert result.trace["recorded_calls"] == 0
    assert result.trace["recorded_unknown_tool_labels"] == 1
    assert result.trace["static_unknown_tool_labels"] == 0
    assert _tool(result, "shell")["offered_unused"] is None


def test_unknown_derived_name_is_counted_separately_and_exec_prefix_is_stripped() -> None:
    from dataclasses import replace

    evidence = trace.analyze_trace(
        _doc(
            [
                _step(
                    [
                        _call(
                            "exec",
                            {"input": 'await tools.exec_command({cmd: "echo synthetic"});'},
                        )
                    ]
                )
            ],
            family="codex",
        )
    )
    step = evidence.steps[0]
    derived = step.calls[1]
    modified = replace(step, calls=(*step.calls, replace(derived, index=2, name="exec>bad\nlabel")))
    result = metrics.measure(replace(evidence, steps=(modified,)))
    assert result.trace["recorded_calls"] == 1
    assert result.trace["static_call_sites"] == 1
    assert result.trace["static_unknown_tool_labels"] == 1
    assert _tool(result, "exec_command")["static_call_sites"] == 1
    assert all(row["tool"] != "unknown_tool_label" for row in result.tools)


def _claude_result(recorded: str, native: object, call_id: str = "synthetic-call") -> JsonObject:
    result = _result(recorded, call_id)
    result["extra"] = {
        "tool_result_metadata": {
            "raw_tool_result": {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": native,
            }
        }
    }
    return result


def test_native_preview_measures_2220_not_appended_exporter_stdout() -> None:
    native = "é" * 2220
    recorded = native + "\n[stdout]\n" + "x" * 30_000
    result, _ = _audited(
        _doc(
            [_step([_call("Bash")], [_claude_result(recorded, native)])],
            family="claude-code",
        )
    )
    row = result.outputs[0]
    assert row["returned_chars"] == row["model_visible_chars"] == 2220
    assert row["returned_bytes"] == row["model_visible_bytes"] == 4440
    assert row["recorded_chars"] == len(recorded)
    assert row["recorded_bytes"] == len(recorded.encode("utf-8"))
    assert row["visibility_basis"] == "claude_native_tool_result"
    assert row["returned_basis"] == "model_visible"
    assert native not in json.dumps(row)


def test_unknown_visibility_uses_recorded_evidence_without_claiming_delivery() -> None:
    text = "synthetic recorded output"
    result, _ = _audited(_doc([_step([_call()], [_result(text)])]))
    row = result.outputs[0]
    assert row["returned_chars"] == row["recorded_chars"] == len(text)
    assert row["returned_bytes"] == row["recorded_bytes"] == len(text.encode("utf-8"))
    assert row["model_visible_chars"] is row["model_visible_bytes"] is None
    assert row["visibility_basis"] == "unknown"
    assert row["returned_basis"] == "recorded"


def test_native_rejection_is_recognized_despite_appended_exporter_stdout() -> None:
    native = "<tool_use_error>Error: No such tool available: Bash</tool_use_error>"
    recorded = native + "\n[stdout]\n" + "x" * 30_000
    result, audit = _audited(
        _doc(
            [_step([_call("Bash")], [_claude_result(recorded, native)])],
            family="claude-code",
        )
    )
    assert len(audit.attempts) == len(audit.rejected) == 1
    assert result.trace["recorded_calls"] == 0
    assert result.tools == result.outputs == ()


def test_native_spool_and_native_followup_outcome_ignore_exporter_extras() -> None:
    native = (
        f"<persisted-output>\nOutput too large (30KB). Full output saved to: {SPOOL}\n\n"
        "Preview (first 2000 bytes):\nsynthetic preview\n</persisted-output>"
    )
    doc = _doc(
        [
            _step(
                [_call("Bash")],
                [_claude_result(native + "\n[stdout]\n" + "x" * 30_000, native)],
            ),
            _step(
                [_call("Read", {"file_path": SPOOL})],
                [_claude_result("synthetic exported content", "")],
            ),
        ],
        family="claude-code",
    )
    result, _ = _audited(doc)
    row = result.outputs[0]
    assert row["retention"] == "file" and row["followup_accesses"] == 1
    assert row["returned_chars"] == len(native)
    assert row["recorded_chars"] > 30_000
    assert row["content_returned_accesses"] == 0
    assert result.followups[0]["outcome"] == "unknown"


def test_rejected_spool_reader_is_not_a_harness_followup() -> None:
    result, audit = _audited(
        _doc(
            [
                _step([_call()], [_result(_spooled())]),
                _step(
                    [_call("read_text_file", {"path": SPOOL})],
                    [_result("Unknown tool: read_text_file")],
                ),
            ]
        )
    )
    assert len(audit.attempts) == 2 and len(audit.rejected) == 1
    assert result.trace["recorded_calls"] == 1
    assert result.outputs[0]["followup_accesses"] == 0
    assert result.followups == ()


@pytest.mark.parametrize(
    "native",
    [
        [{"type": "image", "source": {"type": "url", "url": "https://synthetic.invalid/image"}}],
        [{"type": "text", "text": "caption"}, {"type": "image"}],
    ],
)
def test_native_media_projection_length_is_unknown_not_zero(native: object) -> None:
    result, _ = _audited(
        _doc(
            [_step([_call("Bash")], [_claude_result("synthetic exported stdout", native)])],
            family="claude-code",
        )
    )
    row = result.outputs[0]
    assert row["visibility_basis"] == "claude_native_tool_result"
    assert row["returned_basis"] == "model_visible"
    assert row["model_visible_chars"] is row["model_visible_bytes"] is None
    assert row["returned_chars"] is row["returned_bytes"] is None
    assert row["recorded_chars"] == len("synthetic exported stdout")
    assert row["has_media"] is True


def test_legacy_trace_without_visibility_still_measures_recorded_evidence() -> None:
    from dataclasses import replace

    evidence = trace.analyze_trace(_doc([_step([_call()], [_result("synthetic")])]))
    result = metrics.measure(replace(evidence, result_visibility=()))
    assert result.outputs[0]["recorded_chars"] == result.outputs[0]["returned_chars"] == 9
    assert result.outputs[0]["model_visible_chars"] is None
    assert result.outputs[0]["visibility_basis"] == "unknown"


@pytest.mark.parametrize("recorded", [False, True])
@pytest.mark.parametrize("retention", ["file", "process_handle"])
def test_static_spool_references_remain_sites_not_observed_accesses(
    recorded: bool, retention: str
) -> None:
    name = "read_text_file" if retention == "file" else "process"
    args: JsonObject = (
        {"path": SPOOL} if retention == "file" else {"process_id": HANDLE, "action": "read_output"}
    )
    output = _spooled() if retention == "file" else f"process_id: {HANDLE}\n{_retained()}"
    steps = [
        _step([_call()], [_result(output)]),
        _step([_call("exec", {"input": f"if(false){{await tools.{name}({json.dumps(args)});}}"})]),
    ]
    if recorded:
        steps.append(_step([_call(name, args)], [_result("synthetic retained text")]))
    result = metrics.measure(trace.analyze_trace(_doc(steps)))
    assert _tool(result, name)["static_call_sites"] == 1
    assert _tool(result, name)["recorded_calls"] == int(recorded)
    row = result.outputs[0]
    assert row["retention"] == retention
    assert row["followup_observed"] is recorded
    assert row["followup_accesses"] == len(result.followups) == int(recorded)
    assert row["first_followup_delay_calls"] == (2 if recorded else None)


def test_redacted_input_cannot_certify_offered_unused() -> None:
    from dataclasses import replace

    evidence = trace.analyze_trace(_doc([], declarations=_definitions("shell", "process")))
    assert all(row["offered_unused"] is True for row in metrics.measure(evidence).tools)
    result = metrics.measure(replace(evidence, issues=(*evidence.issues, "redacted_input")))
    assert all(row["offered_unused"] is None for row in result.tools)
    assert result.trace["recorded_calls"] == 0


def test_nullable_serialized_truncation_counts_only_positive_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harness_outputs import OutputEvidence

    original = OutputEvidence.to_doc

    def nullable_doc(evidence: OutputEvidence) -> Doc:
        # Exercise the serializer contract independently of the output-parser edit.
        doc = original(evidence)
        if evidence.kind in {"unsupported", "possible_signal"}:
            doc["truncation"] = None
        return doc

    monkeypatch.setattr(OutputEvidence, "to_doc", nullable_doc)
    known = metrics.measure(
        trace.analyze_trace(
            _doc(
                [
                    _step([_call()], [_result(_spooled())]),
                    _step([_call()], [_result("[Output truncated: unrecognized synthetic]")]),
                    _step([_call()], [_result("synthetic plain output")]),
                ]
            )
        )
    )
    unsupported = metrics.measure(
        trace.analyze_trace(
            _doc([_step([_call()], [_result("synthetic")])], family="synthetic-harness")
        )
    )
    assert [row["truncation"] for row in known.outputs] == [True, None, False]
    assert known.trace["truncated_output_records"] == 1
    assert unsupported.outputs[0]["truncation"] is None
    assert unsupported.trace["truncated_output_records"] == 0
    assert json.loads(json.dumps(known.outputs))[1]["truncation"] is None
