"""Synthetic authored-payload accounting, independent of observations and envelopes."""

import json
from types import MappingProxyType

import pytest

from atif_scan import Status
from atif_scan.data.jsonval import compact_json
from atif_scan.data.loader import parse_calls, parse_trace
from atif_scan.detectors.integrity import output_ratio, output_token_ratio


def call(args):
    return {"tool_call_id": "c", "function_name": "poll", "arguments": args}


def parsed(args, detail=None):
    return parse_calls([call(args)], False, {"c": detail})[0]


def test_nested_json_and_unicode_characters_not_bytes():
    args = {"n": 12, "yes": True, "no": False, "nil": None, "nested": [{"é": '\n"😀'}]}
    expected = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result = parsed(args)
    assert result.raw_argument_chars == len(expected)
    assert result.raw_argument_basis == "compact_json"
    assert len(expected) < len(expected.encode())
    assert compact_json(result.arguments) == expected
    assert compact_json(MappingProxyType({"a": (MappingProxyType({"b": None}),)})) == (
        '{"a":[{"b":null}]}'
    )


@pytest.mark.parametrize("raw", ['{ "n" : 1, "s": "\\u00e9" }', '{"s":"é","n":1}'])
def test_validated_json_metadata_and_direct_serialization(raw):
    args = {"n": 1, "s": "é"}
    detail = {"raw_arguments": raw, "item_type": "function_call", "status": "completed"}
    result = parsed(args, detail)
    assert (result.raw_argument_chars, result.raw_argument_basis) == (len(raw), "recorded_json")
    assert parsed(raw).raw_argument_chars == len(raw)


@pytest.mark.parametrize(
    "detail",
    [
        None,
        {"raw_arguments": " " * 500},
        {"raw_arguments": '{"n":true}', "item_type": "function_call"},
        {"raw_arguments": '{"n":2}', "item_type": "function_call"},
        {"raw_arguments": '{"n":1}', "item_type": "custom_tool_call"},
        {"raw_arguments": None, "item_type": "function_call"},
        {"raw_arguments": "{broken", "item_type": "function_call"},
    ],
)
def test_unvalidated_metadata_cannot_inflate_payload(detail):
    result = parsed({"n": 1}, detail)
    assert (result.raw_argument_chars, result.raw_argument_basis) == (7, "compact_json")


@pytest.mark.parametrize("args", [None, [], 4, False, "{broken", "[broken", "null", "42"])
def test_unknown_arguments_do_not_invent_counts(args):
    result = parsed(args, {"raw_arguments": "x" * 500, "item_type": "custom_tool_call"})
    assert result.raw_argument_chars is None
    assert result.raw_argument_basis is None


@pytest.mark.parametrize("wrapped", [False, True])
def test_raw_patch_text_once(wrapped):
    raw = "*** Begin Patch\n*** Add File: synthetic.txt\n+é\n*** End Patch"
    args = {"input": raw} if wrapped else raw
    result = parsed(args, {"raw_arguments": raw, "item_type": "custom_tool_call"})
    assert (result.raw_argument_chars, result.raw_argument_basis) == (len(raw), "raw_text")
    assert parsed(args).raw_argument_chars == len(raw)


def test_function_input_object_metadata_takes_precedence_over_raw_convention():
    raw = '{"input":"hello"}'
    result = parsed({"input": "hello"}, {"raw_arguments": raw, "item_type": "function_call"})
    assert (result.raw_argument_chars, result.raw_argument_basis) == (len(raw), "recorded_json")


def test_custom_exec_derived_calls_and_results_not_counted():
    program = 'const r = await tools.exec_command({cmd: "echo synthetic"});'
    outer = {**call({"input": program}), "function_name": "exec"}
    calls = parse_calls([outer], False)
    assert len(calls) == 2
    assert calls[0].raw_argument_chars == len(program)
    assert calls[1].raw_argument_chars is None


def test_empty_poll_ratio_counts_structure_excludes_copies_and_metadata():
    args = {"session_id": 123, "chars": "", "yield_time_ms": 1000}
    raw = json.dumps(args)
    step = {
        "source": "agent",
        "tool_calls": [call(args)],
        "extra": {
            "tool_call_details": {
                "c": {
                    "item_type": "function_call",
                    "raw_arguments": raw,
                    "status": "completed",
                }
            }
        },
        "observation": {"results": [{"source_call_id": "c", "content": "x" * 10000}]},
    }
    trace = parse_trace(
        {
            "schema_version": "ATIF-v1.0",
            "steps": [step, step, {**step, "is_copied_context": True}],
            "final_metrics": {
                "total_completion_tokens": 40,
                "extra": {"total_reasoning_tokens": 0},
            },
        }
    )
    ratio = output_ratio(trace)
    assert ratio is not None
    assert ratio.chars == 2 * len(raw)
    assert ratio.tokens == 40
    assert output_token_ratio(trace).status == Status.NO_MATCH
