"""Synthetic failures, refusals and count units; no private trace contents."""

import json

import pytest

from atif_scan import Status
from atif_scan.detectors.integrity import incomplete_tool_generation
from atif_scan.facts import trace_facts
from atif_scan.loader import parse_trace
from atif_scan.output.brief import brief, brief_text
from atif_scan.output.document import recording_gaps
from atif_scan.web_gaps import web_gaps


def trajectory(status="incomplete", error="Error: synthetic rejected arguments"):
    return parse_trace(
        {
            "steps": [
                {
                    "source": "agent",
                    "tool_calls": [
                        {
                            "tool_call_id": "bad",
                            "function_name": "poll",
                            "arguments": {"input": "PRIVATE_SYNTHETIC" * 100},
                        },
                        {"tool_call_id": "repair", "function_name": "poll", "arguments": {}},
                    ],
                    "extra": {"tool_call_details": {"bad": {"status": status}}},
                    "observation": {
                        "results": [
                            {"source_call_id": "bad", "content": error},
                            {"source_call_id": "repair", "content": "success"},
                        ]
                    },
                }
            ],
            "final_metrics": {"total_completion_tokens": 1},
        }
    )


def test_failed_generation_with_same_step_repair():
    trace = trajectory()
    result = incomplete_tool_generation(trace)
    assert result.status is Status.MATCH
    assert [(e.step, e.call) for e in result.evidence] == [(0, 0)]
    assert "PRIVATE_SYNTHETIC" not in json.dumps(trace_facts(trace))


@pytest.mark.parametrize(
    ("status", "error"),
    [
        ("completed", "Error: synthetic failure"),
        ("incomplete", "normal result"),
        ("incomplete", None),
        ("private-unknown-status", "Error: synthetic failure"),
    ],
)
def test_no_unsupported_failure_claim(status, error):
    trace = trajectory(status, error)
    assert incomplete_tool_generation(trace).status is not Status.MATCH
    if status == "private-unknown-status":
        assert trace.steps[0].calls[0].status is None


def web_trace(results=None, code=False):
    calls = [
        {
            "tool_call_id": "",
            "function_name": "web_search_call",
            "arguments": {"action_type": "search", "query": "PRIVATE_QUERY"},
        },
    ]
    if code:
        calls = [
            {
                "tool_call_id": "parent",
                "function_name": "exec",
                "arguments": {
                    "input": (
                        'await tools.web_search({query: "one"});'
                        'await tools.web_search({query: "two"});'
                    )
                },
            }
        ]
    return parse_trace(
        {
            "steps": [
                {
                    "source": "agent",
                    "tool_calls": calls,
                    "observation": {"results": results or []},
                }
            ]
        }
    )


def test_web_counts_empty_ids_and_null_contents_and_trials():
    trace = web_trace([{"source_call_id": "", "content": None}])
    facts = trace_facts(trace)
    assert facts["web_result_gaps"] == {
        "calls": 1,
        "missing_result_ids": 1,
        "no_emitted_contents": 1,
    }
    gaps = recording_gaps([facts, facts])
    assert gaps["web_calls_without_usable_result"] == 2
    assert gaps["web_call_gap_trials"] == 2
    assert "PRIVATE_QUERY" not in json.dumps(gaps)


def test_error_is_recorded_outcome_and_unlinked_content_not_absent():
    assert web_gaps(web_trace([{"content": "Error: unable to fetch"}])).calls == 0
    assert web_gaps(web_trace([{"content": "status: completed"}])).document() == {
        "calls": 1,
        "missing_result_ids": 1,
        "no_emitted_contents": 0,
    }


def test_shared_parent_counts_once():
    trace = web_trace(code=True)
    assert len(trace.steps[0].calls) == 3
    assert web_gaps(trace).calls == 1


@pytest.mark.parametrize("steps", [0, 1])
def test_refusal_without_usage_even_with_zero_cost(steps):
    item = {
        "input_id": "synthetic",
        "input_status": "available",
        "incomplete": False,
        "assessments": [],
        "error_type": "AgentSafetyRefusalError",
        "agent_steps": steps,
        "llm_calls": steps,
        "cost_usd": 0,
    }
    result = brief({"scanner_version": "dev", "inputs": [item], "coverage": {}})
    assert result["usage"]["refusals_without_token_counts"] == 1
    text = brief_text(result)
    assert "consumption remains unknown" in text
    assert "recorded refusal" in text
    assert "zero consumption" not in text


def test_web_gap_brief_uses_calls_not_actions():
    row = {
        **trace_facts(web_trace()),
        "input_id": "synthetic",
        "input_status": "available",
        "incomplete": True,
        "assessments": [],
        "recording_gaps": {"web_results_not_recorded": 1},
    }
    result = brief({"scanner_version": "dev", "inputs": [row], "coverage": {}})
    text = " ".join(brief_text(result).split())
    assert "1 recorded web calls have no usable result across 1 trial" in text
    assert "calls are not web actions or backend requests" in text


def test_error_on_other_call_does_not_link_failure():
    trace = trajectory(error="normal result")
    assert incomplete_tool_generation(trace).status is Status.NO_MATCH


def test_refusal_with_recorded_tokens_is_not_missing_usage():
    row = {
        "input_id": "synthetic",
        "input_status": "available",
        "incomplete": False,
        "assessments": [],
        "error_type": "AgentSafetyRefusalError",
        "input_tokens": 10,
        "output_tokens": 1,
        "cost_usd": 0,
    }
    result = brief({"scanner_version": "dev", "inputs": [row], "coverage": {}})
    assert result["usage"]["refusals_without_token_counts"] == 0


def test_unlinked_and_unreadable_results_are_not_no_emitted_contents():
    from atif_scan.model import Content, Observation, Step, ToolCall, Trace

    call = ToolCall(0, "web_search", (), "c", "web_search", {})
    for content in (Content("unlinked body"), Content(understood=False)):
        trace = Trace(
            None,
            (
                Step(
                    0,
                    "agent",
                    False,
                    Content(),
                    Content(),
                    (call,),
                    (Observation("other", content),),
                ),
            ),
        )
        assert web_gaps(trace).calls == 1
        assert web_gaps(trace).no_emitted_contents == 0


SCRIPT_BLOCKS = [
    {"type": "input_text", "text": "Script failed\nWall time 0.0 seconds\nOutput:\n"},
    {"type": "input_text", "text": "Script error:\nSyntaxError: Invalid or unexpected token"},
]


@pytest.mark.parametrize("serialize", [repr, json.dumps])
def test_serialized_script_failure_corrobates_incomplete(serialize):
    result = incomplete_tool_generation(trajectory(error=serialize(SCRIPT_BLOCKS)))
    assert result.status is Status.MATCH
    assert len(result.evidence) == 1


@pytest.mark.parametrize(
    "error",
    [
        repr(SCRIPT_BLOCKS[1:]),
        "A tutorial mentions Script failed and SyntaxError: Invalid or unexpected token",
        "Synthetic page\n" + "\n".join(b["text"] for b in SCRIPT_BLOCKS),
        repr([{"type": "input_text", "text": "Script completed\nWall time 0 seconds\nOutput:\n"}]),
        repr([*SCRIPT_BLOCKS, {"type": "image"}]),
    ],
)
def test_incomplete_without_recorded_script_failure(error):
    assert incomplete_tool_generation(trajectory(error=error)).status is Status.NO_MATCH


def test_script_failure_requires_ratio_anomaly_and_incomplete_status():
    from dataclasses import replace

    trace = trajectory(error=repr(SCRIPT_BLOCKS))
    assert incomplete_tool_generation(replace(trace, usage=None)).status is not Status.MATCH
    assert (
        incomplete_tool_generation(trajectory("completed", repr(SCRIPT_BLOCKS))).status
        is Status.NO_MATCH
    )


@pytest.mark.parametrize(
    "serialize", [lambda text: text, lambda text: repr([{"type": "input_text", "text": text}])]
)
@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("Script completed\nWall time 0 seconds\nOutput:\n", 1),
        ("pending", 0),
        ("status: completed", 0),
        ("Error: retrieval failed", 0),
        ("Synthetic page", 0),
    ],
)
def test_linked_runner_empty_web_counts(serialize, text, count):
    trace = web_trace([{"source_call_id": "parent", "content": serialize(text)}], code=True)
    assert web_gaps(trace).no_emitted_contents == count


def test_multiple_missing_ids_do_not_associate_null_results_or_shell():
    trace = parse_trace(
        {
            "steps": [
                {
                    "source": "agent",
                    "tool_calls": [
                        {
                            "tool_call_id": "",
                            "function_name": "web_search_call",
                            "arguments": {"query": "one"},
                        },
                        {
                            "tool_call_id": "",
                            "function_name": "web_search_call",
                            "arguments": {"query": "two"},
                        },
                        {"tool_call_id": "shell", "function_name": "shell", "arguments": {}},
                    ],
                    "observation": {
                        "results": [
                            {"content": None},
                            {"source_call_id": "shell", "content": "Synthetic shell output"},
                        ]
                    },
                }
            ]
        }
    )
    step = trace.steps[0]
    assert not step.results_for(step.calls[0])
    assert not step.results_for(step.calls[1])
    assert web_gaps(trace).document() == {
        "calls": 2,
        "missing_result_ids": 2,
        "no_emitted_contents": 2,
    }
