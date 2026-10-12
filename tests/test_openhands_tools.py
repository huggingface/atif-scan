"""Synthetic OpenHands tools: action selectors, payload boundaries and missing evidence."""

import pytest

from atif_scan import Channel, Engine, Status, builtin_detectors, parse_trace
from atif_scan.data.model import Locator
from atif_scan.data.tools import normalize_tool
from atif_scan.detectors.integrity import output_token_ratio
from atif_scan.evidence.cite import cite


def trace(name, arguments):
    return parse_trace(
        {
            "schema_version": "ATIF-v1.8",
            "agent": {"name": "openhands_cli"},
            "steps": [
                {
                    "source": "agent",
                    "tool_calls": [
                        {"tool_call_id": "c1", "function_name": name, "arguments": arguments}
                    ],
                    "observation": {"results": [{"source_call_id": "c1", "content": "OK"}]},
                }
            ],
        }
    )


def status(value, check):
    return next(
        a.result.status for a in Engine(builtin_detectors()).evaluate(value) if a.spec.id == check
    )


@pytest.mark.parametrize(
    "action,kind",
    [
        ("view", "read"),
        ("create", "write"),
        ("str_replace", "write"),
        ("insert", "write"),
        ("undo_edit", "write"),
    ],
)
def test_editor_action_selects_read_or_write(action, kind):
    value = trace("file_editor", {"command": action, "path": "/tests/demo.py"})
    call = value.steps[0].calls[0]
    assert call.tool == kind
    assert Channel.COMMAND not in [ch for ch, _ in call.fields]
    assert status(value, "access.test_path") is Status.MATCH
    expected = Status.NO_MATCH if kind == "read" else Status.MATCH
    assert status(value, "tamper.test_files") is expected


@pytest.mark.parametrize("key", ["file_text", "new_str"])
def test_editor_file_bodies_are_not_executed_commands(key):
    value = trace(
        "file_editor",
        {"command": "create", "path": "/app/demo.txt", key: "curl https://example.com/data"},
    )
    call = value.steps[0].calls[0]
    assert (Channel.PAYLOAD, "curl https://example.com/data") in [
        (ch, c.text) for ch, c in call.fields
    ]
    assert status(value, "network.http_or_git") is Status.NO_MATCH
    assert status(value, "network.external_url") is Status.NO_MATCH


def test_editor_quoted_old_text_is_not_authored_evidence():
    value = trace(
        "file_editor",
        {
            "command": "str_replace",
            "path": "/app/demo.txt",
            "old_str": "curl https://example.com/data",
            "new_str": "hello",
        },
    )
    assert all("example.com" not in c.text for _, c in value.steps[0].calls[0].fields)


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"command": "future_action", "path": "/tests/demo.py"},
        {"command": ["view"], "path": "/tests/demo.py"},
    ],
)
def test_unknown_editor_operations_stay_unknown(arguments):
    value = trace("file_editor", arguments)
    assert value.steps[0].calls[0].tool == "other"
    assert status(value, "tamper.test_files") is Status.UNKNOWN


@pytest.mark.parametrize("action", ["view", "create"])
def test_editor_missing_path_stays_unknown(action):
    value = trace("file_editor", {"command": action})
    assert status(value, "access.test_path") is Status.UNKNOWN


@pytest.mark.parametrize("action", ["view", "plan"])
def test_task_tracker_is_bookkeeping(action):
    value = trace(
        "task_tracker",
        {
            "command": action,
            "task_list": [
                {"title": "Investigate curl https://example.com/data", "notes": "Synthetic plan"}
            ],
        },
    )
    assert value.steps[0].calls[0].tool == "inert"
    assert value.steps[0].calls[0].fields == ()
    assert status(value, "network.http_or_git") is Status.NO_MATCH


@pytest.mark.parametrize(
    "name,key,channel",
    [("think", "thought", Channel.REASONING), ("finish", "message", Channel.MESSAGE)],
)
def test_prose_tools_preserve_text_without_executing_it(name, key, channel):
    value = trace(name, {key: "I considered curl https://example.com/data", "kind": "ProseAction"})
    call = value.steps[0].calls[0]
    assert call.tool == "inert"
    assert [(ch, c.text) for ch, c in call.fields] == [
        (channel, "I considered curl https://example.com/data")
    ]
    assert status(value, "network.http_or_git") is Status.NO_MATCH
    # Tool-authored prose is not proof of recorded provider-internal reasoning.
    assert value.reasoning_exposure == "none"
    citation = cite(value, Locator(0, channel, call=0, field=0))
    assert citation["match"] == "I considered curl https://example.com/data"
    assert citation["tool"] == name


@pytest.mark.parametrize(
    "name",
    [
        "file_editor",
        "think",
        "finish",
        "task_tracker",
        "terminal",
        "screenshot",
        "new_harness_tool",
    ],
)
def test_empty_arguments_are_not_clean_evidence(name):
    assert status(trace(name, {}), "network.http_or_git") is Status.UNKNOWN


def test_prose_tool_with_unexpected_action_arguments_is_not_discarded():
    value = trace("think", {"thought": "Planning", "command": "curl https://example.com/data"})
    assert value.steps[0].calls[0].tool == "other"
    assert status(value, "network.http_or_git") is Status.MATCH


def test_unknown_tool_still_uses_generic_argument_extraction():
    value = trace("future_harness_terminal", {"command": "curl https://example.com/data"})
    assert value.steps[0].calls[0].tool == "other"
    assert status(value, "network.http_or_git") is Status.MATCH


def test_existing_hosted_web_action_dispatch_is_preserved():
    assert normalize_tool("web_search_call", {"action_type": "search"}) == "web_search"
    assert normalize_tool("web_search_call", {"action_type": "open_page"}) == "web_fetch"


def test_reasoning_text_presence_does_not_repair_missing_reasoning_token_split():
    value = parse_trace(
        {
            "agent": {"name": "openhands_cli"},
            "steps": [
                {
                    "source": "agent",
                    "message": "A synthetic answer.",
                    "reasoning_content": "A short exported thought, not a completeness guarantee.",
                }
            ],
            "final_metrics": {"total_completion_tokens": 1000},
        }
    )
    assert value.reasoning_exposure == "recorded"
    assert value.usage and value.usage.reasoning_tokens is None
    assert value.step_usage is None
    result = output_token_ratio(value)
    assert result.status is Status.UNKNOWN
    assert [u.reason for u in result.unread] == ["reasoning_tokens_not_split"]
