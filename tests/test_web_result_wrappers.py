"""Synthetic runner wrappers must not substitute for returned web evidence."""

import json

import pytest

from atif_scan.model import Content, Observation, Step, ToolCall
from atif_scan.web_results import (
    WebResultState,
    recorded_web_content,
    web_outcomes_recorded,
    web_result_state,
    web_results_complete,
)

RUNNER = "Script completed\nWall time 1.9 seconds\nOutput:\n"


@pytest.mark.parametrize(
    "serialize",
    [
        lambda s: s,
        lambda s: repr([{"type": "text", "text": s}]),
        lambda s: json.dumps([{"type": "text", "text": s}]),
    ],
)
@pytest.mark.parametrize(
    ("text", "state"),
    [
        (RUNNER, WebResultState.STATUS_ONLY),
        (RUNNER + " \t\n", WebResultState.STATUS_ONLY),
        (" \n" + RUNNER + "\n ", WebResultState.STATUS_ONLY),
        (RUNNER.replace("\n", "\r\n"), WebResultState.STATUS_ONLY),
        ("Script completed", WebResultState.STATUS_ONLY),
        ("Script completed\n\tWall time  1.9 seconds\n Output:  \t", WebResultState.STATUS_ONLY),
        ("Script completed\nWall time: 2 seconds", WebResultState.STATUS_ONLY),
        ("status: OK", WebResultState.STATUS_ONLY),
        ("pending", WebResultState.STATUS_ONLY),
        (" \t\n", WebResultState.UNAVAILABLE),
        (RUNNER + "Synthetic page body.", WebResultState.CONTENT),
        ("Synthetic page body mentions failed to fetch.", WebResultState.CONTENT),
        ("Script completed is a phrase on this page.", WebResultState.CONTENT),
        ("Error: retrieval timed out", WebResultState.ERROR),
        ("Failed to fetch https://example.org/page", WebResultState.ERROR),
        ("HTTP 403 Forbidden", WebResultState.ERROR),
        ("status: failed", WebResultState.ERROR),
        (RUNNER + "Error: retrieval timed out", WebResultState.ERROR),
    ],
)
def test_result_state(text, state, serialize):
    content = Content(serialize(text))
    assert web_result_state(content) is state
    assert recorded_web_content(content) is (state is WebResultState.CONTENT)


@pytest.mark.parametrize(
    "content",
    [
        Content(RUNNER, understood=False),
        Content("Synthetic body", media=True),
        Content("Synthetic body", understood=False),
        Content(repr([{"type": "text", "text": "Synthetic"}, {"type": "image"}])),
        Content(repr([{"type": "text", "text": "Synthetic"}, {"unknown": "block"}])),
        Content("[{'type': 'text', 'text': unknown()}]"),
        Content("[{'type': 'text', 'text': 'synthetic'}] + arbitrary()"),
        Content("[{'type': 'text', 'text': 'synthetic', 'text': ''}]"),
        Content("[{'type': 'text', 'text':"),
        Content("[]"),
        Content(repr([{"type": "text", "text": "[Image 1]"}])),
    ],
)
def test_unknown_and_media_are_not_cleared(content):
    assert web_result_state(content) is WebResultState.UNAVAILABLE
    assert not recorded_web_content(content)


def result_step(contents):
    call = ToolCall(0, "web_fetch", (), "a", "web_fetch", {"url": "https://example.org"})
    step = Step(
        0,
        "agent",
        False,
        Content(),
        Content(),
        (call,),
        tuple(Observation("a", c) for c in contents),
    )
    return step, call


@pytest.mark.parametrize(
    ("texts", "complete", "recorded"),
    [
        ([], False, False),
        (["Synthetic body"], True, True),
        (["Error: retrieval timed out"], False, True),
        (["Synthetic body", "Error: retrieval timed out"], False, True),
        (["Synthetic body", RUNNER], False, False),
        (["Error: retrieval timed out", RUNNER], False, False),
        (["Synthetic body", ""], False, False),
    ],
)
def test_multipart_outcomes_are_distinct_from_returned_content(texts, complete, recorded):
    step, call = result_step([Content(t) for t in texts])
    assert web_results_complete(step, call) is complete
    assert web_outcomes_recorded(step, call) is recorded


@pytest.mark.parametrize("gap", [Content(media=True), Content(understood=False)])
def test_multipart_unknown_cannot_be_hidden(gap):
    step, call = result_step([Content("Synthetic body"), gap])
    assert not web_results_complete(step, call)
    assert not web_outcomes_recorded(step, call)


@pytest.mark.parametrize("status", ["pending", "status: OK", "done"])
def test_serialized_multipart_status_cannot_be_hidden(status):
    blocks = [{"type": "text", "text": text} for text in ("Synthetic body", status)]
    content = Content(repr(blocks))
    assert web_result_state(content) is WebResultState.STATUS_ONLY
    step, call = result_step([content])
    assert not web_results_complete(step, call)
    assert not web_outcomes_recorded(step, call)


METADATA = (
    "\ue200cite\ue202turn3view0\ue201 [wordlim: 200] "
    'Source: find({"ref_id":"turn2search2","pattern":"example"})'
)
PAGE = "Example (https://example.org)\n\ue200cite\ue202turn1view0\ue201 Synthetic page."
PROVIDER_ERROR = "Internal Error ()\n" + METADATA + "\nSynthetic failure detail."


@pytest.mark.parametrize("channel", ["text", "input_text", "output_text"])
@pytest.mark.parametrize("serialize", [repr, json.dumps])
@pytest.mark.parametrize(
    ("payload", "state"),
    [
        (PAGE, WebResultState.CONTENT),
        (PROVIDER_ERROR, WebResultState.ERROR),
        (METADATA + "\nUnable to resolve click call: synthetic target", WebResultState.ERROR),
    ],
)
def test_runner_preface_with_payload(channel, serialize, payload, state):
    blocks = [{"type": channel, "text": text} for text in (RUNNER, payload)]
    content = Content(serialize(blocks))
    assert web_result_state(content) is state
    step, call = result_step([content])
    assert web_outcomes_recorded(step, call)
    assert web_results_complete(step, call) is (state is WebResultState.CONTENT)


@pytest.mark.parametrize("gap", ["", "pending", None])
def test_runner_preface_does_not_hide_other_siblings(gap):
    blocks = [{"type": "input_text", "text": text} for text in (RUNNER, PAGE)]
    blocks.append({"type": "input_text", "text": gap})
    state = web_result_state(Content(repr(blocks)))
    assert state in (WebResultState.UNAVAILABLE, WebResultState.STATUS_ONLY)


@pytest.mark.parametrize(
    "text",
    [
        PROVIDER_ERROR,
        METADATA + "\nUnable to resolve click call: synthetic target",
    ],
)
def test_provider_errors_are_outcomes_not_content(text):
    content = Content(text)
    assert web_result_state(content) is WebResultState.ERROR
    step, call = result_step([content])
    assert web_outcomes_recorded(step, call)
    assert not web_results_complete(step, call)


@pytest.mark.parametrize(
    "text",
    [
        "Internal Error ()\nAn ordinary page discussing an error.",
        "Synthetic page\n" + PROVIDER_ERROR,
        "Internal Error ()\n\ue200cite\ue202turn3view0\ue201 No source metadata.",
        METADATA + "\nA page says Unable to resolve click call.",
        "Unable to resolve click call is a phrase in this tutorial.",
        PAGE + "\nUnable to resolve click call: quoted example.",
    ],
)
def test_error_phrases_in_pages_remain_content(text):
    assert recorded_web_content(Content(text))
