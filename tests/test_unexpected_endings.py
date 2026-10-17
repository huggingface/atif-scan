"""Unexpected endings (`ending.unexpected`): the agent loop stopped before the agent
concluded its turn. Synthetic fixtures modelled on fast-agent recordings: a 413 from
accumulated images, a reasoning-only max_tokens turn, an empty end_turn after a cut-off
stream, safety stops, time limits with tools in flight. Not a recording defect, not a
behaviour finding, never a DQ input."""

from __future__ import annotations

import base64
import json

import pytest

from atif_scan import Engine, Status, builtin_detectors, parse_trace, report
from atif_scan.data.content import content
from atif_scan.data.ending import ending
from atif_scan.data.facts import trace_facts
from atif_scan.output.brief import brief, brief_text
from atif_scan.output.overview import finding_index

SECRET_REPLY = "SYNTHETIC-REPLY-TEXT-must-not-be-exported"
PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG" + b"\x00" * 300).decode()


def agent(message="", *, stop=None, reasoning=None, calls=(), usage=True, retry=None):
    step: dict[str, object] = {"source": "agent", "message": message}
    if reasoning is not None:
        step["reasoning_content"] = reasoning
    if calls:
        step["tool_calls"] = [
            {"tool_call_id": cid, "function_name": "Shell", "arguments": {"command": "ls"}}
            for cid in calls
        ]
    if usage:
        step["metrics"] = {"prompt_tokens": 10, "completion_tokens": 5}
    extra: dict[str, object] = {}
    if stop is not None:
        extra["stop_reason"] = f"LlmStopReason.{stop}"
    if retry is not None:
        extra["retry"] = retry
    if extra:
        step["extra"] = extra
    return step


def answered(step, *cids):
    step["observation"] = {"results": [{"source_call_id": c, "content": "ok"} for c in cids]}
    return step


def user(message):
    return {"source": "user", "message": message}


def trajectory(*steps, termination=None):
    raw = {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {**step, "step_id": i, "timestamp": f"2026-10-14T10:00:{i:02d}Z"}
            for i, step in enumerate([user("Do the task."), *steps], start=1)
        ],
    }
    if termination is not None:
        raw["extra"] = {"termination": {"status": "error", "error_type": termination}}
    return raw


def retry_413(attempts=4):
    return {
        "schema": "fast-agent.retry/v1",
        "provider_attempts": attempts,
        "retries": [
            {
                "attempt": i,
                "error_type": "APIStatusError",
                "error_message": "Error code: 413 - {'error': 'request entity too large'}",
                "stream_events_received": None,
            }
            for i in range(1, attempts)
        ],
    }


def check(raw):
    assessments = Engine(builtin_detectors()).evaluate(parse_trace(raw))
    return next(a for a in assessments if a.spec.id == "ending.unexpected")


def measure(raw):
    return dict(check(raw).result.measure)


def test_a_reply_that_ends_the_turn_is_a_normal_ending():
    found = check(trajectory(agent(SECRET_REPLY, stop="END_TURN")))
    assert found.result.status == Status.NO_MATCH
    assert dict(found.result.measure)["kind"] == "completed"
    assert found.kind == "context" and not found.counts


def test_a_413_after_images_piled_up_is_a_failed_model_call_with_its_media():
    """Regression (tb4 ds41flash): every attach_media image stayed in the history as
    base64 until the provider refused the request (HTTP 413), retried three times, and
    fast-agent wrote its error text as the final agent message."""
    raw = trajectory(
        agent(stop="TOOL_USE", calls=["c1"]),
        user([{"type": "image", "source": {"media_type": "image/png", "path": PNG}}]),
        agent(
            "I hit an internal error while calling the model",
            stop="ERROR",
            usage=False,
            retry=retry_413(),
        ),
    )
    found = check(raw)
    assert found.result.status == Status.MATCH
    m = dict(found.result.measure)
    assert m["kind"] == "model_call_failed" and m["http_status"] == 413
    assert m["media_items"] == 1 and m["media_bytes"] == 304  # decoded, never the payload
    # Located at the final agent step, as metadata (not the error text).
    (at,) = found.result.evidence
    assert at.step == 3 and at.channel.value == "metadata"


def test_reasoning_only_output_limit_is_unexpected():
    """Regression: 131,072 reasoning tokens, no reply, no call: the run ended there."""
    m = measure(trajectory(agent(reasoning="Thinking about routing…", stop="MAX_TOKENS")))
    assert m["kind"] == "output_limit" and m["reasoning_only"] == 1 and m["usage_recorded"] == 1


def test_empty_end_turn_after_cut_off_reasoning_is_unexpected():
    """Regression: the stream stopped mid-sentence, no usage chunk, recorded as end_turn."""
    m = measure(
        trajectory(
            agent(reasoning="Let me try the near-identity pai", stop="END_TURN", usage=False)
        )
    )
    assert m["kind"] == "empty_reply" and m["reasoning_only"] == 1 and m["usage_recorded"] == 0


@pytest.mark.parametrize(
    ("step", "termination", "kind"),
    [
        (agent(stop="SAFETY"), None, "safety_stop"),
        (agent(stop="CANCELLED"), None, "interrupted"),
        (agent(stop="TOOL_USE", calls=["c1"]), None, "stopped_mid_loop"),
        (answered(agent(stop="TOOL_USE", calls=["c1"]), "c1"), None, "stopped_mid_loop"),
        (agent("done", stop="END_TURN", calls=["c1"]), None, "stopped_mid_loop"),
        (agent(stop="TOOL_USE", calls=["c1"]), "RemoteProtocolError", "model_call_failed"),
    ],
)
def test_other_unexpected_endings(step, termination, kind):
    assert measure(trajectory(step, termination=termination))["kind"] == kind


def test_without_a_stop_reason_a_reply_is_unknown_never_cleared():
    found = check(trajectory(agent(SECRET_REPLY)))
    assert found.result.status == Status.UNKNOWN
    assert [u.reason for u in found.result.unread] == ["stop_reason_not_recorded"]


@pytest.mark.parametrize(
    ("step", "status", "kind"),
    [
        # Calls with no result: the run stopped before they returned.
        (agent(calls=["c1"]), Status.MATCH, "stopped_mid_loop"),
        # Calls answered (a harness's final submit tool, say): can't tell.
        (answered(agent(calls=["c1"]), "c1"), Status.UNKNOWN, "undetermined"),
        (agent(), Status.MATCH, "empty_reply"),
    ],
)
def test_without_a_stop_reason_only_the_final_steps_shape_decides(step, status, kind):
    found = check(trajectory(step))
    assert found.result.status == status
    assert dict(found.result.measure)["kind"] == kind
    assert dict(found.result.measure)["basis"] == "structure"


def test_copied_context_or_no_agent_step_is_not_applicable():
    copied = {**agent("from another session", stop="END_TURN"), "is_copied_context": True}
    assert check(trajectory(copied)).result.status == Status.NOT_APPLICABLE
    assert check(trajectory()).result.status == Status.NOT_APPLICABLE
    assert ending(parse_trace(trajectory())) is None


def test_a_partial_recording_keeps_the_ending_figures():
    raw = trajectory(
        user("This session is being continued from a previous conversation"),
        agent(stop="MAX_TOKENS"),
    )
    found = check(raw)
    assert found.result.status == Status.MATCH and not found.result.complete
    assert dict(found.result.measure)["kind"] == "output_limit"


def test_report_carries_codes_only_and_no_reply_or_error_text():
    raw = trajectory(agent(SECRET_REPLY, stop="ERROR", usage=False, retry=retry_413()))
    trace = parse_trace(raw)
    doc = report(Engine(builtin_detectors()).evaluate(trace), trace.step_numbers)
    text = json.dumps(doc)
    assert SECRET_REPLY not in text and "entity too large" not in text
    row = next(a for a in doc["assessments"] if a["id"] == "ending.unexpected")
    assert row["kind"] == "context" and row["score"] is None
    assert row["measure"]["kind"] == "model_call_failed"


def item(raw, label, reward=0.0, error_type=None):
    trace = parse_trace(raw)
    doc = report(Engine(builtin_detectors()).evaluate(trace), trace.step_numbers)
    return {
        **doc,
        **trace_facts(trace),
        "input_id": label,
        "input_status": "available",
        "task": "demo",
        "reward": reward,
        "error_type": error_type,
    }


def summary(*items):
    return brief({"scanner_version": "dev", "inputs": list(items), "coverage": {}}, dq="low")


def evidence(b):
    lines = brief_text(b).splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("EVIDENCE"))
    end = next((i for i in range(start + 1, len(lines)) if not lines[i]), len(lines))
    return " ".join(" ".join(lines[start:end]).split())


def test_brief_shows_unexpected_endings_as_their_own_evidence_line():
    b = summary(
        item(trajectory(agent("done", stop="END_TURN")), "ok", reward=1.0),
        item(
            trajectory(
                user([{"type": "image", "source": {"media_type": "image/png", "path": PNG}}]),
                agent("I hit an internal error", stop="ERROR", usage=False, retry=retry_413()),
            ),
            "crashed",
            reward=1.0,
            error_type="NonZeroAgentExitCodeError",
        ),
        item(trajectory(agent(reasoning="hmm", stop="MAX_TOKENS")), "limit"),
        item(trajectory(agent(reasoning="cut", stop="END_TURN", usage=False)), "empty"),
    )
    e = b["endings"]
    assert e["trials"] == 3 and e["rewarded"] == 1 and e["checked"] == 4
    assert e["kinds"] == {"model_call_failed": 1, "output_limit": 1, "empty_reply": 1}
    text = evidence(b)
    assert "3 trials (75.0%) didn't end normally" in text
    assert "model call failed 1 (HTTP 413 request too large 1)" in text
    assert "output limit hit 1 (1 reasoning only)" in text
    assert "empty final reply 1 (1 reasoning only, 1 with no usage recorded)" in text
    assert "1 of them was rewarded; Harbor recorded an error for 1" in text
    assert "1 failed call carried inline media in the conversation: up to 1 item and 304 B" in (
        text
    )
    # Not a recording defect, and not a behaviour finding or DQ candidate.
    assert "no recording defect detected" in text
    assert b["findings"]["total"] == 0
    assert finding_index([item(trajectory(agent(stop="MAX_TOKENS")), "x")])["flagged"] == 0


def test_brief_says_every_trial_ended_normally_and_names_unknown_endings():
    b = summary(
        item(trajectory(agent("done", stop="END_TURN")), "ok"),
        item(trajectory(agent("done")), "no-stop-reason"),
    )
    text = evidence(b)
    assert "✓ every judged trial ended with the agent's reply" in text
    assert "· Harness does not record model stop reasons for 1 trial:" in text
    assert "didn't end normally" not in text


def test_a_harness_without_stop_reasons_is_named_once_not_counted_as_unknown():
    b = summary(*(item(trajectory(agent("done")), f"t{i}") for i in range(3)))
    text = evidence(b)
    assert "· Harness does not record model stop reasons: only endings the final step shows" in (
        text
    )
    assert "isn't known" not in text and "✓ every" not in text


def test_calls_left_without_results_by_a_stopped_run_are_told_once():
    """A run killed while its calls ran: the ending and the missing results are one event
    (`integrity.final_calls_unanswered`), so EVIDENCE says it on the endings line only,
    and doesn't call the run free of recording defects either."""
    b = summary(item(trajectory(agent(stop="TOOL_USE", calls=["c1"])), "cut", error_type="T"))
    assert b["recording"].get("integrity.final_calls_unanswered") == 1
    text = evidence(b)
    assert "stopped with tool calls outstanding 1 (1 with no results recorded for them)" in text
    assert "no recorded result" not in text
    assert "no recording defect detected" not in text


def test_inline_media_size_is_counted_for_every_recorded_shape():
    block = {"type": "image", "source": {"media_type": "image/png", "path": PNG}}
    assert content([block, {"type": "text", "text": "see"}]).media_bytes == 304
    # A data URI inside a string (a Codex view_image repr).
    assert content(f"[{{'type': 'input_image', 'image_url': '{PNG}'}}]").media_bytes == 304
    # A placeholder carries no payload to count.
    assert content("[Image 1]").media_bytes == 0


def test_a_terminal_batch_with_one_shared_result_is_not_left_unanswered():
    """Regression (Terminus 2): the last turn sends keystrokes plus `mark_task_complete`
    and records one terminal capture for the batch, linked to no call. With no stop
    reason that is undecided, never "stopped with tool calls outstanding"."""
    step = agent("Analysis: all requirements verified.", calls=["k1", "done"])
    step["observation"] = {"results": [{"content": "$ "}]}
    found = check(trajectory(step))
    assert found.result.status == Status.UNKNOWN
    assert dict(found.result.measure)["kind"] == "undetermined"


def provider_error(status=413, error_type="APIStatusError", schema=None):
    return {
        "schema": schema or "fast-agent.provider-error/v1",
        "error_type": error_type,
        "status": status,
    }


def failed(error=None, retry=None):
    step = agent(
        "I hit an internal error while calling the model", stop="ERROR", usage=False, retry=retry
    )
    if error is not None:
        step["extra"]["provider_error"] = error
    return step


def test_a_413_that_was_not_retried_still_reports_its_status():
    """fast-agent stopped retrying 413s (resending the body can't help), so no retry
    record holds the status: it is in `extra.provider_error` (ATIF has no error
    field), read as a code only."""
    m = measure(trajectory(failed(provider_error())))
    assert m["kind"] == "model_call_failed" and m["http_status"] == 413


def test_the_failed_calls_own_status_wins_over_earlier_retried_attempts():
    retried_429 = {
        "schema": "fast-agent.retry/v1",
        "provider_attempts": 2,
        "retries": [{"attempt": 1, "error_message": "Error code: 429 - slow down"}],
    }
    assert measure(trajectory(failed(provider_error(500), retried_429)))["http_status"] == 500
    # Older traces: the status only in the retried attempts.
    assert measure(trajectory(failed(retry=retry_413())))["http_status"] == 413


@pytest.mark.parametrize(
    "error",
    [
        provider_error(schema="someone-else/v1"),
        provider_error(status=999),
        provider_error(status="413"),
        provider_error(status=None, error_type="EmptyCompletionError"),
    ],
)
def test_an_unrecognised_or_absent_status_is_left_unknown(error):
    m = measure(trajectory(failed(error)))
    assert m["kind"] == "model_call_failed" and m["http_status"] is None


def test_empty_replies_asked_again_are_counted_in_the_brief():
    """The harness asks once more after an empty end_turn (`reason: empty_completion`):
    recovered ones no longer end the run, so the brief says how many there were."""
    retry = {
        "schema": "fast-agent.retry/v1",
        "provider_attempts": 2,
        "retries": [
            {"attempt": 1, "reason": "empty_completion", "error_type": "EmptyCompletionError"}
        ],
    }
    recovered = item(trajectory(agent("done", stop="END_TURN", retry=retry)), "recovered")
    assert recovered["empty_completion_retries"] == 1
    b = summary(recovered, item(trajectory(agent("done", stop="END_TURN")), "ok"))
    assert b["endings"]["empty_retries"] == 1 and b["endings"]["empty_retry_trials"] == 1
    text = evidence(b)
    assert "the harness asked again after 1 empty reply" in text
    assert "in 1 trial; endings above are after that" in text
    assert "✓ every trial ended with the agent's reply" in text
