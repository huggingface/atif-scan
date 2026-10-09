"""Terminal-screen results: one capture for a batch of typed input (Terminus 2 shape).

Synthetic traces only. Terminus 2 records one unlinked terminal capture per step for all
of the step's `bash_command` keystrokes; its own markers say when output is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from atif_scan import Context, Status, builtin_detectors, parse_trace
from atif_scan.data.terminal import terminal_gap

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc

PROMPT: Doc = {"source": "user", "message": "Fix /app."}
SCREEN = "New Terminal Output:\nroot@box:/app# ls\nmain.py\nroot@box:/app# cat main.py\nprint(1)\n"
OMITTED = "[... output limited to 10000 bytes; 72 interior bytes omitted ...]"
KEY = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0"


def keys(text: str, cid: str) -> Doc:
    return {
        "tool_call_id": cid,
        "function_name": "bash_command",
        "arguments": {"keystrokes": text, "duration": 0.5},
    }


def batch(screen: str, *calls: Doc, linked: str | None = None) -> Doc:
    result: Doc = {"content": screen}
    if linked:
        result["source_call_id"] = linked
    return {
        "source": "agent",
        "message": "",
        "tool_calls": list(calls) or [keys("ls\n", "call_0_1"), keys("cat main.py\n", "call_0_2")],
        "observation": {"results": [result]},
    }


def check(check_id: str, *steps: Doc):
    detector = next(d for d in builtin_detectors() if d.spec.id == check_id)
    trace = parse_trace({"schema_version": "ATIF-v1.2", "steps": [PROMPT, *steps]})
    return detector.evaluate(trace, Context())


def reasons(result) -> list[tuple[str, int | None, int | None]]:
    return [(u.reason, u.at.step, u.at.observation) for u in result.unread if u.at]


def test_a_batch_capture_records_every_typed_call_but_not_which_printed_what():
    step = parse_trace({"schema_version": "ATIF-v1.2", "steps": [PROMPT, batch(SCREEN)]}).steps[1]
    (screen,) = step.observations
    assert screen.shared_terminal and screen.pairing_unresolved
    assert all(step.results_for(c) == [] for c in step.calls)
    for check_id in ("observation.credentials_exposed", "observation.account_ids_exposed"):
        result = check(check_id, batch(SCREEN))
        assert result.status == Status.NO_MATCH and result.complete


def test_a_credential_on_a_shared_screen_still_matches():
    assert check("observation.credentials_exposed", batch(SCREEN + KEY)).status == Status.MATCH


def test_task_completion_beside_typed_input_shares_the_capture():
    calls = (
        keys("ls\n", "call_3_1"),
        {
            "tool_call_id": "call_3_task_complete",
            "function_name": "mark_task_complete",
            "arguments": {},
        },
    )
    result = check("observation.credentials_exposed", batch(SCREEN, *calls))
    assert result.status == Status.NO_MATCH


def test_cut_output_is_unread_once_per_capture():
    screen = SCREEN + OMITTED + "\nroot@box:/app# "
    result = check("observation.credentials_exposed", batch(screen))
    assert result.status == Status.UNKNOWN
    assert reasons(result) == [("result_truncated", 1, 0)]


def test_visible_screen_only_is_unread():
    screen = "Current Terminal Screen:\nroot@box:/app# vim notes\n~\n~\n"
    result = check("observation.account_claims", batch(screen))
    assert result.status == Status.UNKNOWN
    assert reasons(result) == [("terminal_screen_only", 1, 0)]


def test_a_linked_single_command_capture_with_cut_output_is_unread():
    step = batch(
        SCREEN + OMITTED, keys("apt-get install -y r-base\n", "call_1_1"), linked="call_1_1"
    )
    result = check("observation.credentials_exposed", step)
    assert reasons(result) == [("result_truncated", 1, 0)]


def test_calls_that_do_not_type_into_a_terminal_share_nothing():
    # One unlinked result for a command and a file read: whose output it is is unknown.
    calls = (
        {"tool_call_id": "a", "function_name": "bash", "arguments": {"command": "ls"}},
        {"tool_call_id": "b", "function_name": "read_file", "arguments": {"path": "/app/x"}},
    )
    done: Doc = {"source": "agent", "message": "Done."}
    result = check("observation.credentials_exposed", batch("main.py\n", *calls), done)
    assert result.status == Status.UNKNOWN
    assert {u.reason for u in result.unread} == {"result_not_recorded"}


def test_markers_are_the_harness_lines_not_quoted_text():
    assert terminal_gap(SCREEN + f"echo '{OMITTED}'\n") is None
    assert terminal_gap(SCREEN + "say Current Terminal Screen: here\n") is None
    # A fresh capture after an empty diff reads like a fallback: still unread.
    assert terminal_gap("Current Terminal Screen:\nroot@box:/app# \n") == "terminal_screen_only"
    assert terminal_gap(f"Current terminal state:\n{SCREEN}\nAre you sure?") is None


def test_quoted_markers_in_a_non_terminal_result_are_ignored():
    step: Doc = {
        "source": "agent",
        "message": "",
        "tool_calls": [
            {"tool_call_id": "c", "function_name": "bash", "arguments": {"command": "cat log"}}
        ],
        "observation": {
            "results": [{"source_call_id": "c", "content": "Current Terminal Screen:\n" + OMITTED}]
        },
    }
    assert check("observation.credentials_exposed", step).status == Status.NO_MATCH
