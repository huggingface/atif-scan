"""Synthetic code-review regressions; program text is never executed."""

from __future__ import annotations

import pytest
from test_ptc_ledger import dispatch, recorded
from test_python_programs import calls, program_trace, results

from atif_scan import Channel, Status, parse_trace
from atif_scan.data.jslit import UNREAD
from atif_scan.data.pylit import MAX_VALUE_CHARS
from atif_scan.data.web_gaps import web_gaps


@pytest.mark.parametrize(
    "prefix",
    [
        "def helper(shell):\n    pass\n",
        "def helper():\n    shell = None\n",
        "unused = lambda shell: shell\n",
        "[shell for shell in []]\n",
        "class Helper:\n    shell = None\n",
    ],
)
def test_unrelated_binding_never_hides_an_unresolved_tool_call(prefix):
    program = prefix + "await shell(command=runtime_command)"
    assert ("?", UNREAD) in calls(program)
    assert results(program_trace(program))["access.test_path"].status == Status.UNKNOWN


def test_shadowed_reference_not_silently_discarded():
    program = "def helper(shell):\n    pass\nf = shell\nawait f(command=runtime_command)"
    assert ("?", UNREAD) in calls(program)


def test_exponential_literal_aliases_stop_at_finite_work_budget():
    program = 'x0 = "a"\n'
    program += "\n".join(f"x{i} = [x{i - 1}, x{i - 1}]" for i in range(1, 25))
    program += "\nawait shell(command=x24)"
    assert calls(program) == [("?", UNREAD)]


def test_join_cannot_materialize_unbounded_separator_copies():
    separator = "x" * (MAX_VALUE_CHARS // 2)
    program = f'await shell(command={separator!r}.join(["a", "b", "c", "d"]))'
    assert calls(program) == [("shell", {"command": UNREAD})]


def test_oversized_integer_fstring_is_unknown_not_an_exception():
    program = 'await shell(command=f"{' + "0x" + "f" * 5000 + '}")'
    assert calls(program) == [("shell", {"command": UNREAD})]


def test_partial_ledger_static_web_sibling_cannot_borrow_parent_output():
    raw = recorded(
        'await web_search(query="synthetic")',
        [dispatch(status="in_flight", arguments={"command": "echo synthetic"})],
    )
    trace = parse_trace(raw)
    step = trace.steps[0]
    inferred = next(c for c in step.calls if c.provenance == "static_inferred")
    assert not step.results_for(inferred)
    gaps = web_gaps(trace)
    assert gaps.calls == 1
    assert gaps.missing_result_ids == 1


@pytest.mark.parametrize(
    "message",
    [
        "Syntax error; the code was NOT executed.",
        "Type check failed; the code was NOT executed.",
    ],
)
def test_empty_ledger_does_not_reintroduce_rejected_unknown_arguments(message):
    raw = recorded('await shell(command="ls"', [])
    raw["steps"][0]["observation"]["results"][0]["content"] = message
    trace = parse_trace(raw)
    assert all(content.understood for _, c in trace.agent_calls() for _, content in c.fields)
    assert results(raw)["access.test_path"].status == Status.NO_MATCH


def test_recorded_dispatch_is_not_cleared_by_contradictory_parent_refusal_text():
    raw = recorded(
        "await shell(command=runtime_command)",
        [dispatch(arguments=None, arguments_error="not_json_serializable")],
    )
    raw["steps"][0]["observation"]["results"][0]["content"] = (
        "Type check failed; the code was NOT executed."
    )
    trace = parse_trace(raw)
    child = next(c for _, c in trace.agent_calls() if c.provenance == "runtime_recorded")
    assert any(ch == Channel.COMMAND and not value.understood for ch, value in child.fields)
    assert results(raw)["access.test_path"].status == Status.UNKNOWN
