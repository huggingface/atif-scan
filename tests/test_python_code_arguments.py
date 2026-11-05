"""Python code-envelope and private-inspection regressions; synthetic programs only."""

from __future__ import annotations

import json

import pytest
from test_ptc_ledger import dispatch, recorded
from test_python_programs import program_trace, results

from atif_scan import Channel, Context, Engine, Status, builtin_detectors, parse_trace
from atif_scan.data.tools import program_calls, python_program_source
from atif_scan.evidence.extract import read_segment
from atif_scan.output.document import report


@pytest.mark.parametrize("name", ["execute_python", "exec_sandbox"])
@pytest.mark.parametrize("key", ["input", "code"])
def test_known_python_envelopes_preserve_commands_and_full_source(name, key):
    program = 'await shell(command="cat /tests/synthetic.py")\nprint("synthetic-marker")'
    raw = program_trace(program)
    raw["steps"][0]["tool_calls"][0].update(function_name=name, arguments={key: program})
    trace = parse_trace(raw)
    parent, child = trace.steps[0].calls
    assert child.name == f"{name}>shell"
    assert any(ch == Channel.PAYLOAD and c.text == program for ch, c in parent.fields)
    assert results(raw)["access.test_path"].status == Status.MATCH


@pytest.mark.parametrize(
    "args",
    [
        {"code": None},
        {"code": 7},
        {"code": []},
        {"code": "print(1)", "input": "print(2)"},
        {"code": "print(1)", "unexpected": True},
    ],
)
def test_ambiguous_or_invalid_code_envelopes_are_not_python_sources(args):
    assert python_program_source(args) is None
    assert program_calls("exec_sandbox", args) is None


def test_arbitrary_code_payloads_are_not_program_tools():
    assert program_calls("write_text_file", {"code": 'await shell(command="ls")'}) is None


def test_full_program_is_masked_before_private_paging_and_absent_from_reports():
    secret = "syntheticCredential987654"
    program = f'API_TOKEN = "{secret}"\nprint(API_TOKEN)\n'
    raw = recorded(program, [dispatch(arguments={"command": "echo synthetic"})])
    raw["steps"][0]["step_id"] = 7
    raw["steps"][0]["tool_calls"][0].update(
        function_name="exec_sandbox", arguments={"code": program}
    )
    trace = parse_trace(raw)
    parent = trace.steps[0].calls[0]
    index = next(i for i, (ch, _) in enumerate(parent.fields) if ch == Channel.PAYLOAD)
    segment = read_segment(trace, 7, "call", index=0, field=index, limit=6000)
    private = json.dumps(segment)
    assert "print(API_TOKEN)" in private
    assert secret not in private
    engine = Engine(builtin_detectors())
    public = json.dumps(report(engine.evaluate(trace, Context())))
    assert secret not in public
    assert "print(API_TOKEN)" not in public
    # A dispatch ledger does not prove that direct Python I/O was absent.
    assert results(raw)["access.test_path"].status == Status.UNKNOWN


def test_source_payload_does_not_turn_quoted_file_contents_into_executed_reads():
    program = 'await write_text_file(path="/app/example.py", content="cat /tests/synthetic.py")'
    raw = program_trace(program)
    raw["steps"][0]["tool_calls"][0]["arguments"] = {"code": program}
    assert results(raw)["access.test_path"].status == Status.NO_MATCH
