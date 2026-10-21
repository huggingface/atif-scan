"""Code-mode programs keeping state between calls (`store`/`load`): counted, never read.
Synthetic programs only."""

from __future__ import annotations

import pytest

from atif_scan import parse_trace
from atif_scan.data import jslit
from atif_scan.data.jslit import state_helper_calls
from atif_scan.data.program_state import stateful_programs


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        ('store("rows", r.output);', {"store"}),
        ('const d = load("rows"); text(d.length);', {"load"}),
        ('store ("a", 1); const b = load("a");', {"store", "load"}),
        # Not the harness's: a string, a comment, a method, a property or its own function.
        ('text("store(x) later");', set()),
        ("// load(prev) next time\ntext(1);", set()),
        ("const x = cache.load(k); JSON.store(1);", set()),
        ("const reload = (k) => k; reload(1); preload(2);", set()),
        ("function load(k) { return k; }\nconst v = load(1);", set()),
        ("const store = new Map(); store(1);", set()),
        # An unterminated string runs to the line end, as in JavaScript: a syntax error,
        # so nothing ran.
        ('const s = "unterminated; load("k");', set()),
    ],
)
def test_state_helper_calls(program, expected):
    assert state_helper_calls(program) == frozenset(expected)


def test_an_unlexable_program_is_searched_as_text(monkeypatch):
    def fail(_):
        raise ValueError

    monkeypatch.setattr(jslit, "_lex", fail)
    assert state_helper_calls('text("x"); load("k");') == {"load"}


def step(*programs):
    calls = [
        {"tool_call_id": f"p{i}", "function_name": "exec", "arguments": {"input": p}}
        for i, p in enumerate(programs)
    ]
    return {"source": "agent", "message": "", "tool_calls": calls}


def trace(*steps):
    return parse_trace({"schema_version": "ATIF-v1.7", "steps": list(steps)})


def test_stateful_programs_are_counted_in_code_mode_traces():
    # Regression (TB4 production-planning): 17 table dumps were stored, then a later
    # program with no tool call loaded and printed parts of them.
    stored = 'const r = await tools.exec_command({cmd: "ls"}); store("rows", r.output);'
    assert stateful_programs(trace(step(stored), step('text(load("rows"));'))) == 2
    assert stateful_programs(trace(step('await tools.exec_command({cmd: "ls"});'))) == 0


def test_not_code_mode_is_zero_and_no_trace_is_unknown():
    # A plain `exec` shell tool whose command happens to say load( is not a program.
    plain = {
        "source": "agent",
        "message": "",
        "tool_calls": [
            {"tool_call_id": "s", "function_name": "exec", "arguments": {"input": "load(1)"}}
        ],
    }
    assert stateful_programs(trace(plain)) == 0
    assert stateful_programs(None) is None
