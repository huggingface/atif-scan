"""PTC counts and limits from synthetic evidence only; no program is executed."""

from __future__ import annotations

import io
import json
import re
from dataclasses import replace
from unittest.mock import patch

import pytest
from test_ptc_ledger import dispatch, recorded
from test_python_programs import program_trace

from atif_scan import Context, Engine, Severity, builtin_detectors, document, parse_trace, to_text
from atif_scan.cli.scan import scanned_item
from atif_scan.data.ptc_usage import PtcUsage, ptc_usage
from atif_scan.output.coverage import PTC_LIMITS, ptc_notice
from atif_scan.output.document import filter_findings, report
from atif_scan.output.summary import summary, summary_text
from atif_scan.output.views import render_rich, render_summary_rich
from atif_scan.sources.inputs import Source

PRIVATE = "synthetic-private-marker"


def item_for(trace, *, evaluate=False):
    context = Context(task="synthetic")
    rows = Engine(builtin_detectors()).evaluate(trace, context) if evaluate else ()
    item = scanned_item(Source("trial", lambda: trace), trace, None, context, rows)
    return item, rows


def runtime_trace(n=1):
    raw = recorded(
        f'await shell(command="{PRIVATE}")',
        [dispatch(arguments={"command": f"echo {PRIVATE}"})],
    )
    # Missing program input leaves the parent classified as other; a valid ledger
    # still recovers its child. Do not change classifications just for presentation.
    raw["steps"][0]["tool_calls"][0]["arguments"] = None
    raw["steps"] *= n
    return parse_trace(raw)


def test_forty_wrappers_and_forty_recorded_children_keep_accounting():
    trace = runtime_trace(40)
    assert ptc_usage(trace) == PtcUsage(40, 40, 0, 40, 0)
    item, rows = item_for(trace, evaluate=True)
    assert item["tool_calls"] == 80
    assert item["unrecognized_tool_calls"] == 40
    assert item["ptc_usage"] == {
        "wrappers": 40,
        "runtime_recorded": 40,
        "static_inferred": 0,
        "unrecognized_wrappers": 40,
        "remaining_unrecognized": 0,
    }
    base = report(rows, trace.step_numbers)
    for key in ("assessments", "coverage_gaps", "coverage_causes", "score", "incomplete"):
        assert item[key] == base[key]
    assert item["coverage_gaps"]["behavioural"]
    exported = json.dumps(document([item], "test"))
    assert PRIVATE not in exported
    assert "execute_python" not in exported
    assert "c1::ptc1" not in exported


@pytest.mark.parametrize("program", ["await shell(command=dynamic)", "invalid python !!!"])
def test_static_and_fallback_wrappers_are_recorded_calls_not_executions(program):
    trace = parse_trace(program_trace(program))
    usage = ptc_usage(trace)
    assert usage is not None
    assert usage.wrappers == 1
    assert usage.runtime_recorded == 0
    assert usage.static_inferred == 1
    item, _ = item_for(trace)
    text = to_text(document([item], "test"))
    assert "PTC: 1 wrappers · 0 runtime-recorded children" in text
    assert "not proof execution completed" in text
    assert "1 static-inferred children (not execution evidence)" in text


@pytest.mark.parametrize("status", ["raised", "in_flight", "cancelled"])
def test_partial_ledgers_keep_inferred_and_runtime_counts_separate(status):
    trace = parse_trace(recorded(entries=[dispatch(status=status)]))
    usage = ptc_usage(trace)
    assert usage is not None
    assert (usage.wrappers, usage.runtime_recorded, usage.static_inferred) == (1, 1, 1)
    assert trace.tool_calls == 3


def test_other_fallback_and_unknown_children_are_not_conflated():
    raw = recorded(entries=[dispatch(tool=PRIVATE)])
    raw["steps"][0]["tool_calls"][0]["arguments"] = None
    raw["steps"][0]["tool_calls"].append(
        {"tool_call_id": "unknown", "function_name": PRIVATE, "arguments": None}
    )
    trace = parse_trace(raw)
    assert ptc_usage(trace) == PtcUsage(1, 1, 0, 1, 2)
    item, _ = item_for(trace)
    assert item["unrecognized_tool_calls"] == 3
    assert "unrecognized tool calls 3 (1 PTC wrappers; 2 remaining)" in to_text(
        document([item], "test")
    )
    assert PRIVATE not in json.dumps(item)


@pytest.mark.parametrize("entries", [[], [dispatch()]])
def test_recognized_wrappers_are_usage_not_unrecognized_calls(entries):
    trace = parse_trace(recorded(entries=entries))
    usage = ptc_usage(trace)
    assert usage == PtcUsage(1, len(entries), 0, 0, 0)
    item, _ = item_for(trace)
    text = to_text(document([item], "test"))
    assert f"PTC: 1 wrappers · {len(entries)} runtime-recorded children" in text


def test_missing_program_without_ledger_still_counts_wrapper():
    raw = program_trace("")
    raw["steps"][0]["tool_calls"][0]["arguments"] = None
    trace = parse_trace(raw)
    assert ptc_usage(trace) == PtcUsage(1, 0, 0, 1, 0)


def test_counts_follow_authorship_parent_binding_and_provenance_not_name_fragments():
    trace = runtime_trace()
    step = trace.steps[0]
    parent, child = step.calls
    # A child named like a wrapper remains a child; an orphan is not PTC evidence.
    altered = replace(
        step,
        calls=(
            replace(parent, tool="inert"),
            replace(child, name="execute_python"),
            replace(child, index=2, result_id="unrelated"),
        ),
    )
    assert ptc_usage(replace(trace, steps=(altered,))) == PtcUsage(1, 1, 0, 0, 0)
    # Reusing a call ID in a different step does not bind that step's children.
    orphan = replace(step, calls=(child,))
    assert ptc_usage(replace(trace, steps=(step, orphan))) == PtcUsage(1, 1, 0, 1, 0)
    renamed = replace(step, calls=(replace(parent, name="exec"), child))
    assert ptc_usage(replace(trace, steps=(renamed,))) == PtcUsage(0, 0, 0, 0, 1)
    assert ptc_usage(replace(trace, steps=(replace(step, copied=True),))) == PtcUsage(0, 0, 0, 0, 0)
    for source in ("user", "system"):
        assert ptc_usage(replace(trace, steps=(replace(step, source=source),))) == PtcUsage(
            0, 0, 0, 0, 0
        )


def test_no_trace_is_null_and_no_ptc_is_zero():
    assert ptc_usage(None) is None
    missing = scanned_item(
        Source("missing", lambda: parse_trace({"steps": []})),
        None,
        "unreadable_trace",
        Context(task="synthetic"),
        (),
    )
    assert missing["ptc_usage"] is None
    trace = parse_trace({"steps": [{"source": "agent", "message": "hello"}]})
    assert ptc_usage(trace) == PtcUsage(0, 0, 0, 0, 0)
    item, _ = item_for(trace)
    assert "PTC" not in to_text(document([item, missing], "test"))


def rendered(doc, *, rich, rollup):
    if rollup:
        doc = summary(doc)
    if not rich:
        return summary_text(doc) if rollup else to_text(doc)
    buffer = io.StringIO()
    if rollup:
        # Summary's rich renderer is deliberately terminal-only.
        with patch.object(buffer, "isatty", return_value=True):
            assert render_summary_rich(doc, file=buffer)
    else:
        render_rich(doc, file=buffer)
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", buffer.getvalue()).split())


@pytest.mark.parametrize("rich", [False, True])
@pytest.mark.parametrize("rollup", [False, True])
def test_usage_is_near_headings_and_limits_once_even_when_findings_filtered(rich, rollup):
    item, _ = item_for(runtime_trace(40), evaluate=True)
    doc = document([item, dict(item, input_id="trial-2")], "test")
    shown = filter_findings(doc, Severity.INFO, checks=("awareness.*",), hide_empty=True)
    assert shown["coverage"] == doc["coverage"]
    assert shown["inputs"][0]["ptc_usage"] == item["ptc_usage"]
    text = rendered(shown, rich=rich, rollup=rollup)
    assert text.count(PTC_LIMITS) == 1
    assert PRIVATE not in text
    if rollup:
        assert text.count("PTC:") == 1
        assert "PTC: 80 wrappers · 80 runtime-recorded children" in text
        assert "2 input(s) using PTC" in text
        assert "Unrecognized calls: 80 PTC wrappers; 0 remaining" in text
        assert "tool calls 80" not in text
    else:
        assert text.count("PTC: 40 wrappers · 40 runtime-recorded children") == 2
        assert "tool calls 80" in text
        assert "unrecognized tool calls 40 (40 PTC wrappers; 0 remaining)" in text
        assert text.index("PTC: 40") < text.index("Behavioural coverage incomplete")


def test_summary_counts_only_known_usage_and_keeps_coverage():
    item, _ = item_for(runtime_trace())
    legacy = dict(item, input_id="legacy")
    legacy.pop("ptc_usage")
    missing = dict(item, input_id="missing", ptc_usage=None)
    zero, _ = item_for(parse_trace({"steps": []}))
    doc = document([item, legacy, missing, zero], "test")
    rolled = summary(doc)
    assert rolled["ptc_usage"] == {
        **item["ptc_usage"],
        "inputs_using_ptc": 1,
        "inputs_counted": 2,
        "inputs_without_counts": 2,
    }
    assert rolled["coverage"] == doc["coverage"]
    assert "2 without counts" in summary_text(rolled)


@pytest.mark.parametrize("rich", [False, True])
@pytest.mark.parametrize("rollup", [False, True])
def test_legacy_unknown_counts_never_infer_ptc(rich, rollup):
    item, _ = item_for(runtime_trace())
    item.pop("ptc_usage")
    item["unrecognized_tool_calls"] = 40
    assert ptc_notice([item]) == ""
    doc = document([item], "test")
    assert "PTC" not in rendered(doc, rich=rich, rollup=rollup)
