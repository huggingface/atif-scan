"""Synthetic references: recorded targets are not automatically source provenance."""

import json

import pytest

from atif_scan import Context, builtin_detectors, parse_trace
from atif_scan.brief import brief, brief_text
from atif_scan.detectors.integrity import output_ratio
from atif_scan.engine import Engine
from atif_scan.report import document, report
from atif_scan.web_inputs import web_input

URL = "https://example.org/synthetic-private-page"
REF = "turn0search12"


def step(args, body="Synthetic ordinary body.", derived=False):
    name = "web__run"
    if derived:
        name, args = "exec", {"input": f"const r = await tools.web__run({json.dumps(args)});"}
    return {
        "source": "agent",
        "tool_calls": [{"tool_call_id": "w", "function_name": name, "arguments": args}],
        "observation": {"results": [{"source_call_id": "w", "content": body}]},
    }


def scan(*steps):
    trace = parse_trace({"steps": [{"source": "user", "message": "Synthetic task."}, *steps]})
    assessments = Engine(builtin_detectors()).evaluate(trace, Context("synthetic"))
    return trace, assessments, {a.spec.id: a.result.status.value for a in assessments}


@pytest.mark.parametrize("derived", [False, True])
@pytest.mark.parametrize("action", ["open", "click", "find"])
@pytest.mark.parametrize("target", [URL, REF, "turn1view0"])
def test_static_targets_and_independent_outcomes(derived, action, target):
    trace, _, states = scan(step({action: [{"ref_id": target}]}, derived=derived))
    call = trace.steps[-1].calls[-1]
    assert web_input(call).recorded
    assert web_input(call).source_known == (target == URL)
    if derived:
        assert call.arguments is None
    assert states["integrity.web_results_not_recorded"] == "no_match"
    assert states["integrity.web_input_unresolved"] == ("no_match" if target == URL else "match")
    assert states["lookup.search_surfaced_benchmark"] == (
        "no_match" if target == URL else "unknown"
    )


@pytest.mark.parametrize(
    "body,missing",
    [
        ("Error: retrieval failed", False),
        ("Script completed\nWall time 1 seconds\nOutput:\nError: retrieval failed", False),
        ("Script completed\nWall time 1 seconds\nOutput:\n", True),
        ("", True),
    ],
)
def test_errors_are_outcomes_not_content(body, missing):
    _, _, states = scan(step({"open": [{"ref_id": REF}]}, body, derived=True))
    assert states["integrity.web_results_not_recorded"] == ("match" if missing else "no_match")
    assert states["integrity.web_input_unresolved"] == "match"
    assert states["lookup.search_surfaced_benchmark"] == "unknown"


@pytest.mark.parametrize(
    "targets,recorded",
    [
        ([URL, REF], True),
        ([REF, "turn99view99"], True),
        ([URL, None], False),
        ([URL, "dynamic.value"], False),
    ],
)
def test_mixed_targets_do_not_hide_unknown(targets, recorded):
    trace, _, states = scan(step({"open": [{"ref_id": t} for t in targets]}))
    assert web_input(trace.steps[-1].calls[0]).recorded == recorded
    assert states["integrity.web_input_unresolved"] == "match"


@pytest.mark.parametrize(
    "prior",
    [
        [],
        [f"Synthetic ({URL})\n【{REF}】 [wordlim: 200] Synthetic body"],
        [f"【{REF}】 {URL}", f"【{REF}】 https://example.net/conflicting"],
    ],
)
@pytest.mark.parametrize("forward", [False, True])
def test_reference_resolution_is_explicitly_deferred(prior, forward):
    # Even plausible earlier provider metadata is not yet a validated index.
    # Nonexistent, conflicting and forward references cannot clear provenance.
    providers = [step({"search_query": [{"q": "synthetic"}]}, text) for text in prior]
    target = step({"open": [{"ref_id": REF}]})
    ordered = [target, *providers] if forward else [*providers, target]
    trace, _, states = scan(*ordered)
    target_call = trace.steps[1 if forward else -1].calls[0]
    assert web_input(target_call).recorded
    assert not web_input(target_call).source_known
    assert states["lookup.search_surfaced_benchmark"] == "unknown"


def test_dynamic_derived_input_stays_unknown_and_source_is_unchanged():
    raw = step({}, derived=True)
    raw["tool_calls"][0]["arguments"]["input"] = (
        "const r = await tools.web__run({open: [{ref_id: dynamicRef}]});"
    )
    before = json.dumps(raw)
    trace, _, states = scan(raw)
    assert json.dumps(raw) == before
    assert trace.steps[-1].calls[-1].arguments is None
    assert not web_input(trace.steps[-1].calls[-1]).recorded
    assert states["integrity.web_input_unresolved"] == "match"


def test_allowlisted_report_contains_no_targets_or_body():
    trace, assessments, _ = scan(step({"open": [{"ref_id": REF}, {"ref_id": URL}]}))
    exported = json.dumps(report(assessments))
    assert REF not in exported and URL not in exported
    assert "Synthetic ordinary body." not in exported
    assert trace.steps[-1].calls[0].arguments is not None


def test_derived_metadata_does_not_count_source_twice():
    raw = step({"open": [{"ref_id": REF}]}, derived=True)
    raw["metrics"] = {"completion_tokens": 100}
    trace, _, _ = scan(raw)
    ratio = output_ratio(trace)
    assert ratio is not None
    assert ratio.chars == len(raw["tool_calls"][0]["arguments"]["input"])


def test_brief_counts_input_gap_as_recording_not_behaviour():
    _, assessments, _ = scan(step({"open": [{"ref_id": REF}]}))
    row = report(assessments)
    row.update({"input_id": "synthetic", "input_status": "available", "task": "synthetic"})
    summary = brief(document([row], "test", Engine(builtin_detectors()).catalog()))
    assert summary["recording"]["integrity.web_input_unresolved"] == 1
    assert "reference provenance unresolved" in " ".join(brief_text(summary).split())


def test_recorded_error_with_direct_url_has_neither_integrity_gap():
    _, _, states = scan(step({"open": [{"ref_id": URL}]}, "Error: retrieval failed"))
    assert states["integrity.web_results_not_recorded"] == "no_match"
    assert states["integrity.web_input_unresolved"] == "no_match"
    assert states["lookup.search_surfaced_benchmark"] == "unknown"


def test_missing_query_is_input_gap_even_with_recorded_body():
    _, _, states = scan(step({"search_query": [{}]}))
    assert states["integrity.web_input_unresolved"] == "match"
    assert states["integrity.web_results_not_recorded"] == "no_match"
