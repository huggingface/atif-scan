"""Synthetic exporter-specific completion semantics; never normalize recorded usage."""

import json

import pytest

from atif_scan import Status, parse_trace
from atif_scan.brief import RATIO_BASIS, output_ratios
from atif_scan.cache import checks_signature
from atif_scan.cli import main
from atif_scan.detectors.integrity import output_ratio, output_token_ratio
from atif_scan.facts import trace_facts


def payload(reasoning=2000, completion=1000, name="grok-build", version="1.0.34"):
    return {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": name, "version": version, "model_name": "xai/grok-synthetic"},
        "steps": [{"source": "agent", "message": "x" * 3000, "reasoning_content": "y" * 9000}],
        "final_metrics": {
            "total_completion_tokens": completion,
            "total_cost_usd": 0.25,
            "extra": {"total_reasoning_tokens": reasoning},
        },
    }


@pytest.mark.parametrize("reasoning", [0, None, 900, 2000, True, -1, 1.5, "2000"])
def test_separate_visible_denominator_and_raw_counts(reasoning):
    trace = parse_trace(payload(reasoning))
    ratio = output_ratio(trace)
    assert ratio is not None
    assert (ratio.tokens, ratio.chars, ratio.value) == (1000, 3000, 3)
    assert ratio.answer_only and ratio.low_verifiable and ratio.separate_visible
    assert output_token_ratio(trace).status == Status.NO_MATCH
    assert trace.usage is not None
    assert trace.usage.completion_tokens == 1000
    assert trace.usage.reasoning_tokens == (
        reasoning if type(reasoning) is int and reasoning >= 0 else None
    )
    assert trace.usage.cost_usd == 0.25
    facts = trace_facts(trace)
    assert facts["trajectory_completion_token_basis"] == "separate_visible"
    assert facts["output_ratio_basis"] == "separate_visible"
    assert facts["usage"]["output_tokens"] == 1000


def test_absent_reasoning_and_compaction():
    raw = payload()
    del raw["final_metrics"]["extra"]
    assert output_token_ratio(parse_trace(raw)).status == Status.NO_MATCH
    raw["steps"].insert(0, {"source": "system", "message": "[COMPACTED HISTORY]"})
    raw["steps"][1]["message"] = "small"
    trace = parse_trace(raw)
    assert output_token_ratio(trace).status == Status.UNKNOWN
    ratio = output_ratio(trace)
    assert ratio and not ratio.low_verifiable
    raw["steps"][1]["message"] = "x" * 9000
    assert output_token_ratio(parse_trace(raw)).status == Status.MATCH


@pytest.mark.parametrize("completion", [None, True, False, -1, 1.5, "1000", 0])
def test_unusable_completion_is_unknown(completion):
    trace = parse_trace(payload(completion=completion))
    assert output_ratio(trace) is None
    assert output_token_ratio(trace).status == Status.UNKNOWN


@pytest.mark.parametrize(
    ("name", "version"),
    [
        ("fast-agent", "1.0.34"),
        ("other", "1.0.34"),
        ("Grok-Build", "1.0.34"),
    ],
)
def test_other_exporters_unchanged(name, version):
    trace = parse_trace(payload(name=name, version=version))
    assert trace.usage and trace.usage.completion_basis == "includes_reasoning"
    assert output_ratio(trace) is None  # standard reasoning > completion is unusable
    trace = parse_trace(payload(reasoning=900, name=name, version=version))
    ratio = output_ratio(trace)
    assert ratio and ratio.tokens == 100
    assert output_token_ratio(trace).status == Status.MATCH


def test_step_fallback_not_blindly_corrected():
    raw = payload(completion=None)
    raw["steps"][0]["metrics"] = {"completion_tokens": 1000}
    trace = parse_trace(raw)
    ratio = output_ratio(trace)
    assert ratio and not ratio.separate_visible and not ratio.answer_only
    assert ratio.chars == 12000
    assert trace_facts(trace)["trajectory_completion_token_basis"] is None


def test_report_brief_and_cache_signature(tmp_path, capsys):
    from atif_scan import Engine, builtin_detectors

    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(payload()))
    assert main([str(path), "--format", "json", "--no-cache"]) == 0
    item = json.loads(capsys.readouterr().out)["inputs"][0]
    assert item["trajectory_completion_token_basis"] == "separate_visible"
    assert item["output_ratio_basis"] == "separate_visible"
    assert item["output_tokens"] == 1000
    assert item["cost_usd"] == 0.25
    ratios = output_ratios([item])
    assert ratios is not None
    assert ratios["separate_visible"]["median"] == 3
    assert "not billed completion" in RATIO_BASIS["separate_visible"]
    signature = checks_signature(Engine(builtin_detectors()))
    assert next(s for s in signature if s[0] == "integrity.output_token_ratio")[1] == "3"


def test_missing_final_completion_and_repeat_parse():
    raw = payload(reasoning=900)
    first = parse_trace(raw)
    assert first == parse_trace(raw)
    assert raw["final_metrics"]["total_completion_tokens"] == 1000
    del raw["final_metrics"]["total_completion_tokens"]
    assert output_token_ratio(parse_trace(raw)).status == Status.UNKNOWN


def test_exporter_scope_does_not_depend_on_model():
    raw = payload()
    raw["agent"]["model_name"] = "synthetic/other"
    assert output_token_ratio(parse_trace(raw)).status == Status.NO_MATCH


@pytest.mark.parametrize("version", [None, "unknown", "1.0.33", "1.0.35", 1.034])
def test_grok_build_convention_is_independent_of_version(version):
    trace = parse_trace(payload(version=version))
    assert trace.usage and trace.usage.completion_basis == "separate_visible"
    ratio = output_ratio(trace)
    assert ratio and ratio.tokens == 1000
    assert output_token_ratio(trace).status == Status.NO_MATCH
