"""Synthetic regressions for inclusive ATIF prompt/cache accounting."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from atif_scan import Context, Engine, Severity, Status, builtin_detectors, parse_trace, report

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc

CHECK = "integrity.cached_tokens_exceed_prompt"


def detector():
    return next(d for d in builtin_detectors() if d.spec.id == CHECK)


def recording(metrics: Doc | None) -> Doc:
    return {
        "schema_version": "ATIF-v1.7",
        "steps": [{"source": "agent", "message": "synthetic-private-marker"}],
        "final_metrics": metrics,
    }


@pytest.mark.parametrize(
    ("prompt", "cached", "status"),
    [
        (100, 101, Status.MATCH),
        (0, 1, Status.MATCH),
        (100, 100, Status.NO_MATCH),
        (100, 0, Status.NO_MATCH),
        (0, 0, Status.NO_MATCH),
        (100, 99, Status.NO_MATCH),
    ],
)
def test_inclusive_prompt_boundary(prompt, cached, status):
    trace = parse_trace(recording({"total_prompt_tokens": prompt, "total_cached_tokens": cached}))
    result = detector().evaluate(trace, Context())
    assert result.status == status
    assert result.complete and not result.evidence
    assert dict(result.measure) == {"prompt_tokens": prompt, "cached_tokens": cached}
    assert trace.usage is not None
    assert trace.usage.prompt_tokens == prompt  # never normalize incompatible totals


@pytest.mark.parametrize("bad", [None, -1, True, "100", 1.5])
@pytest.mark.parametrize("field", ["total_prompt_tokens", "total_cached_tokens"])
def test_missing_or_invalid_operand_is_unknown(field, bad):
    metrics: Doc = {"total_prompt_tokens": 100, "total_cached_tokens": 101}
    metrics[field] = bad
    result = detector().evaluate(parse_trace(recording(metrics)), Context())
    assert result.status == Status.UNKNOWN and not result.complete
    assert [u.reason for u in result.unread] == ["usage_not_recorded"]


@pytest.mark.parametrize("metrics", [None, {}, {"total_completion_tokens": 100}])
def test_no_comparable_final_counts_is_unknown_even_with_step_usage(metrics):
    raw = recording(metrics)
    raw["steps"][0]["metrics"] = {"prompt_tokens": 1, "cached_tokens": 900}
    result = detector().evaluate(parse_trace(raw), Context())
    assert result.status == Status.UNKNOWN
    assert dict(result.measure) == {"prompt_tokens": None, "cached_tokens": None}


def test_final_comparison_does_not_require_step_usage_or_reasoning():
    raw = recording({"total_prompt_tokens": 100, "total_cached_tokens": 900})
    result = detector().evaluate(parse_trace(raw), Context())
    assert result.status == Status.MATCH and result.complete
    assert not result.unread


def test_registration_and_report_are_telemetry_only_and_snippet_free():
    check = detector()
    assert check.spec.version == "1" and check.spec.severity == Severity.LOW
    for metrics in (None, {"total_prompt_tokens": 100, "total_cached_tokens": 900}):
        trace = parse_trace(recording(metrics))
        item = report(Engine([check]).evaluate(trace), trace.step_numbers)
        assert "synthetic-private-marker" not in json.dumps(item)
        assert item["coverage_gaps"]["behavioural"] == []
        assert item["coverage_gaps"]["telemetry"] == ([CHECK] if metrics is None else [])
        assert item["assessments"][0]["id"] == CHECK
