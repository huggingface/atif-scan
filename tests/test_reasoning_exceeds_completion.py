"""Synthetic regressions: reasoning is part of the completion total that reports it."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from atif_scan import Context, Engine, Severity, Status, builtin_detectors, parse_trace, report
from atif_scan.output.document import TELEMETRY_CHECKS

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc

CHECK = "integrity.reasoning_tokens_exceed_completion"


def detector():
    return next(d for d in builtin_detectors() if d.spec.id == CHECK)


def recording(metrics: Doc | None, name: str = "synthetic-harness") -> Doc:
    return {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": name, "version": "1", "model_name": "synthetic/model"},
        "steps": [{"source": "agent", "message": "synthetic-private-marker"}],
        "final_metrics": metrics,
    }


def metrics(completion: object, reasoning: object) -> Doc:
    return {"total_completion_tokens": completion, "extra": {"total_reasoning_tokens": reasoning}}


@pytest.mark.parametrize(
    ("completion", "reasoning", "status"),
    [
        (1000, 1001, Status.MATCH),
        (0, 1, Status.MATCH),
        (8166, 89954, Status.MATCH),
        (1000, 1000, Status.NO_MATCH),
        (1000, 0, Status.NO_MATCH),
        (0, 0, Status.NO_MATCH),
    ],
)
def test_boundary(completion, reasoning, status):
    trace = parse_trace(recording(metrics(completion, reasoning)))
    result = detector().evaluate(trace, Context())
    assert result.status == status
    assert result.complete and not result.evidence
    assert dict(result.measure) == {"completion_tokens": completion, "reasoning_tokens": reasoning}
    assert trace.usage is not None
    assert trace.usage.completion_tokens == completion  # never normalized


@pytest.mark.parametrize("bad", [None, -1, True, "100", 1.5])
@pytest.mark.parametrize("field", ["completion", "reasoning"])
def test_missing_or_invalid_operand_is_unknown(field, bad):
    values = {"completion": 1000, "reasoning": 2000, field: bad}
    trace = parse_trace(recording(metrics(values["completion"], values["reasoning"])))
    result = detector().evaluate(trace, Context())
    assert result.status == Status.UNKNOWN and not result.complete
    assert [u.reason for u in result.unread] == ["usage_not_recorded"]


@pytest.mark.parametrize("name", ["grok-build", "fast-agent", "synthetic-harness"])
def test_applies_to_every_exporter(name):
    trace = parse_trace(recording(metrics(1000, 2000), name=name))
    assert detector().evaluate(trace, Context()).status == Status.MATCH


def test_step_usage_is_not_substituted():
    raw = recording(None)
    raw["steps"][0]["metrics"] = {"completion_tokens": 10, "extra": {"reasoning_tokens": 900}}
    result = detector().evaluate(parse_trace(raw), Context())
    assert result.status == Status.UNKNOWN


def test_registration_and_report_are_telemetry_only_and_snippet_free():
    check = detector()
    assert check.spec.version == "1" and check.spec.severity == Severity.MEDIUM
    assert CHECK in TELEMETRY_CHECKS
    for final in (None, metrics(1000, 2000)):
        trace = parse_trace(recording(final))
        item = report(Engine([check]).evaluate(trace), trace.step_numbers)
        assert "synthetic-private-marker" not in json.dumps(item)
        assert item["coverage_gaps"]["behavioural"] == []
        assert item["coverage_gaps"]["telemetry"] == ([CHECK] if final is None else [])
        assert item["assessments"][0]["id"] == CHECK
