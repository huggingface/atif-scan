"""Synthetic compaction interpretations, not companion-archive verification."""

import json
from io import StringIO

import pytest

from atif_scan.cli import main
from atif_scan.report import COMPACTED_USAGE_EXPLANATION, to_text
from atif_scan.views import render_rich

CHECK = "integrity.tokens_exceed_recorded_calls"


@pytest.mark.parametrize("compacted_count", [0, 1, 2])
def test_token_messaging_cohorts(tmp_path, capsys, compacted_count):
    for index in range(2):
        folder = tmp_path / f"trial-{index}"
        folder.mkdir()
        raw = {
            "schema_version": "ATIF-v1.7",
            "steps": [
                {
                    "step_id": 1,
                    "source": "user",
                    "message": (
                        "[COMPACTED HISTORY] synthetic private text"
                        if index < compacted_count
                        else "synthetic private text"
                    ),
                },
                {"step_id": 2, "source": "agent", "message": "Working."},
            ],
            "final_metrics": {"total_prompt_tokens": 3_000_000, "total_completion_tokens": 100},
        }
        (folder / "trajectory.json").write_text(json.dumps(raw))
    args = [str(tmp_path), "--no-cache", "--packs", "none", "--format", "json"]
    assert main(args) == 0
    doc = json.loads(capsys.readouterr().out)
    for item in doc["inputs"]:
        assessment = next(a for a in item["assessments"] if a["id"] == CHECK)
        assert assessment["status"] == "match"
        assert assessment["severity"] == "low" and assessment["score"] == 25
        assert assessment.get("explanation") == (
            COMPACTED_USAGE_EXPLANATION if item["compacted"] else None
        )
    plain = to_text(doc)
    rich = StringIO()
    render_rich(doc, file=rich)
    assert plain.count(COMPACTED_USAGE_EXPLANATION) == compacted_count
    assert " ".join(rich.getvalue().split()).count(COMPACTED_USAGE_EXPLANATION) == compacted_count
    assert "synthetic private text" not in json.dumps(doc) + plain + rich.getvalue()
    assert main([*args, "--brief"]) == 0
    brief = json.loads(capsys.readouterr().out)
    assert brief["recording"][CHECK] == 2
    assert brief["compacted_token_hits"] == compacted_count
    assert main([*args[:-2], "--brief", "--format", "text"]) == 0
    text = " ".join(capsys.readouterr().out.split())
    assert (COMPACTED_USAGE_EXPLANATION in text) == bool(compacted_count)
    if compacted_count:
        unit = "trial" if compacted_count == 1 else "trials"
        assert f"{compacted_count} {unit} ({compacted_count * 50:.1f}%), compacted:" in text
    assert ("token totals exceed the recorded calls" in text) == (compacted_count < 2)
