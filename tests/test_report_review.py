"""Regressions from the report/cache/CLI review. Synthetic fixtures only."""

from __future__ import annotations

import importlib
import io
import json
import re
import sys

import pytest
from test_brief_cache import write_run
from test_harbor_files import harbor_job, result_json

from atif_scan import sources
from atif_scan.cli import main
from atif_scan.report import (
    _m,
    citation_lines,
    overview,
    overview_text,
    summary,
    summary_text,
)
from atif_scan.views import render_rich, render_summary_rich


def _no_reload(monkeypatch):
    def boom(path):
        raise AssertionError("trace was reloaded")

    monkeypatch.setattr(sources, "load_trace", boom)


def _rule_status(out: str) -> list[str]:
    items = json.loads(out)["inputs"]
    return [a["status"] for i in items for a in i["assessments"] if a["id"] == "review.rule"]


def test_cache_misses_when_a_rule_expression_changes(tmp_path, capsys):
    # Regression: the signature hashed only id/version/tasks/severity, so editing a rule's
    # `when` without bumping its version returned the old cached results.
    root = write_run(tmp_path)
    rules = tmp_path / "rules.json"
    args = [str(root), "--format", "json", "--cache", str(tmp_path / "c"), "--rules", str(rules)]

    def scan(when):
        rules.write_text(
            json.dumps({"rules": [{"id": "review.rule", "severity": "low", "when": when}]})
        )
        main(args)
        return _rule_status(capsys.readouterr().out)

    first = scan("tamper.reward_write")
    assert "match" in first
    flip = {"match": "no_match", "no_match": "match"}
    assert scan({"not": "tamper.reward_write"}) == [flip.get(s, s) for s in first]


def test_cache_misses_when_an_allowance_expression_or_plugin_source_changes(tmp_path, monkeypatch):
    from atif_scan.cache import checks_signature
    from atif_scan.checks import CheckSpec, Detection, Status
    from atif_scan.engine import Engine
    from atif_scan.policy import load_rules

    def allow(when):
        return load_rules({"allow": [{"id": "ok", "covers": ["r"], "when": when}]})

    base = load_rules({"rules": [{"id": "r", "severity": "high", "when": "d"}]})

    class D:
        spec = CheckSpec("d")

        def evaluate(self, trace, context):
            return Detection(Status.NO_MATCH)

    one = checks_signature(Engine([D(), *base, *allow("d")]))
    two = checks_signature(Engine([D(), *base, *allow({"not": "d"})]))
    assert one != two
    # Plugin code lives outside the package: its source file is part of the signature.
    plugin = tmp_path / "review_plugin_mod.py"
    source = (
        "from atif_scan.checks import CheckSpec, Detection, Status\n"
        "class P:\n    spec = CheckSpec('p')\n"
        "    def evaluate(self, trace, context):\n        return Detection(Status.{})\n"
    )
    plugin.write_text(source.format("NO_MATCH"))
    monkeypatch.syspath_prepend(str(tmp_path))
    review_plugin_mod = importlib.import_module("review_plugin_mod")
    before = checks_signature(Engine([review_plugin_mod.P()]))
    plugin.write_text(source.format("MATCH"))
    assert checks_signature(Engine([review_plugin_mod.P()])) != before
    sys.modules.pop("review_plugin_mod", None)


def test_cached_items_take_fresh_run_facts_from_result_json(tmp_path, capsys, monkeypatch):
    # Regression: cost/error/duration from result.json were stored in the cached item and
    # went stale when result.json changed (the trajectory, and so the key, didn't).
    job = harbor_job(tmp_path, costs=(1.0, None, None))
    args = [str(job), "--format", "json", "--cache", str(tmp_path / "c")]
    main(args)
    first = {i["input_id"]: i for i in json.loads(capsys.readouterr().out)["inputs"]}
    assert first["weird-folder-0/agent"]["cost_usd"] == 1.0
    (job / "weird-folder-0" / "result.json").write_text(
        json.dumps(result_json("alpha", 1.0, "AgentTimeoutError", cost=7.5))
    )
    _no_reload(monkeypatch)  # still a cache hit: only run facts changed
    main(args)
    second = {i["input_id"]: i for i in json.loads(capsys.readouterr().out)["inputs"]}
    assert second["weird-folder-0/agent"]["cost_usd"] == 7.5
    assert second["weird-folder-0/agent"]["error_type"] == "AgentTimeoutError"
    # Same layout (key order) as a fresh scan; unchanged trials are unchanged.
    assert list(second["weird-folder-0/agent"]) == list(first["weird-folder-0/agent"])
    assert second["weird-folder-1/agent"] == first["weird-folder-1/agent"]


def test_cached_and_fresh_scans_agree_on_exit_codes(tmp_path, capsys, monkeypatch):
    root = write_run(tmp_path)
    args = [str(root), "--format", "json", "--cache", str(tmp_path / "c"), "--fail-on", "high"]
    assert main(args) == 1
    _no_reload(monkeypatch)
    assert main(args) == 1
    capsys.readouterr()


def _item(i, model, high=False, unknown=False):
    return {
        "input_id": f"t{i}",
        "input_status": "available",
        "incomplete": unknown,
        "task": f"task{i % 50}",
        "reward": 1.0,
        "model_name": model,
        "cost_usd": 1.0,
        "assessments": [
            {
                "id": "tamper.test_files",
                "kind": "detector",
                "severity": "high",
                "score": 75 if high else None,
                "status": "match" if high else "unknown" if unknown else "no_match",
                "expected_by": [],
                "evidence": [],
            }
        ],
    }


def test_overview_is_linear_in_items(linear):
    # Regression: DQ candidate de-duplication and the not-cleared lookup were quadratic.
    def run(n):
        items = [
            _item(i, "a" if i < n * 0.52 else "b", high=i % 2 == 0, unknown=i % 2 == 1)
            for i in range(n)
        ]
        return overview({"scanner_version": "dev", "inputs": items, "coverage": {}})

    linear(run, 1_250)
    ov = run(5000)
    d = ov["disqualification"]
    assert d["candidates"] == 2500 + 1200 and len(set(d["candidate_ids"])) == d["candidates"]
    # The 1200 unknown trials on the other model are already DQ candidates.
    assert d["rewarded_not_cleared"] == 1300
    assert set(d["candidate_ids"]).isdisjoint(d["rewarded_not_cleared_ids"])


def test_token_units_are_picked_after_rounding():
    assert _m(999) == "999" and _m(999_499) == "999k" and _m(999_600) == "1.0M"
    assert _m(12_345_678) == "12.3M" and _m(999_960_000) == "1.00B" and _m(2_500_000_000) == "2.50B"


def test_more_trials_than_planned_is_not_negative_missing():
    doc = {
        "scanner_version": "dev",
        "inputs": [_item(i, "a") for i in range(3)],
        "coverage": {},
        "runs": [{"job_id": "0" * 8, "planned_trials": 2}],
    }
    ov = overview(doc)
    assert ov["trials"]["missing"] == 0
    assert "0 missing · 1 more than planned" in "\n".join(overview_text(ov))


CITATION = {
    "step": 0,
    "step_id": 1,
    "channel": "command",
    "tool": "bash",
    "before": "a⟦b",  # trace text can itself contain the bracket characters
    "match": "M",
    "after": "c⟧d",
}


def test_citation_match_is_structured_not_bracket_joined(monkeypatch):
    # Regression: rows were joined as before⟦match⟧after and re-split by the rich view,
    # which mis-highlighted a match whose context contained ⟦ or ⟧.
    assert citation_lines(CITATION)[-1] == (">", ("a⟦b", "M", "c⟧d"))
    item = {
        "input_id": "t1",
        "input_status": "available",
        "severity": "high",
        "score": 75,
        "incomplete": False,
        "agent_steps": None,
        "assessments": [
            {
                "id": "x.check",
                "kind": "detector",
                "status": "match",
                "severity": "high",
                "score": 75,
                "expected_by": [],
                "evidence": [],
            }
        ],
        "citations": {"x.check": [CITATION]},
    }
    doc = {
        "scanner_version": "dev",
        "inputs": [item],
        "coverage": {"inputs": 1, "available": 1, "incomplete": 0},
    }
    assert "│ > a⟦b⟦M⟧c⟧d" in summary_text(summary(doc))
    pytest.importorskip("rich")
    plain = io.StringIO()
    render_rich(doc, file=plain)  # the rich view highlights the match instead of bracketing
    assert "match a⟦bMc⟧d" in plain.getvalue()
    # Off a terminal the summary stays plain text (the caller prints summary_text).
    assert not render_summary_rich(summary(doc), file=io.StringIO())
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    for render in (render_rich, render_summary_rich):
        styled = io.StringIO()
        render(summary(doc) if render is render_summary_rich else doc, file=styled)
        assert "a⟦b\x1b[1;7mM\x1b[0mc⟧d" in styled.getvalue()  # exactly the match reversed


def test_view_flags_are_mutually_exclusive(tmp_path, capsys):
    root = write_run(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main([str(root), "--brief", "--summary"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        main([str(root), "--overview", "--cite"])
    capsys.readouterr()
    main([str(root), "--format", "text", "--no-cache"])  # a run's text default: the brief
    assert "run integrity" in capsys.readouterr().out
    main([str(root), "--format", "text", "--no-cache", "--cite", "high"])  # citing: detail
    assert "run integrity" not in capsys.readouterr().out
    main([str(root), "--format", "json", "--no-cache"])  # piped default: the JSON document
    assert "inputs" in json.loads(capsys.readouterr().out)


def test_brief_ties_are_ordered_by_check_id():
    # Regression: ties followed set iteration order (string hashing), so the same run
    # could print its recording lines in a different order on every invocation.
    from atif_scan.brief import brief

    def a(check):
        return {
            "id": check,
            "kind": "detector",
            "status": "match",
            "severity": "low",
            "score": 25,
            "expected_by": [],
            "evidence": [],
        }

    names = [f"integrity.z{n}" for n in range(9, 0, -1)]
    items = [dict(_item(0, "a"), assessments=[a(c) for c in names])]
    b = brief({"scanner_version": "dev", "inputs": items, "coverage": {}})
    assert list(b["recording"]) == sorted(names)


def test_citation_rows_say_what_the_context_is():
    def labels(channel):
        c = dict(CITATION, channel=channel, context_before="b", context_after="a")
        return [label for label, _ in citation_lines(c)]

    assert labels("command") == ["@", "why", ">", "result"]  # intent, then what came back
    assert labels("observation") == ["@", "call", ">", "after"]  # the producing call
    assert labels("reasoning") == ["@", "before", ">", "then ran"]  # the step's first call


def test_citation_collapses_blank_lines_but_keeps_match_boundaries():
    c = dict(CITATION, before="one\n\n\n  two ", match="M", after=" three\r\n\nfour")
    assert citation_lines(c)[-1] == (">", ("one ⏎ two ", "M", " three ⏎ four"))


def test_rich_summary_wraps_citations_inside_their_column(monkeypatch):
    pytest.importorskip("rich")
    long = "word " * 60
    cite = dict(CITATION, channel="reasoning", before=long, match="M", after=long)
    cite["context_after"] = "ls /app"
    s = {
        "scanner_version": "dev",
        "checks": {"x.check": {"severity": "low", "traces": 1, "events": 1}},
        "highest_severity": {"low": 1},
        "details": [
            {
                "check": "x.check",
                "severity": "low",
                "traces": [{"input_id": "t1", "reward": 1, "evidence": [], "citations": [cite]}],
            }
        ],
        "expected": {},
        "unresolved": {},
        "coverage": {"inputs": 1, "available": 1, "incomplete": 0},
    }
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = io.StringIO()
    assert render_summary_rich(s, file=out)
    lines = re.sub(r"\x1b\[[0-9;]*m", "", out.getvalue()).splitlines()
    rows = [line for line in lines if "word" in line]
    assert len(rows) > 3 and all(len(line) <= 80 for line in lines)
    # Every wrapped line starts at the text column, never back at column 0.
    column = rows[0].index("word")
    assert all(line[:column].strip() in ("", "match") for line in rows)
    assert all(line.index("word") == column for line in rows)
    assert "⟦" not in out.getvalue() and any("then ran ls /app" in line for line in lines)
