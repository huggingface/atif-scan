"""ATIF recording defects found on a real TB4 leaderboard job (synthetic reproductions)."""

from __future__ import annotations

import json

from atif_scan import Engine, Status, builtin_detectors, parse_trace
from atif_scan.cli import main
from atif_scan.detectors.integrity import output_ratio

NOTICE = (
    "This session is being continued from a previous conversation that ran out of context. "
    "The summary below covers the earlier portion of the conversation."
)


def trace(steps, final_metrics=None):
    raw = {"schema_version": "ATIF-v1.7", "steps": steps}
    if final_metrics is not None:
        raw["final_metrics"] = final_metrics
    return raw


def agent(command="ls", reasoning=None, llm_calls=None, cid="c1"):
    step = {
        "source": "agent",
        "message": "working",
        "tool_calls": [
            {"tool_call_id": cid, "function_name": "bash", "arguments": {"command": command}}
        ],
    }
    if reasoning is not None:
        step["reasoning_content"] = reasoning
    if llm_calls is not None:
        step["llm_call_count"] = llm_calls
    return step


def results(raw, context=None):
    return {a.spec.id: a for a in Engine(builtin_detectors()).evaluate(parse_trace(raw), context)}


GROK_SHAPE = [
    {"source": "system", "message": "You are an autonomous agent."},
    {"source": "user", "message": "<user_info> OS Version: linux </user_info>"},
    {"source": "user", "message": "<user_query> Do the task. </user_query>"},
    {"source": "user", "message": NOTICE + " Summary: 1. Primary Request ..."},
    agent(),
]


def test_compacted_history_is_detected_and_scanned_as_partial(tmp_path, capsys):
    t = parse_trace(trace(GROK_SHAPE))
    assert t.compacted == (3,)
    r = results(trace(GROK_SHAPE))
    assert r["integrity.history_compacted"].result.status == Status.MATCH
    path = tmp_path / "t.json"
    path.write_text(json.dumps(trace(GROK_SHAPE)))
    assert main([str(path), "--format", "json"]) == 0
    item = json.loads(capsys.readouterr().out)["inputs"][0]
    assert item["compacted"] and item["partial"] and item["incomplete"]
    rows = {a["id"]: a for a in item["assessments"]}
    # A negative over the recorded remainder doesn't clear the missing history.
    assert rows["tamper.reward_write"]["status"] == "unknown"
    main([str(path), "--overview", "--format", "json"])
    assert json.loads(capsys.readouterr().out)["trials"]["compacted"] == 1


def test_compaction_markers_only_count_in_prompts():
    said = [agent(), {"source": "agent", "message": NOTICE}]
    assert parse_trace(trace(said)).compacted == ()  # the agent quoting it is not a notice
    other = [{"source": "user", "message": "[COMPACTED HISTORY] earlier turns removed"}, agent()]
    assert parse_trace(trace(other)).compacted == (0,)


def test_reasoning_reported_but_not_recorded_makes_prose_checks_incomplete():
    hidden = trace([agent()], {"extra": {"total_reasoning_tokens": 5000}})
    r = results(hidden)
    assert r["integrity.reasoning_not_recorded"].result.status == Status.MATCH
    assert r["awareness.benchmark"].result.status == Status.UNKNOWN  # can't rule it out
    shown = trace(
        [agent(reasoning="thinking about the task")], {"extra": {"total_reasoning_tokens": 5000}}
    )
    r = results(shown)
    assert r["integrity.reasoning_not_recorded"].result.status == Status.NO_MATCH
    assert r["awareness.benchmark"].result.status == Status.NO_MATCH


def test_cost_missing_and_tokens_exceeding_recorded_calls():
    no_cost = results(
        trace([agent()], {"total_prompt_tokens": 25_000_000, "total_completion_tokens": 9})
    )
    assert no_cost["integrity.cost_missing"].result.status == Status.MATCH
    priced = results(
        trace(
            [agent()],
            {"total_prompt_tokens": 1000, "total_completion_tokens": 9, "total_cost_usd": 0.1},
        )
    )
    assert priced["integrity.cost_missing"].result.status == Status.NO_MATCH
    assert results(trace([agent()]))["integrity.cost_missing"].result.status == Status.UNKNOWN
    # 122M prompt tokens over 8 recorded calls: the steps can't account for the totals.
    steps = [agent(llm_calls=1, cid=f"c{i}") for i in range(8)]
    big = results(trace(steps, {"total_prompt_tokens": 122_000_000, "total_cost_usd": 80.0}))
    assert big["integrity.tokens_exceed_recorded_calls"].result.status == Status.MATCH
    ok = results(trace(steps, {"total_prompt_tokens": 2_000_000, "total_cost_usd": 1.0}))
    assert ok["integrity.tokens_exceed_recorded_calls"].result.status == Status.NO_MATCH


def test_search_glob_is_not_written_code():
    # Regression: a grep glob "**/tests/**/*.py" was flagged as code referencing verifier paths.
    raw = trace(
        [
            {
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {
                        "tool_call_id": "c1",
                        "function_name": "grep",
                        "arguments": {
                            "pattern": "vllm",
                            "glob": "**/tests/**/*.py",
                            "path": "/app",
                        },
                    }
                ],
            }
        ]
    )
    assert results(raw)["code.verifier_path_reference"].result.status == Status.NO_MATCH


# --- integrity.output_token_ratio: recorded authored text vs reported completion tokens ---

RATIO = "integrity.output_token_ratio"


def said(chars, reasoning=None, tokens=None, cid="c1"):
    """An agent step with ~`chars` authored characters (message + command)."""
    step = agent(command="x" * (chars - len("working")), reasoning=reasoning, cid=cid)
    if tokens is not None:
        step["metrics"] = {"prompt_tokens": 10, "completion_tokens": tokens}
    return step


def ratio_status(raw):
    return results(raw)[RATIO].result.status


def test_ratio_above_bound_flags_even_with_hidden_reasoning():
    # More text than the tokens could encode: unrecorded output can't explain it.
    raw = trace([said(10_000)], {"total_completion_tokens": 1_000})
    assert ratio_status(raw) == Status.MATCH


def test_ratio_low_side_unknown_without_reasoning_count():
    # Hidden reasoning with no reported count (Codex/Claude Code shape): ~0.3 chars/token
    # is normal there, so it's unknown, never flagged.
    raw = trace([said(300)], {"total_completion_tokens": 1_000})
    assert ratio_status(raw) == Status.UNKNOWN
    # Plausible text is still unknown: the lower bound can't be checked.
    raw = trace([said(3_000)], {"total_completion_tokens": 1_000})
    assert ratio_status(raw) == Status.UNKNOWN


def test_ratio_answer_only_ignores_reasoning_summaries():
    # Regression: Gemini records short thought summaries; counting them against all
    # completion tokens read as 0.2-0.3 chars/token. Reasoning tokens are reported, so
    # reasoning text and tokens are both excluded.
    raw = trace(
        [said(3_000, reasoning="brief summary")],
        {"total_completion_tokens": 11_000, "extra": {"total_reasoning_tokens": 10_000}},
    )
    assert ratio_status(raw) == Status.NO_MATCH
    r = output_ratio(parse_trace(raw))
    assert r.answer_only and r.chars == 3_000 and r.tokens == 1_000


def test_ratio_below_bound_flags_when_reasoning_is_accounted_for():
    raw = trace(
        [said(300)],
        {"total_completion_tokens": 6_000, "extra": {"total_reasoning_tokens": 1_000}},
    )
    assert ratio_status(raw) == Status.MATCH  # 300 chars for 5,000 answer tokens


def test_ratio_low_side_unknown_when_history_compacted():
    steps = [*GROK_SHAPE[:4], said(300)]
    raw = trace(steps, {"total_completion_tokens": 6_000, "extra": {"total_reasoning_tokens": 0}})
    assert ratio_status(raw) == Status.UNKNOWN  # totals include unrecorded calls


def test_ratio_falls_back_to_step_metrics():
    # No final_metrics: only steps that report completion tokens are compared.
    steps = [said(9_000, tokens=1_000, cid="a"), said(50_000, cid="b")]
    t = parse_trace(trace(steps))
    assert [s.completion_tokens for s in t.steps] == [1_000, None]
    assert results(trace(steps))[RATIO].result.status == Status.MATCH  # 9.0 > 8
    steps = [said(3_000, tokens=1_000, cid="a"), said(50_000, cid="b")]
    assert ratio_status(trace(steps)) == Status.UNKNOWN  # 3.0: fine, low side unchecked


def test_ratio_unknown_without_usable_tokens():
    assert ratio_status(trace([said(100)])) == Status.UNKNOWN
    raw = trace(
        [said(100)], {"total_completion_tokens": 50, "extra": {"total_reasoning_tokens": 50}}
    )
    assert ratio_status(raw) == Status.UNKNOWN


def test_ratio_in_report_and_brief(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    good = trace(
        [said(2_500)],
        {"total_completion_tokens": 2_000, "extra": {"total_reasoning_tokens": 1_000}},
    )
    bad = trace([said(20_000)], {"total_completion_tokens": 1_000})
    (tmp_path / "a" / "trajectory.json").write_text(json.dumps(good))
    (tmp_path / "b" / "trajectory.json").write_text(json.dumps(bad))
    main([str(tmp_path), "--format", "json", "--no-cache"])
    items = {i["input_id"]: i for i in json.loads(capsys.readouterr().out)["inputs"]}
    assert items["a"]["chars_per_output_token"] == 2.5
    assert items["a"]["output_ratio_basis"] == "answer_only"
    assert items["b"]["chars_per_output_token"] == 20.0
    assert items["b"]["output_ratio_basis"] == "all_text"
    main([str(tmp_path), "--format", "text", "--no-cache"])
    out = capsys.readouterr().out
    assert "recorded text doesn't fit reported output tokens: 1 (50.0%)" in out
    assert (
        "chars/output token: median 2.50 (p5–p95 2.50–2.50) over 1 traces, excl. reasoning" in out
    )
    assert "median 20.00" in out and "incl. reasoning; hidden reasoning lowers it" in out
