"""The one-screen integrity brief, its estimates, and the per-trace result cache."""

from __future__ import annotations

import json

import pytest

from atif_scan import sources
from atif_scan.cli import main
from atif_scan.estimates import cost_estimate, missing_activity

SECRET = "sk-live-abcdefghijklmnop1234"
RATES = (2.5, 0.6, 10.0)  # $/M uncached, cached, output


def priced(i, inp, cache, out, cost=True):
    c = (RATES[0] * (inp - cache) + RATES[1] * cache + RATES[2] * out) / 1e6
    return {
        "input_id": f"t{i}",
        "input_tokens": inp,
        "cache_tokens": cache,
        "output_tokens": out,
        "cost_usd": c if cost else None,
    }


def test_cost_estimate_recovers_the_runs_own_prices():
    items = [
        priced(i, 1_000_000 + 37_000 * i, 400_000 + 21_000 * (i % 7), 20_000 + 900 * i)
        for i in range(30)
    ]
    missing = [
        priced(100, 2_000_000, 1_500_000, 50_000, cost=False),
        priced(101, 900_000, 100_000, 9_000, cost=False),
    ]
    truth = sum(
        (
            RATES[0] * (m["input_tokens"] - m["cache_tokens"])
            + RATES[1] * m["cache_tokens"]
            + RATES[2] * m["output_tokens"]
        )
        / 1e6
        for m in missing
    )
    est = cost_estimate(items + missing)
    assert est["unpriced"] == 2 and est["unpriced_ids"] == ["t100", "t101"]
    assert est["estimate_usd"] == pytest.approx(truth, rel=0.01)
    assert est["rates_per_mtok"]["output"] == pytest.approx(10.0, rel=0.01)
    # Too little data: say so rather than guess.
    few = cost_estimate(items[:5] + missing)
    assert few["estimate_usd"] is None and "fewer than" in few["method"]
    # $0 despite tokens counts as unpriced; no tokens at all is "no usage", not unpriced.
    zero = dict(priced(102, 1_000_000, 0, 1000), cost_usd=0.0)
    blank = {
        "input_id": "t103",
        "input_tokens": None,
        "cache_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
    }
    est = cost_estimate(items + [zero, blank])
    assert est["unpriced_ids"] == ["t102"] and est["no_usage"] == 1


def test_missing_activity_range_brackets_the_truth():
    refs = [
        {
            "input_id": f"r{i}",
            "input_tokens": 100_000 * (20 + i),
            "llm_calls": 20 + i,
            "compacted": False,
        }
        for i in range(20)
    ]  # ~100k prompt tokens per recorded call
    # A compacted trial that really made 400 calls but recorded 10.
    comp = {"input_id": "c1", "input_tokens": 100_000 * 400, "llm_calls": 10, "compacted": True}
    est = missing_activity(refs + [comp])
    lo, hi = est["compacted_recorded_pct"]
    assert lo <= 100 * 10 / 400 <= hi * 1.1
    run_lo, run_hi = est["missing_calls_pct"]
    total = sum(r["llm_calls"] for r in refs) + 400
    assert run_lo - 0.1 <= 100 * 390 / total <= run_hi + 1  # estimates are rounded
    assert missing_activity(refs[:5] + [comp])["missing_calls_pct"] is None  # too few references
    assert missing_activity(refs)["compacted"] == 0


def trace(command="ls", message="working", compacted=False, cost=None, tokens=None):
    steps = [{"source": "system", "message": "You are an agent."}]
    if compacted:
        steps.append(
            {
                "source": "user",
                "message": "This session is being continued from a previous conversation "
                "that ran out of context.",
            }
        )
    steps.append(
        {
            "source": "agent",
            "message": message,
            "tool_calls": [
                {"tool_call_id": "c1", "function_name": "bash", "arguments": {"command": command}}
            ],
        }
    )
    raw = {
        "schema_version": "ATIF-v1.7",
        "steps": steps,
        "agent": {"name": "demo-agent", "version": "1.0", "model_name": "demo/model"},
    }
    if tokens is not None:
        raw["final_metrics"] = {"total_prompt_tokens": tokens, "total_completion_tokens": 10}
        if cost is not None:
            raw["final_metrics"]["total_cost_usd"] = cost
    return raw


def write_run(tmp_path):
    root = tmp_path / "run"
    for i, raw in enumerate(
        [
            trace(f"echo {SECRET}", cost=1.0, tokens=1000),
            trace("echo 1 > /logs/verifier/reward.txt", cost=1.2, tokens=1200),
            trace(compacted=True, tokens=5000),
        ]
    ):
        d = root / f"task{i}__x{i}" / "agent"
        d.mkdir(parents=True)
        (d / "trajectory.json").write_text(json.dumps(raw))
        (d.parent / "verifier").mkdir()
        (d.parent / "verifier" / "reward.txt").write_text("1")
    return root


def test_brief_is_the_default_text_view_for_a_run(tmp_path, capsys):
    root = write_run(tmp_path)
    assert main([str(root), "--task-from", "trial-dir", "--format", "text"]) == 0
    out = capsys.readouterr().out
    assert "run integrity" in out and "agent  demo-agent / 1.0 / demo/model" in out
    assert "RESULT     100.0%" in out and "if 1 DQ candidate(s) are disqualified" in out
    assert "TRACES     ⚠ 1 (33.3%) compacted history" in out
    assert "COST       $2.20 reported" in out and "1 unpriced trial(s)" in out
    assert "tamper.reward_write" in out
    assert SECRET not in out and "echo" not in out
    # --detail restores the per-trace view; a single input defaults to it.
    main([str(root), "--detail", "--format", "text"])
    assert "run integrity" not in capsys.readouterr().out
    one = next(root.rglob("trajectory.json"))
    main([str(one), "--format", "text"])
    assert "run integrity" not in capsys.readouterr().out


def test_brief_json(tmp_path, capsys):
    root = write_run(tmp_path)
    main([str(root), "--brief", "--format", "json"])
    b = json.loads(capsys.readouterr().out)
    assert b["kind"] == "integrity_brief" and b["agents"] == {"demo-agent / 1.0 / demo/model": 3}
    assert b["missing_activity"]["compacted"] == 1 and b["cost_estimate"]["unpriced"] == 1
    assert b["findings"]["checks"]["tamper.reward_write"] == {"severity": "high", "traces": 1}
    assert "integrity.history_compacted" in b["recording"]


def test_result_cache_skips_rescans_and_invalidates(tmp_path, capsys, monkeypatch):
    root = write_run(tmp_path)
    cache = tmp_path / "cache"
    args = [str(root), "--format", "json", "--cache", str(cache)]
    main(args)
    first = json.loads(capsys.readouterr().out)["inputs"]
    assert len(list(cache.rglob("*.json"))) == 3

    def boom(path):  # any trace load now means a cache miss
        raise AssertionError("trace was reloaded")

    monkeypatch.setattr(sources, "load_trace", boom)
    main(args)
    assert json.loads(capsys.readouterr().out)["inputs"] == first
    # A different check set, or --cite, must not reuse cached results.
    with pytest.raises(AssertionError, match="reloaded"):
        main([*args, "--plugin", "atif_scan.packs.tb21:checks"])
    with pytest.raises(AssertionError, match="reloaded"):
        main([*args, "--cite"])
    monkeypatch.undo()
    # Touching a trajectory changes its fingerprint.
    target = next(root.rglob("trajectory.json"))
    target.write_text(target.read_text() + " ")
    main(args)
    capsys.readouterr()
    assert len(list(cache.rglob("*.json"))) == 4


def test_trajectory_cost_gap_is_not_a_defect_when_the_source_has_cost():
    from atif_scan.brief import brief

    def item(i, cost):
        return {
            "input_id": f"t{i}",
            "input_status": "available",
            "incomplete": False,
            "task": "t",
            "reward": 1.0,
            "cost_usd": cost,
            "assessments": [
                {
                    "id": "integrity.cost_missing",
                    "kind": "detector",
                    "status": "match",
                    "severity": "low",
                    "score": 25,
                    "expected_by": [],
                    "evidence": [],
                }
            ],
        }

    doc = {"scanner_version": "dev", "inputs": [item(1, 0.5), item(2, None)], "coverage": {}}
    assert brief(doc)["recording"] == {"integrity.cost_missing": 1}


def test_brief_colour_only_on_terminals(tmp_path, capsys):
    import io

    from rich.console import Console

    from atif_scan.brief import brief, brief_text, colourise, print_brief

    root = write_run(tmp_path)
    main([str(root), "--brief", "--format", "json"])
    text = brief_text(json.loads(capsys.readouterr().out))
    # Piped / non-terminal: byte-identical plain text, no escape codes.
    buffer = io.StringIO()
    print_brief(text, file=buffer)
    assert buffer.getvalue() == text and "\x1b[" not in buffer.getvalue()
    # Terminal: same characters, styled.
    styled = colourise(text)
    assert styled.plain == text
    styles = {str(span.style) for span in styled.spans}
    assert {"bold cyan", "bold yellow", "bold magenta"} <= styles
    console = Console(file=io.StringIO(), force_terminal=True, color_system="standard")
    console.print(styled, end="")
    assert "\x1b[" in console.file.getvalue()
    assert brief  # the JSON brief itself is uncoloured by construction


def test_brief_names_the_task_pack_for_a_known_dataset_without_loading_it():
    from atif_scan.brief import brief, brief_text

    def doc(check_id):
        item = {
            "input_id": "t1",
            "input_status": "available",
            "incomplete": False,
            "task": "t",
            "reward": 1.0,
            "assessments": [
                {
                    "id": check_id,
                    "kind": "detector",
                    "status": "no_match",
                    "severity": "low",
                    "score": None,
                    "expected_by": [],
                    "evidence": [],
                }
            ],
        }
        run = {
            "source": "harbor_hub",
            "job_id": "j",
            "datasets": ["terminal-bench/terminal-bench-2-1"],
        }
        return {"scanner_version": "dev", "inputs": [item], "coverage": {}, "runs": [run]}

    b = brief(doc("awareness.benchmark"))
    assert b["suggested_packs"] == ["atif_scan.packs.tb21:checks"]
    assert "PACKS" in brief_text(b)
    assert brief(doc("tb21.recall.task_catalog"))["suggested_packs"] == []


def test_cache_key_changes_with_scanner_code(tmp_path, monkeypatch):
    # Regression: parsing fixes (tool aliases, [REDACTED] values) reused results cached
    # before them, because no version string was bumped.
    from atif_scan import cache
    from atif_scan.checks import Context as Ctx

    before = cache.ResultCache(tmp_path, "0", []).key("f", Ctx())
    monkeypatch.setattr(cache, "code_fingerprint", lambda: "changed")
    assert cache.ResultCache(tmp_path, "0", []).key("f", Ctx()) != before


def _item(i, model, reward, status="available", matched=(), error=None, error_type=None):
    assessments = [
        {
            "id": c,
            "kind": "detector",
            "status": "match",
            "severity": "low",
            "score": 10,
            "expected_by": [],
            "evidence": [],
        }
        for c in matched
    ] + [
        {
            "id": "tamper.test_files",
            "kind": "detector",
            "severity": "high",
            "score": None,
            "status": "unknown" if matched or status != "available" else "no_match",
            "expected_by": [],
            "evidence": [],
        }
    ]
    return {
        "input_id": f"t{i}",
        "input_status": status,
        "input_error": error,
        "incomplete": bool(matched),
        "task": f"task{i % 3}",
        "reward": reward,
        "model_name": model,
        "cost_usd": 2.0,
        "error_type": error_type,
        "assessments": assessments,
    }


def test_trials_on_another_model_are_critical_dq_candidates():
    # Regression (TB4 Fable 5.1 row): 12 trials ran Opus 5 end to end (a safety-classifier
    # fallback); the brief only listed the second model in its agent line.
    from atif_scan.brief import brief, brief_text

    items = [_item(i, "main-model", 1.0) for i in range(8)]
    items += [_item(8, "fallback-model", 1.0), _item(9, "fallback-model", 0.0)]
    b = brief({"scanner_version": "dev", "inputs": items, "coverage": {}, "runs": []})
    mm, d = b["overview"]["model_mismatch"], b["overview"]["disqualification"]
    assert mm["expected"] == "main-model" and mm["trial_ids"] == ["t8", "t9"]
    assert d["candidate_ids"] == ["t8"]
    text = brief_text(b)
    assert "MODEL      ✗ critical  2 trial(s)" in text and "fallback-model 2" in text
    assert "(1 DQ candidates: 1 ran another model)" in text


def test_comparison_job_models_are_planned_not_substitutions():
    # Regression (TB2.1 5-agent job c8fcaaeb): each configured agent/model is ~20% of the
    # trials; reading the most common as "the" model made 1,347 rewarded trials critical.
    from atif_scan.brief import brief

    items = [_item(i, "model-a", 1.0) for i in range(4)]
    items += [_item(i, "model-b", 1.0) for i in range(4, 8)]
    items += [_item(8, "fallback-model", 1.0)]
    runs = [{"source": "harbor_hub", "job_id": "j", "configured_agents": 2}]
    b = brief({"scanner_version": "dev", "inputs": items, "coverage": {}, "runs": runs})
    mm, d = b["overview"]["model_mismatch"], b["overview"]["disqualification"]
    assert mm["planned_models"] == ["model-a", "model-b"] and mm["trial_ids"] == ["t8"]
    assert mm["other_models"] == {"fallback-model": 1}
    assert d["candidate_ids"] == ["t8"]
    two = [dict(r, configured_agents=3) for r in runs]  # every model planned: no mismatch
    b = brief({"scanner_version": "dev", "inputs": items, "coverage": {}, "runs": two})
    assert b["overview"]["model_mismatch"] is None


def test_configured_agents_from_job_config():
    from atif_scan.harbor_files import configured_agents

    agents = [{"name": "terminus-2", "model_name": m} for m in ("a/x", "b/y")]
    assert configured_agents({"agents": agents}) == 2
    assert configured_agents({"agents": "bad"}) is None and configured_agents({}) is None


def test_brief_says_why_trials_cannot_be_cleared_and_which_errors_occurred():
    from atif_scan.brief import brief, brief_text

    items = [
        _item(0, "m", 1.0, matched=("integrity.web_results_not_recorded",)),
        _item(1, "m", 1.0, status="unavailable_or_invalid", error="trace_too_large"),
        _item(2, "m", 0.0, error_type="UnknownApiError"),
        _item(3, "m", None, error_type="OutputTokenExceededError"),
    ]
    text = brief_text(
        brief({"scanner_version": "dev", "inputs": items, "coverage": {}, "runs": []})
    )
    assert (
        "can't be cleared: web results/URLs not recorded 1 · no trajectory (trace_too_large) 1"
        in text
    )
    assert "2 errored (50.0%): UnknownApiError 1, OutputTokenExceededError 1" in text


def _worked(i, calls, cost=None, tokens=True, reward=0.0, error_type=None, tools=None):
    return {
        "input_id": f"w{i}",
        "llm_calls": calls,
        "tool_calls": calls // 2 if tools is None else tools,
        "duration_sec": 60.0,
        "reward": reward,
        "error_type": error_type,
        "compacted": False,
        "input_tokens": 1000 * calls if tokens else None,
        "output_tokens": 10 * calls if tokens else None,
        "cache_tokens": None,
        "cost_usd": cost,
    }


def test_work_without_usage_is_flagged_and_estimated_not_silently_dropped():
    """Regression: an agent that died before writing usage did real (even rewarded) work,
    but the run total omitted it and the brief said "every trial priced"."""
    from atif_scan.estimates import unmetered_work

    a, b = 0.01, 0.0002  # true cost = a*calls + b*calls^2
    priced_ = [_worked(i, c, cost=a * c + b * c * c) for i, c in enumerate(range(10, 70, 2))]
    died = [
        _worked(100, 200, tokens=False, error_type="NonZeroAgentExitCodeError"),
        _worked(101, 40, tokens=False, reward=1.0, error_type="NonZeroAgentExitCodeError"),
    ]
    idle = _worked(102, 0, tokens=False, tools=0)  # nothing ran: not unmetered work
    hub_cost_only = _worked(103, 30, tokens=False, cost=0.5)  # cost known: not unmetered

    um = unmetered_work(priced_ + died + [idle, hub_cost_only])
    assert um["ids"] == ["w100", "w101"]
    assert (um["llm_calls"], um["tool_calls"], um["rewarded"], um["errored"]) == (240, 120, 1, 2)
    truth = sum(a * c + b * c * c for c in (200, 40))
    assert um["estimate_usd"] == pytest.approx(truth, rel=0.02)
    # Too few references: counted, but not guessed.
    few = unmetered_work(priced_[:5] + died)
    assert few["trials"] == 2 and few["estimate_usd"] is None and "fewer than" in few["method"]


def test_brief_warns_about_work_without_usage():
    from atif_scan.brief import brief, brief_text

    def item(i, calls, cost, tokens=True, reward=1.0, error_type=None):
        it = _item(i, "m", reward, error_type=error_type)
        it.update(_worked(i, calls, cost, tokens, reward, error_type))
        it["input_id"] = f"t{i}"
        return it

    items = [item(i, 20 + i, 0.1 + 0.01 * i) for i in range(25)]
    items.append(item(90, 120, None, tokens=False, error_type="NonZeroAgentExitCodeError"))
    text = brief_text(
        brief({"scanner_version": "dev", "inputs": items, "coverage": {}, "runs": []})
    )
    assert "every trial priced" not in text
    assert "every trial with usage priced" in text
    assert "⚠ 1 trial(s) did work but report no usage or cost (1 errored, 1 rewarded)" in text
    assert "120 LLM calls" in text and "not in the total" in text
    assert "1 rewarded, counted in RESULT without a cost" in text
    # A fully metered run keeps the plain "every trial priced".
    clean = brief_text(
        brief({"scanner_version": "dev", "inputs": items[:-1], "coverage": {}, "runs": []})
    )
    assert "every trial priced" in clean and "did work" not in clean


def _served(i, reward, steps, header="anthropic/main-model", cost=None, tokens=True):
    it = _item(i, header, reward)
    it.update(
        step_models=steps,
        cost_usd=cost,
        input_tokens=1_000_000 + 1000 * i if tokens else None,
        cache_tokens=500_000,
        output_tokens=20_000 + 100 * i if tokens else None,
    )
    return it


def test_step_models_reveal_a_fallback_the_header_hides():
    """Regression (TB2.1 Terminus 2 / Fable 5 row): every header said claude-fable-5, but
    77 trials' steps ran claude-opus-4-8 end to end and 2 switched mid-trial. Only those
    were priced, so the cost fit priced Fable at Opus rates."""
    from atif_scan.brief import brief, brief_text

    items = [_served(i, 1.0, {"main-model": 10}) for i in range(30)]
    items += [_served(30 + i, 1.0, {"fallback-model": 10}, cost=1.0) for i in range(25)]
    items.append(_served(60, 0.0, {"main-model": 4, "fallback-model": 6}, cost=1.0))
    b = brief({"scanner_version": "dev", "inputs": items, "coverage": {}, "runs": []})
    mm, ce = b["overview"]["model_mismatch"], b["cost_estimate"]
    assert mm["expected"] == "main-model" and len(mm["trial_ids"]) == 26
    assert mm["switched_ids"] == ["t60"] and mm["other_models"] == {"fallback-model": 26}
    assert len(b["overview"]["disqualification"]["candidate_ids"]) == 25
    # 26 priced trials ran the fallback: no fit on them, even though there are >= 20.
    assert ce["priced_other_model"] == 26 and ce["estimate_usd"] is None
    text = brief_text(b)
    assert "ran another model than main-model: fallback-model 26" in text
    assert "1 switched mid-trial" in text
    assert "(all of it on another model)" in text
    assert "not estimated: only 26 trial(s) on another model are priced" in text
    assert "unpriced tokens:" in text and "--price" in text


def test_model_names_differing_only_in_prefix_date_or_case_are_the_same_model():
    from atif_scan.brief import brief
    from atif_scan.report import model_key

    assert model_key("anthropic/Claude-Fable-5") == model_key("claude-fable-5")
    assert model_key("gpt-5.5-2026-04-23") == model_key("openai/gpt-5.5") == "gpt-5.5"
    assert model_key("claude-3-5-sonnet-20241022") == "claude-3-5-sonnet"
    assert model_key("claude-opus-4-8") != model_key("claude-fable-5")
    items = [_served(i, 1.0, {"openai/gpt-5.5": 5}, header="gpt-5.5") for i in range(3)]
    items += [_served(3, 1.0, {"gpt-5.5-2026-04-23": 5}, header="openai/gpt-5.5")]
    items += [_served(4, 1.0, None, header="GPT-5.5")]  # no step models: header decides
    b = brief({"scanner_version": "dev", "inputs": items, "coverage": {}, "runs": []})
    assert b["overview"]["model_mismatch"] is None


def test_step_model_names_are_parsed_label_safe(tmp_path, capsys):
    """Claude Code's `<synthetic>` placeholder (locally generated messages) is not a model."""
    trial = tmp_path / "run" / "t1"
    trial.mkdir(parents=True)
    steps = [
        {"step_id": 1, "source": "user", "message": "do it"},
        {"step_id": 2, "source": "agent", "message": "ok", "model_name": "claude-main-1"},
        {"step_id": 3, "source": "agent", "message": "ok", "model_name": "claude-other-2"},
        {"step_id": 4, "source": "agent", "message": "err", "model_name": "<synthetic>"},
        {"step_id": 5, "source": "agent", "message": "ok", "model_name": "claude-main-1"},
    ]
    (trial / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.2",
                "session_id": "s",
                "agent": {"name": "a", "version": "1", "model_name": "anthropic/claude-main-1"},
                "steps": steps,
            }
        )
    )
    main([str(tmp_path / "run"), "--format", "json"])
    item = json.loads(capsys.readouterr().out)["inputs"][0]
    assert item["step_models"] == {"claude-main-1": 2, "claude-other-2": 1}
