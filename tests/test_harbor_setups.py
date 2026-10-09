"""Comparison jobs: several configured harness/model setups in one job (synthetic).

Regression (a TB2.1 Hub comparison job: terminus-2 with three models, claude-code and
codex, 445 trials each): the job's one accuracy pooled five different agents, and the
Hub's model or agent grouping still blended two or three. Each setup now gets its own
score; the setup is what the job configured, so a model fallback inside one setup
stays a model mismatch rather than a setup of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from atif_scan.output.brief_view.run import _setup_texts
from atif_scan.output.overview import overview, overview_text
from atif_scan.sources.harbor.hub import inspect_job

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc

JOB = "00000000-0000-4000-8000-0000000000ee"


def row(n: int, agent: str, model: str, task: str, reward: float) -> Doc:
    return {
        "id": f"00000000-0000-4000-8000-{n:012d}",
        "name": f"{task}__{n:08x}",
        "task_name": f"terminal-bench/{task}",
        "agent_name": agent,
        "model_name": model,
        "reward": reward,
        "started_at": "2026-04-30T06:00:00+00:00",
    }


def comparison() -> list[Doc]:
    rows, n = [], 0
    for agent, model, rewards in (
        ("terminus-2", "gpt-5.5", [1, 1, 1, 0]),
        ("codex", "gpt-5.5", [1, 1, 1, 1]),
        ("terminus-2", "claude-opus-4-7", [1, 0, 0, 0]),
    ):
        for task, reward in zip(("a", "a", "b", "b"), rewards, strict=True):
            n += 1
            rows.append(row(n, agent, model, task, reward))
    return rows


@dataclass(frozen=True)
class ListingCLI:
    rows: list[Doc]

    def json(self, *args: str) -> object:
        if args[:3] == ("hub", "job", "show"):
            return {"name": "demo", "n_planned_trials": len(self.rows)}
        if args[:3] == ("hub", "job", "trials"):
            return {"items": self.rows, "total_pages": 1}
        raise AssertionError(args)

    def run(self, *args: str) -> str:
        raise AssertionError(args)


def card(rows: list[Doc], **extra: object) -> Doc:
    job = inspect_job(f"harbor://jobs/{JOB}", cli=ListingCLI(rows))
    listed = {"input_status": "listed", "incomplete": False, "assessments": [], **extra}
    items = [{**t, **listed} for t in job["trials"]]
    return overview({"inputs": items, "runs": [job["run"]]}, scanned=False)


def test_each_configured_setup_gets_its_own_score():
    ov = card(comparison())
    assert ov["accuracy"][0] == 66.67  # pooled: 8 of 12, not a score for any agent
    scores = {(s["agent"], s["model"]): s["accuracy"][0] for s in ov["setups"]}
    assert scores == {
        ("codex", "gpt-5.5"): 100.0,
        ("terminus-2", "claude-opus-4-7"): 25.0,
        ("terminus-2", "gpt-5.5"): 75.0,
    }
    # Same model, different harness: not merged (the Hub's model grouping merges them).
    assert len({s["model"] for s in ov["setups"]}) == 2 and len(ov["setups"]) == 3
    text = "\n".join(overview_text(ov))
    assert "pooled across setups: not one score" in text
    assert "codex / gpt-5.5: 100.0%" in text


def test_the_brief_leads_with_setups_and_labels_the_pool():
    rows = card(comparison())["setups"]
    texts = _setup_texts(rows, "66.7% · 8 of 12 scored trials rewarded", has_se=False)
    assert texts[0].startswith("codex / gpt-5.5: 100.0%")
    assert "3 harness/model setups configured: pooled 66.7%" in texts[-1]


def test_one_setup_is_not_split():
    rows = [r for r in comparison() if r["agent_name"] == "codex"]
    assert card(rows)["setups"] is None


def test_a_model_fallback_is_not_a_setup():
    # The listing says what was configured; a trajectory that ran another model (a
    # safety fallback) is a model mismatch inside the one setup.
    rows = [r for r in comparison() if r["agent_name"] == "codex"]
    assert card(rows, model_name="claude-opus-4-7")["setups"] is None


def served(rows: list[Doc], fallback: dict[int, str] | None = None) -> Doc:
    """The comparison job scanned: each trajectory ran its configured model, except the
    listed trials (by row number), which ran a fallback."""
    job = inspect_job(f"harbor://jobs/{JOB}", cli=ListingCLI(rows))
    by_name = {f"{r['task_name'].rsplit('/', 1)[-1]}__{n:08x}": n for n, r in enumerate(rows, 1)}
    items = []
    for t in job["trials"]:
        n = by_name[t["input_id"]]
        model = (fallback or {}).get(n) or t["configured_model"]
        items.append({**t, "input_status": "listed", "incomplete": False, "assessments": []})
        items[-1]["model_name"] = model
    return overview({"inputs": items, "runs": [job["run"]]}, scanned=True)


def test_each_setup_is_checked_for_a_model_fallback_on_its_own():
    # Regression (TB2.1 job c8fcaaeb, read without its job config): the most common model
    # was "the" model, so every rewarded trial of the other setups was a critical DQ.
    assert served(comparison())["model_mismatch"] is None
    # Row 9 is terminus-2/claude-opus-4-7, rewarded, but ran gpt-5.5: gpt-5.5 is another
    # setup's model, not this one's.
    ov = served(comparison(), {9: "gpt-5.5"})
    mm = ov["model_mismatch"]
    assert mm["trial_ids"] == ["a__00000009"]
    assert mm["other_models"] == {"gpt-5.5": 1} and len(mm["rewarded_ids"]) == 1
    assert mm["planned_models"] == ["claude-opus-4-7", "gpt-5.5"]
    assert ov["disqualification"]["by_model_only"] == 1


def test_a_saved_listing_keeps_each_trials_setup():
    # Regression: the saved Hub listing dropped agent_name/model_name, so a cached rescan
    # of a comparison job lost its setups.
    import json

    from atif_scan.sources.harbor.hub import _reduced
    from atif_scan.sources.harbor.listing import saved_listing

    saved = {"version": 1, "job": JOB, **_reduced({"name": "demo"}, comparison())}
    _, facts = saved_listing(json.dumps(saved).encode())
    setups = {(f["configured_agent"], f["configured_model"]) for f in facts.values()}
    assert setups == {
        ("terminus-2", "gpt-5.5"),
        ("codex", "gpt-5.5"),
        ("terminus-2", "claude-opus-4-7"),
    }
