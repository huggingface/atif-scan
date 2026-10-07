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
