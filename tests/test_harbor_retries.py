"""Harbor retry chains: `<trial>__retry_<id>` attempts of one trial (synthetic listings).

Regression (a TB2.1 Hub job, 546 trials for 89 tasks x 5): Harbor re-ran each trial
whose agent crashed at start-up and kept every attempt in the job, so tasks had 5 to 10
trials and the accuracy (71.98%) scored each crash as well as the attempt that replaced
it. Only each chain's last attempt is scored now (88.31% there), on the stated
assumption that the attempts before it were infrastructure failures; attempts that argue
against it are listed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atif_scan.output.brief_view.evidence import _retry_texts
from atif_scan.output.overview import overview, overview_text
from atif_scan.sources.harbor.hub import inspect_job
from atif_scan.sources.harbor.listing import retry_chains, retry_summary

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc

JOB = "00000000-0000-4000-8000-0000000000ff"
CRASH = "NonZeroAgentExitCodeError"


def row(n: int, name: str, start: str | None, reward: float, error: str | None = None) -> Doc:
    return {
        "id": f"00000000-0000-4000-8000-{n:012d}",
        "name": name,
        "task_name": f"terminal-bench/{name.split('__', maxsplit=1)[0]}",
        "reward": reward,
        "error_type": error,
        "started_at": f"2026-07-09T{start}:00+00:00" if start else None,
    }


def job_rows() -> list[Doc]:
    return [
        # Crashed twice; the suffix-free name is the *second* attempt, as Harbor names it.
        row(1, "alpha__aa11bb22__retry_0a0a0a0a", "00:00", 0, CRASH),
        row(2, "alpha__aa11bb22", "01:00", 0, CRASH),
        row(3, "alpha__aa11bb22__retry_0b0b0b0b", "02:00", 1),
        row(4, "alpha__cc33dd44", "00:00", 1),
        row(5, "beta__ee55ff66", "00:00", 0),
        row(6, "beta__0011aabb", "00:00", 1),
    ]


@dataclass(frozen=True)
class ListingCLI:
    rows: list[Doc]
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def json(self, *args: str) -> object:
        if args[:3] == ("hub", "job", "show"):
            return {"name": "demo", "n_planned_trials": len(self.rows)}
        if args[:3] == ("hub", "job", "trials"):
            return {"items": self.rows, "total_pages": 1}
        raise AssertionError(args)

    def run(self, *args: str) -> str:
        raise AssertionError(args)


def card(rows: list[Doc]) -> Doc:
    """The --inspect scorecard: the listing only, like `cli.inspect._harbor_cards`."""
    job = inspect_job(f"harbor://jobs/{JOB}", cli=ListingCLI(rows))
    items = [
        dict(t, input_status="listed", incomplete=False, assessments=[]) for t in job["trials"]
    ]
    return overview({"inputs": items, "runs": [job["run"]]}, scanned=False)


def test_only_the_last_attempt_of_a_chain_is_superseding():
    facts = retry_chains(job_rows())
    assert {k[-1]: v["retry_superseded"] for k, v in facts.items()} == {
        "1": True,
        "2": True,  # suffix-free, but not last
        "3": False,
    }
    assert all(v["retry_attempts"] == 3 for v in facts.values())
    assert retry_summary(job_rows()) == {
        "chains": 1,
        "attempts": 3,
        "superseded": 2,
        "unordered_chains": 0,
        "superseded_errors": {CRASH: 2},
        "superseded_without_error": 0,
        "superseded_with_usage": 0,
    }


def test_superseded_attempts_are_present_but_not_scored():
    ov = card(job_rows())
    assert ov["trials"]["present"] == 6 and ov["trials"]["errored"] == 2
    assert ov["accuracy"][0] == 75.0  # alpha 2/2, beta 1/2 (every attempt: 3/6)
    assert ov["tasks"]["max_trials"] == 2
    rt = ov["retries"]
    assert rt["assumption"] == "superseded_attempts_were_infrastructure_failures"
    assert (rt["superseded"], rt["every_attempt_accuracy"]) == (2, 50.0)
    assert rt["superseded_without_error"] == [] and rt["superseded_with_work"] == []
    text = "\n".join(overview_text(ov))
    assert "2 earlier attempts not scored (assumed infrastructure failures)" in text


def test_a_retried_result_or_real_work_argues_against_the_assumption():
    rows = job_rows()
    rows[0]["error_type"] = None  # a clean failure, then retried: a re-roll, not a crash
    rows[1]["cost_usd"] = 0.4  # paid model calls before it crashed
    rt = card(rows)["retries"]
    assert rt["superseded_without_error"] == ["alpha__aa11bb22__retry_0a0a0a0a"]
    assert rt["superseded_with_work"] == ["alpha__aa11bb22"]
    texts = _retry_texts(rt)
    assert any("ended without an error" in t for t in texts)
    assert any("did agent work" in t for t in texts)


def test_a_superseded_attempt_with_agent_time_counts_as_work():
    # A full scan reads each trial's result.json: an agent that ran minutes before it
    # crashed did more than fail to start (TB2.1: two attempts ran 266 s and 343 s).
    items = [
        {
            "input_id": "alpha__1",
            "retry_superseded": True,
            "error_type": CRASH,
            "reward": 0,
            "agent_duration_sec": 266.5,
        },
        {"input_id": "alpha__1__retry_ab12cd34", "retry_superseded": False, "reward": 1},
    ]
    docs = [dict(i, input_status="available", incomplete=False, assessments=[]) for i in items]
    rt = overview({"inputs": docs, "runs": []}, scanned=False)["retries"]
    assert rt["superseded_with_work"] == ["alpha__1"]


def test_unordered_chains_keep_every_attempt_scored():
    rows = job_rows()
    rows[2]["started_at"] = rows[1]["started_at"]  # tied: which one was last?
    ov = card(rows)
    assert ov["retries"]["superseded"] == 0 and ov["retries"]["unordered_chains"] == 1
    assert ov["accuracy"][0] == 50.0
    assert any("can't be ordered" in t for t in _retry_texts(ov["retries"]))


def test_no_retry_names_no_retry_section():
    rows = [r for r in job_rows() if "alpha__aa11bb22" not in r["name"]]
    assert retry_chains(rows) == {}
    assert card(rows)["retries"] is None
