"""Harbor Hub jobs (via a fake `harbor` CLI) and the run overview scorecard."""

from __future__ import annotations

import json
import stat
import sys
import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from test_brief import section

from atif_scan.cli import main
from atif_scan.output.overview import accuracy
from atif_scan.sources.harbor.files import overrides
from atif_scan.sources.harbor.hub import job_id

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc

JOB = "1fead079-8e3b-4fef-b893-1944558a3949"
SECRET = "sk-live-abcdefghijklmnop1234"


def trajectory(command):
    return {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {
                "source": "agent",
                "message": "working",
                "tool_calls": [
                    {
                        "tool_call_id": "c1",
                        "function_name": "bash",
                        "arguments": {"command": command},
                    }
                ],
            }
        ],
        "final_metrics": {"total_cost_usd": 9.99},  # Hub cost wins when present
    }


def trial(i, task, reward, error=None, cost=0.5, tokens=1000, has_trajectory=True):
    return {
        "id": f"00000000-0000-4000-8000-{i:012d}",
        "name": f"{task}__T{i}",
        "task_name": f"terminal-bench/{task}",
        "reward": reward,
        "error_type": error,
        "status": "completed",
        "cost_usd": cost,
        "input_tokens": tokens,
        "cache_tokens": tokens // 2,
        "output_tokens": 100,
        "started_at": "2026-08-27T16:12:00+00:00",
        "finished_at": "2026-08-27T16:17:00+00:00",
        "config_values": {},
        "_trajectory": has_trajectory,
    }


TRIALS = [
    trial(1, "alpha", 1),
    trial(2, "alpha", 0),
    trial(3, "beta", 1),  # rewarded + reward-file write -> DQ candidate
    # No Hub cost: the trajectory's own cost (9.99) is reported instead.
    trial(4, "beta", None, error="AgentTimeoutError", cost=None),
    trial(5, "gamma", 1, cost=0, tokens=500),  # tokens but no cost
    trial(6, "gamma", 1, has_trajectory=False),  # rewarded, trajectory unavailable
]
COMMANDS = {3: "echo 1 > /logs/verifier/reward.txt"}

FAKE = textwrap.dedent(
    """
    import json, os, sys, pathlib
    data = json.loads(pathlib.Path(os.environ["FAKE_HARBOR_DATA"]).read_text())
    with open(os.environ["FAKE_HARBOR_LOG"], "a") as log:
        log.write(" ".join(sys.argv[1:]) + "\\n")
    args = sys.argv[1:]
    if os.environ.get("FAKE_HARBOR_FAIL"):
        sys.stderr.write("401 https://hub.example/x?token=" + os.environ["FAKE_HARBOR_FAIL"])
        sys.exit(1)
    if args[:3] == ["hub", "job", "show"]:
        print(json.dumps(data["show"]))
    elif args[:3] == ["hub", "job", "trials"]:
        page = int(args[args.index("--page") + 1]); size = 4
        items = data["trials"][(page - 1) * size: page * size]
        pages = (len(data["trials"]) + size - 1) // size
        print(json.dumps({"items": items, "total": len(data["trials"]), "total_pages": pages}))
    elif args[:3] == ["hub", "trial", "download"]:
        row = next(t for t in data["trials"] if t["id"] == args[3])
        if not row["_trajectory"]:
            sys.exit(1)
        out = pathlib.Path(args[args.index("-o") + 1]) / row["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "trajectory.json").write_text(json.dumps(data["trajectories"][row["id"]]))
    elif args[:3] == ["hub", "job", "download"]:
        root = pathlib.Path(args[args.index("-o") + 1]) / "my-job"
        for row in data["trials"]:
            if row["_trajectory"]:
                d = root / row["name"] / "agent"; d.mkdir(parents=True)
                (d / "trajectory.json").write_text(json.dumps(data["trajectories"][row["id"]]))
    else:
        sys.exit(2)
    """
)


@pytest.fixture
def harbor(tmp_path, monkeypatch):
    data = {
        "show": {
            "name": "tb21-demo-job",
            "n_planned_trials": 8,
            "n_total_trials": 6,
            "n_completed_trials": 6,
            "n_errors": 1,
            "cost_usd": 2.0,
            "config": {
                "n_attempts": 2,
                "datasets": [
                    {
                        "name": "terminal-bench/terminal-bench-2-1",
                        "ref": "sha256:" + "7d" * 32,
                        "task_names": ["terminal-bench/alpha", "terminal-bench/beta"],
                    }
                ],
                "agents": [{"override_timeout_sec": 21600.0, "timeout_multiplier": 1.0}],
            },
        },
        "trials": TRIALS,
        "trajectories": {
            t["id"]: trajectory(COMMANDS.get(i + 1, "ls")) for i, t in enumerate(TRIALS)
        },
    }
    (tmp_path / "data.json").write_text(json.dumps(data))
    script = tmp_path / "harbor"
    script.write_text(f"#!{sys.executable}\n{FAKE}")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.log"
    monkeypatch.setenv("ATIF_SCAN_HARBOR", str(script))
    monkeypatch.setenv("FAKE_HARBOR_DATA", str(tmp_path / "data.json"))
    monkeypatch.setenv("FAKE_HARBOR_LOG", str(log))
    return log


def calls(log, verb):
    return [c for c in log.read_text().splitlines() if verb in c] if log.exists() else []


@pytest.mark.parametrize(
    "value",
    [
        f"harbor://jobs/{JOB}",
        f"https://hub.harborframework.com/jobs/{JOB}",
        f"https://hub.harborframework.com/jobs/{JOB.upper()}/trials/abc?tab=x",
    ],
)
def test_job_references(value):
    assert job_id(value) == JOB


def test_invalid_job_reference(capsys, monkeypatch):
    # Validated before looking for the harbor CLI (CI has none installed).
    monkeypatch.delenv("ATIF_SCAN_HARBOR", raising=False)
    monkeypatch.setenv("PATH", "")
    assert main(["harbor://jobs/not-a-uuid"]) == 2
    err = capsys.readouterr().err
    assert "invalid_harbor_reference" in err and "harbor://rows/<uuid>" in err


def test_overrides_follow_leaderboard_rules():
    config = {
        "timeout_multiplier": 1.0,
        "agents": [{"override_timeout_sec": 10, "max_timeout_sec": None}],
        "environment": {"override_cpus": 4},
        "verifier": {"agent_timeout_multiplier": 2.0},
    }
    assert overrides(config) == [
        "agents[].override_timeout_sec",
        "environment.override_cpus",
        "verifier.agent_timeout_multiplier",
    ]


def test_accuracy_matches_leaderboard_formula():
    by_task = {"a": [True, False], "b": [True, True, False]}
    result = accuracy(by_task)
    assert result is not None
    acc, se = result
    assert acc == 60.0
    # s^2 = (1/n^2) * (0.5*0.5/1 + (2/3)(1/3)/2)
    assert se == round(100 * ((0.25 + (2 / 9) / 2) / 4) ** 0.5, 2)


def test_scan_harbor_job_overview(harbor, capsys):
    assert main([f"harbor://jobs/{JOB}", "--overview", "--format", "json"]) == 0
    ov = json.loads(capsys.readouterr().out)
    assert ov["kind"] == "overview"
    t = ov["trials"]
    assert (t["present"], t["planned"], t["missing"], t["errored"]) == (6, 8, 2, 1)
    assert t["error_types"] == {"AgentTimeoutError": 1} and t["without_trajectory"] == 1
    assert ov["tasks"]["count"] == 3 and ov["tasks"]["expected_per_task"] == 2
    # 4 successes / 6 trials; errored counts as 0.
    assert ov["accuracy"][0] == round(100 * 4 / 6, 2)
    d = ov["disqualification"]
    assert d["candidate_ids"] == ["beta__T3"] and d["rate_pct"] == round(100 / 6, 2)
    assert d["accuracy_if_disqualified"][0] == 50.0
    assert d["rewarded_not_cleared_ids"] == ["gamma__T6"]  # rewarded, no trajectory
    c = ov["cost"]
    # Hub costs 0.5 x 4 + 0 (trial 5) + trial 4's trajectory cost 9.99: cost is decided
    # separately from tokens, so recorded tokens don't hide the trajectory's cost.
    assert c["total_usd"] == 11.99 and c["missing"] == 0 and c["tokens_without_cost"] == 1
    assert c["hub_job_total_usd"] == 2.0
    assert ov["overrides"] == ["agents[].override_timeout_sec"]
    assert len(calls(harbor, "trial download")) == 6


def test_summary_text_and_task_from_hub(harbor, capsys, tmp_path):
    args = [f"harbor://jobs/{JOB}", "--summary", "--format", "text", "--expect-tasks", "4"]
    assert main([*args, "--plugin", "atif_scan.packs.tb21:checks"]) == 0
    out = capsys.readouterr().out
    assert "run overview" in out and "tb21-demo-job" in out
    assert "6 present / 8 planned · 2 missing" in out
    assert "1 of 4 expected tasks missing" in out
    assert "0 trial(s) missing cost" in out and "1 with tokens but no cost" in out
    assert "overrides set: agents[].override_timeout_sec" in out
    assert "tamper.reward_write" in out and "beta__T3" in out
    main([f"harbor://jobs/{JOB}", "--format", "json"])
    items = {i["input_id"]: i for i in json.loads(capsys.readouterr().out)["inputs"]}
    assert items["beta__T3"]["task"] == "beta"  # recorded by the Hub, no --task-from needed
    assert items["alpha__T1"]["cost_usd"] == 0.5  # Hub cost preferred over final_metrics
    # ...and checked against it: the trajectory says 9.99.
    assert items["alpha__T1"]["cost_vs_trajectory"] == "differs"
    assert items["beta__T4"]["cost_usd"] == 9.99  # no Hub cost: the trajectory's
    assert items["beta__T4"]["cost_vs_trajectory"] is None


def test_sync_to_reuses_downloads(harbor, tmp_path, capsys):
    keep = tmp_path / "keep"
    main([f"harbor://jobs/{JOB}", "--sync-to", str(keep), "--format", "json"])
    first = len(calls(harbor, "trial download"))
    main([f"harbor://jobs/{JOB}", "--sync-to", str(keep), "--format", "json"])
    capsys.readouterr()
    # Only the trial without a trajectory is retried.
    assert len(calls(harbor, "trial download")) - first == 1
    assert (keep / "harbor" / JOB / "alpha__T1" / "trajectory.json").is_file()


def test_full_archive_mode(harbor, capsys):
    assert main([f"harbor://jobs/{JOB}", "--full", "--overview", "--format", "json"]) == 0
    ov = json.loads(capsys.readouterr().out)
    assert ov["disqualification"]["candidate_ids"] == ["beta__T3"]
    assert calls(harbor, "job download") and not calls(harbor, "trial download")


def test_inspect_harbor_job_downloads_nothing(harbor, capsys):
    assert (
        main(["--inspect", f"https://hub.harborframework.com/jobs/{JOB}", "--format", "text"]) == 0
    )
    out = capsys.readouterr().out
    assert "6 present / 8 planned" in out and "DQ" not in out
    assert not calls(harbor, "download")


def test_harbor_errors_are_withheld(harbor, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_HARBOR_FAIL", SECRET)
    assert main([f"harbor://jobs/{JOB}"]) == 2
    captured = capsys.readouterr()
    assert "harbor_command_failed" in captured.err and SECRET not in captured.out + captured.err


def test_missing_harbor_cli(monkeypatch, capsys):
    monkeypatch.delenv("ATIF_SCAN_HARBOR", raising=False)
    monkeypatch.setenv("PATH", "")
    assert main([f"harbor://jobs/{JOB}"]) == 2
    assert "harbor_cli_not_found" in capsys.readouterr().err


def test_local_traces_use_final_metrics_and_exception_marker(tmp_path, capsys):
    trial_dir = tmp_path / "job" / "alpha__X1"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(trajectory("ls")))
    (trial_dir / "exception.txt").write_text("boom")
    assert (
        main([str(tmp_path / "job"), "--task-from", "trial-dir", "--overview", "--format", "json"])
        == 0
    )
    ov = json.loads(capsys.readouterr().out)
    assert ov["cost"]["total_usd"] == 9.99 and ov["trials"]["errored"] == 1
    assert ov["trials"]["error_types"] == {"exception": 1}


# --- leaderboard rows ---------------------------------------------------------------------

ROW = "0000feed-0000-4000-8000-00000000c0de"
JOB2 = "2c89a14f-d14a-4ea8-ad8a-d08d90c67a5d"

ROW_FAKE = textwrap.dedent(
    """
    import json, os, sys, pathlib
    data = json.loads(pathlib.Path(os.environ["FAKE_HARBOR_DATA"]).read_text())
    with open(os.environ["FAKE_HARBOR_LOG"], "a") as log:
        log.write(" ".join(sys.argv[1:]) + "\\n")
    args = sys.argv[1:]
    def job_of(tid):
        return next((j for j, ts in data["jobs"].items() if any(t["id"] == tid for t in ts)), None)
    if args[:4] == ["hub", "leaderboard", "row", "show"]:
        print(json.dumps(data["row"]))
    elif args[:5] == ["hub", "leaderboard", "row", "trial", "list"]:
        items = [{"trial_id": t} for t in data["row_trials"]]
        print(json.dumps({"items": items, "total_pages": 1}))
    elif args[:3] == ["hub", "trial", "show"]:
        print(json.dumps({"id": args[3], "job_id": job_of(args[3])}))
    elif args[:3] == ["hub", "job", "show"]:
        print(json.dumps({"name": "job-" + args[3][:4], "config": {"n_attempts": 2}}))
    elif args[:3] == ["hub", "job", "trials"]:
        print(json.dumps({"items": data["jobs"][args[3]], "total_pages": 1}))
    elif args[:3] == ["hub", "trial", "download"]:
        row = next(t for ts in data["jobs"].values() for t in ts if t["id"] == args[3])
        out = pathlib.Path(args[args.index("-o") + 1]) / row["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "trajectory.json").write_text(json.dumps(data["trajectory"]))
    else:
        sys.exit(2)
    """
)


@pytest.fixture
def row_harbor(tmp_path, monkeypatch):
    job1 = [trial(i, "alpha", 1) for i in range(1, 4)]  # the row uses trials 1-2 only
    job2 = [trial(10, "beta", 0), trial(11, "beta", 1)]
    data = {
        "row": {
            "rank": 9,
            "metadata": {
                "agent_display": {"label": "Demo CLI"},
                "model_display": {"label": "Model X"},
                "reasoning_effort": "high",
            },
            "metrics": {
                "accuracy": 50.0,
                "n_trials": 5,
                "total_cost_usd": 2.5,
                "reward_hacks": 20.0,
            },
        },
        "row_trials": [t["id"] for t in job1[:2]]
        + [t["id"] for t in job2]
        + ["00000000-0000-4000-8000-999999999999"],
        "jobs": {JOB: job1, JOB2: job2},
        "trajectory": json.loads(json.dumps(trajectory("ls"))),
    }
    (tmp_path / "data.json").write_text(json.dumps(data))
    script = tmp_path / "harbor"
    script.write_text(f"#!{sys.executable}\n{ROW_FAKE}")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.log"
    monkeypatch.setenv("ATIF_SCAN_HARBOR", str(script))
    monkeypatch.setenv("FAKE_HARBOR_DATA", str(tmp_path / "data.json"))
    monkeypatch.setenv("FAKE_HARBOR_LOG", str(log))
    return log


@pytest.mark.parametrize(
    "value",
    [
        f"https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest/leaderboards/main/rows/{ROW}",
        f"harbor://rows/{ROW}",
    ],
)
def test_leaderboard_row_scans_exactly_its_trials(row_harbor, capsys, value):
    assert main([value, "--format", "json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    ids = sorted(i["input_id"] for i in doc["inputs"])
    assert ids == ["alpha__T1", "alpha__T2", "beta__T10", "beta__T11"]  # T3 isn't in the row
    run = doc["runs"][0]
    assert run["source"] == "harbor_leaderboard_row" and run["planned_trials"] == 5
    assert run["unresolved_trials"] == 1  # listed by the row, in no job
    assert run["leaderboard"]["jobs"] == sorted([JOB, JOB2]) or set(run["leaderboard"]["jobs"]) == {
        JOB,
        JOB2,
    }
    # One `trial show` per job, not per trial (plus the unresolved one).
    assert len(calls(row_harbor, "trial show")) == 3


def test_leaderboard_row_brief_compares_with_reported(row_harbor, capsys):
    main([f"harbor://rows/{ROW}", "--format", "text"])
    out = capsys.readouterr().out
    flat = " ".join(out.split())  # unwrapped: phrases may span lines
    assert "RUN        leaderboard row #9 · id 0000feed" in out
    assert "leaderboard row Demo CLI / Model X (high) · jobs 1fead079, 2c89a14f" in out
    assert (
        "· the leaderboard reports 50.0% after its own reward-hack disqualifications"
        " (20.0% of trials) over 5 trials"
    ) in flat
    assert "(⚠ 4 trials scanned here)" in flat  # the row reports 5, the scan found 4
    assert "⚠ 1 row trial not found in any job" in section(out, "SCORE")


def test_inspect_leaderboard_row(row_harbor, capsys):
    assert main(["--inspect", f"harbor://rows/{ROW}", "--format", "json"]) == 0
    card = json.loads(capsys.readouterr().out)["harbor_jobs"][0]
    assert card["trials"]["present"] == 4 and not calls(row_harbor, "download")


def test_leaderboard_row_reports_progress_without_identifiers(row_harbor, tmp_path):
    from atif_scan.sources.harbor.hub import harbor_sources

    messages = []
    harbor_sources(f"harbor://rows/{ROW}", tmp_path / "dl", progress=messages.append)
    assert messages[0] == "listing leaderboard row"
    assert "leaderboard row lists 5 trials · finding their jobs" in messages
    assert "listing job 1 trials · page 1/1 · 3 trials" in messages
    downloads = [m for m in messages if m.startswith("downloading")]
    assert downloads[-1].startswith("downloading trajectories 2/2")  # per job
    joined = "\n".join(messages)
    assert (
        ROW not in joined and JOB not in joined and "alpha" not in joined and "0000" not in joined
    )

    # A resumed run downloads nothing and says nothing about downloading.
    again = []
    harbor_sources(f"harbor://rows/{ROW}", tmp_path / "dl", progress=again.append)
    assert not [m for m in again if m.startswith("downloading")]


def test_harbor_download_progress_counts_failures(harbor, tmp_path):
    from atif_scan.sources.harbor.hub import harbor_sources

    messages = []
    harbor_sources(f"harbor://jobs/{JOB}", tmp_path / "dl", workers=1, progress=messages.append)
    assert "listing job trials · page 2/2 · 6 trials" in messages
    assert messages[-1] == "downloading trajectories 6/6 · 1 failed"  # trial 6 has none


def test_harbor_status_line_only_on_terminals(harbor, capsys, monkeypatch):
    main([f"harbor://jobs/{JOB}", "--format", "json", "--no-cache"])
    assert capsys.readouterr().err == ""  # piped: silent
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    main([f"harbor://jobs/{JOB}", "--format", "json", "--no-cache", "--refresh"])
    err = capsys.readouterr().err
    assert "\r\033[Katif-scan: listing job" in err
    assert "atif-scan: downloading trajectories 6/6" in err and "scanning 6/6" in err
    assert JOB not in err


@dataclass(frozen=True)
class _PagedCLI:
    """Stub `harbor` CLI: one job of 25 trials, served `size` per page; it runs nothing."""

    size: int = 10
    pages: list[int] = field(default_factory=list)
    trials: list[Doc] = field(
        default_factory=lambda: [{"id": f"t{i:02d}", "name": f"task__T{i:02d}"} for i in range(25)]
    )

    def run(self, *args: str) -> str:
        raise AssertionError(args)

    def json(self, *args: str) -> object:
        if args[:3] == ("hub", "job", "show"):
            return {"name": "paged-job"}
        page, limit = int(args[args.index("--page") + 1]), int(args[args.index("--limit") + 1])
        self.pages.append(page)
        size = min(limit, self.size)
        total_pages = (len(self.trials) + size - 1) // size
        return {"items": self.trials[(page - 1) * size : page * size], "total_pages": total_pages}


def test_job_listing_asks_for_large_pages_and_follows_a_smaller_server_cap():
    from atif_scan.sources.harbor.hub import PAGE_SIZE, listing

    assert PAGE_SIZE == 1000  # a 2,225-trial job is 3 calls, not 23
    cli = _PagedCLI(size=10)  # the server caps pages lower: still lists everything
    _, rows = listing(cli, JOB)
    assert len(rows) == 25 and cli.pages == [1, 2, 3]


def test_row_listing_stops_paging_once_the_rows_trials_are_found():
    from atif_scan.sources.harbor.hub import listing

    cli = _PagedCLI(size=10)  # the row holds one agent's share, all on page 1
    _, rows = listing(cli, JOB, want={"t01", "t05"})
    assert cli.pages == [1] and {"t01", "t05"} <= {r["id"] for r in rows}
    cli = _PagedCLI(size=10)  # a wanted trial on the last page: every page is read
    listing(cli, JOB, want={"t01", "t24"})
    assert cli.pages == [1, 2, 3]


def test_local_copy_of_a_hub_job_keeps_hub_facts(harbor, tmp_path, capsys):
    """A synced job scanned from its folder reports the same task/reward/error facts as a
    harbor:// scan (the listing is saved next to the trajectories): reward-gated rules and
    DQ candidates don't turn unknown on a rescan."""
    from atif_scan.sources.harbor.listing import SAVED_LISTING

    data = json.loads((tmp_path / "data.json").read_text())
    data["show"]["config"]["agents"][0]["env"] = {"OPENAI_API_KEY": "sk-proj-secret1234567890"}
    (tmp_path / "data.json").write_text(json.dumps(data))
    sync = tmp_path / "sync"
    assert main([f"harbor://jobs/{JOB}", "--sync-to", str(sync), "--format", "json"]) in (0, 1)
    live = json.loads(capsys.readouterr().out)
    saved = sync / "harbor" / JOB / SAVED_LISTING
    assert saved.is_file() and "sk-proj" not in saved.read_text()  # env/keys aren't kept

    assert main([str(sync / "harbor" / JOB), "--format", "json"]) in (0, 1)
    local = json.loads(capsys.readouterr().out)
    keys = ("task", "reward", "error_type", "cost_usd", "hub_trial_id")
    facts = {
        i["input_id"]: tuple(i[k] for k in keys)
        for i in live["inputs"]
        if i["input_status"] == "available"
    }
    assert {i["input_id"]: tuple(i[k] for k in keys) for i in local["inputs"]} == facts
    assert any(r for _, r, *_ in facts.values())  # the fixture has rewards to lose
    rewarded = {
        i["input_id"]: next(a["status"] for a in i["assessments"] if a["id"] == "context.rewarded")
        for i in local["inputs"]
    }
    assert "unknown" not in {s for k, s in rewarded.items() if facts[k][1] is not None}
    (run,) = [r for r in local["runs"] if r.get("source") == "harbor_hub"]
    assert run["job_id"] == JOB and run["overrides"] == live["runs"][0]["overrides"]
    assert run["listed_trials"] == len(live["inputs"]) > len(local["inputs"])  # T6: none

    # A damaged or foreign sidecar is ignored, never trusted or fatal.
    for junk in ("not json", json.dumps({"version": 1, "job": "../x", "show": {}, "rows": []})):
        saved.write_text(junk)
        assert main([str(sync / "harbor" / JOB), "--format", "json"]) in (0, 1)
        doc = json.loads(capsys.readouterr().out)
        assert all(i["reward"] is None for i in doc["inputs"])


def test_row_syncs_sharing_a_job_keep_each_others_listing_rows(tmp_path):
    """Several leaderboard rows can share one job, and each row sync lists only its own
    trials. The sidecar used to be replaced by the last row's rows, so a later scan of
    the job folder lost the other rows' tasks and every task-scoped check went unknown
    (TB2.1: 1,779 of 2,219 trials of one combined job)."""
    from atif_scan.sources.harbor.hub import save_listing
    from atif_scan.sources.harbor.listing import SAVED_LISTING, saved_listing

    def row(n: int, task: str) -> Doc:
        tid = f"00000000-0000-4000-8000-{n:012d}"
        return {"id": tid, "name": f"{task}__t{n}", "task_name": task, "reward": 1.0}

    folder = tmp_path / JOB
    save_listing(folder, JOB, {}, [row(1, "task-a"), row(2, "task-a")])
    save_listing(folder, JOB, {}, [row(2, "task-b"), row(3, "task-c")])  # row 2 updated
    _, trials = saved_listing((folder / SAVED_LISTING).read_bytes())
    assert {k: t["task"] for k, t in trials.items()} == {
        "task-a__t1": "task-a",
        "task-b__t2": "task-b",
        "task-c__t3": "task-c",
    }

    # A sidecar for another job is replaced, not merged; stray keys never survive.
    saved = json.loads((folder / SAVED_LISTING).read_text())
    saved["rows"][0]["env"] = "sk-proj-secret1234567890"
    (folder / SAVED_LISTING).write_text(json.dumps(saved))
    save_listing(folder, JOB, {}, [row(4, "task-d")])
    assert "sk-proj" not in (folder / SAVED_LISTING).read_text()
    other = "11111111-1111-4111-8111-111111111111"
    save_listing(folder, other, {}, [row(5, "task-e")])
    _, trials = saved_listing((folder / SAVED_LISTING).read_bytes())
    assert list(trials) == ["task-e__t5"]


def test_harbor_sync_permissions_are_private(harbor, tmp_path):
    from atif_scan.sources.harbor.hub import harbor_sources

    dest = tmp_path / "copy"
    dest.mkdir(mode=0o755)
    harbor_sources(f"harbor://jobs/{JOB}", dest)
    for path in (dest, *dest.rglob("*")):
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
    # Existing mirrors are tightened too, including trajectories reused without download.
    for path in dest.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    harbor_sources(f"harbor://jobs/{JOB}", dest)
    for path in (dest, *dest.rglob("*")):
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
