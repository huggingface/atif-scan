"""A run's `trials.jsonl` ledger (task, reward, error, cost, tokens per trial folder).

Regression: a bucket export of trajectories plus a trials.jsonl ledger (no Harbor
result.json files) was reported as "tasks unknown" and "rewards unknown": the ledger
was neither synced nor read. Synthetic fixtures only.
"""

from __future__ import annotations

import json

from test_sync import FS, trajectory

from atif_scan.cli import main
from atif_scan.harbor_files import trial_ledger
from atif_scan.sync import sync_remote


def row(trial, task, reward, error=None, cost=0.5, **extra):
    return {
        "schema_version": 1,
        "trial_name": trial,
        "task_name": task,
        "reward": reward,
        "error_type": error,
        "cost_usd": cost,
        "input_tokens": 1000,
        "cached_input_tokens": 800,
        "output_tokens": 50,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:40+00:00",
        "job_name": "demo",
        **extra,
    }


def ledger(*rows):
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()


def test_ledger_rows_become_allowlisted_trial_facts():
    facts = trial_ledger(
        ledger(
            row("alpha__1", "tb/alpha", 1.0),
            row("beta__2", "beta", 0.0, error="AgentTimeoutError"),
        )
        + b"not json\n"
        + ledger(
            row("gamma__3", "gamma", 1.0) | {"schema_version": 2},  # unknown schema: skipped
            row("../etc", "delta", 1.0),  # not a folder name: skipped
            row("eps__5", "has space", 1.0, error="Bad Error!", cost=-1),
            row("dup__6", "dup", 1.0),
            row("dup__6", "dup", 0.0),  # listed twice: ambiguous, dropped
        )
    )
    assert set(facts) == {"alpha__1", "beta__2", "eps__5"}
    assert facts["alpha__1"] == {
        "task": "alpha",
        "reward": 1.0,
        "cost_usd": 0.5,
        "input_tokens": 1000,
        "cache_tokens": 800,
        "output_tokens": 50,
        "duration_sec": 100.0,
    }
    assert facts["beta__2"]["error_type"] == "AgentTimeoutError"
    # Unsafe labels and values never pass through as text; a bad cost is unknown.
    assert "task" not in facts["eps__5"] and "cost_usd" not in facts["eps__5"]
    assert facts["eps__5"]["error_type"] == "other"
    assert trial_ledger(b"\xff\xfe") == {} and trial_ledger(None) == {}


def bundle(tmp_path):
    (tmp_path / "trials.jsonl").write_bytes(
        ledger(
            row("alpha__1", "alpha", 1.0, cost=0.25),
            row("beta__2", "beta", 0.0, error="AgentTimeoutError", cost=0.75),
        )
    )
    for name in ("alpha__1", "beta__2"):
        (tmp_path / "trials" / name).mkdir(parents=True)
        (tmp_path / "trials" / name / "trajectory.json").write_bytes(trajectory())


def test_local_bundle_scan_uses_the_ledger(tmp_path, capsys):
    bundle(tmp_path)
    main([str(tmp_path), "--format", "json", "--no-cache"])
    items = {i["input_id"]: i for i in json.loads(capsys.readouterr().out)["inputs"]}
    alpha, beta = items["trials/alpha__1"], items["trials/beta__2"]
    assert (alpha["task"], alpha["reward"], alpha["cost_usd"]) == ("alpha", 1.0, 0.25)
    assert (beta["task"], beta["reward"], beta["error_type"]) == (
        "beta",
        0.0,
        "AgentTimeoutError",
    )

    main([str(tmp_path), "--format", "text", "--no-cache"])
    out = capsys.readouterr().out
    assert "tasks unknown" not in out and "rewards unknown" not in out
    # One attempt per task: the per-task SE is 0 by construction, so it isn't shown.
    assert "RESULT     50.0% (1/2)" in out and "± 0.0" not in out
    assert "$1.00 reported" in out


def test_remote_bundle_syncs_the_ledger(tmp_path, capsys):
    root = "buckets/o/b/bundle"
    fs = FS(
        {
            f"{root}/trials.jsonl": ledger(row("alpha__1", "alpha", 1.0)),
            f"{root}/README.md": b"not needed",
            f"{root}/trials/alpha__1/trajectory.json": trajectory(),
        }
    )
    dest = tmp_path / "copy"
    _, counts = sync_remote(f"hf://{root}", dest, fs=fs)
    assert counts["downloaded"] == 2 and (dest / "trials.jsonl").is_file()
    assert not (dest / "README.md").exists()
    main([str(dest), "--format", "json", "--no-cache"])
    (item,) = json.loads(capsys.readouterr().out)["inputs"]
    assert (item["task"], item["reward"]) == ("alpha", 1.0)
