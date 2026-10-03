"""Regression tests for review findings in sources, harbor_files, harbor_hub and layout.

Synthetic fixtures only; no network, no real traces.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from atif_scan.jsonval import Doc, number
from atif_scan.sources import layout
from atif_scan.sources.harbor import hub as harbor_hub
from atif_scan.sources.harbor import runs as harbor_runs
from atif_scan.sources.harbor.files import dataset_source, job_meta, primary_reward, trial_result
from atif_scan.sources.harbor.hub import HarborCLI, harbor_sources, inspect_job
from atif_scan.sources.harbor.listing import run_meta, trial_meta
from atif_scan.sources.inputs import Entry, Listing, confined, list_input, resolve
from atif_scan.sources.sync import sync_remote

if TYPE_CHECKING:
    from collections.abc import Callable

    from atif_scan.model import Trace

JOB = "1d6abb23-0000-4000-8000-000000000000"
ROOT = "buckets/o/b/job"
TRAJ = json.dumps(
    {
        "schema_version": "ATIF-v1.4",
        "session_id": "s",
        "agent": {"name": "a", "version": "1"},
        "steps": [{"step_id": 1, "source": "user", "message": "hi"}],
    }
).encode()


class FS:
    """HfFileSystem stand-in whose `find` returns exactly the given paths."""

    def __init__(self, files):
        self.files = dict(files)

    def info(self, path):
        if path in self.files:
            return {"type": "file", "size": len(self.files[path])}
        return {"type": "directory"}

    def find(self, path, detail=False):
        return {p: {"type": "file", "size": len(d)} for p, d in self.files.items()}

    def get_file(self, path, local):
        Path(local).write_bytes(self.files[path])

    def open(self, path, mode):
        return io.BytesIO(self.files[path])


# 1, 2: remote listings stay under the root, and sync never writes outside dest.


def test_confined_rejects_traversal(tmp_path):
    assert confined(tmp_path, "a/b.json") == tmp_path / "a" / "b.json"
    for bad in ("", ".", "/abs", "a/../../x", "../x", "a//b", "a/./b", "a\0b"):
        assert confined(tmp_path, bad) is None


def test_remote_listing_drops_entries_outside_the_root(tmp_path):
    fs = FS(
        {
            f"{ROOT}/ok/trajectory.json": TRAJ,
            f"{ROOT}/../evil/trajectory.json": TRAJ,  # `..` inside the root prefix
            "elsewhere/secret-root/trajectory.json": TRAJ,  # not under the root at all
        }
    )
    listing = list_input(f"hf://{ROOT}", fs)
    assert [e.path for e in listing.entries] == ["ok/trajectory.json"]
    labels = [s.label for s in resolve([f"hf://{ROOT}"], fs=fs)]
    assert labels == ["ok"] and not any("secret-root" in x for x in labels)
    dest = tmp_path / "sync" / "dest"
    _, counts = sync_remote(f"hf://{ROOT}", dest, fs=fs)
    assert counts["downloaded"] == 1 and not (tmp_path / "sync" / "evil").exists()


def test_unsafe_listed_paths_are_never_written(tmp_path):
    # An absolute path or `..` in a bucket listing never leaves the sync folder: the listing
    # drops them (sync's own path check is a second line of defence behind it).
    outside = tmp_path / "outside.json"
    for name in (str(outside), f"{ROOT}/a/../../x/trajectory.json"):
        fs = FS({name: b"{}", f"{ROOT}/ok.json": b"{}"})
        dest = tmp_path / "dest"
        _, counts = sync_remote(f"hf://{ROOT}", dest, pattern="*.json", fs=fs)
        assert counts["files"] == 1 and not outside.exists()
        assert sorted(p.name for p in dest.rglob("*.json") if p.name != ".atif-sync.json") == [
            "ok.json"
        ]


# 10: listed files over their size cap are never downloaded.


def test_sync_skips_oversize_reward_files(tmp_path):
    fs = FS(
        {
            f"{ROOT}/a/verifier/reward.txt": b"1" * (harbor_runs.REWARD_BYTES + 1),
            f"{ROOT}/b/trajectory.json": TRAJ[:100],
            f"{ROOT}/b/verifier/reward.txt": b"1",
        }
    )
    _, counts = sync_remote(f"hf://{ROOT}", tmp_path, fs=fs)
    assert counts == {"files": 3, "downloaded": 2, "up_to_date": 0, "failed": 1}
    assert not (tmp_path / "a/verifier/reward.txt").exists()


# 3: Hub trial ids and names are validated before becoming CLI args or folders.


def never_opened(entry: Entry) -> Callable[[], Trace]:
    """A listing opener for tests that must not open any trace."""
    raise AssertionError(entry)


@dataclass(frozen=True)
class FakeCLI:
    """A `harbor` CLI double: every `--json` call returns `reply`; it never runs a process."""

    reply: object = None

    def json(self, *args: str) -> object:
        return self.reply

    def run(self, *args: str) -> str:
        raise AssertionError(args)


@dataclass(frozen=True)
class StubCLI:
    rows: list[Doc] = field(default_factory=list)
    row_items: list[Doc] = field(default_factory=list)
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def json(self, *args: str) -> object:
        self.calls.append(args)
        if args[:3] == ("hub", "job", "show"):
            return {"name": "demo"}
        if args[:3] == ("hub", "job", "trials"):
            return {"items": self.rows, "total_pages": 1}
        if args[:5] == ("hub", "leaderboard", "row", "trial", "list"):
            return {"items": self.row_items, "total_pages": 1}
        raise AssertionError(args)

    def run(self, *args: str) -> str:
        self.calls.append(args)
        out = Path(args[args.index("-o") + 1])
        row = next(r for r in self.rows if r["id"] == args[3])
        (out / row["name"]).mkdir(parents=True, exist_ok=True)
        (out / row["name"] / "trajectory.json").write_bytes(TRAJ)
        return ""


def test_hub_rows_with_unsafe_ids_or_names_are_dropped_or_relabelled(tmp_path):
    good = "00000000-0000-4000-8000-000000000001"
    renamed = "00000000-0000-4000-8000-000000000002"
    rows = [
        {"id": good, "name": "task__A1"},
        {"id": "-oevil", "name": "task__B1"},  # option injection
        {"id": "../../evil", "name": "task__C1"},
        {"id": renamed, "name": "a/../../x"},  # traversal in the name
    ]
    cli = StubCLI(rows=rows)
    dest = tmp_path / "dl"
    found, _ = harbor_sources(f"harbor://jobs/{JOB}", dest, workers=1, cli=cli)
    assert sorted(s.label for s in found) == sorted(["task__A1", renamed])
    downloads = [c[3] for c in cli.calls if c[:3] == ("hub", "trial", "download")]
    assert sorted(downloads) == [good, renamed]
    assert not (tmp_path / "x").exists() and not (tmp_path / "evil").exists()
    assert {t["input_id"] for t in inspect_job(f"harbor://jobs/{JOB}", cli=cli)["trials"]} == {
        "task__A1",
        renamed,
    }


def test_row_trial_ids_must_be_uuids():
    good = "00000000-0000-4000-8000-000000000001"
    cli = StubCLI(row_items=[{"trial_id": good}, {"trial_id": "-o/tmp/x"}, {"x": 1}])
    assert harbor_hub.row_trials(cli, "r") == [good]


# 4: listing lookups are linear, not quadratic.


def test_many_trials_resolve_and_inspect_quickly(linear):
    def job(n):
        files = {}
        for i in range(n):
            trial = f"{ROOT}/t{i:04d}"
            files[f"{trial}/agent/trajectory.json"] = b"{}"
            files[f"{trial}/result.json"] = b"{}"
            files[f"{trial}/config.json"] = b"{}"
            files[f"{trial}/verifier/reward.txt"] = b"1"
        files[f"{ROOT}/config.json"] = files[f"{ROOT}/result.json"] = b"{}"
        return FS(files)

    jobs = {n: job(n) for n in (750, 3000)}

    def scan(n):
        resolve([f"hf://{ROOT}"], fs=jobs[n])
        layout.inspect_listing(list_input(f"hf://{ROOT}", jobs[n]), "trajectory.json", 1)

    linear(scan, 750)
    fs = jobs[3000]
    found = resolve([f"hf://{ROOT}"], fs=fs)
    card = layout.inspect_listing(list_input(f"hf://{ROOT}", fs), "trajectory.json", 1)
    assert len(found) == 3000 and card["harbor"]["trials"] == 3000 and card["harbor"]["jobs"] == 1


# 5: --inspect reports never carry raw unvalidated paths.


def test_inspect_report_has_no_raw_paths():
    names = ["we ird!", "sp ace", "tri al"]
    entries = (
        Entry(f"{names[0]}/trajectory.json", 2),  # positional label
        Entry(f"{names[1]}/old-trajectory.json", 2),  # unscanned trajectory-like
        Entry(f"{names[2]}/trial.log", 1),
        Entry(f"{names[2]}/exception.txt", 1),
        Entry(f"{names[2]}/agent/trajectory.json", 2),
        Entry(f"{names[2]}/user-agent/trajectory.json", 2),
    )
    listing = Listing(False, True, entries, never_opened)
    card = layout.inspect_listing(listing, "trajectory.json", 1)
    text = json.dumps(card)
    assert not any(n in text for n in names)
    codes = {a["code"] for a in card["anomalies"]}
    assert {
        "positional_labels",
        "unscanned_trajectory_like_files",
        "trials_with_exception",
    } <= codes


# 6: one validator for free-text labels (no URLs, no control characters).


def test_free_text_labels_are_validated():
    assert dataset_source({"name": "https://evil.example/x"}) == (None, None, None)
    assert dataset_source({"name": "terminal-bench/x", "ref": "a\nb"})[1] is None
    config = {"datasets": [{"name": "ok/x", "ref": "sha256:ab"}, {"name": "x://y", "ref": "\x1b"}]}
    meta = run_meta(JOB, {"name": "https://evil.example", "config": config}, [])
    assert meta["job_name"] is None
    assert meta["datasets"] == ["ok/x"] and meta["dataset_refs"] == ["sha256:ab"]


def test_row_labels_and_display_cost_are_validated():
    show = {
        "rank": 1,
        "metadata": {"agent_display": {"label": "Demo CLI (v2)"}, "model_display": "see https://x"},
        "metrics": {"display_cost": "$1.20 (partial)\x1b[2J", "accuracy": float("nan")},
    }
    cli = FakeCLI(reply=show)  # every call answers `show`: the row lists no trials
    run, _ = harbor_hub.row_listing(cli, "r")
    lb = run["leaderboard"]
    assert lb["agent"] == "Demo CLI (v2)" and lb["model"] is None
    assert lb["display_cost"] is None and lb["reported_accuracy"] is None


# 7: one finite-number helper.


def test_numbers_are_finite_and_rewards_may_be_negative():
    assert (
        trial_meta({"id": "x", "reward": float("nan"), "cost_usd": float("inf")})["reward"] is None
    )
    assert trial_meta({"id": "x", "cost_usd": float("inf")})["cost_usd"] is None
    assert trial_meta({"id": "x", "reward": -1})["reward"] == -1.0
    assert (
        primary_reward({"reward": -1}) == -1.0 and harbor_runs.parse_reward(b"-1", "r.txt") == -1.0
    )
    assert number(True) is None and number(-1, 0) is None and number(2, 0) == 2.0
    data = json.dumps({"task_name": "t", "agent_result": {"cost_usd": -3}}).encode()
    assert "cost_usd" not in trial_result(data)


# 8: the CLI runner and pagination.


def test_cli_run_is_non_interactive_and_decodes_leniently(monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert HarborCLI("harbor").json("hub", "job", "show", JOB) == {}
    assert seen["stdin"] is subprocess.DEVNULL
    assert seen["encoding"] == "utf-8" and seen["errors"] == "replace"


@pytest.mark.parametrize("total", ["not-a-number <https://x>", None, -2, 1.5])
def test_bad_total_pages_is_one_page(total):
    cli = FakeCLI(reply={"items": [], "total_pages": total})
    assert list(harbor_hub._pages(cli, "hub")) == [(1, 1, {"items": [], "total_pages": total})]


def test_pagination_is_capped():
    cli = FakeCLI(reply={"items": [], "total_pages": 10**9})
    pages = [p for p, _, _ in harbor_hub._pages(cli, "hub")]
    assert pages[-1] == harbor_hub.MAX_PAGES == len(pages)


# 9: deeply nested JSON doesn't abort the scan.


def test_deeply_nested_json_is_unknown():
    deep = b"[" * 200_000 + b"]" * 200_000
    assert job_meta(deep, deep) is None and trial_result(deep) == {}


# 11: unreadable local reward/result files are unknown, not an error.


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs file permissions")
def test_unreadable_local_side_files_are_unknown(tmp_path):
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier").mkdir()
    path = trial / "agent" / "trajectory.json"
    path.write_bytes(TRAJ)
    for name in ("verifier/reward.txt", "result.json"):
        (trial / name).write_text("1")
        (trial / name).chmod(0)
    try:
        (source,) = resolve([str(path)])
        assert source.reward() is None and source.details() == {}
    finally:
        for name in ("verifier/reward.txt", "result.json"):
            (trial / name).chmod(0o600)
