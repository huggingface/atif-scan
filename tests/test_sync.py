"""--sync (default) mirrors remote inputs locally; re-runs fetch only what changed."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from atif_scan import sources
from atif_scan.cli import main
from atif_scan.sources import SourceError, sync_remote, sync_target


def trajectory(command="ls"):
    return json.dumps(
        {
            "schema_version": "ATIF-v1.7",
            "steps": [
                {
                    "source": "agent",
                    "message": "x",
                    "tool_calls": [
                        {
                            "tool_call_id": "c1",
                            "function_name": "bash",
                            "arguments": {"command": command},
                        }
                    ],
                }
            ],
        }
    ).encode()


class FS:
    """Minimal HfFileSystem stand-in that counts downloads."""

    def __init__(self, files):
        self.files = dict(files)
        self.downloads = []

    def info(self, path):
        if path in self.files:
            return {
                "type": "file",
                "size": len(self.files[path]),
                "xet_hash": hashlib.sha256(self.files[path]).hexdigest(),
            }
        if any(p.startswith(path + "/") for p in self.files):
            return {"type": "directory"}
        raise FileNotFoundError(path)

    def find(self, path, detail=False):
        return {
            p: {"type": "file", "size": len(d), "xet_hash": hashlib.sha256(d).hexdigest()}
            for p, d in self.files.items()
            if p.startswith(path + "/")
        }

    def get_file(self, path, local):
        self.downloads.append(path)
        Path(local).write_bytes(self.files[path])

    def open(self, path, mode):
        return io.BytesIO(self.files[path])


ROOT = "buckets/o/b/job"


def job_files():
    return {
        f"{ROOT}/config.json": json.dumps(
            {"job_name": "demo", "n_attempts": 1, "datasets": []}
        ).encode(),
        f"{ROOT}/result.json": json.dumps(
            {"id": "1d6abb23-0000-4000-8000-000000000000", "n_total_trials": 2, "stats": {}}
        ).encode(),
        f"{ROOT}/alpha__1/agent/trajectory.json": trajectory(),
        f"{ROOT}/alpha__1/result.json": json.dumps(
            {
                "trial_name": "alpha__1",
                "task_name": "x/alpha",
                "verifier_result": {"rewards": {"reward": 1}},
            }
        ).encode(),
        f"{ROOT}/alpha__1/agent/huge-session.log": b"x" * 10_000,  # not needed by the scan
        f"{ROOT}/beta__2/agent/trajectory.json": trajectory("echo 1 > /logs/verifier/reward.txt"),
        f"{ROOT}/beta__2/verifier/reward.txt": b"1",
    }


def test_sync_downloads_only_what_the_scan_needs_and_resumes(tmp_path):
    fs = FS(job_files())
    dest = tmp_path / "copy"
    path, counts = sync_remote(f"hf://{ROOT}", dest, fs=fs)
    assert path == dest and counts["downloaded"] == 6 and counts["failed"] == 0
    assert not any("huge-session" in d for d in fs.downloads)
    assert (dest / "alpha__1" / "result.json").is_file()
    # Re-run: everything up to date, nothing fetched.
    fs.downloads.clear()
    _, counts = sync_remote(f"hf://{ROOT}", dest, fs=fs)
    assert (
        counts == {"files": 6, "downloaded": 0, "up_to_date": 6, "failed": 0} and not fs.downloads
    )
    # A remote file that changed size is fetched again; --refresh fetches everything.
    fs.files[f"{ROOT}/beta__2/verifier/reward.txt"] = b"0.5"
    _, counts = sync_remote(f"hf://{ROOT}", dest, fs=fs)
    assert counts["downloaded"] == 1
    _, counts = sync_remote(f"hf://{ROOT}", dest, fs=fs, refresh=True)
    assert counts["downloaded"] == 6
    assert not list(dest.rglob("*.part"))


def test_sync_target_is_confined_to_the_sync_root(tmp_path):
    assert (
        sync_target("hf://buckets/o/b/job", tmp_path)
        == tmp_path / "hf" / "buckets" / "o" / "b" / "job"
    )
    assert (
        sync_target("https://huggingface.co/buckets/o/b/tree/job", tmp_path)
        == tmp_path / "hf/buckets/o/b/job"
    )
    with pytest.raises(SourceError):
        sync_target("hf://buckets/o/../../etc", tmp_path)


def test_cli_syncs_by_default_and_rescans_from_the_copy(tmp_path, monkeypatch, capsys):
    fs = FS(job_files())
    monkeypatch.setattr(sources, "hf_filesystem", lambda: fs)
    root = Path(os.environ["ATIF_SCAN_SYNC_DIR"])
    assert main([f"hf://{ROOT}", "--format", "json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert (root / "hf" / ROOT / "alpha__1" / "agent" / "trajectory.json").is_file()
    assert {i["input_id"]: i["task"] for i in first["inputs"]} == {
        "alpha__1/agent": "alpha",
        "beta__2/agent": None,
    }
    assert first["runs"][0]["job_name"] == "demo"  # job facts from the synced job folder
    fs.downloads.clear()
    assert main([f"hf://{ROOT}", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["inputs"] == first["inputs"]
    assert not fs.downloads  # second run: no downloads
    assert list((root / "results").rglob("*.json"))  # results cached under the sync root


def test_no_sync_streams_without_keeping_files(tmp_path, monkeypatch, capsys):
    fs = FS(job_files())
    monkeypatch.setattr(sources, "hf_filesystem", lambda: fs)
    root = Path(os.environ["ATIF_SCAN_SYNC_DIR"])
    assert main([f"hf://{ROOT}", "--no-sync", "--format", "json"]) == 0
    items = json.loads(capsys.readouterr().out)["inputs"]
    assert len(items) == 2 and not fs.downloads
    assert not (root / "hf").exists()


def test_sync_dir_flag(tmp_path, monkeypatch, capsys):
    fs = FS(job_files())
    monkeypatch.setattr(sources, "hf_filesystem", lambda: fs)
    assert main([f"hf://{ROOT}", "--sync-dir", str(tmp_path / "mine"), "--format", "json"]) == 0
    capsys.readouterr()
    assert (tmp_path / "mine" / "hf" / ROOT / "beta__2" / "verifier" / "reward.txt").is_file()
