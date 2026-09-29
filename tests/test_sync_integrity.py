"""Sync never shrinks the evidence population or silently reuses stale run facts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_sync import FS, ROOT, job_files, trajectory
from typing_extensions import override

from atif_scan import sources, sync
from atif_scan.cli import main
from atif_scan.loader import TraceError
from atif_scan.sources import SourceError, resolve
from atif_scan.sync import sync_remote


class FailingFS(FS):
    fail: set[str]

    def __init__(self, files):
        super().__init__(files)
        self.fail = set()

    @override
    def get_file(self, path, local):
        if path in self.fail:
            Path(local).write_bytes(b"partial synthetic download")
            raise OSError("synthetic-private-remote-error")
        super().get_file(path, local)


def test_download_failure_remains_unknown_in_remote_and_offline_reports(monkeypatch, capsys):
    fs = FailingFS(job_files())
    fs.fail = {f"{ROOT}/beta__2/agent/trajectory.json"}
    monkeypatch.setattr(sources, "hf_filesystem", lambda: fs)
    dest = Path(os.environ["ATIF_SCAN_SYNC_DIR"]) / "hf" / ROOT
    for location in (f"hf://{ROOT}", str(dest)):
        assert main([location, "--format", "json"]) == 2
        captured = capsys.readouterr()
        doc = json.loads(captured.out)
        assert doc["coverage"]["inputs"] == 2
        assert doc["coverage"]["available"] == 1
        assert doc["coverage"]["sync_failed_files"] == 1
        failed = next(i for i in doc["inputs"] if i["input_id"] == "beta__2/agent")
        assert failed["input_status"] == "unavailable_or_invalid"
        assert failed["incomplete"]
        assert all(a["status"] in ("unknown", "not_applicable") for a in failed["assessments"])
        assert "sync file(s) unavailable" in captured.err
        assert "synthetic-private" not in captured.out + captured.err
        assert not list(dest.rglob("*.part"))
    fs.fail.clear()
    assert main([f"hf://{ROOT}", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["coverage"]["available"] == 2


@pytest.mark.parametrize("view", ["detail", "brief", "overview", "summary"])
@pytest.mark.parametrize("fmt", ["json", "text"])
def test_metadata_failure_is_visible_in_every_view(view, fmt, monkeypatch, capsys):
    fs = FailingFS(job_files())
    fs.fail = {f"{ROOT}/beta__2/verifier/reward.txt"}
    monkeypatch.setattr(sources, "hf_filesystem", lambda: fs)
    assert main([f"hf://{ROOT}", f"--{view}", "--format", fmt]) == 2
    captured = capsys.readouterr()
    assert "sync file(s) unavailable" in captured.err
    if fmt == "json":
        json.loads(captured.out)
        assert '"sync_failed_files": 1' in captured.out
    else:
        assert "file(s) unavailable" in captured.out


@pytest.mark.parametrize("refresh", [False, True])
def test_deleted_trials_and_metadata_are_removed(tmp_path, refresh):
    fs = FS(job_files())
    sync_remote(f"hf://{ROOT}", tmp_path, fs=fs)
    for path in list(fs.files):
        if "/beta__2/" in path or path.endswith("alpha__1/result.json"):
            del fs.files[path]
    (tmp_path / "notes.txt").write_text("unrelated user file")
    sync_remote(f"hf://{ROOT}", tmp_path, fs=fs, refresh=refresh)
    found = resolve([str(tmp_path)])
    assert [s.label for s in found] == ["alpha__1/agent"]
    assert not (tmp_path / "beta__2/agent/trajectory.json").exists()
    assert not (tmp_path / "alpha__1/result.json").exists()
    assert found[0].details() == {}
    assert (tmp_path / "notes.txt").read_text() == "unrelated user file"


def test_same_size_reward_and_trajectory_changes_invalidate_results(monkeypatch, capsys):
    fs = FS(job_files())
    monkeypatch.setattr(sources, "hf_filesystem", lambda: fs)
    args = [f"hf://{ROOT}", "--format", "json"]
    assert main(args) == 0
    before = json.loads(capsys.readouterr().out)
    fs.files[f"{ROOT}/beta__2/verifier/reward.txt"] = b"0"
    old = fs.files[f"{ROOT}/beta__2/agent/trajectory.json"]
    command = b"echo 1 > /logs/verifier/reward.txt"
    fs.files[f"{ROOT}/beta__2/agent/trajectory.json"] = old.replace(
        command, b"ls" + b" " * (len(command) - 2)
    )
    assert len(fs.files[f"{ROOT}/beta__2/agent/trajectory.json"]) == len(old)
    fs.downloads.clear()
    assert main(args) == 0
    after = json.loads(capsys.readouterr().out)
    assert len(fs.downloads) == 2
    assert before["inputs"][1]["reward"] == 1
    assert after["inputs"][1]["reward"] == 0
    old_check = next(
        a for a in before["inputs"][1]["assessments"] if a["id"] == "tamper.reward_write"
    )
    new_check = next(
        a for a in after["inputs"][1]["assessments"] if a["id"] == "tamper.reward_write"
    )
    assert old_check["status"] == "match"
    assert new_check["status"] != "match"


class NoRevisionFS(FS):
    @override
    def info(self, path):
        return {k: v for k, v in super().info(path).items() if k != "xet_hash"}

    @override
    def find(self, path, detail=False):
        return {
            p: {k: v for k, v in info.items() if k != "xet_hash"}
            for p, info in super().find(path, detail).items()
        }


def test_missing_revision_is_never_assumed_current(tmp_path):
    fs = NoRevisionFS(job_files())
    sync_remote(f"hf://{ROOT}", tmp_path, fs=fs)
    fs.files[f"{ROOT}/beta__2/verifier/reward.txt"] = b"0"
    _, counts = sync_remote(f"hf://{ROOT}", tmp_path, fs=fs)
    assert counts["downloaded"] == 6 and counts["up_to_date"] == 0
    assert (tmp_path / "beta__2/verifier/reward.txt").read_bytes() == b"0"


def test_failed_refresh_cannot_reuse_cached_trace_or_stale_reward(monkeypatch, capsys):
    fs = FailingFS(job_files())
    monkeypatch.setattr(sources, "hf_filesystem", lambda: fs)
    args = [f"hf://{ROOT}", "--format", "json"]
    assert main(args) == 0
    capsys.readouterr()
    fs.fail = {f"{ROOT}/beta__2/agent/trajectory.json", f"{ROOT}/beta__2/verifier/reward.txt"}
    assert main([*args, "--refresh"]) == 2
    doc = json.loads(capsys.readouterr().out)
    item = doc["inputs"][1]
    assert item["reward"] is None
    assert item["input_status"] == "unavailable_or_invalid"
    assert doc["coverage"]["sync_failed_files"] == 2


def test_all_downloads_fail_and_single_file_failure_still_report(monkeypatch, capsys):
    fs = FailingFS({f"{ROOT}/trace.data": trajectory()})
    fs.fail = set(fs.files)
    monkeypatch.setattr(sources, "hf_filesystem", lambda: fs)
    assert main([f"hf://{ROOT}/trace.data", "--format", "json"]) == 2
    doc = json.loads(capsys.readouterr().out)
    assert doc["coverage"]["inputs"] == 1
    assert doc["coverage"]["available"] == 0
    assert doc["inputs"][0]["input_id"] == "input-0001"
    fs.fail.clear()
    assert main([f"hf://{ROOT}/trace.data", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["coverage"]["available"] == 1


def test_oversize_trace_stays_in_population(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "MAX_BYTES", 8)
    fs = FS({f"{ROOT}/a/trajectory.json": b"x" * 9})
    _, counts = sync_remote(f"hf://{ROOT}", tmp_path, fs=fs)
    assert counts["failed"] == 1 and not fs.downloads
    found = resolve([str(tmp_path)])
    assert len(found) == 1
    with pytest.raises(TraceError, match="sync_failed"):
        found[0].load()


def test_sync_tightens_permissions_and_keeps_temporary_files_private(tmp_path):
    dest = tmp_path / "copy"
    dest.mkdir(mode=0o755)

    class PermissionsFS(FS):
        @override
        def get_file(self, path, local):
            assert Path(local).stat().st_mode & 0o777 == 0o600
            assert Path(local).parent.stat().st_mode & 0o777 == 0o700
            super().get_file(path, local)

    fs = PermissionsFS(job_files())
    old_umask = os.umask(0)
    try:
        sync_remote(f"hf://{ROOT}", dest, fs=fs)
        for path in dest.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        sync_remote(f"hf://{ROOT}", dest, fs=fs)
    finally:
        os.umask(old_umask)
    assert dest.stat().st_mode & 0o777 == 0o700
    for path in dest.rglob("*"):
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)


@pytest.mark.parametrize("kind", ["root", "directory", "file", "inventory"])
def test_sync_refuses_symlink_destinations(tmp_path, kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "trajectory.json"
    sentinel.write_bytes(b"private synthetic sentinel")
    dest = tmp_path / "copy"
    if kind == "root":
        dest.symlink_to(outside, target_is_directory=True)
    else:
        dest.mkdir()
        if kind == "directory":
            (dest / "a").symlink_to(outside, target_is_directory=True)
        elif kind == "inventory":
            (dest / sources.SYNC_STATE).symlink_to(sentinel)
        else:
            (dest / "a").mkdir()
            (dest / "a/trajectory.json").symlink_to(sentinel)
    fs = FS({f"{ROOT}/a/trajectory.json": trajectory()})
    with pytest.raises(SourceError):
        sync_remote(f"hf://{ROOT}", dest, fs=fs)
    assert sentinel.read_bytes() == b"private synthetic sentinel"
    assert not fs.downloads


def test_local_edit_is_not_reused_and_malformed_inventory_fails_closed(tmp_path):
    fs = FS(job_files())
    sync_remote(f"hf://{ROOT}", tmp_path, fs=fs)
    local = tmp_path / "beta__2/verifier/reward.txt"
    local.write_bytes(b"0")
    _, counts = sync_remote(f"hf://{ROOT}", tmp_path, fs=fs)
    assert counts["downloaded"] == 1
    assert local.read_bytes() == b"1"
    (tmp_path / sources.SYNC_STATE).write_text("not json")
    with pytest.raises(SourceError, match="invalid_sync_inventory"):
        resolve([str(tmp_path)])


def test_legacy_mirror_does_not_reuse_size_or_retain_deleted_trials(tmp_path):
    (tmp_path / "old/agent").mkdir(parents=True)
    (tmp_path / "old/agent/trajectory.json").write_bytes(trajectory())
    (tmp_path / "beta__2/verifier").mkdir(parents=True)
    (tmp_path / "beta__2/verifier/reward.txt").write_bytes(b"0")
    fs = FS(job_files())
    _, counts = sync_remote(f"hf://{ROOT}", tmp_path, fs=fs)
    assert counts["downloaded"] == 6
    assert not (tmp_path / "old/agent/trajectory.json").exists()
    assert (tmp_path / "beta__2/verifier/reward.txt").read_bytes() == b"1"


def test_provider_listing_cache_is_invalidated_before_resync(tmp_path):
    class CachedFS(FS):
        def __init__(self, files):
            super().__init__(files)
            self.cached = None
            self.invalidations = 0

        def invalidate_cache(self):
            self.cached = None
            self.invalidations += 1

        @override
        def find(self, path, detail=False):
            if self.cached is None:
                self.cached = super().find(path, detail)
            return self.cached

    fs = CachedFS(job_files())
    sync_remote(f"hf://{ROOT}", tmp_path, fs=fs)
    fs.files[f"{ROOT}/beta__2/verifier/reward.txt"] = b"0"
    sync_remote(f"hf://{ROOT}", tmp_path, fs=fs)
    assert (tmp_path / "beta__2/verifier/reward.txt").read_bytes() == b"0"
    assert fs.invalidations == 2


@pytest.mark.parametrize("field", ["xet_hash", "blob_id", "etag"])
def test_provider_content_identity_supported(tmp_path, field):
    class RevisionFS(FS):
        @override
        def find(self, path, detail=False):
            found = super().find(path, detail)
            for info in found.values():
                info[field] = info.pop("xet_hash")
            return found

    fs = RevisionFS(job_files())
    sync_remote(f"hf://{ROOT}", tmp_path, fs=fs)
    _, counts = sync_remote(f"hf://{ROOT}", tmp_path, fs=fs)
    assert counts["up_to_date"] == 6
    fs.files[f"{ROOT}/beta__2/verifier/reward.txt"] = b"0"
    _, counts = sync_remote(f"hf://{ROOT}", tmp_path, fs=fs)
    assert counts["downloaded"] == 1


def test_download_size_disagreement_is_not_published(tmp_path):
    class ChangedFS(FS):
        @override
        def get_file(self, path, local):
            Path(local).write_bytes(b"different size from inventory")

    fs = ChangedFS({f"{ROOT}/a/trajectory.json": trajectory()})
    _, counts = sync_remote(f"hf://{ROOT}", tmp_path, fs=fs)
    assert counts["failed"] == 1
    assert not (tmp_path / "a/trajectory.json").exists()
    assert not list(tmp_path.rglob("*.part"))
    with pytest.raises(TraceError, match="sync_failed"):
        resolve([str(tmp_path)])[0].load()


def test_interrupted_sync_leaves_inventory_unknown_not_cached(monkeypatch, capsys):
    fs = FS(job_files())
    monkeypatch.setattr(sources, "hf_filesystem", lambda: fs)
    assert main([f"hf://{ROOT}", "--format", "json"]) == 0
    capsys.readouterr()
    dest = Path(os.environ["ATIF_SCAN_SYNC_DIR"]) / "hf" / ROOT
    state = sources.read_sync_state(dest)
    assert state is not None
    for row in state["files"].values():
        row["failed"] = True
        row["stamp"] = None
    # The conservative inventory published before downloading, with old files still present.
    sync.write_sync_state(dest, state["files"], None)
    assert main([str(dest), "--format", "json"]) == 2
    doc = json.loads(capsys.readouterr().out)
    assert doc["coverage"]["available"] == 0
    assert doc["coverage"]["sync_failed_files"] == 6
    assert all(i["reward"] is None for i in doc["inputs"])


def test_scanning_parent_of_mirrors_keeps_failed_inputs(tmp_path, capsys):
    fs = FailingFS(job_files())
    fs.fail = {f"{ROOT}/beta__2/agent/trajectory.json"}
    sync_remote(f"hf://{ROOT}", tmp_path / "copy", fs=fs)
    assert main([str(tmp_path), "--format", "json"]) == 2
    doc = json.loads(capsys.readouterr().out)
    assert doc["coverage"]["inputs"] == 2
    assert doc["coverage"]["available"] == 1
    assert doc["coverage"]["sync_failed_files"] == 1
