"""The atif-scan home: one root, older per-part variables still win."""

from __future__ import annotations

from pathlib import Path

from atif_scan.data import paths
from atif_scan.sources.sync import default_sync_root


def test_home_falls_back_to_the_xdg_then_user_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("ATIF_SCAN_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert paths.home() == tmp_path / "atif-scan"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    assert paths.home() == tmp_path / "user" / ".cache" / "atif-scan"


def test_every_folder_lives_under_the_home(monkeypatch, tmp_path):
    monkeypatch.setenv("ATIF_SCAN_HOME", str(tmp_path))
    monkeypatch.delenv("ATIF_SCAN_SYNC_DIR")
    assert default_sync_root() == tmp_path
    assert paths.labels_dir() == tmp_path / "labels"
    assert paths.gold_dir() == tmp_path / "gold"
    assert paths.bundles_dir() == tmp_path / "bundles"


def test_older_variables_still_win_for_their_part(monkeypatch, tmp_path):
    monkeypatch.setenv("ATIF_SCAN_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ATIF_SCAN_SYNC_DIR", str(tmp_path / "sync"))
    monkeypatch.setenv("ATIF_SCAN_GOLD_DIR", str(tmp_path / "gold"))
    assert default_sync_root() == tmp_path / "sync"
    assert paths.gold_dir() == tmp_path / "gold"
    assert paths.labels_dir() == tmp_path / "home" / "labels"


def test_a_tilde_in_the_home_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ATIF_SCAN_HOME", "~/atif")
    assert paths.home() == Path(tmp_path) / "atif"
