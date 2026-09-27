import pytest


@pytest.fixture(autouse=True)
def isolated_sync_dir(tmp_path_factory, monkeypatch):
    """Syncing is on by default: never let a test write to the real ~/.cache."""
    monkeypatch.setenv("ATIF_SCAN_SYNC_DIR", str(tmp_path_factory.mktemp("sync")))
