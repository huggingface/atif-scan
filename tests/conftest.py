import pytest


@pytest.fixture(autouse=True)
def isolated_sync_dir(tmp_path_factory, monkeypatch):
    """Syncing is on by default: never let a test write to the real ~/.cache."""
    monkeypatch.setenv("ATIF_SCAN_SYNC_DIR", str(tmp_path_factory.mktemp("sync")))


def _best(fn, size, runs=3):
    import time

    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        fn(size)
        best = min(best, time.perf_counter() - start)
    return best


@pytest.fixture
def linear():
    """`linear(fn, n)`: `fn(size)` must scale linearly, not quadratically. Times n and 4n
    (best of 3): linear work grows ~4x, quadratic ~16x. Machine speed doesn't matter, so
    the check holds on slow CI runners; anything done in under 50 ms at 4n passes."""

    def check(fn, n, ratio=8.0):
        small, large = _best(fn, n), _best(fn, 4 * n)
        assert large < 0.05 or large < ratio * small, f"{small:.4f}s -> {large:.4f}s at 4x"

    return check
