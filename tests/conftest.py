import pytest


@pytest.fixture(autouse=True)
def isolated_sync_dir(tmp_path_factory, monkeypatch):
    """Syncing is on by default: never let a test write to the real ~/.cache."""
    monkeypatch.setenv("ATIF_SCAN_SYNC_DIR", str(tmp_path_factory.mktemp("sync")))


def _best(fn, size, runs=3):
    """Best of `runs`, with the cyclic GC paused: its full collections grow with the heap
    and make linear code that allocates many objects look superlinear (a runtime effect,
    not an algorithmic one, and noisy on CI runners)."""
    import gc
    import time

    best = float("inf")
    for _ in range(runs):
        gc.collect()
        gc.disable()
        try:
            start = time.perf_counter()
            fn(size)
            best = min(best, time.perf_counter() - start)
        finally:
            gc.enable()
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


# The two seams tests replace, named once: moving a module means editing these paths only.
# Library calls take `fs=` directly; these are for tests that drive the whole CLI.
HF_FILESYSTEM = "atif_scan.sources.inputs.hf_filesystem"
LOAD_TRACE = "atif_scan.sources.inputs.load_trace"


@pytest.fixture
def fake_hub(monkeypatch):
    """`fake_hub(factory)`: the CLI's Hugging Face filesystem becomes `factory()` (a fake
    with the HfFileSystem calls the code uses)."""

    def install(factory):
        monkeypatch.setattr(HF_FILESYSTEM, factory)

    return install


@pytest.fixture
def forbid_trace_loads(monkeypatch):
    """Call it to make any further trace load fail: what follows must be a cache hit."""

    def install():
        def boom(path):
            raise AssertionError("trace was reloaded")

        monkeypatch.setattr(LOAD_TRACE, boom)

    return install
