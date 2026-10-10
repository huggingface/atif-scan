"""Mirror a remote (hf://) input locally, keeping an inventory of what synced.

Only matching trajectories and the small run files a scan reads beside them are
downloaded, each up to a size cap. The inventory (`sources.SYNC_STATE`) lists every
current remote file with its provider identity and whether it failed, so a scan of the
copy keeps failed files as unavailable inputs and never lets a stale file stand in for
one. Everything is private (0700/0600) and symlinks are refused, never followed.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, TypedDict

from ..data import paths
from ..data.loader import MAX_BYTES
from .datacurve import INDEX_BYTES, TRIAL_INDEX
from .harbor.files import (
    ATTEMPT_COST_BYTES,
    ATTEMPT_COSTS,
    LEDGER_BYTES,
    RUN_MANIFEST,
    RUN_MANIFEST_BYTES,
    TRIAL_LEDGER,
)
from .harbor.runs import EXCEPTION_MARKER, RESULT_BYTES, REWARD_BYTES
from .inputs import (
    DEFAULT_PATTERN,
    HF_PREFIX,
    SYNC_STATE,
    SourceError,
    confined,
    list_input,
    normalize,
    read_sync_state,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .inputs import Entry, Listing, RemoteFS, SyncRow

# Besides matching trajectories, the small files the scan reads next to them, and the
# largest size synced for each (exception.txt is only checked for presence).
SYNC_CAPS = {
    "result.json": RESULT_BYTES,
    "config.json": RESULT_BYTES,
    "reward.txt": REWARD_BYTES,
    "reward.json": REWARD_BYTES,
    EXCEPTION_MARKER: RESULT_BYTES,
    TRIAL_LEDGER: LEDGER_BYTES,
    TRIAL_INDEX: INDEX_BYTES,
    RUN_MANIFEST: RUN_MANIFEST_BYTES,
}
SYNC_NAMES = frozenset(SYNC_CAPS)
PRIVATE_DIR = 0o700
PRIVATE_FILE = 0o600
Outcome = Literal["downloaded", "up_to_date", "failed"]


class SyncCounts(TypedDict):
    files: int
    downloaded: int
    up_to_date: int
    failed: int


@dataclass(frozen=True)
class Wanted:
    """One remote file to mirror: where it goes and the largest size accepted."""

    entry: Entry
    relative: str
    path: Path
    cap: int


def sync_cap(relative: str) -> int | None:
    """Largest size synced for a listed file that isn't a matching trajectory, or None
    when the scan never reads it."""
    path = PurePosixPath(relative)
    if path.parent.name == ATTEMPT_COSTS and path.suffix == ".json":
        return ATTEMPT_COST_BYTES
    return SYNC_CAPS.get(path.name)


def default_sync_root() -> Path:
    """Where remote inputs are synced (see `data.paths`): $ATIF_SCAN_SYNC_DIR, else the
    atif-scan home. Contents are real traces: documented, and safe to delete."""
    return paths.sync_root()


def sync_target(value: str, root: Path) -> Path:
    """Local mirror folder for an hf:// input: <root>/hf/<path>, no traversal."""
    parts = [p for p in normalize(value)[len(HF_PREFIX) :].split("/") if p]
    if not parts or any(p in (".", "..") for p in parts):
        raise SourceError("invalid_hf_path")
    return root.joinpath("hf", *parts)


def private_directory(path: Path) -> None:
    """Create private parents; tighten the owned directory, never follow symlinks."""
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise SourceError("unsafe_sync_path")
    if not path.exists():
        if not path.parent.exists():
            private_directory(path.parent)
        path.mkdir(mode=PRIVATE_DIR, exist_ok=True)
    if not path.is_dir():
        raise SourceError("unsafe_sync_path")
    path.chmod(PRIVATE_DIR)


def private_tree(root: Path) -> None:
    """Tighten an owned download tree, refusing links instead of following them."""
    private_directory(root)
    for folder, dirs, files in os.walk(root):
        for name in dirs:
            private_directory(Path(folder) / name)
        for name in files:
            path = Path(folder) / name
            if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
                raise SourceError("unsafe_sync_path")
            path.chmod(PRIVATE_FILE)


def private_sync_path(root: Path, path: Path) -> None:
    """Validate/create only directories inside the mirror; reject linked targets."""
    relative = path.relative_to(root)
    parent = root
    private_directory(parent)
    for part in relative.parts[:-1]:
        parent /= part
        private_directory(parent)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise SourceError("unsafe_sync_path")


def write_sync_state(root: Path, files: Mapping[str, SyncRow], single: str | None) -> None:
    """Publish an inventory atomically; private metadata, never report fields."""
    private_sync_path(root, root / SYNC_STATE)
    fd, name = tempfile.mkstemp(prefix=".atif-sync-", suffix=".tmp", dir=root)
    path = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump({"schema_version": 1, "files": files, "single": single}, handle)
        path.replace(root / SYNC_STATE)
    finally:
        path.unlink(missing_ok=True)


def sync_stamp(path: Path) -> str | None:
    try:
        st = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        return None
    return f"{st.st_size}:{st.st_mtime_ns}"


def _wanted(listing: Listing, dest: Path, single: str | None, pattern: str) -> list[Wanted]:
    """The listed files a scan reads, with their confined local paths."""
    wanted = []
    for entry in listing.entries:
        relative = single if single is not None else entry.path
        name = PurePosixPath(relative).name
        cap = (
            MAX_BYTES if not listing.directory or fnmatchcase(name, pattern) else sync_cap(relative)
        )
        if cap is None:
            continue
        path = confined(dest, relative)
        if path is None or relative == SYNC_STATE:
            raise SourceError("invalid_hf_path")
        private_sync_path(dest, path)
        wanted.append(Wanted(entry, relative, path, cap))
    return wanted


def _remove_obsolete(
    dest: Path, current: Mapping[str, object], previous: Mapping[str, object], pattern: str
) -> None:
    """Remove only scan-owned file kinds (including legacy mirrors without an inventory)
    that the current inventory doesn't list. Other user files are left alone and
    excluded by the authoritative inventory."""
    for folder, _, files in os.walk(dest):
        for name in files:
            path = Path(folder) / name
            relative = path.relative_to(dest).as_posix()
            owned = (
                relative in previous or sync_cap(relative) is not None or fnmatchcase(name, pattern)
            )
            if relative not in current and owned and name != SYNC_STATE:
                private_sync_path(dest, path)
                path.unlink()


def _reusable(item: Wanted, old: SyncRow | None, stamp: str | None) -> bool:
    """Reuse needs a provider content identity AND an unchanged local fingerprint."""
    entry = item.entry
    return (
        old is not None
        and entry.revision is not None
        and old["revision"] == entry.revision
        and not old["failed"]
        and stamp is not None
        and old["stamp"] == stamp
        and old["size"] == entry.size
    )


def _download(fetch: Callable[[Entry, Path], None], item: Wanted) -> None:
    """Fetch into a private temporary file, then move it into place if its size fits."""
    fd, name = tempfile.mkstemp(prefix=".atif-download-", suffix=".part", dir=item.path.parent)
    os.close(fd)
    partial = Path(name)
    try:
        fetch(item.entry, partial)
        size = partial.stat().st_size
        if size > item.cap or (item.entry.size is not None and size != item.entry.size):
            raise ValueError("sync_size_mismatch")
        partial.chmod(PRIVATE_FILE)
        partial.replace(item.path)
    finally:
        partial.unlink(missing_ok=True)


def _sync_one(
    fetch: Callable[[Entry, Path], None],
    item: Wanted,
    old: SyncRow | None,
    refresh: bool,
) -> tuple[Outcome, str | None]:
    """(outcome, local stamp) for one wanted file: up_to_date, downloaded or failed."""
    try:
        stamp = sync_stamp(item.path)
        if item.entry.size is not None and item.entry.size > item.cap:
            raise ValueError("sync_too_large")
        if not refresh and _reusable(item, old, stamp):
            item.path.chmod(PRIVATE_FILE)
            return "up_to_date", stamp
        # Failure must not leave an old copy usable by direct local scans.
        item.path.unlink(missing_ok=True)
        _download(fetch, item)
        return "downloaded", sync_stamp(item.path)
    except Exception:  # noqa: BLE001 - any provider/file error is a failed (unknown) file
        item.path.unlink(missing_ok=True)
        return "failed", None


def sync_remote(
    value: str,
    dest: Path,
    pattern: str = DEFAULT_PATTERN,
    fs: RemoteFS | None = None,
    workers: int = 16,
    refresh: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, SyncCounts]:
    """Mirror the current remote inventory, retaining failures as unknown inputs.

    Reuse requires a provider content identity AND an unchanged local fingerprint.
    Without an identity re-download. The saved inventory excludes obsolete files even
    when scanning the copy offline; no stale file can stand in for a failed download.
    """
    listing = list_input(value, fs)
    fetch = listing.fetch
    if fetch is None:
        raise SourceError("not_a_remote_input")
    private_directory(dest)
    before = read_sync_state(dest)
    previous = before["files"] if before is not None else {}
    single = None if listing.directory else PurePosixPath(normalize(value)).name
    wanted = _wanted(listing, dest, single, pattern)
    state: dict[str, SyncRow] = {
        w.relative: {
            "size": w.entry.size,
            "revision": w.entry.revision,
            "stamp": None,
            "failed": True,
        }
        for w in wanted
    }
    # Publish unknowns before changing files: interruption cannot resurrect stale facts.
    write_sync_state(dest, state, single)
    _remove_obsolete(dest, state, previous, pattern)

    counts: SyncCounts = {"files": len(wanted), "downloaded": 0, "up_to_date": 0, "failed": 0}

    def one(item: Wanted) -> tuple[Wanted, Outcome, str | None]:
        return (item, *_sync_one(fetch, item, previous.get(item.relative), refresh))

    with ThreadPoolExecutor(max(1, workers)) as pool:
        for done, (item, outcome, stamp) in enumerate(pool.map(one, wanted), 1):
            counts[outcome] += 1
            state[item.relative]["failed"] = outcome == "failed"
            state[item.relative]["stamp"] = stamp
            if progress is not None:
                progress(done, len(wanted))
    write_sync_state(dest, state, single)
    return (dest / single if single is not None else dest), counts
