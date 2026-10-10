"""Resolve CLI inputs to loadable trace sources. The only module that reads input files.

Accepted inputs, per positional argument:

- a local file: scanned as-is, whatever its name;
- a local directory: every file below it whose *name* matches `pattern` (default
  `trajectory.json`), recursively, in sorted order;
- `hf://buckets/<ns>/<bucket>/<path>`, `hf://datasets/<ns>/<repo>[@rev]/<path>`, ...:
  the same file/directory rules on the Hugging Face Hub;
- a Hub web URL (`https://huggingface.co/buckets/.../tree/...`, `.../blob/...`,
  `.../resolve/...`), which is translated to the `hf://` form above.

A path that doesn't exist, or a directory with no matching files, is an error rather
than an empty (and misleadingly clean) report. `huggingface_hub` is imported only when a
Hub input is used, so local scans never touch the network. A local mirror written by
`sync` is listed from its inventory, so files that failed to sync stay in the scan as
unavailable inputs. The Harbor run files next to each trajectory are found by
`harbor_runs`; which of them wins for each run fact is `facts`' decision.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, cast
from urllib.parse import unquote, urlsplit

from ..data.facts import listed_facts
from ..data.jsonval import Doc, count, identifier
from ..data.loader import MAX_BYTES, TraceError, load_bytes, load_trace
from .harbor.runs import (
    REWARD_BYTES,
    SavedRun,
    errored,
    hub_trial,
    job_folders,
    listed_without_trajectory,
    reward_lookup,
    saved_hub_listings,
    trial_details,
    trial_folder,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import BinaryIO

    from ..data.model import Trace

HF_PREFIX = "hf://"
HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
DEFAULT_PATTERN = "trajectory.json"
WEB_VIEWS = {"tree", "blob", "resolve"}
URL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
# Hub URL kinds whose repo is <kind>/<ns>/<name>; models are just <ns>/<name>.
HUB_KINDS = {"buckets", "datasets", "spaces"}
# Content identities supplied by Hub repos and buckets (not size or timestamps).
REVISION_KEYS = ("xet_hash", "blob_id", "etag")

# A local mirror's inventory (see `sync`), next to the mirrored files.
SYNC_STATE = ".atif-sync.json"
SYNC_STATE_BYTES = 16 * 1024 * 1024
SYNC_ROW_KEYS = {"size", "revision", "stamp", "failed"}


class SourceError(ValueError):
    """Fixed error codes only; never input paths or remote messages."""


class RemoteFS(Protocol):
    """The part of `huggingface_hub.HfFileSystem` (fsspec) a scan uses."""

    def info(self, path: str, /) -> Mapping[str, Any]: ...

    def find(self, path: str, /, detail: bool = False) -> Mapping[str, Mapping[str, Any]]: ...

    def open(self, path: str, mode: str, /) -> BinaryIO: ...

    def get_file(self, rpath: str, lpath: str, /) -> None: ...


class SyncRow(TypedDict):
    size: int | None
    revision: str | None  # provider content identity
    stamp: str | None  # the local copy's size and mtime when written
    failed: bool


class SyncState(TypedDict):
    schema_version: int
    files: dict[str, SyncRow]
    single: str | None  # a mirrored single file's name


def _no_reward() -> float | None:
    return None


def _no_fingerprint() -> str | None:
    return None


def _no_details() -> Doc:
    return {}


@dataclass(frozen=True)
class Source:
    # Report label. Relative to the scanned root for expanded inputs, else `input-NNNN`.
    label: str
    load: Callable[[], Trace] = field(repr=False)
    # Verifier reward recorded next to the trajectory (or given by a manifest), read
    # lazily; None when unknown.
    reward: Callable[[], float | None] = field(default=_no_reward, repr=False)
    # The input path as given plus the entry's relative path. Used only to derive a task
    # from Harbor folder names (--task-from); never reported.
    hint: str = field(default="", repr=False)
    # Allowlisted run facts about this trial from its run listing (task, reward,
    # error_type, cost, tokens, ...): the Harbor Hub listing or a trials.jsonl ledger,
    # plus Harbor's exception.txt marker and job membership (facts.listed_facts).
    meta: Mapping[str, object] = field(default_factory=dict, repr=False)
    # Cheap identity of the trajectory file for the result cache (None: don't cache).
    fingerprint: Callable[[], str | None] = field(default=_no_fingerprint, repr=False)
    # Recorded run facts read lazily from Harbor's trial result.json ({} when absent).
    details: Callable[[], Doc] = field(default=_no_details, repr=False)
    # The trajectory's local file, when there is one (never reported; used to point
    # follow-up tooling such as the read-only trace tool at the same file).
    local: Path | None = field(default=None, repr=False)
    # Whether this trial counts (`role`: canonical, replaced, not_selected) and its
    # replacement lineage (codes and trial names), from --run/--release; {} otherwise.
    selection: Mapping[str, object] = field(default_factory=dict, repr=False)


def local_fingerprint(path: Path) -> Callable[[], str | None]:
    def fingerprint() -> str | None:
        try:
            st = path.stat()
        except OSError:
            return None
        return f"{path.resolve()}:{st.st_size}:{st.st_mtime_ns}"

    return fingerprint


def confined(root: Path, relative: str) -> Path | None:
    """`root/relative` for a plain relative POSIX path; None if it could leave `root`
    (absolute, empty, `.` or `..` parts, NUL)."""
    parts = relative.split("/")
    if "\0" in relative or any(p in ("", ".", "..") for p in parts):
        return None
    return root.joinpath(*parts)


def hf_filesystem() -> RemoteFS:
    from huggingface_hub import HfFileSystem  # noqa: PLC0415 - optional, hf:// inputs only
    from huggingface_hub.utils import disable_progress_bars  # noqa: PLC0415 - as above

    disable_progress_bars()  # one atif-scan progress line, not a bar per file
    # Trust boundary with fsspec's untyped/overloaded signatures: RemoteFS is the subset
    # used here, with the same parameters.
    return cast("RemoteFS", HfFileSystem())


def normalize(value: str) -> str:
    """Translate a Hub web URL to `hf://`; reject other URL schemes. Paths pass through."""
    if value.startswith(HF_PREFIX) or not URL_SCHEME.match(value):
        return value
    url = urlsplit(value)
    if url.scheme not in {"http", "https"} or url.hostname not in HF_HOSTS:
        raise SourceError("unsupported_url_use_a_path_or_hf_url")
    return HF_PREFIX + "/".join(_hub_path([p for p in url.path.split("/") if p]))


def _hub_path(parts: list[str]) -> list[str]:
    """`<kind>/<ns>/<repo>[@rev]/<path>` parts from a Hub web URL's path parts."""
    kind = parts[0] if parts and parts[0] in HUB_KINDS else None
    size = 3 if kind else 2
    head, rest = parts[:size], parts[size:]
    if len(head) < size:
        raise SourceError("invalid_hf_url")
    if rest and rest[0] in WEB_VIEWS:
        if kind == "buckets":  # buckets have no revisions: /tree/<path>
            rest = rest[1:]
        elif len(rest) > 1:  # repos: /tree/<revision>/<path>
            head[-1] += "@" + rest[1]
            rest = rest[2:]
        else:
            raise SourceError("invalid_hf_url")
    return [*head, *(unquote(p) for p in rest)]


def _remote_load(fs: RemoteFS, path: str, size: int | None) -> Callable[[], Trace]:
    def load() -> Trace:
        if size is not None and size > MAX_BYTES:
            raise TraceError("trace_too_large")
        try:
            with fs.open(path, "rb") as handle:
                data = handle.read(MAX_BYTES + 1)
        except Exception:  # noqa: BLE001 - provider errors vary; details are withheld
            # Remote errors can carry URLs or tokens in their message; withhold them.
            raise TraceError("unreadable_trace") from None
        return load_bytes(data)

    return load


@dataclass(frozen=True)
class Entry:
    path: str  # POSIX path relative to the listed root
    size: int | None
    revision: str | None = None  # provider content identity, never report text
    failed: bool = False


@dataclass(frozen=True)
class Listing:
    """Names and sizes under one input; no file contents are read to build it."""

    remote: bool
    directory: bool
    entries: tuple[Entry, ...]  # sorted; a single file is one entry with path ""
    opener: Callable[[Entry], Callable[[], Trace]] = field(repr=False)
    # Read (up to `limit` bytes of) a small listed metadata file by relative path.
    reader: Callable[[str, int], bytes] | None = field(default=None, repr=False)
    # Local listings: the file path of an entry (for cache fingerprints).
    local_path: Callable[[Entry], Path] | None = field(default=None, repr=False)
    # Remote listings: download an entry to a local path (for --sync).
    fetch: Callable[[Entry, Path], None] | None = field(default=None, repr=False)

    sync_failed: int = 0

    @cached_property
    def paths(self) -> frozenset[str]:
        """Every listed path, for O(1) "is this file present" lookups."""
        return frozenset(e.path for e in self.entries)

    def read(self, relative: str, limit: int) -> bytes | None:
        """Up to `limit` bytes of a listed metadata file; None when it can't be read,
        which callers treat as unknown (never an error, a zero or a negative)."""
        if self.reader is None:
            return None
        try:
            return self.reader(relative, limit)
        except Exception:  # noqa: BLE001 - local or provider errors; the file is unknown
            return None


def _has_state(folder: Path) -> bool:
    state = folder / SYNC_STATE
    return state.exists() or state.is_symlink()


def _list_local(value: str) -> Listing:
    root = Path(value)
    if root.is_dir() and _has_state(root):
        return synced_listing(root)
    if _has_state(root.parent) and root.name in _sync_state(root.parent)["files"]:
        return synced_listing(root.parent, root.name)
    if root.is_file():
        entry = Entry("", root.stat().st_size)
        return Listing(
            False, False, (entry,), lambda _: lambda: load_trace(root), local_path=lambda _: root
        )
    if not root.is_dir():
        raise SourceError("path_not_found")
    return _local_tree(root)


def _regular_files(root: Path, folder: Path, names: list[str]) -> list[Entry]:
    """Regular files only (no symlinks), as entries relative to `root`."""
    entries = []
    for name in names:
        path = folder / name
        try:
            st = path.lstat()
        except OSError:
            continue
        if stat.S_ISREG(st.st_mode):
            entries.append(Entry(path.relative_to(root).as_posix(), st.st_size))
    return entries


@dataclass
class _Tree:
    """What a walk of a local folder found, including mirrors nested below it."""

    entries: list[Entry] = field(default_factory=list)
    synced_loaders: dict[str, Callable[[], Trace]] = field(default_factory=dict)
    failed_paths: set[str] = field(default_factory=set)
    sync_failed: int = 0

    def add_mirror(self, root: Path, folder: Path) -> None:
        """A parent of several mirrors must retain their missing inputs too."""
        nested, local = _synced(folder)
        self.sync_failed += nested.sync_failed
        for entry in nested.entries:
            relative = local(entry).relative_to(root).as_posix()
            self.entries.append(Entry(relative, entry.size, failed=entry.failed))
            self.synced_loaders[relative] = nested.opener(entry)
            if entry.failed:
                self.failed_paths.add(relative)


def _local_tree(root: Path) -> Listing:
    tree = _Tree()
    # os.walk without following symlinks: the same traversal on every Python version.
    for folder, dirs, files in os.walk(root):
        dirs.sort()
        if SYNC_STATE in files:
            tree.add_mirror(root, Path(folder))
            dirs.clear()
        else:
            tree.entries.extend(_regular_files(root, Path(folder), files))
    entries = sorted(tree.entries, key=lambda e: e.path)
    synced_loaders, failed_paths = tree.synced_loaders, tree.failed_paths

    def opener(entry: Entry) -> Callable[[], Trace]:
        return synced_loaders.get(entry.path) or (lambda: load_trace(root / entry.path))

    def read(relative: str, limit: int = REWARD_BYTES) -> bytes:
        if relative in failed_paths:
            raise OSError("sync_failed")
        with (root / relative).open("rb") as handle:
            return handle.read(limit)

    return Listing(
        False,
        True,
        tuple(entries),
        opener,
        read,
        local_path=lambda e: root / e.path,
        sync_failed=tree.sync_failed,
    )


def remote_revision(info: Mapping[str, object]) -> str | None:
    """Content identities supplied by Hub repos and buckets, not size or timestamps."""
    for key in REVISION_KEYS:
        value = info.get(key)
        if isinstance(value, str) and value:
            return f"{key}:{value}"
    return None


def _list_remote(value: str, fs: RemoteFS) -> Listing:
    root = value[len(HF_PREFIX) :].rstrip("/")
    try:
        # fsspec may reuse HfFileSystem instances, including their directory caches.
        # A fresh mirror decision must use a fresh remote inventory.
        invalidate = getattr(fs, "invalidate_cache", None)
        if callable(invalidate):
            invalidate()
        info = fs.info(root)
    except FileNotFoundError:
        raise SourceError("hf_path_not_found_or_no_access") from None
    except Exception:  # noqa: BLE001 - provider errors vary; details are withheld
        raise SourceError("hf_request_failed") from None
    if info.get("type") != "directory":
        entry = Entry("", info.get("size"), remote_revision(info))
        return Listing(
            True,
            False,
            (entry,),
            lambda e: _remote_load(fs, root, e.size),
            fetch=lambda _, path: fs.get_file(root, str(path)),
        )
    return _remote_tree(fs, root)


def _remote_files(found: Mapping[str, Mapping[str, Any]], root: str) -> dict[str, str]:
    """{relative path: remote path} for the listed files under `root`."""
    full: dict[str, str] = {}
    for path, meta in found.items():
        if meta.get("type") != "file":
            continue
        try:
            relative = PurePosixPath(path).relative_to(root).as_posix()
        except ValueError:  # not under the root (e.g. revision-qualified): skip, so the
            continue  # root never reaches a label or a sync path
        if confined(Path(), relative) is not None:
            full[relative] = path
    return full


def _remote_tree(fs: RemoteFS, root: str) -> Listing:
    try:
        found = fs.find(root, detail=True)
    except Exception:  # noqa: BLE001 - provider errors vary; details are withheld
        raise SourceError("hf_listing_failed") from None
    full = _remote_files(found, root)
    entries = [
        Entry(r, found[full[r]].get("size"), remote_revision(found[full[r]])) for r in sorted(full)
    ]

    def opener(entry: Entry) -> Callable[[], Trace]:
        return _remote_load(fs, full[entry.path], entry.size)

    def read(relative: str, limit: int = REWARD_BYTES) -> bytes:
        with fs.open(full[relative], "rb") as handle:
            return handle.read(limit)

    def fetch(entry: Entry, path: Path) -> None:
        fs.get_file(full[entry.path], str(path))

    return Listing(True, True, tuple(entries), opener, read, fetch=fetch)


def list_input(value: str, fs: RemoteFS | None = None) -> Listing:
    """List one CLI input (see module docstring). The only traversal code."""
    value = normalize(value)
    if value.startswith(HF_PREFIX):
        return _list_remote(value, fs or hf_filesystem())
    return _list_local(value)


def selected(listing: Listing, pattern: str) -> list[Entry]:
    """What a scan reads: the file itself, or directory entries whose *name* matches."""
    if not listing.directory:
        return list(listing.entries)
    return [e for e in listing.entries if fnmatchcase(PurePosixPath(e.path).name, pattern)]


def label_for(entry: Entry, pattern: str) -> str | None:
    """Relative path minus an implied file name; None means "use a positional label"."""
    if not entry.path:
        return None
    folder, _, name = entry.path.rpartition("/")
    relative = folder if folder and name == pattern else entry.path
    try:
        return identifier(relative)
    except ValueError:
        return None


def _valid_row(root: Path, name: object, row: object) -> bool:
    if not isinstance(name, str) or confined(root, name) is None or name == SYNC_STATE:
        return False
    if not isinstance(row, dict) or set(row) != SYNC_ROW_KEYS:
        return False
    return (
        (row["size"] is None or count(row["size"]) is not None)
        and (row["revision"] is None or isinstance(row["revision"], str))
        and (row["stamp"] is None or isinstance(row["stamp"], str))
        and type(row["failed"]) is bool
    )


def _load_state(root: Path) -> SyncState:
    """The inventory as written by `sync.write_sync_state`; ValueError (or KeyError,
    TypeError, OSError) when it isn't exactly that."""
    path = root / SYNC_STATE
    if path.is_symlink():
        raise ValueError
    with path.open("rb") as handle:
        data = handle.read(SYNC_STATE_BYTES + 1)
    if len(data) > SYNC_STATE_BYTES:
        raise ValueError
    value = json.loads(data)
    if value["schema_version"] != 1 or not isinstance(value["files"], dict):
        raise ValueError
    if value["single"] is not None and value["single"] not in value["files"]:
        raise ValueError
    if not all(_valid_row(root, name, row) for name, row in value["files"].items()):
        raise ValueError
    # Trust boundary: every field was checked above, so the JSON is a SyncState.
    return cast("SyncState", value)


def read_sync_state(root: Path) -> SyncState | None:
    """A mirror's inventory, None without one. Malformed state fails closed, never
    becomes an ordinary directory scan."""
    if not _has_state(root):
        return None
    try:
        return _load_state(root)
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        raise SourceError("invalid_sync_inventory") from None


def _sync_state(root: Path) -> SyncState:
    """The inventory of a folder known to hold one (it can't vanish into a plain scan)."""
    state = read_sync_state(root)
    if state is None:
        raise SourceError("invalid_sync_inventory")
    return state


def synced_listing(root: Path, single: str | None = None) -> Listing:
    return _synced(root, single)[0]


def _refuse_links(root: Path, names: list[str]) -> None:
    """A mirror's listed files and their folders must not be symlinks."""
    for name in names:
        path = root / name
        while path != root:
            if path.is_symlink():
                raise SourceError("unsafe_sync_path")
            path = path.parent


def _synced(root: Path, single: str | None = None) -> tuple[Listing, Callable[[Entry], Path]]:
    """A mirror's listing, from its inventory, and each entry's local file."""
    state = _sync_state(root)
    single = single or state["single"]
    files = state["files"]
    names = [single] if single is not None else sorted(files)
    _refuse_links(root, names)
    entries = tuple(
        Entry("" if single is not None else name, files[name]["size"], failed=files[name]["failed"])
        for name in names
    )

    def local(entry: Entry) -> Path:
        return root / (single or entry.path)

    def opener(entry: Entry) -> Callable[[], Trace]:
        def load() -> Trace:
            if entry.failed:
                raise TraceError("sync_failed")
            return load_trace(local(entry))

        return load

    def read(relative: str, limit: int = REWARD_BYTES) -> bytes:
        if files[relative]["failed"]:
            raise OSError("sync_failed")
        with (root / relative).open("rb") as handle:
            return handle.read(limit)

    listing = Listing(
        False,
        single is None,
        entries,
        opener,
        read,
        local,
        sync_failed=sum(row["failed"] for row in files.values()),
    )
    return listing, local


def file_source(label: str, location: str, fs: RemoteFS | None = None) -> Source:
    """A single named file (manifest entries): no expansion; failures surface on load."""
    location = normalize(location)
    if location.startswith(HF_PREFIX):
        fs = fs or hf_filesystem()
        return Source(label, _remote_load(fs, location[len(HF_PREFIX) :], None))
    return Source(label, lambda: load_trace(Path(location)), local=Path(location))


def _trial_source(
    listing: Listing,
    entry: Entry,
    value: str,
    label: str,
    saved: dict[str, SavedRun],
    membership: Mapping[str, bool],
) -> Source:
    """One selected trajectory with the run files beside it (see harbor_runs, facts)."""
    meta = listed_facts(
        hub_trial(saved, entry), errored(listing, entry), membership.get(trial_folder(entry))
    )
    local = listing.local_path(entry) if listing.local_path is not None else None
    fingerprint = (
        local_fingerprint(local) if local is not None and not entry.failed else _no_fingerprint
    )
    return Source(
        label,
        listing.opener(entry),
        reward_lookup(listing, entry, value),
        "/".join(p for p in (value.rstrip("/"), entry.path) if p),
        meta,
        fingerprint,
        trial_details(listing, entry, value),
        local,
    )


def _untraced(label: str, facts: Doc) -> Source:
    """A trial its saved Hub listing names but whose trajectory isn't in the folder:
    unavailable with its listed facts, exactly as a harbor:// scan reports it."""

    def load() -> Trace:
        raise TraceError("no_trajectory_downloaded")

    return Source(label, load, meta=facts)


def _listing_sources(
    listing: Listing,
    value: str,
    pattern: str,
    saved: dict[str, SavedRun],
    membership: Mapping[str, bool],
    before: int,
) -> list[Source]:
    """One input's selected trajectories, then the trials its saved Hub listing names
    without one (unavailable, as a harbor:// scan reports them). `before` counts the
    sources already resolved, for positional labels."""
    entries = selected(listing, pattern)
    if not entries:
        raise SourceError("no_files_match_pattern")
    sources = [
        _trial_source(
            listing,
            entry,
            value,
            label_for(entry, pattern) or f"input-{before + n + 1:04d}",
            saved,
            membership,
        )
        for n, entry in enumerate(entries)
    ]
    if pattern == DEFAULT_PATTERN:
        sources += [
            _untraced(label, facts) for label, facts in listed_without_trajectory(saved, entries)
        ]
    return sources


def resolve(
    values: Sequence[str],
    pattern: str = DEFAULT_PATTERN,
    fs: RemoteFS | None = None,
    runs: list[Doc] | None = None,
    sync_failures: list[int] | None = None,
) -> list[Source]:
    """Expand each value in order (see module docstring).

    Expanded files are labelled by their path relative to the given root (the part the
    caller chose to scan), minus the file name when it is the literal pattern. Single
    files, and relative paths that are not valid identifiers, get a positional
    `input-NNNN` label, so absolute paths never reach a report. Harbor job run facts
    found on the way are appended to `runs`, mirror files that failed to sync counted
    per input in `sync_failures`.
    """
    sources: list[Source] = []
    for given in values:
        value = normalize(given)
        listing = list_input(value, fs)
        if sync_failures is not None:
            sync_failures.append(listing.sync_failed)
        saved = saved_hub_listings(listing)
        found_runs, membership = job_folders(listing)
        if runs is not None:
            runs.extend(found_runs)
            runs.extend(run for run, _ in saved.values() if run)
        sources += _listing_sources(listing, value, pattern, saved, membership, len(sources))
    if len({s.label for s in sources}) != len(sources):
        raise SourceError("duplicate_input_labels")
    return sources
