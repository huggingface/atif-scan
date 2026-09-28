"""Resolve CLI inputs to loadable trace sources. The only module that does input I/O.

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
Hub input is used, so local scans never touch the network.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from functools import cached_property
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from .checks import identifier
from .harbor_files import job_meta, number, trial_result
from .loader import MAX_BYTES, TraceError, load_bytes, load_trace
from .model import Trace

HF_PREFIX = "hf://"
HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
DEFAULT_PATTERN = "trajectory.json"
WEB_VIEWS = {"tree", "blob", "resolve"}


class SourceError(ValueError):
    """Fixed error codes only; never input paths or remote messages."""


def _no_reward() -> float | None:
    return None


@dataclass(frozen=True)
class Source:
    # Report label. Relative to the scanned root for expanded inputs, else `input-NNNN`.
    label: str
    load: Callable[[], Trace] = field(repr=False)
    # Verifier reward recorded next to the trajectory, read lazily; None when unknown.
    reward: Callable[[], float | None] = field(default=_no_reward, repr=False)
    # The input path as given plus the entry's relative path. Used only to derive a task
    # from Harbor folder names (--task-from); never reported.
    hint: str = field(default="", repr=False)
    # Allowlisted run facts about this trial (task, reward, error_type, cost, tokens, ...)
    # from its source, e.g. the Harbor Hub listing or Harbor's exception.txt marker.
    meta: Mapping[str, object] = field(default_factory=dict, repr=False)
    # Cheap identity of the trajectory file for the result cache (None: don't cache).
    fingerprint: Callable[[], str | None] = field(default=lambda: None, repr=False)
    # Recorded run facts read lazily from Harbor's trial result.json ({} when absent).
    details: Callable[[], dict] = field(default=dict, repr=False)
    # The trajectory's local file, when there is one (never reported; used to point
    # follow-up tooling such as the read-only trace tool at the same file).
    local: Path | None = field(default=None, repr=False)


def local_fingerprint(path: Path) -> Callable[[], str | None]:
    def fingerprint() -> str | None:
        try:
            st = path.stat()
        except OSError:
            return None
        return f"{path.resolve()}:{st.st_size}:{st.st_mtime_ns}"

    return fingerprint


REWARD_FILES = ("reward.json", "reward.txt")  # Harbor reads reward.json first
REWARD_BYTES = 4096
RESULT_BYTES = 1024 * 1024  # Harbor result.json / config.json


def parse_reward(data: bytes, name: str) -> float | None:
    """A finite number from Harbor's reward file; anything else is unknown (None)."""
    try:
        text = data[:REWARD_BYTES].decode("utf-8").strip()
        value = json.loads(text) if name.endswith(".json") else float(text)
    except (UnicodeError, ValueError):
        return None
    if isinstance(value, dict):
        value = value.get("reward")
    return number(value)


REWARD_NAMES = tuple(f"verifier/{name}" for name in REWARD_FILES)


def _near(path, names):
    """`<dir>/<name>` for the file's folder, then one level up (Harbor's trial folder:
    `<trial>/agent/trajectory.json` -> `<trial>/verifier/reward.*`). Path or PurePath."""
    parent = path.parent
    folders = [parent] + ([parent.parent] if parent != parent.parent else [])
    return [folder / name for folder in folders for name in names]


def reward_candidates(path: str) -> list[str]:
    """Relative reward-file paths for a listed trajectory, in lookup order."""
    return [p.as_posix() for p in _near(PurePosixPath(path), REWARD_NAMES)]


def _read_local(path: Path, limit: int) -> bytes | None:
    """Up to `limit` bytes of a local metadata file; None when missing or unreadable."""
    try:
        with open(path, "rb") as handle:
            return handle.read(limit)
    except OSError:
        return None


def confined(root: Path, relative: str) -> Path | None:
    """`root/relative` for a plain relative POSIX path; None if it could leave `root`
    (absolute, empty, `.` or `..` parts, NUL)."""
    parts = relative.split("/")
    if "\0" in relative or any(p in ("", ".", "..") for p in parts):
        return None
    return root.joinpath(*parts)


def hf_filesystem():
    from huggingface_hub import HfFileSystem
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()  # one atif-scan progress line, not a bar per file
    return HfFileSystem()


def normalize(value: str) -> str:
    """Translate a Hub web URL to `hf://`; reject other URL schemes. Paths pass through."""
    if value.startswith(HF_PREFIX):
        return value
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        return value
    url = urlsplit(value)
    if url.scheme not in {"http", "https"} or url.hostname not in HF_HOSTS:
        raise SourceError("unsupported_url_use_a_path_or_hf_url")
    parts = [p for p in url.path.split("/") if p]
    kind = parts[0] if parts and parts[0] in {"buckets", "datasets", "spaces"} else None
    head = parts[:3] if kind else parts[:2]  # models are just <ns>/<repo>
    rest = parts[len(head) :]
    if len(head) < (3 if kind else 2):
        raise SourceError("invalid_hf_url")
    if rest and rest[0] in WEB_VIEWS:
        if kind == "buckets":  # buckets have no revisions: /tree/<path>
            rest = rest[1:]
        elif len(rest) >= 2:  # repos: /tree/<revision>/<path>
            head[-1] += "@" + rest[1]
            rest = rest[2:]
        else:
            raise SourceError("invalid_hf_url")
    return HF_PREFIX + "/".join([*head, *(unquote(p) for p in rest)])


def _remote_load(fs, path: str, size: int | None) -> Callable[[], Trace]:
    def load() -> Trace:
        if size is not None and size > MAX_BYTES:
            raise TraceError("trace_too_large")
        try:
            with fs.open(path, "rb") as handle:
                data = handle.read(MAX_BYTES + 1)
        except Exception:
            # Remote errors can carry URLs or tokens in their message; withhold them.
            raise TraceError("unreadable_trace") from None
        return load_bytes(data)

    return load


@dataclass(frozen=True)
class Entry:
    path: str  # POSIX path relative to the listed root
    size: int | None


@dataclass(frozen=True)
class Listing:
    """Names and sizes under one input; no file contents are read to build it."""

    remote: bool
    directory: bool
    entries: tuple[Entry, ...]  # sorted; a single file is one entry with path ""
    opener: Callable[[Entry], Callable[[], Trace]] = field(repr=False)
    # Read (up to `limit` bytes of) a small listed metadata file by relative path.
    reader: Callable[..., bytes] | None = field(default=None, repr=False)
    # Local listings: the file path of an entry (for cache fingerprints).
    local_path: Callable[[Entry], Path] | None = field(default=None, repr=False)
    # Remote listings: download an entry to a local path (for --sync).
    fetch: Callable[[Entry, Path], None] | None = field(default=None, repr=False)

    @cached_property
    def paths(self) -> frozenset[str]:
        """Every listed path, for O(1) "is this file present" lookups."""
        return frozenset(e.path for e in self.entries)


def _list_local(value: str) -> Listing:
    root = Path(value)
    if root.is_file():
        entry = Entry("", root.stat().st_size)
        return Listing(
            False, False, (entry,), lambda e: lambda: load_trace(root), local_path=lambda e: root
        )
    if not root.is_dir():
        raise SourceError("path_not_found")
    entries = []
    # os.walk without following symlinks: the same traversal on every Python version.
    for folder, dirs, files in os.walk(root):
        dirs.sort()
        for name in files:
            path = Path(folder) / name
            try:
                st = path.lstat()
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):  # regular files only: no symlinks
                entries.append(Entry(path.relative_to(root).as_posix(), st.st_size))
    entries.sort(key=lambda e: e.path)

    def read(relative: str, limit: int = REWARD_BYTES) -> bytes:
        with open(root / relative, "rb") as handle:
            return handle.read(limit)

    return Listing(
        False,
        True,
        tuple(entries),
        lambda e: lambda: load_trace(root / e.path),
        read,
        local_path=lambda e: root / e.path,
    )


def _list_remote(value: str, fs) -> Listing:
    root = value[len(HF_PREFIX) :].rstrip("/")
    try:
        info = fs.info(root)
    except FileNotFoundError:
        raise SourceError("hf_path_not_found_or_no_access") from None
    except Exception:
        raise SourceError("hf_request_failed") from None
    if info.get("type") != "directory":
        entry = Entry("", info.get("size"))
        return Listing(
            True,
            False,
            (entry,),
            lambda e: _remote_load(fs, root, e.size),
            fetch=lambda e, path: fs.get_file(root, str(path)),
        )
    try:
        found = fs.find(root, detail=True)
    except Exception:
        raise SourceError("hf_listing_failed") from None
    full: dict[str, str] = {}  # relative -> remote path
    for path, meta in found.items():
        if meta.get("type") != "file":
            continue
        try:
            relative = PurePosixPath(path).relative_to(root).as_posix()
        except ValueError:  # not under the root (e.g. revision-qualified): skip, so the
            continue  # root never reaches a label or a sync path
        if confined(Path(), relative) is not None:
            full[relative] = path
    sizes = {path: meta.get("size") for path, meta in found.items()}
    entries = [Entry(r, sizes[full[r]]) for r in sorted(full)]

    def opener(entry: Entry) -> Callable[[], Trace]:
        return _remote_load(fs, full[entry.path], entry.size)

    def read(relative: str, limit: int = REWARD_BYTES) -> bytes:
        with fs.open(full[relative], "rb") as handle:
            return handle.read(limit)

    def fetch(entry: Entry, path: Path) -> None:
        fs.get_file(full[entry.path], str(path))

    return Listing(True, True, tuple(entries), opener, read, fetch=fetch)


def list_input(value: str, fs=None) -> Listing:
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


def reward_lookup(listing: Listing, entry: Entry, value: str) -> Callable[[], float | None]:
    """Find the reward next to a selected trajectory without any extra listing call."""
    if not listing.directory:
        if listing.remote:
            return _no_reward
        candidates = _near(Path(normalize(value)), REWARD_NAMES)

        def local() -> float | None:
            for candidate in candidates:
                if candidate.is_file():
                    data = _read_local(candidate, REWARD_BYTES)
                    return None if data is None else parse_reward(data, candidate.name)
            return None

        return local
    found = next((c for c in reward_candidates(entry.path) if c in listing.paths), None)
    if found is None or listing.reader is None:
        return _no_reward
    reader = listing.reader

    def listed() -> float | None:
        try:
            return parse_reward(reader(found), found)
        except Exception:
            return None  # unreadable reward is unknown, never an error or a zero

    return listed


def trial_details(listing: Listing, entry: Entry, value: str) -> Callable[[], dict]:
    """Lazy reader for the trial's Harbor result.json (trajectory folder or one up)."""
    if listing.directory:
        near = _near(PurePosixPath(entry.path), ["result.json"])
        found = [p for p in (f.as_posix() for f in near) if p in listing.paths]
        reader = listing.reader
        if not found or reader is None:
            return dict

        def listed() -> dict:
            for path in found:
                try:
                    facts = trial_result(reader(path, RESULT_BYTES))
                except Exception:
                    continue
                if facts:
                    return facts
            return {}

        return listed
    if listing.remote:
        return dict
    candidates = _near(Path(normalize(value)), ["result.json"])

    def local() -> dict:
        for candidate in candidates:
            if candidate.is_file() and (
                facts := trial_result(_read_local(candidate, RESULT_BYTES))
            ):
                return facts
        return {}

    return local


def job_runs(listing: Listing) -> list[dict]:
    """Harbor job facts for every job folder in the listing, at any depth.

    A job folder has config.json and at least one subfolder with a result.json (its
    trials); trial folders' own agent/ and verifier/ subfolders never do, so trial
    configs aren't read. job_meta then rejects anything that isn't a job config.
    """
    if not listing.directory or listing.reader is None:
        return []
    present = listing.paths
    has_trial_child = {
        str(PurePosixPath(p).parent.parent)
        for p in present
        if PurePosixPath(p).name == "result.json" and "/" in p
    }
    candidates = sorted(
        d
        for d in {
            str(PurePosixPath(p).parent) for p in present if PurePosixPath(p).name == "config.json"
        }
        if d in has_trial_child
    )
    runs = []
    for folder in candidates:
        prefix = "" if folder == "." else folder + "/"
        try:
            config = listing.reader(prefix + "config.json", RESULT_BYTES)
            result_path = prefix + "result.json"
            result = listing.reader(result_path, RESULT_BYTES) if result_path in present else None
        except Exception:
            continue
        if (run := job_meta(config, result)) is not None:
            runs.append(run)
    return runs


def errored(listing: Listing, entry: Entry) -> bool:
    """Harbor writes `<trial>/exception.txt` when a trial raised; seen in the listing."""
    if not listing.directory:
        return False
    near = _near(PurePosixPath(entry.path), ["exception.txt"])
    return any(p.as_posix() in listing.paths for p in near)


# Besides matching trajectories, the small files the scan reads next to them, and the
# largest size synced for each (exception.txt is only checked for presence).
SYNC_CAPS = {
    "result.json": RESULT_BYTES,
    "config.json": RESULT_BYTES,
    "reward.txt": REWARD_BYTES,
    "reward.json": REWARD_BYTES,
    "exception.txt": RESULT_BYTES,
}
SYNC_NAMES = frozenset(SYNC_CAPS)


def default_sync_root() -> Path:
    """Where remote inputs are synced: $ATIF_SCAN_SYNC_DIR, $XDG_CACHE_HOME/atif-scan, or
    ~/.cache/atif-scan. Contents are real traces: documented, and safe to delete."""
    if os.environ.get("ATIF_SCAN_SYNC_DIR"):
        return Path(os.environ["ATIF_SCAN_SYNC_DIR"])
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "atif-scan"


def sync_target(value: str, root: Path) -> Path:
    """Local mirror folder for an hf:// input: <root>/hf/<path>, no traversal."""
    parts = [p for p in normalize(value)[len(HF_PREFIX) :].split("/") if p]
    if not parts or any(p in (".", "..") for p in parts):
        raise SourceError("invalid_hf_path")
    return root.joinpath("hf", *parts)


def sync_remote(
    value: str,
    dest: Path,
    pattern: str = DEFAULT_PATTERN,
    fs=None,
    workers: int = 16,
    refresh: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, dict]:
    """Mirror what a scan needs from an hf:// input into `dest` (parallel, resumable:
    files already present with the listed size are kept unless `refresh`). Returns the
    local path to scan and download counts."""
    listing = list_input(value, fs)
    if listing.fetch is None:
        raise SourceError("not_a_remote_input")
    if not listing.directory:
        name = PurePosixPath(normalize(value)).name
        scan_path = confined(dest, name)
        if scan_path is None:
            raise SourceError("invalid_hf_path")
        wanted = [(listing.entries[0], scan_path, MAX_BYTES)]
    else:
        wanted = []
        for e in listing.entries:
            name = PurePosixPath(e.path).name
            cap = MAX_BYTES if fnmatchcase(name, pattern) else SYNC_CAPS.get(name)
            if cap is not None:
                wanted.append((e, confined(dest, e.path), cap))
        scan_path = dest
    counts = {"files": len(wanted), "downloaded": 0, "up_to_date": 0, "failed": 0}

    def one(item: tuple[Entry, Path | None, int]) -> str:
        entry, path, cap = item
        if path is None or (entry.size or 0) > cap:
            return "failed"  # would leave `dest`, or too large to read: never written
        if (
            not refresh
            and path.is_file()
            and (entry.size is None or path.stat().st_size == entry.size)
        ):
            return "up_to_date"
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        try:
            listing.fetch(entry, partial)
            partial.replace(path)
        except Exception:
            partial.unlink(missing_ok=True)
            return "failed"  # the scan then reports it as unreadable
        return "downloaded"

    dest.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for done, outcome in enumerate(pool.map(one, wanted), 1):
            counts[outcome] += 1
            if progress is not None:
                progress(done, len(wanted))
    return scan_path, counts


def file_source(label: str, location: str, fs=None) -> Source:
    """A single named file (manifest entries): no expansion; failures surface on load."""
    location = normalize(location)
    if location.startswith(HF_PREFIX):
        fs = fs or hf_filesystem()
        return Source(label, _remote_load(fs, location[len(HF_PREFIX) :], None))
    return Source(label, lambda: load_trace(Path(location)), local=Path(location))


def resolve(
    values: list[str], pattern: str = DEFAULT_PATTERN, fs=None, runs: list | None = None
) -> list[Source]:
    """Expand each value in order (see module docstring).

    Expanded files are labelled by their path relative to the given root (the part the
    caller chose to scan), minus the file name when it is the literal pattern. Single
    files, and relative paths that are not valid identifiers, get a positional
    `input-NNNN` label, so absolute paths never reach a report.
    """
    sources: list[Source] = []
    for value in values:
        value = normalize(value)
        listing = list_input(value, fs)
        if runs is not None:
            runs.extend(job_runs(listing))
        entries = selected(listing, pattern)
        if not entries:
            raise SourceError("no_files_match_pattern")
        for entry in entries:
            label = label_for(entry, pattern) or f"input-{len(sources) + 1:04d}"
            reward = reward_lookup(listing, entry, value)
            hint = "/".join(p for p in (value.rstrip("/"), entry.path) if p)
            meta = {"error_type": "exception"} if errored(listing, entry) else {}
            local = listing.local_path(entry) if listing.local_path is not None else None
            fingerprint = local_fingerprint(local) if local is not None else (lambda: None)
            details = trial_details(listing, entry, value)
            sources.append(
                Source(
                    label, listing.opener(entry), reward, hint, meta, fingerprint, details, local
                )
            )
    if len({s.label for s in sources}) != len(sources):
        raise SourceError("duplicate_input_labels")
    return sources
