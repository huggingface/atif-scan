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
import tempfile
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from functools import cached_property
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from .checks import identifier
from .harbor_files import (
    ATTEMPT_COST_BYTES,
    ATTEMPT_COSTS,
    LEDGER_BYTES,
    RUN_MANIFEST,
    RUN_MANIFEST_BYTES,
    TRIAL_LEDGER,
    attempt_cost,
    declared_prices,
    job_listed_trials,
    job_meta,
    number,
    trial_ledger,
    trial_result,
)
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
    reader: Callable[..., bytes] | None = field(default=None, repr=False)
    # Local listings: the file path of an entry (for cache fingerprints).
    local_path: Callable[[Entry], Path] | None = field(default=None, repr=False)
    # Remote listings: download an entry to a local path (for --sync).
    fetch: Callable[[Entry, Path], None] | None = field(default=None, repr=False)

    sync_failed: int = 0

    @cached_property
    def paths(self) -> frozenset[str]:
        """Every listed path, for O(1) "is this file present" lookups."""
        return frozenset(e.path for e in self.entries)


def _list_local(value: str) -> Listing:
    root = Path(value)
    if root.is_dir() and ((root / SYNC_STATE).exists() or (root / SYNC_STATE).is_symlink()):
        return synced_listing(root)
    if (root.parent / SYNC_STATE).exists() or (root.parent / SYNC_STATE).is_symlink():
        state = read_sync_state(root.parent)
        if root.name in state["files"]:
            return synced_listing(root.parent, root.name)
    if root.is_file():
        entry = Entry("", root.stat().st_size)
        return Listing(
            False, False, (entry,), lambda e: lambda: load_trace(root), local_path=lambda e: root
        )
    if not root.is_dir():
        raise SourceError("path_not_found")
    entries = []
    synced_loaders = {}
    failed_paths = set()
    sync_failed = 0
    # os.walk without following symlinks: the same traversal on every Python version.
    for folder, dirs, files in os.walk(root):
        dirs.sort()
        if SYNC_STATE in files:
            # Scanning a parent of several mirrors must retain their missing inputs too.
            nested = synced_listing(Path(folder))
            sync_failed += nested.sync_failed
            for entry in nested.entries:
                relative = nested.local_path(entry).relative_to(root).as_posix()
                entries.append(Entry(relative, entry.size, failed=entry.failed))
                synced_loaders[relative] = nested.opener(entry)
                if entry.failed:
                    failed_paths.add(relative)
            dirs.clear()
            continue
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
        if relative in failed_paths:
            raise OSError("sync_failed")
        with open(root / relative, "rb") as handle:
            return handle.read(limit)

    return Listing(
        False,
        True,
        tuple(entries),
        lambda e: synced_loaders.get(e.path, lambda: load_trace(root / e.path)),
        read,
        local_path=lambda e: root / e.path,
        sync_failed=sync_failed,
    )


def remote_revision(info: Mapping) -> str | None:
    """Content identities supplied by Hub repos and buckets, not size or timestamps."""
    for key in ("xet_hash", "blob_id", "etag"):
        value = info.get(key)
        if isinstance(value, str) and value:
            return f"{key}:{value}"
    return None


def _list_remote(value: str, fs) -> Listing:
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
    except Exception:
        raise SourceError("hf_request_failed") from None
    if info.get("type") != "directory":
        entry = Entry("", info.get("size"), remote_revision(info))
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
    entries = [Entry(r, sizes[full[r]], remote_revision(found[full[r]])) for r in sorted(full)]

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
                    return with_attempt_cost(listing, path, facts)
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


def with_attempt_cost(listing: Listing, result_path: str, facts: dict) -> dict:
    """Trial facts plus the cost harbor-hf recorded beside the trial
    (`<run>/attempt-costs/<attempt id>.json`, the job folder being `<run>/job/`) as
    `attempt_cost_usd`. The attempt id is a lookup key only and is dropped."""
    facts = dict(facts)
    attempt = facts.pop("attempt_id", None)
    if attempt is None or listing.reader is None:
        return facts
    trial = PurePosixPath(result_path).parent
    for run in (trial.parent.parent, trial.parent):
        path = (run / ATTEMPT_COSTS / f"{attempt}.json").as_posix()
        if path in listing.paths:
            try:
                cost = attempt_cost(listing.reader(path, ATTEMPT_COST_BYTES), attempt, trial.name)
            except Exception:
                return facts  # unreadable: unknown, never a zero
            if cost is not None:
                facts["attempt_cost_usd"] = cost
            return facts
    return facts


def run_prices(listing: Listing, folder: str) -> dict | None:
    """Prices declared by harbor-hf's `run.json` beside (or inside) a job folder."""
    job = PurePosixPath(folder)
    for run in (job.parent, job) if folder != "." else (job,):
        path = (run / RUN_MANIFEST).as_posix()
        if path in listing.paths and listing.reader is not None:
            try:
                return declared_prices(listing.reader(path, RUN_MANIFEST_BYTES))
            except Exception:
                return None
    return None


def job_runs(listing: Listing) -> list[dict]:
    """Harbor job facts for every job folder in the listing, at any depth."""
    return job_folders(listing)[0]


def job_folders(listing: Listing) -> tuple[list[dict], dict[str, bool]]:
    """(run facts per Harbor job folder, {trial folder: listed in its job's result.json}).

    A job folder has config.json and at least one subfolder with a result.json (its
    trials); trial folders' own agent/ and verifier/ subfolders never do, so trial
    configs aren't read. job_meta then rejects anything that isn't a job config.

    Trial folders the job's result.json doesn't account for usually come from another
    execution of the same job (a rerun or resume writing into the folder, possibly while
    the first one still ran). They are still scanned; the run facts count them. When the
    job's own listing is missing or incomplete, membership is unknown (not recorded).
    """
    if not listing.directory or listing.reader is None:
        return [], {}
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
    runs, membership = [], {}
    for folder in candidates:
        prefix = "" if folder == "." else folder + "/"
        try:
            config = listing.reader(prefix + "config.json", RESULT_BYTES)
            result_path = prefix + "result.json"
            result = listing.reader(result_path, RESULT_BYTES) if result_path in present else None
        except Exception:
            continue
        if (run := job_meta(config, result)) is None:
            continue
        if (prices := run_prices(listing, folder)) is not None:
            run["declared_prices"] = prices
        listed = job_listed_trials(result)
        if listed is not None:
            trials, traced = set(), set()
            for p in present:
                if not p.startswith(prefix):
                    continue
                parts = PurePosixPath(p[len(prefix) :]).parts
                if len(parts) == 2 and parts[1] == "result.json":
                    trials.add(parts[0])
                elif parts[1:] in (("agent", "trajectory.json"), ("trajectory.json",)):
                    trials.add(parts[0])
                    traced.add(parts[0])
            unlisted = trials - listed
            run["trial_folders"] = len(trials)
            run["unlisted_trials"] = len(unlisted)
            run["unlisted_with_trajectory"] = len(unlisted & traced)
            membership.update({prefix + t: t in listed for t in trials})
        runs.append(run)
    return runs, membership


def trial_folder(entry: Entry) -> str:
    """The trial folder of a trajectory: `<trial>/trajectory.json` or `<trial>/agent/...`."""
    trial = PurePosixPath(entry.path).parent
    return str(trial.parent if trial.name == "agent" else trial)


LISTING_BYTES = 16 * 1024 * 1024  # a saved Hub listing: ~500 bytes per trial


def saved_hub_listings(listing: Listing) -> dict[str, tuple[dict | None, dict[str, dict]]]:
    """Saved Harbor Hub listings (harbor_hub.save_listing) and `trials.jsonl` run ledgers
    by the folder holding them: (run facts, {trial folder: trial facts}), so a scan of a
    synced job keeps its recorded facts (reward, task, error, cost). A ledger lists the
    trials of the folder it sits in or of its `trials/` subfolder (see hub_trial)."""
    from .harbor_hub import SAVED_LISTING, saved_listing  # harbor_hub imports this module

    if not listing.directory or listing.reader is None:
        return {}
    out = {}
    for path in sorted(listing.paths):
        if PurePosixPath(path).name == TRIAL_LEDGER:
            try:
                trials = trial_ledger(listing.reader(path, LEDGER_BYTES))
            except Exception:
                continue  # unreadable: the trials simply lack ledger facts
            folder = str(PurePosixPath(path).parent)
            run, known = out.get(folder, (None, {}))
            out[folder] = (run, {**trials, **known})  # a saved Hub listing wins
        elif PurePosixPath(path).name == SAVED_LISTING:
            try:
                folder = str(PurePosixPath(path).parent)
                run, trials = saved_listing(listing.reader(path, LISTING_BYTES))
                out[folder] = (run, {**out.get(folder, (None, {}))[1], **trials})
            except Exception:
                continue  # unreadable: the trials simply lack Hub facts
    return out


def hub_trial(saved: dict, entry: Entry) -> dict:
    """The saved Hub facts of the trial whose trajectory is `entry` (`<job>/<trial>/
    trajectory.json`, or `<job>/job/<trial>/agent/trajectory.json` for a full archive)."""
    trial = PurePosixPath(entry.path).parent
    if trial.name == "agent":
        trial = trial.parent
    for folder in (trial.parent, trial.parent.parent):
        found = saved.get(str(folder))
        if found and trial.name in found[1]:
            return found[1][trial.name]
    return {}


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
    TRIAL_LEDGER: LEDGER_BYTES,
    RUN_MANIFEST: RUN_MANIFEST_BYTES,
}
SYNC_NAMES = frozenset(SYNC_CAPS)


def sync_cap(relative: str) -> int | None:
    """Largest size synced for a listed file that isn't a matching trajectory, or None
    when the scan never reads it."""
    path = PurePosixPath(relative)
    if path.parent.name == ATTEMPT_COSTS and path.suffix == ".json":
        return ATTEMPT_COST_BYTES
    return SYNC_CAPS.get(path.name)


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


SYNC_STATE = ".atif-sync.json"


def private_directory(path: Path) -> None:
    """Create private parents; tighten the owned directory, never follow symlinks."""
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise SourceError("unsafe_sync_path")
    if not path.exists():
        if not path.parent.exists():
            private_directory(path.parent)
        path.mkdir(mode=0o700, exist_ok=True)
    if not path.is_dir():
        raise SourceError("unsafe_sync_path")
    path.chmod(0o700)


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
            path.chmod(0o600)


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


def read_sync_state(root: Path) -> dict | None:
    """Malformed state fails closed, never becomes an ordinary directory scan."""
    path = root / SYNC_STATE
    if not path.exists() and not path.is_symlink():
        return None
    try:
        if path.is_symlink():
            raise ValueError
        with path.open("rb") as handle:
            data = handle.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise ValueError
        value = json.loads(data)
        if value["schema_version"] != 1 or not isinstance(value["files"], dict):
            raise ValueError
        if value["single"] is not None and value["single"] not in value["files"]:
            raise ValueError
        for name, row in value["files"].items():
            if (
                not isinstance(name, str)
                or confined(root, name) is None
                or name == SYNC_STATE
                or not isinstance(row, dict)
                or set(row) != {"size", "revision", "stamp", "failed"}
                or (row["size"] is not None and (type(row["size"]) is not int or row["size"] < 0))
                or (row["revision"] is not None and not isinstance(row["revision"], str))
                or (row["stamp"] is not None and not isinstance(row["stamp"], str))
                or type(row["failed"]) is not bool
            ):
                raise ValueError
        return value
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        raise SourceError("invalid_sync_inventory") from None


def write_sync_state(root: Path, files: dict, single: str | None) -> None:
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


def synced_listing(root: Path, single: str | None = None) -> Listing:
    state = read_sync_state(root)
    single = single or state["single"]
    files = state["files"]
    names = [single] if single is not None else sorted(files)
    for name in names:
        path = root / name
        while path != root:
            if path.is_symlink():
                raise SourceError("unsafe_sync_path")
            path = path.parent
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
        with open(root / relative, "rb") as handle:
            return handle.read(limit)

    return Listing(
        False,
        single is None,
        entries,
        opener,
        read,
        local,
        sync_failed=sum(row["failed"] for row in files.values()),
    )


def sync_stamp(path: Path) -> str | None:
    try:
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            return None
        return f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return None


def sync_remote(
    value: str,
    dest: Path,
    pattern: str = DEFAULT_PATTERN,
    fs=None,
    workers: int = 16,
    refresh: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, dict]:
    """Mirror the current remote inventory, retaining failures as unknown inputs.

    Reuse requires a provider content identity AND an unchanged local fingerprint.
    Without an identity re-download. The saved inventory excludes obsolete files even
    when scanning the copy offline; no stale file can stand in for a failed download.
    """
    listing = list_input(value, fs)
    if listing.fetch is None:
        raise SourceError("not_a_remote_input")
    private_directory(dest)
    previous = (read_sync_state(dest) or {}).get("files", {})
    single = None if listing.directory else PurePosixPath(normalize(value)).name
    wanted = []
    for entry in listing.entries:
        relative = entry.path if listing.directory else single
        name = PurePosixPath(relative).name
        cap = (
            MAX_BYTES if not listing.directory or fnmatchcase(name, pattern) else sync_cap(relative)
        )
        if cap is not None:
            path = confined(dest, relative)
            if path is None or relative == SYNC_STATE:
                raise SourceError("invalid_hf_path")
            private_sync_path(dest, path)
            wanted.append((entry, relative, path, cap))
    state = {
        relative: {"size": entry.size, "revision": entry.revision, "stamp": None, "failed": True}
        for entry, relative, _, _ in wanted
    }
    # Publish unknowns before changing files: interruption cannot resurrect stale facts.
    write_sync_state(dest, state, single)
    # Remove only scan-owned file kinds (including legacy mirrors without an inventory).
    # Other user files are left alone and excluded by the authoritative inventory.
    for folder, _, files in os.walk(dest):
        for name in files:
            path = Path(folder) / name
            relative = path.relative_to(dest).as_posix()
            if (
                relative not in state
                and (
                    relative in previous
                    or sync_cap(relative) is not None
                    or fnmatchcase(name, pattern)
                )
                and name != SYNC_STATE
            ):
                private_sync_path(dest, path)
                path.unlink()

    counts = {"files": len(wanted), "downloaded": 0, "up_to_date": 0, "failed": 0}

    def one(item: tuple[Entry, str, Path, int]) -> tuple[str, str, str | None]:
        entry, relative, path, cap = item
        partial = None
        try:
            old = previous.get(relative, {})
            stamp = sync_stamp(path)
            if entry.size is not None and entry.size > cap:
                raise ValueError("sync_too_large")
            if (
                not refresh
                and entry.revision is not None
                and old.get("revision") == entry.revision
                and not old.get("failed", True)
                and stamp is not None
                and old.get("stamp") == stamp
                and old.get("size") == entry.size
            ):
                path.chmod(0o600)
                return relative, "up_to_date", stamp
            # Failure must not leave an old copy usable by direct local scans.
            path.unlink(missing_ok=True)
            fd, name = tempfile.mkstemp(prefix=".atif-download-", suffix=".part", dir=path.parent)
            os.close(fd)
            partial = Path(name)
            listing.fetch(entry, partial)
            size = partial.stat().st_size
            if size > cap or (entry.size is not None and size != entry.size):
                raise ValueError("sync_size_mismatch")
            partial.chmod(0o600)
            partial.replace(path)
            return relative, "downloaded", sync_stamp(path)
        except Exception:
            path.unlink(missing_ok=True)
            return relative, "failed", None
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)

    with ThreadPoolExecutor(max(1, workers)) as pool:
        for done, (relative, outcome, stamp) in enumerate(pool.map(one, wanted), 1):
            counts[outcome] += 1
            state[relative]["failed"] = outcome == "failed"
            state[relative]["stamp"] = stamp
            if progress is not None:
                progress(done, len(wanted))
    write_sync_state(dest, state, single)
    return (dest / single if single is not None else dest), counts


def file_source(label: str, location: str, fs=None) -> Source:
    """A single named file (manifest entries): no expansion; failures surface on load."""
    location = normalize(location)
    if location.startswith(HF_PREFIX):
        fs = fs or hf_filesystem()
        return Source(label, _remote_load(fs, location[len(HF_PREFIX) :], None))
    return Source(label, lambda: load_trace(Path(location)), local=Path(location))


def _either(read: Callable[[], float | None], fallback: object) -> Callable[[], float | None]:
    """The reward file's value, else the saved Hub listing's."""
    if not isinstance(fallback, float | int):
        return read

    def reward() -> float | None:
        found = read()
        return found if found is not None else float(fallback)

    return reward


def resolve(
    values: list[str],
    pattern: str = DEFAULT_PATTERN,
    fs=None,
    runs: list | None = None,
    sync_failures: list[int] | None = None,
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
        if sync_failures is not None:
            sync_failures.append(listing.sync_failed)
        saved = saved_hub_listings(listing)
        found_runs, membership = job_folders(listing)
        if runs is not None:
            runs.extend(found_runs)
            runs.extend(run for run, _ in saved.values() if run)
        entries = selected(listing, pattern)
        if not entries:
            raise SourceError("no_files_match_pattern")
        for entry in entries:
            label = label_for(entry, pattern) or f"input-{len(sources) + 1:04d}"
            meta = hub_trial(saved, entry)
            reward = _either(reward_lookup(listing, entry, value), meta.get("reward"))
            hint = "/".join(p for p in (value.rstrip("/"), entry.path) if p)
            if errored(listing, entry) and not meta.get("error_type"):
                meta = {**meta, "error_type": "exception"}
            if (listed := membership.get(trial_folder(entry))) is not None:
                meta = {**meta, "in_job_result": listed}
            local = listing.local_path(entry) if listing.local_path is not None else None
            fingerprint = (
                local_fingerprint(local)
                if local is not None and not entry.failed
                else (lambda: None)
            )
            details = trial_details(listing, entry, value)
            sources.append(
                Source(
                    label, listing.opener(entry), reward, hint, meta, fingerprint, details, local
                )
            )
    if len({s.label for s in sources}) != len(sources):
        raise SourceError("duplicate_input_labels")
    return sources
