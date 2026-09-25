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

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from .checks import identifier
from .loader import MAX_BYTES, TraceError, load_bytes, load_trace
from .model import Trace

HF_PREFIX = "hf://"
HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
DEFAULT_PATTERN = "trajectory.json"
WEB_VIEWS = {"tree", "blob", "resolve"}


class SourceError(ValueError):
    """Fixed error codes only; never input paths or remote messages."""


@dataclass(frozen=True)
class Source:
    # Report label. Relative to the scanned root for expanded inputs, else `input-NNNN`.
    label: str
    load: Callable[[], Trace] = field(repr=False)


def hf_filesystem():
    from huggingface_hub import HfFileSystem

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


def _list_local(value: str) -> Listing:
    root = Path(value)
    if root.is_file():
        entry = Entry("", root.stat().st_size)
        return Listing(False, False, (entry,), lambda e: lambda: load_trace(root))
    if not root.is_dir():
        raise SourceError("path_not_found")
    entries = []
    # os.walk without following symlinks: the same traversal on every Python version.
    for folder, dirs, files in os.walk(root):
        dirs.sort()
        for name in files:
            path = Path(folder) / name
            if path.is_file() and not path.is_symlink():
                entries.append(Entry(path.relative_to(root).as_posix(), path.stat().st_size))
    entries.sort(key=lambda e: e.path)
    return Listing(False, True, tuple(entries), lambda e: lambda: load_trace(root / e.path))


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
        return Listing(True, False, (entry,), lambda e: _remote_load(fs, root, e.size))
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
        except ValueError:  # e.g. revision-qualified listings; keep the full path
            relative = path
        full[relative] = path
    sizes = {path: meta.get("size") for path, meta in found.items()}
    entries = [Entry(r, sizes[full[r]]) for r in sorted(full)]

    def opener(entry: Entry) -> Callable[[], Trace]:
        return _remote_load(fs, full[entry.path], entry.size)

    return Listing(True, True, tuple(entries), opener)


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


def file_source(label: str, location: str, fs=None) -> Source:
    """A single named file (manifest entries): no expansion; failures surface on load."""
    location = normalize(location)
    if location.startswith(HF_PREFIX):
        fs = fs or hf_filesystem()
        return Source(label, _remote_load(fs, location[len(HF_PREFIX) :], None))
    return Source(label, lambda: load_trace(Path(location)))


def resolve(values: list[str], pattern: str = DEFAULT_PATTERN, fs=None) -> list[Source]:
    """Expand each value in order (see module docstring).

    Expanded files are labelled by their path relative to the given root (the part the
    caller chose to scan), minus the file name when it is the literal pattern. Single
    files, and relative paths that are not valid identifiers, get a positional
    `input-NNNN` label, so absolute paths never reach a report.
    """
    sources: list[Source] = []
    for value in values:
        listing = list_input(value, fs)
        entries = selected(listing, pattern)
        if not entries:
            raise SourceError("no_files_match_pattern")
        for entry in entries:
            label = label_for(entry, pattern) or f"input-{len(sources) + 1:04d}"
            sources.append(Source(label, listing.opener(entry)))
    if len({s.label for s in sources}) != len(sources):
        raise SourceError("duplicate_input_labels")
    return sources
