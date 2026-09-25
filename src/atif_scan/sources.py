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

import re
from collections.abc import Callable, Iterator
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


def _local(value: str, pattern: str) -> Iterator[tuple[str | None, Callable[[], Trace]]]:
    root = Path(value)
    if root.is_file():
        yield None, lambda: load_trace(root)
    elif root.is_dir():
        for path in sorted(p for p in root.rglob(pattern) if p.is_file()):
            yield path.relative_to(root).as_posix(), lambda path=path: load_trace(path)
    else:
        raise SourceError("path_not_found")


def _remote(value: str, pattern: str, fs) -> Iterator[tuple[str | None, Callable[[], Trace]]]:
    root = value[len(HF_PREFIX) :].rstrip("/")
    try:
        info = fs.info(root)
    except FileNotFoundError:
        raise SourceError("hf_path_not_found_or_no_access") from None
    except Exception:
        raise SourceError("hf_request_failed") from None
    if info.get("type") != "directory":
        yield None, _remote_load(fs, root, info.get("size"))
        return
    try:
        found = fs.find(root, detail=True)
    except Exception:
        raise SourceError("hf_listing_failed") from None
    for path in sorted(found):
        entry = found[path]
        if entry.get("type") != "file" or not fnmatchcase(PurePosixPath(path).name, pattern):
            continue
        try:
            relative: str | None = PurePosixPath(path).relative_to(root).as_posix()
        except ValueError:  # e.g. revision-qualified paths; fall back to a positional label
            relative = None
        yield relative, _remote_load(fs, path, entry.get("size"))


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
        value = normalize(value)
        if value.startswith(HF_PREFIX):
            fs = fs or hf_filesystem()
            items = list(_remote(value, pattern, fs))
        else:
            items = list(_local(value, pattern))
        if not items:
            raise SourceError("no_files_match_pattern")
        for relative, load in items:
            label = f"input-{len(sources) + 1:04d}"
            if relative is not None:
                # `run/trial-1/trajectory.json` -> `run/trial-1`: the name is implied.
                folder, _, name = relative.rpartition("/")
                if folder and name == pattern:
                    relative = folder
                try:
                    label = identifier(relative)
                except ValueError:
                    pass
            sources.append(Source(label, load))
    if len({s.label for s in sources}) != len(sources):
        raise SourceError("duplicate_input_labels")
    return sources
