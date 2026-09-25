"""Resolve CLI inputs to readable trace sources: local files/directories or `hf://` paths.

Only this module does input I/O. Directory and remote-prefix expansion is explicit (a
filename pattern the caller controls), sorted, and never follows anything a trace says.
`hf://` support needs the optional `hub` extra (`huggingface_hub`); it is imported lazily
so the core stays dependency-free and offline.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

from .checks import identifier
from .loader import MAX_BYTES, TraceError, load_bytes, load_trace
from .model import Trace

HF_PREFIX = "hf://"
DEFAULT_PATTERN = "trajectory.json"


class SourceError(ValueError):
    """Fixed error codes only; never input paths or remote messages."""


@dataclass(frozen=True)
class Source:
    # Report label. Relative to the scanned root for expanded inputs, else `input-NNNN`.
    label: str
    load: Callable[[], Trace] = field(repr=False)


def is_remote(value: str) -> bool:
    return value.startswith(HF_PREFIX)


def hf_filesystem():
    try:
        from huggingface_hub import HfFileSystem
    except ImportError:
        raise SourceError("hf_paths_require_atif_scan_hub_extra") from None
    return HfFileSystem()


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
    if not root.is_dir():
        yield None, lambda: load_trace(root)
        return
    for path in sorted(p for p in root.rglob(pattern) if p.is_file()):
        relative = path.relative_to(root).as_posix()
        yield relative, lambda path=path: load_trace(path)


def _remote(value: str, pattern: str, fs) -> Iterator[tuple[str | None, Callable[[], Trace]]]:
    root = value[len(HF_PREFIX) :].rstrip("/")
    try:
        directory = fs.isdir(root)
    except Exception:
        directory = False  # unreadable: surfaces as an unavailable input below
    if not directory:
        yield None, _remote_load(fs, root, None)
        return
    try:
        found = fs.find(root, detail=True)
    except Exception:
        raise SourceError("hf_listing_failed") from None
    for path in sorted(found):
        info = found[path]
        if info.get("type") != "file" or not fnmatchcase(PurePosixPath(path).name, pattern):
            continue
        try:
            relative: str | None = PurePosixPath(path).relative_to(root).as_posix()
        except ValueError:  # e.g. revision-qualified paths; fall back to a positional label
            relative = None
        yield relative, _remote_load(fs, path, info.get("size"))


def resolve(values: list[str], pattern: str = DEFAULT_PATTERN, fs=None) -> list[Source]:
    """Expand each value in order. A directory/prefix that yields nothing is an error.

    Expanded files are labelled by their path relative to the given root (the part the
    caller chose to scan). Single files, and relative paths that are not valid
    identifiers, get a positional `input-NNNN` label, so absolute paths never reach a report.
    """
    sources: list[Source] = []
    for value in values:
        if is_remote(value):
            fs = fs or hf_filesystem()
            items = list(_remote(value, pattern, fs))
        else:
            items = list(_local(value, pattern))
        if not items:
            raise SourceError("no_traces_found")
        for relative, load in items:
            label = f"input-{len(sources) + 1:04d}"
            if relative is not None:
                try:
                    label = identifier(relative)
                except ValueError:
                    pass
            sources.append(Source(label, load))
    if len({s.label for s in sources}) != len(sources):
        raise SourceError("duplicate_input_labels")
    return sources
