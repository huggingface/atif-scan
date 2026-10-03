"""Local companion-history inventory and bounded inspection, never ATIF reconstruction.

Only fixed Grok session locations beside a selected trajectory are considered. Summary
paths are untrusted text, never filesystem instructions. Absence locally says nothing
about availability in a provider's full trial archive. Markdown is untrusted evidence,
not parsed into invented steps or used to clear deterministic coverage gaps.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING

from .. import credentials
from .cite import mask

if TYPE_CHECKING:
    from pathlib import Path

    from ..jsonval import Doc

MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_FILES = 32
MAX_SESSIONS = 32
MAX_ENTRIES = 256
MAX_PAGE_CHARS = 6000
MAX_SEARCH_CHARS = 150
MAX_SEARCH_HITS = 10
SEARCH_WINDOW = 200
SEGMENT = re.compile(r"segment_[0-9]+\.md")


@dataclass(frozen=True)
class HistoryArchive:
    files: tuple[Path, ...] = ()
    rejected: int = 0
    checked: bool = True

    def counts(self) -> Doc:
        """Explicit report allowlist; no names, paths, text or content hashes."""
        return {
            "status": "available"
            if self.files
            else "unreadable"
            if self.rejected
            else "not_found"
            if self.checked
            else "not_checked",
            "format": "grok_markdown",
            "segments": sum(p.name != "INDEX.md" for p in self.files),
            "indexes": sum(p.name == "INDEX.md" for p in self.files),
            "rejected": self.rejected,
            "scanned": False,
        }

    def digest(self) -> str:
        """Private answer binding: contents, ordering and unavailable files matter."""
        # Canonical records bind file boundaries and read status, not raw concatenation.
        material = json.dumps(
            ["history-archive/v2", self.checked, self.rejected, [_binding(p) for p in self.files]],
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def outline(self) -> Doc:
        return {
            **self.counts(),
            "files": [
                {"file": i, "kind": "index" if p.name == "INDEX.md" else "segment"}
                for i, p in enumerate(self.files)
            ],
        }

    def page(
        self, file: int, offset: int = 0, limit: int = 3000, known: frozenset[str] = frozenset()
    ) -> Doc:
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be non-negative")
        if type(limit) is not int or not 1 <= limit <= MAX_PAGE_CHARS:
            raise ValueError("limit must be 1-6000")
        # Mask the WHOLE field before slicing: credentials crossing a page stay masked.
        text = self._text(file, known)
        if offset > len(text):
            raise ValueError("offset exceeds masked file length")
        end = min(offset + limit, len(text))
        return {
            "file": file,
            "offset": offset,
            "end_offset": end,
            "total_length": len(text),
            "next_offset": end if end < len(text) else None,
            "text": text[offset:end],
        }

    def _text(self, file: int, known: frozenset[str]) -> str:
        if type(file) is not int or not 0 <= file < len(self.files):
            raise ValueError("no such archive file")
        text = _read(self.files[file]).decode("utf-8")
        return mask(text, known | credentials.values((text,)))

    def search(self, file: int, term: str, known: frozenset[str] = frozenset()) -> Doc:
        """Ten literal windows in one bound, fully masked file; no regex execution."""
        if not term or len(term) > MAX_SEARCH_CHARS:
            raise ValueError("search needs 1-150 characters")
        text = self._text(file, known)
        return {"file": file, "matches": _search_windows(text, term)}


def _binding(path: Path) -> tuple[str, str, str | None]:
    name = "/".join(path.parts[-4:])
    try:
        content_digest = hashlib.sha256(_read(path)).hexdigest()
    except ValueError:
        return name, "unavailable", None
    return name, "readable", content_digest


def _search_windows(text: str, term: str) -> list[Doc]:
    hits = []
    start = 0
    while len(hits) < MAX_SEARCH_HITS:
        at = text.lower().find(term.lower(), start)
        if at < 0:
            break
        lo, hi = max(0, at - SEARCH_WINDOW), min(len(text), at + len(term) + SEARCH_WINDOW)
        hits.append({"offset": lo, "end_offset": hi, "text": text[lo:hi]})
        start = at + len(term)
    return hits


def _read(path: Path) -> bytes:
    try:
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise ValueError("unsafe archive path")
        with path.open("rb") as handle:
            data = handle.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("archive file too large")
        data.decode("utf-8")
        return data
    except (OSError, UnicodeError) as error:
        raise ValueError("archive file unavailable") from error


def _directories(path: Path) -> tuple[list[Path], int]:
    """One fixed directory level, capped and never following symlinks."""
    if path.is_symlink():
        return [], 1
    if not path.exists():
        return [], 0
    try:
        entries = sorted(islice(path.iterdir(), MAX_ENTRIES + 1))
    except OSError:
        return [], 1
    rejected = int(len(entries) > MAX_ENTRIES) + sum(p.is_symlink() for p in entries)
    return [p for p in entries[:MAX_ENTRIES] if p.is_dir() and not p.is_symlink()], rejected


def _accepted_files(entries: list[Path]) -> HistoryArchive:
    found = []
    rejected = int(len(entries) > MAX_ENTRIES)
    candidates = [
        p for p in entries[:MAX_ENTRIES] if p.name == "INDEX.md" or SEGMENT.fullmatch(p.name)
    ]
    for path in candidates[:MAX_FILES]:
        try:
            _read(path)
        except ValueError:
            rejected += 1
        else:
            found.append(path)
    return HistoryArchive(tuple(found), rejected + int(len(candidates) > MAX_FILES))


def _files(folder: Path) -> HistoryArchive:
    if folder.is_symlink():
        return HistoryArchive(rejected=1)
    if not folder.exists():
        return HistoryArchive()
    try:
        entries = sorted(islice(folder.iterdir(), MAX_ENTRIES + 1))
    except OSError:
        return HistoryArchive(rejected=1)
    return _accepted_files(entries)


def discover_history(local: Path | None) -> HistoryArchive:
    """Inventory beside this local trajectory, not paths mentioned by trace text."""
    if local is None:
        return HistoryArchive(checked=False)
    root = local.parent if local.parent.name == "agent" else local.parent / "agent"
    workspaces, rejected = _directories(root / "sessions")
    sessions = []
    for workspace in workspaces[:MAX_SESSIONS]:
        found, bad = _directories(workspace)
        sessions.extend(found)
        rejected += bad
    files = []
    for session in sessions[:MAX_SESSIONS]:
        archive = _files(session / "compaction")
        files.extend(archive.files)
        rejected += archive.rejected
    capped = (
        len(sessions) > MAX_SESSIONS or len(workspaces) > MAX_SESSIONS or len(files) > MAX_FILES
    )
    return HistoryArchive(tuple(files[:MAX_FILES]), rejected + int(capped))


def history_counts(local: Path | None) -> Doc:
    return discover_history(local).counts()
