"""Private, append-only reviewer annotations, never scan verdicts.

The journal is deliberately separate from publication reports. Corruption fails closed;
it must not silently turn an existing annotation into an unreviewed finding.
Schema 1 additively supports optional saved_at (timezone-aware ISO datetime) and
context metadata; older rows remain readable. New rows always carry a UTC saved_at.
Context contains only human-identifiable finding metadata, never a source path;
finding_sha256 identifies the last raw assessment, not the reviewer annotation.
Concurrent reviewers are last-append-wins in journal order, not timestamp order.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..data.jsonval import as_object, as_str, count, is_object

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ..data.jsonval import Doc

VERDICTS = frozenset({"valid_signal", "false_positive", "unclear", "unreviewed"})
MAX_NOTE = 2000
MAX_ROW = 32_768
JOURNAL = "feedback.jsonl"


def digest(value: object) -> str:
    """Stable identity, including evidence spans even though jumps omit spans."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def annotation(verdict: str, note: str) -> Doc:
    if verdict not in VERDICTS:
        raise ValueError("invalid_feedback_verdict")
    if not isinstance(note, str) or len(note) > MAX_NOTE:
        raise ValueError("invalid_feedback_note")
    return {"verdict": verdict, "note": note}


def _context(value: object) -> Doc:
    if not is_object(value):
        raise ValueError("invalid_feedback_context")
    allowed = {"input_id", "task", "trace_sha256", "check_id", "check_version", "finding_sha256"}
    if value.keys() - allowed:
        raise ValueError("invalid_feedback_context")
    result: Doc = {}
    for key, item in value.items():
        if item is None and key in {"task", "trace_sha256"}:
            result[key] = None
            continue
        text = as_str(item)
        if text is None:
            raise ValueError("invalid_feedback_context")
        if key in {"trace_sha256", "finding_sha256"} and not re.fullmatch(r"[0-9a-fA-F]{64}", text):
            raise ValueError("invalid_feedback_context")
        result[key] = text
    return result


def _metadata(row: Doc) -> None:
    if "context" in row:
        _context(row["context"])
    if "saved_at" in row:
        text = as_str(row["saved_at"])
        if text is None:
            raise ValueError("invalid_feedback_saved_at")
        saved_at = datetime.fromisoformat(text)
        if saved_at.utcoffset() is None:
            raise ValueError("invalid_feedback_saved_at")


def _directory(path: Path) -> int:
    """Walk with directory descriptors: no symlink component, including ancestors."""
    absolute = path.absolute()
    fd = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in absolute.parts[1:]:
            if component == "..":
                raise ValueError("unsafe_feedback_directory")
            try:
                os.mkdir(component, mode=0o700, dir_fd=fd)
                os.fsync(fd)
            except FileExistsError:
                pass
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if info.st_uid != os.getuid():
            raise ValueError("unsafe_feedback_owner")
        os.fchmod(fd, 0o700)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _validate_file(fd: int) -> None:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
        raise ValueError("unsafe_feedback_file")
    os.fchmod(fd, 0o600)


def _row(line: bytes) -> tuple[str, Doc]:
    if len(line) > MAX_ROW or not line.endswith(b"\n"):
        raise ValueError("malformed_feedback")
    try:
        row = as_object(json.loads(line))
        binding = as_str(row.get("binding"))
        verdict, note = as_str(row.get("verdict")), as_str(row.get("note"))
        if (
            count(row.get("schema")) != 1
            or binding is None
            or re.fullmatch(r"[0-9a-f]{64}", binding) is None
            or verdict is None
            or note is None
        ):
            raise ValueError("malformed_feedback")
        _metadata(row)
        return binding, annotation(verdict, note)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("malformed_feedback") from exc


class FeedbackStore:
    """One locked, fsynced journal; reload on every read to observe other reviewers."""

    def __init__(self, directory: Path):
        self.directory = directory
        with self._journal():
            pass

    @contextmanager
    def _journal(self) -> Iterator[int]:
        directory = _directory(self.directory)
        try:
            fd = os.open(
                JOURNAL,
                os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
                dir_fd=directory,
            )
            try:
                _validate_file(fd)
                fcntl.flock(fd, fcntl.LOCK_EX)
                os.fsync(directory)
                yield fd
            finally:
                os.close(fd)
        finally:
            os.close(directory)

    @staticmethod
    def _read(fd: int) -> dict[str, Doc]:
        os.lseek(fd, 0, os.SEEK_SET)
        result: dict[str, Doc] = {}
        with os.fdopen(os.dup(fd), "rb") as stream:
            while line := stream.readline(MAX_ROW + 1):
                binding, value = _row(line)
                result[binding] = value
        return result

    def load(self) -> dict[str, Doc]:
        with self._journal() as fd:
            return self._read(fd)

    def append(self, binding: str, verdict: str, note: str, *, context: Doc | None = None) -> Doc:
        """Append allowlisted optional context; return only the UI annotation."""
        value = annotation(verdict, note)
        row: Doc = {
            "schema": 1,
            "binding": binding,
            **value,
            "saved_at": datetime.now(UTC).isoformat(),
        }
        if context is not None:
            row["context"] = _context(context)
        data = (json.dumps(row, ensure_ascii=True) + "\n").encode()
        _row(data)
        with self._journal() as fd:
            self._read(fd)
            remaining = memoryview(data)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("feedback_write_failed")
                remaining = remaining[written:]
            os.fsync(fd)
        return value
