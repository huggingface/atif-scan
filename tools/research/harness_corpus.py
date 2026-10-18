"""Offline, filename-independent ATIF corpus discovery (stdlib only).

API for research callers:
* ``default_roots()`` names the six cache roots; selection is always explicit:
  ``discover(roots)`` streams Candidate(root_label, path) for every regular
  .json file, including renamed trajectories, metadata and UUID source receipts.
  Missing/unreadable roots raise CorpusError, not a successful empty inventory.
* ``inspect_candidate(candidate)`` reads at most MAX_BYTES (256 MiB), returns an
  Inspection with fixed classification codes and digests, and retains no trace text.
  Structural recognition is not full ATIF validation or evidence of task success.
* ``build_inventory(roots, private_output_dir)`` writes a SQLite *private mapping*,
  NOT an exported report. The directory must be new, outside this repository, and
  have an existing parent. It is 0700; inventory.sqlite3 is 0600. Every candidate,
  including failures/exclusions and duplicate copies, gets a row. Corpus memory is
  bounded to one file/JSON document plus SQLite's bounded cache.
* ``iter_unique_inventory(inventory)`` streams one Inspection per exact byte hash
  of accepted traces, with paths privately available to the parent. Semantic hashes
  and session hashes are grouping aids ONLY: divergent executions are never merged.
  Pin the digest with read_document before subsequently consuming a stored path.
* ``read_document(candidate, expected_sha256=None)`` returns LoadedDocument with
  doc (JsonObject), redacted_values (int) and raw_sha256 (str). It reads once and
  parses once; the optional pin always checks ORIGINAL bytes before parsing.
  Non-object JSON raises invalid_structure; ATIF classification is not imposed.

JSON-semantic hashing sorts object keys, preserves array order and number types,
normalizes whitespace/escaping and UTF-8/16/32 encodings. It is not RFC 8785 and
never equates 1 with 1.0. Duplicate keys/nonfinite numbers are invalid, not guessed.
Bare [REDACTED] values are normalized to null (unknown), never inside JSON strings.
Raw hashes cover original bytes; semantic hashes cover normalized JSON. Positive
redacted_values counts are coverage warnings, not complete evidence. Inspection's
None count means parsing/coverage unavailable, not zero. Summary redacted_documents
and redacted_values count successfully normalized documents/values across ALL copies;
redaction_unknown_documents counts unavailable coverage (including exclusions).
Session identity is a hash of the top-level session_id only (not a task identity).
Only explicit review filenames or agent sessions whose non-copied, actual ATIF
function_name calls are ALL inspection tools are excluded. Mentions, observations,
user tool calls and mixed benchmark/inspection calls do not establish a judge.

No trace commands, URLs, referenced files, models or network are used. No filename
heuristic drops config/result/auth JSON: even those may be renamed trajectories.
Only counts appear in Inventory.summary (an allowlisted jsonval.Doc); the database
contains sensitive local paths/labels and must never be exported or committed.
POSIX descriptor-relative opens refuse symlinks in every input path component.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from atif_scan.data.jsonval import as_list, as_object, as_str, is_object

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from atif_scan.data.jsonval import Doc, JsonObject

MAX_BYTES = 256 * 1024 * 1024
ROOT_NAMES = ("hf", "harbor", "harbor-full", "deepswe", "local-jobs", "archive")
INSPECTION_TOOLS = frozenset({"uv__read_steps", "uv__search_trace", "uv__trace_outline"})
# String alternative takes precedence, including unterminated strings. Disjoint,
# possessive repeats prevent backtracking/restarts within malformed long strings.
_STRING_OR_REDACTED = re.compile(r'"[^"\\]*+(?:\\[\s\S][^"\\]*+)*+(?:"|\\?\Z)|\[REDACTED\]')
VERSION = re.compile(r"ATIF-v[0-9]+\.[0-9]+(?:\.[0-9]+)?")
Classification = Literal[
    "trace",
    "judge",
    "non_trace",
    "invalid_json",
    "invalid_structure",
    "too_large",
    "unsafe_path",
    "unreadable",
    "changed_file",
]
CLASSIFICATIONS: tuple[Classification, ...] = (
    "trace",
    "judge",
    "non_trace",
    "invalid_json",
    "invalid_structure",
    "too_large",
    "unsafe_path",
    "unreadable",
    "changed_file",
)
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_REPOSITORY = Path(__file__).absolute().parents[2]


class CorpusError(ValueError):
    """Fixed codes only; no source paths, snippets or OS exception messages."""


@dataclass(frozen=True)
class Candidate:
    """Private discovery locator, not an exported report field."""

    root_label: str
    path: Path


@dataclass(frozen=True)
class Inspection:
    """Private record; missing digests mean unknown/unreadable, never clean."""

    candidate: Candidate
    classification: Classification
    byte_count: int | None = None
    raw_sha256: str | None = None
    semantic_sha256: str | None = None
    schema_version: str | None = None
    session_sha256: str | None = None
    redacted_values: int | None = None


@dataclass(frozen=True)
class LoadedDocument:
    """Private parsed JSON; never export/serialize this trace-bearing result."""

    doc: JsonObject
    redacted_values: int
    raw_sha256: str


@dataclass(frozen=True)
class Inventory:
    """Private database locator and a separate count-only exportable summary."""

    database_path: Path
    summary: Doc


def default_roots() -> dict[str, Path]:
    """Return choices, without scanning or including any other cache directories."""
    home = Path.home() / ".cache" / "atif-scan"
    return {name: home / name for name in ROOT_NAMES}


def _absolute(path: Path) -> Path:
    if ".." in path.parts:
        raise CorpusError("unsafe_path")
    return path if path.is_absolute() else Path.cwd() / path


@contextmanager
def _directory(path: Path) -> Iterator[int]:
    """Open each ancestor without resolving symlinks, pinning it by descriptor."""
    absolute = _absolute(path)
    fd = os.open(absolute.anchor, _DIRECTORY_FLAGS)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def _walk(label: str, path: Path, fd: int) -> Iterator[Candidate]:
    with os.scandir(fd) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                child = os.open(entry.name, _DIRECTORY_FLAGS, dir_fd=fd)
                try:
                    yield from _walk(label, path / entry.name, child)
                finally:
                    os.close(child)
            elif entry.name.lower().endswith(".json") and entry.is_file(follow_symlinks=False):
                yield Candidate(label, path / entry.name)


def discover(roots: Mapping[str, Path]) -> Iterator[Candidate]:
    """Stream all regular JSON files; order unspecified, no symlinks followed.

    Selection is the supplied mapping, never implicit cache expansion. A traversal
    failure aborts with a fixed code, so incomplete coverage cannot look complete.
    """
    try:
        for label, root in roots.items():
            absolute = _absolute(root)
            with _directory(absolute) as fd:
                yield from _walk(label, absolute, fd)
    except OSError:
        raise CorpusError("discovery_unavailable") from None


def _snapshot(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read(candidate: Candidate, cap: int) -> tuple[bytes | None, Classification, int | None]:
    with _directory(candidate.path.parent) as parent:
        fd = os.open(candidate.path.name, _FILE_FLAGS, dir_fd=parent)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            return None, "unsafe_path", None
        if before.st_size > cap:
            return None, "too_large", before.st_size
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(cap + 1)
        after = os.fstat(fd)
        return _read_result(data, before, after, cap)
    finally:
        os.close(fd)


def _read_result(
    data: bytes, before: os.stat_result, after: os.stat_result, cap: int
) -> tuple[bytes | None, Classification, int | None]:
    if len(data) > cap:
        return None, "too_large", len(data)
    if _snapshot(before) != _snapshot(after) or len(data) != before.st_size:
        return None, "changed_file", len(data)
    return data, "trace", len(data)


def _unique_object(pairs: list[tuple[str, object]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError("nonfinite_json_number")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("nonfinite_json_number")
    return value


def _unredact(text: str) -> tuple[str, int]:
    """Replace only bare markers with null; quoted text stays byte-for-byte text."""
    found = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal found
        if match[0].startswith('"'):
            return match[0]
        found += 1
        return "null"

    return _STRING_OR_REDACTED.sub(replace, text), found


def _parse(data: bytes) -> tuple[object, int]:
    """Normalize then strictly parse once, preserving JSON's supported encodings."""
    try:
        text = data.decode(json.detect_encoding(data), errors="surrogatepass")
        text, redacted = _unredact(text)
        value: object = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (ValueError, UnicodeError, RecursionError):
        raise CorpusError("invalid_json") from None
    return value, redacted


def _io_classification(exc: OSError) -> Classification:
    return "unsafe_path" if exc.errno in (errno.ELOOP, errno.ENOTDIR) else "unreadable"


def read_document(candidate: Candidate, expected_sha256: str | None = None) -> LoadedDocument:
    """Read capped, no-follow, stable original bytes and return one parsed object.

    Fixed CorpusError codes: unsafe_path, unreadable, too_large, changed_file,
    digest_mismatch, invalid_json, invalid_structure. A digest mismatch is refused
    BEFORE normalization/parsing. Unlike inspect_candidate, filenames are not excluded.
    Duplicate keys, nonfinite constants/overflow and all other bad syntax are refused.
    The returned mutable document is private evidence; null redactions remain unknown.
    """
    try:
        data, classification, _ = _read(candidate, MAX_BYTES)
    except OSError as exc:
        raise CorpusError(_io_classification(exc)) from None
    if data is None:
        raise CorpusError(classification)
    raw_hash = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and raw_hash != expected_sha256:
        raise CorpusError("digest_mismatch")
    value, redacted = _parse(data)
    if not is_object(value):
        raise CorpusError("invalid_structure")
    return LoadedDocument(value, redacted, raw_hash)


def _semantic_hash(value: object) -> str:
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("ascii"))
    return digest.hexdigest()


def _trace_shape(doc: JsonObject) -> Classification:
    version = as_str(doc.get("schema_version"))
    declared = version is not None and VERSION.fullmatch(version) is not None
    if not isinstance(doc.get("steps"), list):
        return "invalid_structure" if declared else "non_trace"
    steps = as_list(doc.get("steps"))
    valid = all(is_object(s) and s.get("source") in ("system", "user", "agent") for s in steps)
    if not valid:
        return "invalid_structure" if declared else "non_trace"
    # An arbitrary metadata object with an empty 'steps' array is not enough.
    return "trace" if declared or steps else "non_trace"


def _inspection_calls(step: JsonObject) -> list[object] | None:
    if not isinstance(step.get("is_copied_context", False), bool):
        return None
    calls = step.get("tool_calls")
    return as_list(calls) if calls is None or isinstance(calls, list) else None


def _inspection_session(doc: JsonObject) -> bool:
    found = False
    for raw in as_list(doc.get("steps")):
        step = as_object(raw)
        if step.get("source") != "agent" or step.get("is_copied_context") is True:
            continue
        calls = _inspection_calls(step)
        if calls is None:
            return False
        for raw_call in calls:
            name = as_str(as_object(raw_call).get("function_name"))
            if name not in INSPECTION_TOOLS:
                return False
            found = True
    return found


def _review_filename(path: Path) -> bool:
    name = path.name.lower()
    return name == "review.atif.json" or name.endswith(".review.atif.json")


def _classify(candidate: Candidate, data: bytes) -> Inspection:
    raw_hash = hashlib.sha256(data).hexdigest()
    try:
        value, redacted = _parse(data)
        semantic = _semantic_hash(value)
    except (ValueError, UnicodeError, RecursionError):
        return Inspection(candidate, "invalid_json", len(data), raw_hash)
    doc = as_object(value)
    classification = _trace_shape(doc)
    if classification == "trace" and _inspection_session(doc):
        classification = "judge"
    version = as_str(doc.get("schema_version"))
    version = version if version and VERSION.fullmatch(version) else None
    session = as_str(doc.get("session_id"))
    return Inspection(
        candidate,
        classification,
        len(data),
        raw_hash,
        semantic,
        version,
        _semantic_hash(session) if session else None,
        redacted,
    )


def inspect_candidate(candidate: Candidate, *, max_bytes: int = MAX_BYTES) -> Inspection:
    """Classify one private locator without retaining raw data or following paths.

    Review filenames are excluded before opening. Other failures are explicit,
    including malformed JSON, excessive size, unsafe paths and concurrent changes.
    max_bytes may lower, but never raise, the hard cap.
    """
    if not 0 < max_bytes <= MAX_BYTES:
        raise CorpusError("invalid_size_cap")
    if _review_filename(candidate.path):
        return Inspection(candidate, "judge")
    try:
        data, classification, size = _read(candidate, max_bytes)
    except CorpusError:
        return Inspection(candidate, "unsafe_path")
    except OSError as exc:
        classification = _io_classification(exc)
        return Inspection(candidate, classification)
    return (
        _classify(candidate, data)
        if data is not None
        else Inspection(candidate, classification, size)
    )


def iter_inspections(roots: Mapping[str, Path]) -> Iterator[Inspection]:
    """No whole-corpus list or retained JSON documents."""
    for candidate in discover(roots):
        yield inspect_candidate(candidate)


_SCHEMA = """
CREATE TABLE state (complete INTEGER NOT NULL);
INSERT INTO state VALUES (0);
CREATE TABLE copies (
    id INTEGER PRIMARY KEY,
    root_label TEXT NOT NULL,
    path TEXT NOT NULL,
    classification TEXT NOT NULL,
    byte_count INTEGER,
    raw_sha256 TEXT,
    semantic_sha256 TEXT,
    schema_version TEXT,
    session_sha256 TEXT,
    redacted_values INTEGER CHECK (redacted_values >= 0)
);
CREATE INDEX by_raw ON copies(classification, raw_sha256, id);
CREATE INDEX by_semantic ON copies(classification, semantic_sha256);
CREATE INDEX by_session ON copies(classification, session_sha256);
"""


def _create_private_database(output: Path) -> Path:
    absolute = _absolute(output)
    if absolute == _REPOSITORY or _REPOSITORY in absolute.parents:
        raise CorpusError("output_inside_repository")
    try:
        with _directory(absolute.parent) as parent:
            os.mkdir(absolute.name, mode=0o700, dir_fd=parent)
            folder = os.open(absolute.name, _DIRECTORY_FLAGS, dir_fd=parent)
            try:
                os.fchmod(folder, 0o700)
                fd = os.open(
                    "inventory.sqlite3",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=folder,
                )
                try:
                    os.fchmod(fd, 0o600)
                finally:
                    os.close(fd)
            finally:
                os.close(folder)
    except OSError:
        raise CorpusError("private_output_unavailable") from None
    return absolute / "inventory.sqlite3"


def _insert(connection: sqlite3.Connection, record: Inspection) -> None:
    connection.execute(
        """INSERT INTO copies(root_label, path, classification, byte_count, raw_sha256,
           semantic_sha256, schema_version, session_sha256, redacted_values)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            record.candidate.root_label,
            str(_absolute(record.candidate.path)),
            record.classification,
            record.byte_count,
            record.raw_sha256,
            record.semantic_sha256,
            record.schema_version,
            record.session_sha256,
            record.redacted_values,
        ),
    )


def _summary(connection: sqlite3.Connection) -> Doc:
    counts = dict.fromkeys(CLASSIFICATIONS, 0)
    for code, count in connection.execute("SELECT classification, COUNT(*) FROM copies GROUP BY 1"):
        if code in counts:
            counts[code] = count
    unique = connection.execute(
        """SELECT COUNT(DISTINCT raw_sha256), COUNT(DISTINCT semantic_sha256),
           COUNT(DISTINCT session_sha256) FROM copies WHERE classification='trace'"""
    ).fetchone()
    divergent = connection.execute(
        """SELECT COUNT(*) FROM (SELECT session_sha256 FROM copies
           WHERE classification='trace' AND session_sha256 IS NOT NULL
           GROUP BY session_sha256 HAVING COUNT(DISTINCT semantic_sha256)>1)"""
    ).fetchone()[0]
    redactions = connection.execute(
        """SELECT COUNT(CASE WHEN redacted_values > 0 THEN 1 END),
           COALESCE(SUM(redacted_values), 0),
           COUNT(CASE WHEN redacted_values IS NULL THEN 1 END) FROM copies"""
    ).fetchone()
    return {
        "redacted_documents": redactions[0],
        "redacted_values": redactions[1],
        "redaction_unknown_documents": redactions[2],
        "candidate_count": sum(counts.values()),
        "classifications": counts,
        "unique_byte_traces": unique[0],
        "unique_semantic_traces": unique[1],
        "session_groups": unique[2],
        "divergent_sessions": divergent,
    }


def build_inventory(roots: Mapping[str, Path], private_output_dir: Path) -> Inventory:
    """Create a private, disk-backed inventory; never silently finish a partial scan.

    On failure the new directory may remain with state.complete=0 for diagnosis.
    No raw JSON, arguments, messages, credentials or exception text are persisted.
    SQLite holds only private locators, digests, fixed classifications and ATIF versions.
    """
    database = _create_private_database(private_output_dir)
    connection = sqlite3.connect(database)
    try:
        connection.executescript(_SCHEMA)
        for record in iter_inspections(roots):
            _insert(connection, record)
        summary = _summary(connection)
        connection.execute("UPDATE state SET complete=1")
        connection.commit()
    finally:
        connection.close()
    return Inventory(database, summary)


def _record(row: tuple[object, ...]) -> Inspection:
    # SQLite stores only our own typed records, never the untrusted parsed JSON.
    label, path, code, size, raw, semantic, version, session, redacted = row
    if code not in CLASSIFICATIONS or not isinstance(label, str) or not isinstance(path, str):
        raise CorpusError("invalid_inventory")
    if redacted is not None and (type(redacted) is not int or redacted < 0):
        raise CorpusError("invalid_inventory")
    classification = next(c for c in CLASSIFICATIONS if c == code)
    return Inspection(
        Candidate(label, Path(path)),
        classification,
        size if type(size) is int else None,
        as_str(raw),
        as_str(semantic),
        as_str(version),
        as_str(session),
        redacted if type(redacted) is int else None,
    )


def iter_unique_inventory(inventory: Inventory) -> Iterator[Inspection]:
    """Yield first recorded copy per exact byte hash, NOT per session/semantic hash.

    Paths/labels are private, not report fields. Contents can change after inventory;
    the consumer must pin raw_sha256 with read_document before using the file. Partial inventories
    are refused. SQLite is opened read-only and never loads trace content.
    """
    path = _absolute(inventory.database_path)
    with _directory(path.parent) as parent:
        fd = os.open(path.name, _FILE_FLAGS, dir_fd=parent)
        os.close(fd)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        if connection.execute("SELECT complete FROM state").fetchone() != (1,):
            raise CorpusError("inventory_incomplete")
        rows = connection.execute(
            """SELECT root_label, path, classification, byte_count, raw_sha256,
               semantic_sha256, schema_version, session_sha256, redacted_values FROM copies
               WHERE id IN (SELECT MIN(id) FROM copies
                            WHERE classification='trace' GROUP BY raw_sha256)
               ORDER BY id"""
        )
        for row in rows:
            yield _record(row)
    finally:
        connection.close()
