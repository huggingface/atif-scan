"""Offline harness audit over an explicitly selected, complete private inventory.

Usage: uv run python tools/research/harness_audit.py --inventory PRIVATE_SQLITE
       --output NEW_PRIVATE_DIRECTORY [--limit N]
Outputs are allowlisted, snippet/path-free, and private (0700/0600), outside the
repository. A pilot, any record failure, or interrupted stream stays incomplete.
POSIX no-follow descriptors pin inputs and outputs; SQLite uses Linux /proc/self/fd.
No trace commands, URLs, models, or network calls are executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from harness_corpus import (
    ROOT_NAMES,
    Candidate,
    CorpusError,
    _absolute,
    _directory,
    _io_classification,
    _parse,
    _read,
    _record,
    _semantic_hash,
    read_document,
)
from harness_metrics import measure, tool_label
from harness_model_calls import evaluate_calls
from harness_trace import analyze_trace

from atif_scan.data.jsonval import as_list, as_object, as_str, is_object

if TYPE_CHECKING:
    from collections.abc import Iterator

    from harness_corpus import Inspection
    from harness_metrics import Metrics
    from harness_model_calls import CallAudit, ModelSource

    from atif_scan.data.jsonval import Doc, JsonObject

_SCHEMA = """
CREATE TABLE state(complete INTEGER NOT NULL);
INSERT INTO state VALUES(0);
CREATE TABLE traces(trace_id TEXT PRIMARY KEY, family TEXT, signature TEXT, version TEXT,
 scope TEXT, parent_id TEXT, run_id TEXT, data TEXT);
CREATE TABLE tools(trace_id TEXT, tool TEXT, recorded_calls INTEGER, static_call_sites INTEGER,
 declared INTEGER, offered_unused INTEGER, inventory_source TEXT);
CREATE TABLE outputs(trace_id TEXT, step INTEGER, observation INTEGER, tool TEXT, context TEXT,
 truncation INTEGER, marker TEXT, retention TEXT, followup_observed INTEGER,
 returned_chars INTEGER, returned_bytes INTEGER, data TEXT);
CREATE TABLE followups(trace_id TEXT, origin_step INTEGER, origin_observation INTEGER,
 step INTEGER, call INTEGER, tool TEXT, access_kind TEXT, partial_requested INTEGER,
 outcome TEXT, delay_calls INTEGER);
CREATE TABLE copies(trace_id TEXT, source_id INTEGER, root TEXT, run_id TEXT, raw_sha256 TEXT);
CREATE TABLE failures(trace_id TEXT, code TEXT);
CREATE TABLE native_sources(trace_id TEXT, definition_sha256 TEXT, status TEXT);
CREATE TABLE model_usage(trace_id TEXT, model TEXT, model_source TEXT,
 recorded_requests INTEGER, rejected_requests INTEGER, unknown_tool_labels INTEGER);
CREATE TABLE rejected_tools(trace_id TEXT, step INTEGER, call INTEGER, tool TEXT,
 model TEXT, model_source TEXT, observation INTEGER, reason TEXT,
 declaration_conflict INTEGER, pairing TEXT);
"""
_TOOL_FIELDS = (
    "tool",
    "recorded_calls",
    "static_call_sites",
    "declared",
    "offered_unused",
    "inventory_source",
)
_OUTPUT_FIELDS = (
    "step",
    "observation",
    "tool",
    "context",
    "truncation",
    "marker",
    "retention",
    "followup_observed",
    "returned_chars",
    "returned_bytes",
)
_FOLLOW_FIELDS = (
    "origin_step",
    "origin_observation",
    "step",
    "call",
    "tool",
    "access_kind",
    "partial_requested",
    "outcome",
    "delay_calls",
)

_DIGEST = re.compile(r"[0-9a-f]{64}")
_ALL_RECORDS = """SELECT id,root_label,path,classification,byte_count,raw_sha256,
 semantic_sha256,schema_version,session_sha256,redacted_values FROM copies"""
_STREAM = """SELECT root_label,path,classification,byte_count,raw_sha256,
 semantic_sha256,schema_version,session_sha256,redacted_values FROM copies
 WHERE id IN (SELECT MIN(id) FROM copies WHERE classification='trace' GROUP BY raw_sha256)
 ORDER BY id"""
_FAILURE_CODES = frozenset(
    {
        "unsafe_path",
        "unreadable",
        "too_large",
        "changed_file",
        "digest_mismatch",
        "invalid_json",
        "invalid_structure",
        "invalid_inventory",
    }
)
_PROGRESS_INTERVAL = 100
_NATIVE_CAP = 4 * 1024 * 1024


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="backslashreplace")).hexdigest()


def _run_locator(path: Path, root: str) -> str | None:
    if root == "archive":
        return None
    parent = path.parent.parent.parent if path.parent.name == "agent" else path.parent.parent
    return digest(str(parent))


def _checked_record(row: tuple[object, ...]) -> Inspection:
    record = _record(row)
    if record.classification == "trace":
        hashes = (record.raw_sha256, record.semantic_sha256)
        if not all(value is not None and _DIGEST.fullmatch(value) for value in hashes):
            raise CorpusError("invalid_inventory")
    if record.candidate.root_label not in ROOT_NAMES:
        raise CorpusError("invalid_inventory")
    if not record.candidate.path.is_absolute() or ".." in record.candidate.path.parts:
        raise CorpusError("invalid_inventory")
    return record


def _validate_inventory(source: sqlite3.Connection) -> None:
    if source.execute("SELECT complete FROM state").fetchall() != [(1,)]:
        raise CorpusError("inventory_incomplete")
    for row in source.execute(_ALL_RECORDS):
        if type(row[0]) is not int or row[0] <= 0:
            raise CorpusError("invalid_inventory")
        _checked_record(row[1:])
    # Validate the stream query (including id) before creating any output.
    source.execute(_STREAM)


@contextmanager
def _source(inventory: Path) -> Iterator[sqlite3.Connection]:
    path = _absolute(inventory)
    try:
        with _directory(path.parent) as parent:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            connection = sqlite3.connect(f"file:/proc/self/fd/{fd}?mode=ro", uri=True)
            try:
                connection.execute("PRAGMA query_only=ON")
                _validate_inventory(connection)
                yield connection
            finally:
                connection.close()
        finally:
            os.close(fd)
    except (OSError, sqlite3.Error):
        raise CorpusError("invalid_inventory") from None


def _copy_rows(source: sqlite3.Connection) -> list[tuple[str, int, str, str | None, str]]:
    rows = source.execute(
        "SELECT semantic_sha256,id,root_label,path,raw_sha256 "
        "FROM copies WHERE classification='trace' ORDER BY id"
    )
    return [
        (semantic, number, root, _run_locator(Path(path), root), raw)
        for semantic, number, root, path, raw in rows
    ]


def _canonical_runs(rows: list[tuple[str, int, str, str | None, str]]) -> dict[str, str]:
    parents: dict[str, str] = {}
    representatives: dict[str, str] = {}

    def find(node: str) -> str:
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for semantic, _, _, run_id, _ in rows:
        if run_id is not None:
            first = representatives.setdefault(semantic, run_id)
            a, b = find(first), find(run_id)
            parents[max(a, b)] = min(a, b)
    return {run_id: find(run_id) for run_id in parents}


def _copies(source: sqlite3.Connection, target: sqlite3.Connection) -> dict[str, str | None]:
    rows = _copy_rows(source)
    canonical = _canonical_runs(rows)
    runs: dict[str, str | None] = {}
    for trace_id, number, root, run_id, raw in rows:
        canonical_id = canonical.get(run_id) if run_id else None
        if canonical_id is not None or trace_id not in runs:
            runs[trace_id] = canonical_id
        target.execute(
            "INSERT INTO copies VALUES(?,?,?,?,?)", (trace_id, number, root, canonical_id, raw)
        )
    return runs


def _native_candidates(source: sqlite3.Connection) -> dict[str, list[tuple[Path, str]]]:
    found: dict[str, list[tuple[Path, str]]] = {}
    rows = source.execute(
        "SELECT semantic_sha256,raw_sha256,path FROM copies "
        "WHERE classification='trace' AND root_label='harbor-full' ORDER BY id"
    )
    for semantic, raw, path in rows:
        found.setdefault(semantic, []).append((Path(path), raw))
    return found


def _session_definitions(trace_path: Path, session: str) -> Iterator[Path]:
    # Discover fixed session locations. Never interpolate a trace-supplied session/path.
    sessions = trace_path.parent / "sessions"
    try:
        with _directory(sessions) as fd, os.scandir(fd) as entries:
            branches = sorted(e.name for e in entries if e.is_dir(follow_symlinks=False))
    except FileNotFoundError:
        return
    for branch in branches:
        with _directory(sessions / branch) as fd, os.scandir(fd) as entries:
            names = sorted(
                e.name for e in entries if e.name == session and e.is_dir(follow_symlinks=False)
            )
        for name in names:
            yield sessions / branch / name / "tool_definitions.json"


def _native_value(path: Path, agent_name: object) -> tuple[object, Doc]:
    try:
        # Native definitions are JSON LISTS: read_document intentionally rejects those.
        data, status, _ = _read(Candidate("harbor-full", path), _NATIVE_CAP)
        if data is None:
            return None, {"definition_sha256": None, "status": status}
        value, _ = _parse(data)
    except OSError as error:
        return None, {"definition_sha256": None, "status": _io_classification(error)}
    except CorpusError:
        return None, {"definition_sha256": None, "status": "invalid_json"}
    declarations = analyze_trace({"agent": {"name": agent_name}, "steps": []}, value).declarations
    valid = declarations.status == "known"
    return (value if valid else None), {
        "definition_sha256": hashlib.sha256(data).hexdigest(),
        "status": "matched_session" if valid else "invalid_structure",
    }


def _native_copy(key: str, path: Path, raw: str, session: str) -> tuple[list[object], list[Doc]]:
    values: list[object] = []
    records: list[Doc] = []
    try:
        # Pin the actual trial copy, not merely the inventory's association.
        loaded = read_document(Candidate("harbor-full", path), raw)
        if _semantic_hash(loaded.doc) != key:
            return [], [{"definition_sha256": None, "status": "digest_mismatch"}]
        if loaded.doc.get("session_id") != session:
            return [], [{"definition_sha256": None, "status": "session_mismatch"}]
        for definition in _session_definitions(path, session):
            value, record = _native_value(
                definition, as_object(loaded.doc.get("agent")).get("name")
            )
            records.append(record)
            if value is not None:
                values.append(value)
    except CorpusError as error:
        records.append({"definition_sha256": None, "status": _failure_code(error)})
    except OSError as error:
        records.append({"definition_sha256": None, "status": _io_classification(error)})
    return values, records


def _native(
    key: str, doc: JsonObject, candidates: dict[str, list[tuple[Path, str]]]
) -> tuple[object, list[Doc]]:
    session = as_str(doc.get("session_id"))
    if not session:
        return None, []
    definitions: list[object] = []
    records: list[Doc] = []
    for path, raw in candidates.get(key, []):
        values, sources = _native_copy(key, path, raw, session)
        definitions.extend(values)
        records.extend(sources)
    serial = {_semantic_hash(value) for value in definitions}
    usable = len(serial) == 1 and all(s["status"] == "matched_session" for s in records)
    status = "conflict" if len(serial) > 1 else "not_used"
    for record in records:
        if record["status"] == "matched_session" and not usable:
            record["status"] = status
    return (definitions[0] if usable else None), records


def _insert_metrics(
    db: sqlite3.Connection,
    trace_id: str,
    metrics: Metrics,
    scope: str,
    parent: str | None,
    run_id: str | None,
) -> None:
    row = metrics.trace
    db.execute(
        "INSERT INTO traces VALUES(?,?,?,?,?,?,?,?)",
        (
            trace_id,
            row["family"],
            row["signature"],
            row["version"],
            scope,
            parent,
            run_id,
            json.dumps(row),
        ),
    )
    db.executemany(
        "INSERT INTO tools VALUES(?,?,?,?,?,?,?)",
        [(trace_id, *(row[k] for k in _TOOL_FIELDS)) for row in metrics.tools],
    )
    db.executemany(
        "INSERT INTO outputs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        [(trace_id, *(row[k] for k in _OUTPUT_FIELDS), json.dumps(row)) for row in metrics.outputs],
    )
    db.executemany(
        "INSERT INTO followups VALUES(?,?,?,?,?,?,?,?,?,?)",
        [(trace_id, *(row[k] for k in _FOLLOW_FIELDS)) for row in metrics.followups],
    )


def _insert_call_audit(
    db: sqlite3.Connection, trace_id: str, call_audit: CallAudit, family: str | None
) -> None:
    requests: Counter[tuple[str | None, ModelSource]] = Counter()
    unknown: Counter[tuple[str | None, ModelSource]] = Counter()
    rejected: Counter[tuple[str | None, ModelSource]] = Counter()
    for attempt in call_audit.attempts:
        key = (attempt.model, attempt.model_source)
        requests[key] += 1
        unknown[key] += tool_label(attempt.tool_label or "", family) is None
    for event in call_audit.rejected:
        rejected[event.attempt.model, event.attempt.model_source] += 1
    db.executemany(
        "INSERT INTO model_usage VALUES(?,?,?,?,?,?)",
        [
            (trace_id, model, source, total, rejected[model, source], unknown[model, source])
            for (model, source), total in requests.items()
        ],
    )
    db.executemany(
        "INSERT INTO rejected_tools VALUES(?,?,?,?,?,?,?,?,?,?)",
        [
            (
                trace_id,
                event.attempt.step,
                event.attempt.index,
                tool_label(event.attempt.tool_label or "", family),
                event.attempt.model,
                event.attempt.model_source,
                event.observation,
                event.reason,
                event.declaration_conflict,
                event.pairing,
            )
            for event in call_audit.rejected
        ],
    )


def _children(doc: JsonObject) -> Iterator[JsonObject]:
    raw = doc.get("subagent_trajectories")
    children = list(as_object(raw).values()) if is_object(raw) else as_list(raw)
    for child in children:
        value = as_object(child)
        if isinstance(value.get("steps"), list):
            yield value


def _remove_metrics(db: sqlite3.Connection, key: str) -> None:
    db.execute("DELETE FROM traces WHERE trace_id=?", (key,))
    db.execute("DELETE FROM tools WHERE trace_id=?", (key,))
    db.execute("DELETE FROM outputs WHERE trace_id=?", (key,))
    db.execute("DELETE FROM followups WHERE trace_id=?", (key,))
    db.execute("DELETE FROM model_usage WHERE trace_id=?", (key,))
    db.execute("DELETE FROM rejected_tools WHERE trace_id=?", (key,))


def _analyze_document(
    db: sqlite3.Connection,
    doc: JsonObject,
    key: str,
    native: object,
    run_id: str | None,
    parent: str | None = None,
    *,
    redacted_values: int = 0,
) -> None:
    existing = db.execute("SELECT scope FROM traces WHERE trace_id=?", (key,)).fetchone()
    if existing and (parent is not None or existing[0] == "primary"):
        return
    # A later physical primary supersedes nested attribution AND native fallback.
    trace = analyze_trace(doc, native)
    if redacted_values:
        issues = ("redacted_input",)
        if parent is not None:
            issues += ("redacted_values_parent_inherited",)
        trace = replace(trace, issues=tuple(dict.fromkeys((*trace.issues, *issues))))
    call_audit = evaluate_calls(trace, doc)
    metrics = measure(trace, call_audit)
    # The loader counts whole-document bare markers, not their locations. Nested
    # traces inherit that coverage gap; this count is not a child-local total.
    metrics.trace["redacted_values"] = redacted_values
    metrics.trace["redacted_values_basis"] = "parent_document" if parent else "document"
    if existing:
        _remove_metrics(db, key)
    _insert_metrics(db, key, metrics, "nested" if parent else "primary", parent, run_id)
    _insert_call_audit(db, key, call_audit, trace.family)
    for child in _children(doc):
        _analyze_document(
            db, child, _semantic_hash(child), None, run_id, key, redacted_values=redacted_values
        )


def _has_atif_declarations(doc: JsonObject) -> bool:
    agent = as_object(doc.get("agent"))
    return any(
        key in owner
        for owner, key in (
            (agent, "tool_definitions"),
            (agent, "tools"),
            (doc, "tool_definitions"),
        )
    )


def _analyze_record(
    db: sqlite3.Connection,
    record: Inspection,
    natives: dict[str, list[tuple[Path, str]]],
    runs: dict[str, str | None],
) -> None:
    key = record.semantic_sha256
    if key is None or record.raw_sha256 is None:
        raise CorpusError("invalid_inventory")
    loaded = read_document(record.candidate, record.raw_sha256)
    if _semantic_hash(loaded.doc) != key:
        raise CorpusError("digest_mismatch")
    native, sources = _native(key, loaded.doc, natives)
    if _has_atif_declarations(loaded.doc):
        for source in sources:
            if source["status"] == "matched_session":
                source["status"] = "not_used_atif"
    db.execute("DELETE FROM native_sources WHERE trace_id=?", (key,))
    db.executemany(
        "INSERT INTO native_sources VALUES(?,?,?)",
        [(key, s["definition_sha256"], s["status"]) for s in sources],
    )
    _analyze_document(
        db, loaded.doc, key, native, runs.get(key), redacted_values=loaded.redacted_values
    )


def _failure_code(error: Exception) -> str:
    code = str(error) if isinstance(error, CorpusError) else "analysis_failed"
    return code if code in _FAILURE_CODES else "analysis_failed"


def _attempt_record(
    db: sqlite3.Connection,
    record: Inspection,
    natives: dict[str, list[tuple[Path, str]]],
    runs: dict[str, str | None],
) -> None:
    db.execute("SAVEPOINT record_analysis")
    try:
        _analyze_record(db, record, natives, runs)
    except (
        CorpusError,
        ValueError,
        TypeError,
        RecursionError,
        OSError,
        sqlite3.Error,
        KeyError,
        IndexError,
    ) as error:
        db.execute("ROLLBACK TO record_analysis")
        db.execute(
            "INSERT INTO failures VALUES(?,?)", (record.semantic_sha256, _failure_code(error))
        )
    finally:
        db.execute("RELEASE record_analysis")


def _new_folder(output: Path) -> int:
    absolute = _absolute(output)
    repo = Path(__file__).resolve().parents[2]
    if absolute.is_relative_to(repo):
        raise CorpusError("output_inside_repository")
    with _directory(absolute.parent) as parent:
        os.mkdir(absolute.name, mode=0o700, dir_fd=parent)
        folder = os.open(absolute.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    os.fchmod(folder, 0o700)
    return folder


@contextmanager
def _output(output: Path) -> Iterator[tuple[sqlite3.Connection, int]]:
    try:
        folder = _new_folder(output)
        try:
            fd = os.open(
                "analysis.sqlite3",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=folder,
            )
            os.close(fd)
            db = sqlite3.connect(f"/proc/self/fd/{folder}/analysis.sqlite3")
            try:
                db.executescript(_SCHEMA)
                db.commit()
                yield db, folder
            finally:
                db.close()
        finally:
            os.close(folder)
    except (OSError, sqlite3.Error):
        raise CorpusError("private_output_unavailable") from None


def _process_records(
    source: sqlite3.Connection,
    db: sqlite3.Connection,
    natives: dict[str, list[tuple[Path, str]]],
    runs: dict[str, str | None],
    limit: int | None,
    start: float,
) -> int:
    processed = 0
    records = iter(source.execute(_STREAM))
    while limit is None or processed < limit:
        row = next(records, None)
        if row is None:
            break
        _attempt_record(db, _checked_record(row), natives, runs)
        processed += 1
        if processed % _PROGRESS_INTERVAL == 0:
            db.commit()
            print(
                json.dumps({"processed": processed, "seconds": round(time.monotonic() - start)}),
                flush=True,
            )
    return processed


def _summary(db: sqlite3.Connection, processed: int, limit: int | None, start: float) -> Doc:
    failures = db.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
    return {
        "processed_files": processed,
        "traces": db.execute("SELECT COUNT(*) FROM traces").fetchone()[0],
        "failures": failures,
        "complete_selection": limit is None and failures == 0,
        "elapsed_seconds": round(time.monotonic() - start),
    }


def _write_summary(folder: int, summary: Doc) -> None:
    fd = os.open(
        "analysis-summary.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=folder,
    )
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")


def _finish(db: sqlite3.Connection, folder: int, summary: Doc) -> None:
    db.executescript("""
        CREATE INDEX output_trace ON outputs(trace_id);
        CREATE INDEX tool_trace ON tools(trace_id);
        CREATE INDEX family_trace ON traces(family,signature);
    """)
    # Export must succeed before the database can be marked complete.
    _write_summary(folder, summary)
    db.execute("UPDATE state SET complete=?", (int(summary["complete_selection"]),))
    db.commit()


def _abandon(db: sqlite3.Connection, folder: int) -> None:
    with suppress(FileNotFoundError):
        os.unlink("analysis-summary.json", dir_fd=folder)
    db.rollback()
    db.execute("UPDATE state SET complete=0")
    db.commit()


def run(inventory: Path, output: Path, limit: int | None = None) -> Doc:
    """Analyze a complete readonly inventory; never alter source data or old outputs."""
    if limit is not None and (type(limit) is not int or limit < 0):
        raise CorpusError("invalid_limit")
    start = time.monotonic()
    with _source(inventory) as source, _output(output) as (db, folder):
        finished = False
        try:
            runs = _copies(source, db)
            natives = _native_candidates(source)
            processed = _process_records(source, db, natives, runs, limit, start)
            summary = _summary(db, processed, limit, start)
            _finish(db, folder, summary)
            finished = True
            return summary
        except (
            CorpusError,
            OSError,
            sqlite3.Error,
            ValueError,
            TypeError,
            RecursionError,
            RuntimeError,
            KeyError,
            IndexError,
        ):
            raise CorpusError("audit_incomplete") from None
        finally:
            if not finished:
                _abandon(db, folder)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.inventory, args.output, args.limit), indent=2))


if __name__ == "__main__":
    main()
