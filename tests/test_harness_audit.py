"""Synthetic-only, offline regressions for the private harness audit runner."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import stat
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from atif_scan.data.jsonval import as_list, as_object, as_str

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools" / "research"))
import harness_audit as audit
import harness_corpus as corpus


def _trace(session: str = "synthetic-session", message: str = "synthetic") -> Doc:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": session,
        "agent": {"name": "fast-agent"},
        "steps": [{"source": "agent", "message": message}],
    }


def _shell_steps() -> list[Doc]:
    spool = "/tmp/never-read-synthetic-audit-spool/output.txt"
    result = (
        "[Output truncated: showing first 4 bytes and last 6 bytes of 30 bytes "
        "(~2 of ~7 tokens); omitted 20 middle bytes. "
        f"The complete output is available during this session at {spool}. "
        "Use read_text_file for selected line ranges or run a targeted search against that file; "
        "avoid reading the entire file unless necessary.]"
    )
    return [
        {
            "source": "agent",
            "tool_calls": [
                {
                    "function_name": "shell",
                    "tool_call_id": "origin",
                    "arguments": {"command": "inert synthetic command"},
                }
            ],
            "observation": {"results": [{"source_call_id": "origin", "content": result}]},
        },
        {
            "source": "agent",
            "tool_calls": [
                {
                    "function_name": "read_text_file",
                    "tool_call_id": "reader",
                    "arguments": {"path": spool},
                }
            ],
            "observation": {
                "results": [{"source_call_id": "reader", "content": "synthetic retained text"}]
            },
        },
    ]


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _inventory(tmp_path: Path, entries: list[tuple[str, Path, Doc]]) -> Path:
    path = tmp_path / "inventory.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript(corpus._SCHEMA)
        for label, file, doc in entries:
            _write(file, doc)
            corpus._insert(db, corpus.inspect_candidate(corpus.Candidate(label, file)))
        db.execute("UPDATE state SET complete=1")
    return path


def _definitions(schema: object = None) -> list[object]:
    return [
        {
            "type": "function",
            "function": {
                "name": "tools.Shell",
                "parameters": {} if schema is None else schema,
            },
        }
    ]


def _native_file(trace: Path, session: str, value: object, branch: str = "provider") -> Path:
    return _write(trace.parent / "sessions" / branch / session / "tool_definitions.json", value)


def _query(output: Path, sql: str) -> list[tuple[object, ...]]:
    with sqlite3.connect(output / "analysis.sqlite3") as db:
        return cast("list[tuple[object, ...]]", db.execute(sql).fetchall())


def _data(output: Path) -> Doc:
    row = _query(output, "SELECT data FROM traces")[0]
    return as_object(json.loads(as_str(row[0]) or "{}"))


def test_native_list_exact_session_parent_and_private_allowlisted_outputs(tmp_path: Path) -> None:
    doc = _trace("exact-session", "private synthetic marker /never/export/marker")
    doc["steps"] = _shell_steps()
    mirror = tmp_path / "hf/job/trial/agent/trajectory.json"
    full = tmp_path / "harbor-full/job/trial/agent/trajectory.json"
    inventory = _inventory(tmp_path, [("hf", mirror, doc), ("harbor-full", full, doc)])
    definitions = _native_file(full, "exact-session", _definitions())
    _native_file(full, "prefix-exact-session", _definitions({"type": "string"}))
    _native_file(full, "exact-session-suffix", _definitions({"type": "number"}))
    # A filename containing the session is not session-directory provenance.
    _write(full.parent / "sessions/provider/wrong/exact-session.json", _definitions())
    before = inventory.read_bytes()
    output = tmp_path / "analysis"
    summary = audit.run(inventory, output)
    assert summary["complete_selection"] is True
    assert summary["traces"] == summary["processed_files"] == 1
    assert _query(output, "SELECT complete FROM state") == [(1,)]
    assert _data(output)["inventory_source"] == "native"
    assert _data(output)["inventory_status"] == "known"
    assert _query(output, "SELECT definition_sha256,status FROM native_sources") == [
        (hashlib.sha256(definitions.read_bytes()).hexdigest(), "matched_session")
    ]
    assert inventory.read_bytes() == before
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for file in output.iterdir():
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
        contents = file.read_bytes()
        assert str(tmp_path).encode() not in contents
        assert b"private synthetic marker" not in contents
        assert b"exact-session" not in contents
        assert b"/tmp/never-read-synthetic-audit-spool" not in contents
        assert b"synthetic retained text" not in contents
    assert set(summary) == {
        "processed_files",
        "traces",
        "failures",
        "complete_selection",
        "elapsed_seconds",
    }


def test_native_requires_pinned_matching_trial_copy(tmp_path: Path) -> None:
    doc = _trace()
    mirror = tmp_path / "hf/job/trial/agent/trajectory.json"
    full = tmp_path / "harbor-full/job/trial/agent/trajectory.json"
    inventory = _inventory(tmp_path, [("hf", mirror, doc), ("harbor-full", full, doc)])
    _native_file(full, "synthetic-session", _definitions())
    _write(full, _trace(message="changed after inventory"))
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    assert _data(output)["inventory_status"] == "missing"
    assert _query(output, "SELECT status FROM native_sources") == [("digest_mismatch",)]


@pytest.mark.parametrize("placement", ["file", "session", "branch", "sessions"])
def test_native_symlinks_never_supply_definitions(tmp_path: Path, placement: str) -> None:
    full = tmp_path / "harbor-full/job/trial/agent/trajectory.json"
    inventory = _inventory(tmp_path, [("harbor-full", full, _trace())])
    definition = _native_file(full, "synthetic-session", _definitions())
    target = tmp_path / "untrusted-target"
    if placement == "file":
        definition.rename(target)
        definition.symlink_to(target)
    else:
        link = {
            "session": definition.parent,
            "branch": definition.parent.parent,
            "sessions": definition.parent.parent.parent,
        }[placement]
        link.rename(target)
        link.symlink_to(target, target_is_directory=True)
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    assert _data(output)["inventory_status"] == "missing"
    assert ("matched_session",) not in _query(output, "SELECT status FROM native_sources")


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        [None],
        [{"name": "Shell", "parameters": None}],
        [{"name": "Shell", "parameters": {}}, {"name": "shell", "parameters": {}}],
    ],
)
def test_malformed_native_schema_is_unknown(tmp_path: Path, bad: object) -> None:
    full = tmp_path / "harbor-full/job/trial/agent/trajectory.json"
    inventory = _inventory(tmp_path, [("harbor-full", full, _trace())])
    _native_file(full, "synthetic-session", bad)
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    assert _data(output)["inventory_status"] != "known"
    assert _query(output, "SELECT status FROM native_sources") == [("invalid_structure",)]


@pytest.mark.parametrize("divergent", [False, True])
def test_native_identical_tool_names_do_not_hide_schema_conflict(
    tmp_path: Path, divergent: bool
) -> None:
    full = tmp_path / "harbor-full/job/trial/agent/trajectory.json"
    inventory = _inventory(tmp_path, [("harbor-full", full, _trace())])
    _native_file(full, "synthetic-session", _definitions(), "first")
    _native_file(
        full, "synthetic-session", _definitions({"type": "object"} if divergent else {}), "second"
    )
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    assert _data(output)["inventory_status"] == ("missing" if divergent else "known")
    expected = "conflict" if divergent else "matched_session"
    assert _query(output, "SELECT status FROM native_sources") == [(expected,), (expected,)]


def test_native_not_used_when_atif_declarations_present(tmp_path: Path) -> None:
    full = tmp_path / "harbor-full/job/trial/agent/trajectory.json"
    doc = _trace()
    doc["tool_definitions"] = []
    inventory = _inventory(tmp_path, [("harbor-full", full, doc)])
    _native_file(full, "synthetic-session", _definitions())
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    assert _data(output)["declared_tools"] == 0
    assert _query(output, "SELECT status FROM native_sources") == [("not_used_atif",)]


def test_run_union_find_uses_trace_aliases_not_session_ids(tmp_path: Path) -> None:
    shared_a = _trace("shared-session", "one")
    shared_b = _trace("other-session", "two")
    divergent = _trace("shared-session", "divergent")
    entries = [
        ("hf", tmp_path / "hf/a/t1/agent/trajectory.json", shared_a),
        ("harbor", tmp_path / "harbor/b/t1/agent/trajectory.json", shared_a),
        ("harbor", tmp_path / "harbor/b/t2/agent/trajectory.json", shared_b),
        ("local-jobs", tmp_path / "local-jobs/c/t2/agent/trajectory.json", shared_b),
        ("hf", tmp_path / "hf/d/t1/agent/trajectory.json", divergent),
        ("archive", tmp_path / "archive/copy.json", shared_a),
        ("archive", tmp_path / "archive/only.json", _trace("archive-only")),
    ]
    inventory = _inventory(tmp_path, entries)
    output = tmp_path / "analysis"
    summary = audit.run(inventory, output)
    assert summary["traces"] == summary["processed_files"] == 4
    runs = _query(output, "SELECT run_id FROM copies ORDER BY source_id")
    assert len({row[0] for row in runs[:4]}) == 1
    assert runs[0][0] is not None
    assert runs[4][0] != runs[0][0]
    assert runs[5:] == [(None,), (None,)]
    assert _query(output, "SELECT COUNT(DISTINCT run_id) FROM traces") == [(2,)]


@pytest.mark.parametrize("embedded_first", [False, True])
def test_nested_dictionary_dedup_preserves_primary_attribution_and_native(
    tmp_path: Path, embedded_first: bool
) -> None:
    child = _trace("child-session", "synthetic Unicode \u03bb")
    child["steps"] = [
        {
            "source": "agent",
            "tool_calls": [
                {
                    "function_name": "tools.Shell",
                    "tool_call_id": "synthetic-call",
                    "arguments": {"command": "inert synthetic"},
                }
            ],
        }
    ]
    parent = _trace("parent-session")
    parent["subagent_trajectories"] = {"first": child, "duplicate": child, "bad": None}
    parent_path = tmp_path / "hf/parent-run/trial/agent/trajectory.json"
    child_path = tmp_path / "harbor-full/child-run/trial/agent/trajectory.json"
    entries = [("hf", parent_path, parent), ("harbor-full", child_path, child)]
    inventory = _inventory(tmp_path, entries if embedded_first else list(reversed(entries)))
    _native_file(child_path, "child-session", _definitions())
    output = tmp_path / "analysis"
    summary = audit.run(inventory, output)
    assert summary["traces"] == 2
    key = corpus._semantic_hash(child)
    rows = _query(output, "SELECT trace_id,scope,parent_id,run_id,data FROM traces")
    row = next(row for row in rows if row[0] == key)
    assert row[1:4] == ("primary", None, audit.digest(str(child_path.parent.parent.parent)))
    assert as_object(json.loads(as_str(row[4]) or "{}"))["inventory_source"] == "native"
    assert _query(output, "SELECT SUM(recorded_calls) FROM tools") == [(1,)]
    assert _query(output, "SELECT COUNT(*) FROM native_sources") == [(1,)]


def test_nested_lists_analyzed_separately_and_canonical_aliases_count_once(tmp_path: Path) -> None:
    child = _trace("nested-only", "synthetic \u03bb")
    parent = _trace()
    parent["subagent_trajectories"] = [child, dict(reversed(list(child.items()))), "invalid"]
    file = tmp_path / "hf/job/trial/trajectory.json"
    inventory = _inventory(tmp_path, [("hf", file, parent)])
    output = tmp_path / "analysis"
    assert audit.run(inventory, output)["traces"] == 2
    rows = _query(output, "SELECT trace_id,scope,parent_id FROM traces ORDER BY scope")
    assert rows[0] == (corpus._semantic_hash(child), "nested", corpus._semantic_hash(parent))


@pytest.mark.parametrize("limit", [0, 1, 10])
def test_pilot_is_always_incomplete(tmp_path: Path, limit: int) -> None:
    inventory = _inventory(tmp_path, [("hf", tmp_path / "hf/t.json", _trace())])
    output = tmp_path / "analysis"
    summary = audit.run(inventory, output, limit)
    assert summary["processed_files"] == min(limit, 1)
    assert summary["complete_selection"] is False
    assert _query(output, "SELECT complete FROM state") == [(0,)]


@pytest.mark.parametrize("code", ["digest_mismatch", "private path /never/export/error"])
def test_record_failure_rolls_back_parent_children_sources_and_all_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str
) -> None:
    parent = _trace("parent")
    parent["steps"] = _shell_steps()
    child = _trace("bad-child")
    parent["subagent_trajectories"] = [child]
    full = tmp_path / "harbor-full/job/trial/agent/trajectory.json"
    good = tmp_path / "hf/other/trial/trajectory.json"
    good_doc = _trace("good")
    good_doc["tool_definitions"] = []
    inventory = _inventory(tmp_path, [("harbor-full", full, parent), ("hf", good, good_doc)])
    _native_file(full, "parent", _definitions())
    original = audit.measure

    def fail_child(trace, call_audit=None):
        if trace.declarations.status == "missing":
            raise corpus.CorpusError(code)
        metrics = original(trace, call_audit)
        if trace.declarations.tools:
            assert metrics.outputs and metrics.followups
        return metrics

    monkeypatch.setattr(audit, "measure", fail_child)
    output = tmp_path / "analysis"
    summary = audit.run(inventory, output)
    assert summary["failures"] == 1
    assert summary["traces"] == 1
    assert summary["complete_selection"] is False
    assert _query(output, "SELECT complete FROM state") == [(0,)]
    for sql in (
        "SELECT COUNT(*) FROM tools",
        "SELECT COUNT(*) FROM outputs",
        "SELECT COUNT(*) FROM followups",
        "SELECT COUNT(*) FROM native_sources",
        "SELECT COUNT(*) FROM model_usage",
        "SELECT COUNT(*) FROM rejected_tools",
    ):
        assert _query(output, sql) == [(0,)]
    assert _query(output, "SELECT trace_id FROM traces") == [(corpus._semantic_hash(good_doc),)]
    expected = code if code == "digest_mismatch" else "analysis_failed"
    assert _query(output, "SELECT code FROM failures") == [(expected,)]


def test_changed_primary_is_a_failure_and_not_complete(tmp_path: Path) -> None:
    file = tmp_path / "hf/job/trial/trajectory.json"
    inventory = _inventory(tmp_path, [("hf", file, _trace())])
    _write(file, _trace(message="changed"))
    output = tmp_path / "analysis"
    summary = audit.run(inventory, output)
    assert summary["failures"] == 1
    assert summary["complete_selection"] is False
    assert _query(output, "SELECT code FROM failures") == [("digest_mismatch",)]


@pytest.mark.parametrize("kind", ["incomplete", "schema", "hash", "root", "id"])
def test_bad_inventory_refused_before_creating_output(tmp_path: Path, kind: str) -> None:
    inventory = _inventory(tmp_path, [("hf", tmp_path / "hf/t.json", _trace())])
    with sqlite3.connect(inventory) as db:
        if kind == "incomplete":
            db.execute("UPDATE state SET complete=0")
        elif kind == "schema":
            db.execute("DROP TABLE copies")
        elif kind == "hash":
            db.execute("UPDATE copies SET raw_sha256=NULL")
        elif kind == "root":
            db.execute("UPDATE copies SET root_label='/never/export/root'")
        else:
            db.execute("UPDATE copies SET id=-1")
    output = tmp_path / "analysis"
    with pytest.raises(corpus.CorpusError, match=r"inventory_incomplete|invalid_inventory"):
        audit.run(inventory, output)
    assert not output.exists()


def test_input_and_output_symlink_ancestors_are_refused(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, [("hf", tmp_path / "hf/t.json", _trace())])
    link = tmp_path / "input-link.sqlite3"
    link.symlink_to(inventory)
    output = tmp_path / "analysis"
    with pytest.raises(corpus.CorpusError, match="invalid_inventory"):
        audit.run(link, output)
    assert not output.exists()
    external = tmp_path / "external"
    external.mkdir()
    directory_link = tmp_path / "directory-link"
    directory_link.symlink_to(external, target_is_directory=True)
    with pytest.raises(corpus.CorpusError, match="private_output_unavailable"):
        audit.run(inventory, directory_link / "analysis")
    assert not (external / "analysis").exists()
    final_link = tmp_path / "final-link"
    final_link.symlink_to(external, target_is_directory=True)
    with pytest.raises(corpus.CorpusError, match="private_output_unavailable"):
        audit.run(inventory, final_link)
    assert list(external.iterdir()) == []


def test_output_inside_repository_is_refused(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, [("hf", tmp_path / "hf/t.json", _trace())])
    output = Path(__file__).absolute().parents[1] / "never-created-synthetic-audit"
    with pytest.raises(corpus.CorpusError, match="output_inside_repository"):
        audit.run(inventory, output)
    assert not output.exists()


@pytest.mark.parametrize("limit", [-1, True])
def test_invalid_limit_does_not_create_output(tmp_path: Path, limit: int) -> None:
    output = tmp_path / "analysis"
    with pytest.raises(corpus.CorpusError, match="invalid_limit"):
        audit.run(tmp_path / "absent.sqlite3", output, limit)
    assert not output.exists()


def test_end_stream_failure_retains_incomplete_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = _inventory(tmp_path, [("hf", tmp_path / "hf/t.json", _trace())])
    original = audit._process_records

    def broken_stream(source, db, natives, runs, limit, start):
        original(source, db, natives, runs, limit, start)
        db.commit()
        raise corpus.CorpusError("private stream exception /never/export")

    monkeypatch.setattr(audit, "_process_records", broken_stream)
    output = tmp_path / "analysis"
    with pytest.raises(corpus.CorpusError, match="audit_incomplete"):
        audit.run(inventory, output)
    assert _query(output, "SELECT complete FROM state") == [(0,)]
    assert not (output / "analysis-summary.json").exists()


def test_summary_export_failure_never_marks_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = _inventory(tmp_path, [("hf", tmp_path / "hf/t.json", _trace())])

    original = audit._write_summary

    def unavailable(folder, summary):
        original(folder, summary)
        raise OSError("private exception /never/export")

    monkeypatch.setattr(audit, "_write_summary", unavailable)
    output = tmp_path / "analysis"
    with pytest.raises(corpus.CorpusError, match="audit_incomplete"):
        audit.run(inventory, output)
    assert _query(output, "SELECT complete FROM state") == [(0,)]
    assert not (output / "analysis-summary.json").exists()


def test_formatted_aliases_share_metrics_and_conflicting_native_evidence(tmp_path: Path) -> None:
    doc = _trace()
    one = tmp_path / "harbor-full/one/trial/agent/trajectory.json"
    two = tmp_path / "harbor-full/two/trial/agent/trajectory.json"
    inventory = _inventory(tmp_path, [("harbor-full", one, doc), ("harbor-full", two, doc)])
    two.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    record = corpus.inspect_candidate(corpus.Candidate("harbor-full", two))
    with sqlite3.connect(inventory) as db:
        db.execute("UPDATE copies SET raw_sha256=? WHERE id=2", (record.raw_sha256,))
    _native_file(one, "synthetic-session", _definitions())
    _native_file(two, "synthetic-session", _definitions({"type": "object"}))
    output = tmp_path / "analysis"
    summary = audit.run(inventory, output)
    assert summary["processed_files"] == 2
    assert summary["traces"] == 1
    assert _data(output)["inventory_status"] == "missing"
    assert _query(output, "SELECT status FROM native_sources") == [("conflict",), ("conflict",)]
    assert _query(output, "SELECT COUNT(DISTINCT run_id) FROM copies") == [(1,)]


def test_explicit_empty_native_list_is_known(tmp_path: Path) -> None:
    full = tmp_path / "harbor-full/job/trial/agent/trajectory.json"
    inventory = _inventory(tmp_path, [("harbor-full", full, _trace())])
    _native_file(full, "synthetic-session", [])
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    assert _data(output)["inventory_status"] == "known"
    assert _data(output)["declared_tools"] == 0


def test_directory_swap_cannot_redirect_database_or_summary_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = _inventory(tmp_path, [("hf", tmp_path / "hf/t.json", _trace())])
    output = tmp_path / "analysis"
    pinned = tmp_path / "pinned-analysis"
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    original = audit._process_records

    def swapped(source, db, natives, runs, limit, start):
        output.rename(pinned)
        output.symlink_to(redirected, target_is_directory=True)
        return original(source, db, natives, runs, limit, start)

    monkeypatch.setattr(audit, "_process_records", swapped)
    # SQLite may refuse a moved database; either way, no replacement path is written.
    with pytest.raises(corpus.CorpusError, match=r"audit_incomplete|private_output_unavailable"):
        audit.run(inventory, output)
    assert _query(pinned, "SELECT complete FROM state") == [(0,)]
    assert not (pinned / "analysis-summary.json").exists()
    assert list(redirected.iterdir()) == []


def test_source_database_swap_cannot_change_pinned_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = _inventory(tmp_path, [("hf", tmp_path / "hf/t.json", _trace())])
    original = audit._copies

    def swapped(source, target):
        inventory.rename(tmp_path / "pinned-inventory.sqlite3")
        inventory.write_text("not SQLite", encoding="utf-8")
        return original(source, target)

    monkeypatch.setattr(audit, "_copies", swapped)
    summary = audit.run(inventory, tmp_path / "analysis")
    assert summary["traces"] == 1
    assert summary["complete_selection"] is True


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_stream_interruption_clears_state_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]
) -> None:
    inventory = _inventory(tmp_path, [("hf", tmp_path / "hf/t.json", _trace())])

    def broken(source, db, natives, runs, limit, start):
        raise error("private stream failure /never/export")

    monkeypatch.setattr(audit, "_process_records", broken)
    output = tmp_path / "analysis"
    expected = KeyboardInterrupt if error is KeyboardInterrupt else corpus.CorpusError
    with pytest.raises(expected):
        audit.run(inventory, output)
    assert _query(output, "SELECT complete FROM state") == [(0,)]
    assert not (output / "analysis-summary.json").exists()


def test_partial_valid_native_evidence_is_not_labelled_matched(tmp_path: Path) -> None:
    full = tmp_path / "harbor-full/job/trial/agent/trajectory.json"
    inventory = _inventory(tmp_path, [("harbor-full", full, _trace())])
    _native_file(full, "synthetic-session", _definitions(), "good")
    _native_file(full, "synthetic-session", None, "bad")
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    assert _data(output)["inventory_status"] != "known"
    statuses = _query(output, "SELECT status FROM native_sources")
    assert set(statuses) == {("not_used",), ("invalid_structure",)}


def test_failed_primary_promotion_restores_prior_nested_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = _trace("child")
    child["steps"] = _shell_steps()
    parent = _trace("parent")
    parent["subagent_trajectories"] = [child]
    parent_path = tmp_path / "hf/parent/trial/agent/trajectory.json"
    child_path = tmp_path / "harbor-full/child/trial/agent/trajectory.json"
    inventory = _inventory(
        tmp_path,
        [
            ("hf", parent_path, parent),
            ("harbor-full", child_path, child),
        ],
    )
    _native_file(child_path, "child", _definitions())
    key = corpus._semantic_hash(child)
    original = audit._insert_metrics

    def fail_promotion(db, trace_id, metrics, scope, parent_id, run_id):
        original(db, trace_id, metrics, scope, parent_id, run_id)
        if trace_id == key and scope == "primary":
            raise ValueError("private promotion failure /never/export")

    monkeypatch.setattr(audit, "_insert_metrics", fail_promotion)
    output = tmp_path / "analysis"
    summary = audit.run(inventory, output)
    assert summary["failures"] == 1
    assert summary["traces"] == 2
    assert summary["complete_selection"] is False
    rows = _query(output, "SELECT trace_id,scope,parent_id,data FROM traces")
    row = next(row for row in rows if row[0] == key)
    assert row[1:3] == ("nested", corpus._semantic_hash(parent))
    assert as_object(json.loads(as_str(row[3]) or "{}"))["inventory_status"] == "missing"
    assert _query(output, "SELECT SUM(recorded_calls) FROM tools") == [(2,)]
    assert _query(output, "SELECT COUNT(*) FROM outputs") == [(1,)]
    assert _query(output, "SELECT COUNT(*) FROM followups") == [(1,)]
    assert _query(output, "SELECT COUNT(*) FROM native_sources") == [(0,)]


def _request(name: str, call_id: str, args: Doc | None = None) -> Doc:
    return {
        "function_name": name,
        "tool_call_id": call_id,
        "arguments": {} if args is None else args,
    }


def _request_step(calls: list[Doc], results: list[Doc]) -> Doc:
    return {"source": "agent", "tool_calls": calls, "observation": {"results": results}}


def _model_trace(version: str = "1.0") -> Doc:
    doc = _trace()
    doc["agent"] = {
        "name": "fast-agent",
        "version": version,
        "model_name": "provider/root-model",
        "tool_definitions": [],
    }
    explicit = _request_step(
        [_request("tools.Shell", "valid")],
        [{"source_call_id": "valid", "content": "Error: Invalid arguments: missing command"}],
    )
    explicit["model_name"] = "provider/step-model"
    invalid = _request_step([_request("unsafe\nprivate-name", "unknown")], [])
    invalid["model_name"] = None
    code = _request_step(
        [
            _request(
                "exec",
                "parent",
                {"input": 'await tools.exec_command({cmd: "echo inert synthetic"});'},
            )
        ],
        [{"source_call_id": "parent", "content": "synthetic shared capture"}],
    )
    code["model_name"] = "provider/step-model"
    doc["steps"] = [
        _request_step(
            [_request("functions.missing_tool", "bad"), _request("functions.Read", "good")],
            [
                {"source_call_id": "bad", "content": "Unknown tool: tools.missing_tool"},
                {"source_call_id": "good", "content": "synthetic read result"},
            ],
        ),
        explicit,
        invalid,
        code,
    ]
    return doc


def test_model_tables_exact_schema_full_denominator_and_no_harness_attribution(
    tmp_path: Path,
) -> None:
    doc = _model_trace()
    inventory = _inventory(tmp_path, [("hf", tmp_path / "hf/trace.json", doc)])
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    assert _query(output, "PRAGMA table_info(model_usage)") == [
        (0, "trace_id", "TEXT", 0, None, 0),
        (1, "model", "TEXT", 0, None, 0),
        (2, "model_source", "TEXT", 0, None, 0),
        (3, "recorded_requests", "INTEGER", 0, None, 0),
        (4, "rejected_requests", "INTEGER", 0, None, 0),
        (5, "unknown_tool_labels", "INTEGER", 0, None, 0),
    ]
    assert _query(output, "PRAGMA table_info(rejected_tools)") == [
        (0, "trace_id", "TEXT", 0, None, 0),
        (1, "step", "INTEGER", 0, None, 0),
        (2, "call", "INTEGER", 0, None, 0),
        (3, "tool", "TEXT", 0, None, 0),
        (4, "model", "TEXT", 0, None, 0),
        (5, "model_source", "TEXT", 0, None, 0),
        (6, "observation", "INTEGER", 0, None, 0),
        (7, "reason", "TEXT", 0, None, 0),
        (8, "declaration_conflict", "INTEGER", 0, None, 0),
        (9, "pairing", "TEXT", 0, None, 0),
    ]
    assert _query(
        output,
        "SELECT model,model_source,recorded_requests,rejected_requests,unknown_tool_labels "
        "FROM model_usage ORDER BY model",
    ) == [
        (None, "unknown", 1, 0, 1),
        ("provider/root-model", "root", 2, 1, 0),
        ("provider/step-model", "step", 2, 0, 0),
    ]
    assert _query(output, "SELECT * FROM rejected_tools") == [
        (
            corpus._semantic_hash(doc),
            0,
            0,
            "missing_tool",
            "provider/root-model",
            "root",
            0,
            "not_found",
            0,
            "recorded_id",
        )
    ]
    assert _query(output, "SELECT SUM(recorded_calls),SUM(static_call_sites) FROM tools") == [
        (3, 1)
    ]
    assert _data(output)["recorded_calls"] == 3
    assert _data(output)["recorded_unknown_tool_labels"] == 1
    names = {row[0] for row in _query(output, "SELECT tool FROM tools")}
    assert names == {"Read", "shell", "exec", "exec_command"}
    assert not any("model" in key or "reject" in key for key in _data(output))
    assert b"unsafe\nprivate-name" not in (output / "analysis.sqlite3").read_bytes()


def test_step_models_are_authoritative_across_harness_versions(tmp_path: Path) -> None:
    entries = [
        ("hf", tmp_path / f"hf/v{version}.json", _model_trace(version))
        for version in ("1.0", "2.0")
    ]
    inventory = _inventory(tmp_path, entries)
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    rows = _query(
        output,
        "SELECT t.version,u.model,u.model_source,u.recorded_requests "
        "FROM model_usage u JOIN traces t USING(trace_id) "
        "WHERE u.model_source='step' ORDER BY t.version",
    )
    assert rows == [
        ("1.0", "provider/step-model", "step", 2),
        ("2.0", "provider/step-model", "step", 2),
    ]
    assert _query(output, "SELECT SUM(recorded_requests) FROM model_usage") == [(10,)]


def test_invalid_tool_and_unknown_model_stay_null_in_rejection_export(tmp_path: Path) -> None:
    doc = _trace()
    doc["agent"] = {"name": "fast-agent", "model_name": "https://synthetic.invalid/model"}
    doc["steps"] = [
        _request_step(
            [_request("private label", "bad")],
            [{"source_call_id": "bad", "content": "Unknown tool: 'private label'"}],
        )
    ]
    inventory = _inventory(tmp_path, [("hf", tmp_path / "hf/trace.json", doc)])
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    assert _query(
        output, "SELECT model,model_source,recorded_requests,rejected_requests FROM model_usage"
    ) == [(None, "unknown", 1, 1)]
    assert _query(
        output, "SELECT tool,model,model_source,declaration_conflict FROM rejected_tools"
    ) == [(None, None, "unknown", None)]
    assert _query(output, "SELECT COUNT(*) FROM tools") == [(0,)]
    assert b"private label" not in (output / "analysis.sqlite3").read_bytes()


def test_model_usage_and_rejections_deduplicate_nested_then_primary(tmp_path: Path) -> None:
    child = _model_trace()
    parent = _trace("parent")
    parent["subagent_trajectories"] = [child, child]
    entries = [
        ("hf", tmp_path / "hf/parent.json", parent),
        ("hf", tmp_path / "hf/child.json", child),
    ]
    inventory = _inventory(tmp_path, entries)
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    assert _query(
        output, "SELECT SUM(recorded_requests),SUM(rejected_requests) FROM model_usage"
    ) == [(5, 1)]
    assert _query(output, "SELECT COUNT(*) FROM rejected_tools") == [(1,)]
    scopes = _query(output, "SELECT trace_id,scope,parent_id FROM traces")
    child_scope = next(row[1:] for row in scopes if row[0] == corpus._semantic_hash(child))
    assert child_scope == ("primary", None)


def test_failure_after_model_insert_rolls_back_all_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = _inventory(tmp_path, [("hf", tmp_path / "hf/trace.json", _model_trace())])
    original = audit._insert_call_audit

    def failed_insert(db, trace_id, call_audit, family):
        original(db, trace_id, call_audit, family)
        assert db.execute("SELECT SUM(recorded_requests) FROM model_usage").fetchone() == (5,)
        assert db.execute("SELECT COUNT(*) FROM rejected_tools").fetchone() == (1,)
        raise ValueError("private synthetic failure")

    monkeypatch.setattr(audit, "_insert_call_audit", failed_insert)
    output = tmp_path / "analysis"
    summary = audit.run(inventory, output)
    assert summary["failures"] == 1
    for sql in (
        "SELECT COUNT(*) FROM traces",
        "SELECT COUNT(*) FROM tools",
        "SELECT COUNT(*) FROM outputs",
        "SELECT COUNT(*) FROM followups",
        "SELECT COUNT(*) FROM model_usage",
        "SELECT COUNT(*) FROM rejected_tools",
    ):
        assert _query(output, sql) == [(0,)]


def test_failed_primary_promotion_restores_model_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = _model_trace()
    parent = _trace("parent")
    parent["subagent_trajectories"] = [child]
    inventory = _inventory(
        tmp_path,
        [
            ("hf", tmp_path / "hf/parent.json", parent),
            ("hf", tmp_path / "hf/child.json", child),
        ],
    )
    original = audit._insert_call_audit
    key = corpus._semantic_hash(child)

    def fail_primary(db, trace_id, call_audit, family):
        original(db, trace_id, call_audit, family)
        scope = db.execute("SELECT scope FROM traces WHERE trace_id=?", (trace_id,)).fetchone()
        if trace_id == key and scope == ("primary",):
            raise ValueError("private promotion failure")

    monkeypatch.setattr(audit, "_insert_call_audit", fail_primary)
    output = tmp_path / "analysis"
    summary = audit.run(inventory, output)
    assert summary["failures"] == 1
    assert _query(
        output, "SELECT SUM(recorded_requests),SUM(rejected_requests) FROM model_usage"
    ) == [(5, 1)]
    assert _query(output, "SELECT COUNT(*) FROM rejected_tools") == [(1,)]
    scopes = {row[0]: row[1] for row in _query(output, "SELECT trace_id,scope FROM traces")}
    assert scopes[key] == "nested"


def test_native_visibility_lengths_are_stored_in_outputs_data(tmp_path: Path) -> None:
    doc = _trace()
    doc["agent"] = {"name": "Claude Code"}
    native = "p" * 2220
    recorded = native + "\n[stdout]\n" + "x" * 30_000
    result = {
        "source_call_id": "bash",
        "content": recorded,
        "extra": {
            "tool_result_metadata": {
                "raw_tool_result": {
                    "type": "tool_result",
                    "tool_use_id": "bash",
                    "content": native,
                }
            }
        },
    }
    doc["steps"] = [_request_step([_request("functions.Bash", "bash")], [result])]
    inventory = _inventory(tmp_path, [("hf", tmp_path / "hf/trace.json", doc)])
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    chars, raw = _query(output, "SELECT returned_chars,data FROM outputs")[0]
    data = as_object(json.loads(as_str(raw) or "{}"))
    assert chars == data["returned_chars"] == data["model_visible_chars"] == 2220
    assert data["recorded_chars"] == len(recorded)
    assert data["visibility_basis"] == "claude_native_tool_result"
    assert data["returned_basis"] == "model_visible"
    assert native not in (as_str(raw) or "")


def test_same_model_root_and_step_attribution_have_separate_denominators(tmp_path: Path) -> None:
    root = _request_step([_request("shell", "root")], [])
    explicit = _request_step(
        [_request("shell", "step")],
        [{"source_call_id": "step", "content": "Unknown tool: shell"}],
    )
    explicit["model_name"] = "provider/same-model"
    doc = _trace()
    doc["agent"] = {"name": "fast-agent", "model_name": "provider/same-model"}
    doc["steps"] = [root, explicit]
    inventory = _inventory(tmp_path, [("hf", tmp_path / "hf/trace.json", doc)])
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    assert _query(
        output,
        "SELECT model,model_source,recorded_requests,rejected_requests FROM model_usage "
        "ORDER BY model_source",
    ) == [
        ("provider/same-model", "root", 1, 0),
        ("provider/same-model", "step", 1, 1),
    ]
    assert _data(output)["shell_calls_missing_results"] == 1
    assert _data(output)["recorded_calls"] == 1
    assert _query(output, "SELECT COUNT(*) FROM outputs") == [(0,)]


def _redacted_inventory(tmp_path: Path, doc: Doc) -> Path:
    file = _write(tmp_path / "hf/job/trial/agent/trajectory.json", doc)
    raw = file.read_text(encoding="utf-8").replace('"synthetic-bare-redaction"', "[REDACTED]")
    file.write_text(raw, encoding="utf-8")
    return corpus.build_inventory({"hf": tmp_path / "hf"}, tmp_path / "inventory").database_path


@pytest.mark.parametrize("location", ["tool_calls", "token_only", "unmatched"])
def test_redacted_document_preserves_known_requests_but_cannot_certify_unused(
    tmp_path: Path, location: str
) -> None:
    doc = _trace()
    doc["agent"] = {
        "name": "fast-agent",
        "model_name": "provider/synthetic-model",
        "tool_definitions": [
            {"name": "shell", "parameters": {}},
            {"name": "process", "parameters": {}},
        ],
    }
    doc["steps"] = [
        _request_step([_request("shell", "known", {"command": "echo synthetic"})], []),
        {"source": "agent", "tool_calls": "synthetic-bare-redaction"}
        if location == "tool_calls"
        else {
            "source": "agent",
            "metrics": {"prompt_tokens": "synthetic-bare-redaction"},
        }
        if location == "token_only"
        else {"source": "agent", "message": "synthetic"},
    ]
    if location == "unmatched":
        doc["private_redaction_location_not_exported"] = "synthetic-bare-redaction"
    inventory = _redacted_inventory(tmp_path, doc)
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    data = _data(output)
    assert data["redacted_values"] == 1
    assert data["redacted_values_basis"] == "document"
    assert "redacted_input" in as_list(data["issues"])
    assert data["recorded_calls"] == 1
    assert _query(output, "SELECT tool,offered_unused FROM tools ORDER BY tool") == [
        ("process", None),
        ("shell", None),
    ]
    assert _query(output, "SELECT model,recorded_requests,rejected_requests FROM model_usage") == [
        ("provider/synthetic-model", 1, 0)
    ]
    assert (
        b"private_redaction_location_not_exported" not in (output / "analysis.sqlite3").read_bytes()
    )


def test_redaction_count_conservatively_flags_nested_family_without_localizing(
    tmp_path: Path,
) -> None:
    grandchild = _trace("synthetic-grandchild")
    grandchild["tool_definitions"] = _definitions()
    child = _trace("synthetic-child")
    child["tool_definitions"] = _definitions()
    child["subagent_trajectories"] = [grandchild]
    parent = _trace("synthetic-parent")
    parent["tool_definitions"] = _definitions()
    parent["steps"] = [{"source": "agent", "tool_calls": "synthetic-bare-redaction"}]
    parent["subagent_trajectories"] = [child]
    inventory = _redacted_inventory(tmp_path, parent)
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    rows = _query(output, "SELECT scope,data FROM traces")
    assert len(rows) == 3
    for scope, raw in rows:
        data = as_object(json.loads(as_str(raw) or "{}"))
        assert data["redacted_values"] == 1
        assert "redacted_input" in as_list(data["issues"])
        assert data["redacted_values_basis"] == (
            "parent_document" if scope == "nested" else "document"
        )
        assert ("redacted_values_parent_inherited" in as_list(data["issues"])) is (
            scope == "nested"
        )
    assert _query(output, "SELECT offered_unused FROM tools") == [(None,), (None,), (None,)]
    assert _query(output, "SELECT COUNT(*) FROM model_usage") == [(0,)]


def test_unredacted_document_has_zero_count_without_redaction_coverage_issue(
    tmp_path: Path,
) -> None:
    doc = _trace(message="quoted [REDACTED] is not a bare marker")
    doc["tool_definitions"] = _definitions()
    inventory = _inventory(tmp_path, [("hf", tmp_path / "hf/trace.json", doc)])
    output = tmp_path / "analysis"
    audit.run(inventory, output)
    data = _data(output)
    assert data["redacted_values"] == 0
    assert "redacted_input" not in as_list(data["issues"])
    assert _query(output, "SELECT offered_unused FROM tools") == [(1,)]
