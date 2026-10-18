"""Synthetic-only regressions for the private, offline harness report exporter."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import stat
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from atif_scan.data.jsonval import as_object

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools" / "research"))
import harness_audit as audit
import harness_report as report
import harness_trace as trace
from harness_metrics import measure

_PRIVATE = "synthetic-private-snippet /never/export/synthetic-path https://synthetic.invalid/secret"


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "analysis.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript(audit._SCHEMA)
        db.execute("UPDATE state SET complete=1")
    return path


def _tool(
    name: str,
    recorded: int = 0,
    static: int = 0,
    declared: bool | None = None,
    unused: bool | None = None,
) -> Doc:
    return {
        "tool": name,
        "recorded_calls": recorded,
        "static_call_sites": static,
        "declared": declared,
        "offered_unused": unused,
        "inventory_source": "atif" if declared else "observed",
    }


def _trace(
    db: sqlite3.Connection,
    key: str,
    tools: list[Doc],
    signature: str | None = None,
    version: str = "1.0",
    run: str | None = "run-a",
    scope: str = "primary",
    family: str = "fast-agent",
) -> None:
    data: Doc = {
        "inventory_source": "atif" if signature else "unknown",
        "inventory_status": "known" if signature else "missing",
        "declared_tools": sum(t["declared"] is True for t in tools) if signature else None,
        "recorded_calls": sum(t["recorded_calls"] for t in tools),
        "static_call_sites": sum(t["static_call_sites"] for t in tools),
        "agent_steps": 3,
        "dropped_steps": 0,
        "shell_calls_missing_results": 0,
        "issues": [],
        "raw_text": _PRIVATE,
        "path": _PRIVATE,
    }
    db.execute(
        "INSERT INTO traces VALUES(?,?,?,?,?,?,?,?)",
        (
            key,
            family,
            signature,
            version,
            scope,
            "parent" if scope == "nested" else None,
            run,
            json.dumps(data),
        ),
    )
    db.executemany(
        "INSERT INTO tools VALUES(?,?,?,?,?,?,?)",
        [(key, *(t[k] for k in audit._TOOL_FIELDS)) for t in tools],
    )


def _output(
    db: sqlite3.Connection,
    key: str,
    step: int,
    retention: str = "file",
    followed: bool | None = None,
    length: int | None = None,
    truncated: bool = True,
    context: str = "shell",
) -> None:
    row: Doc = {
        "step": step,
        "observation": 0,
        "tool": "shell",
        "context": context,
        "truncation": truncated,
        "kind": "recognized" if truncated else "none",
        "marker": "synthetic_marker" if truncated else None,
        "retention": retention,
        "followup_observed": followed,
        "returned_chars": length,
        "returned_bytes": length,
        "raw_text": _PRIVATE,
        "spool_paths": [_PRIVATE],
        "arguments": {"command": _PRIVATE},
        "extra": {"snippet": _PRIVATE},
    }
    db.execute(
        "INSERT INTO outputs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (key, *(row[k] for k in audit._OUTPUT_FIELDS), json.dumps(row)),
    )


def _csv(output: Path, name: str) -> list[dict[str, str]]:
    with (output / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_counts_do_not_multiply_tool_output_joins(database: Path, tmp_path: Path) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "t1", [_tool("shell", 2), _tool("reader", 1, 4)])
        _trace(db, "t2", [_tool("shell", 3)], version="2.0", scope="nested")
        for step in range(3):
            _output(db, "t1", step, length=10, followed=True)
        _output(db, "t2", 0, length=20, followed=False)
    output = tmp_path / "report"
    summary = report.export(database, output)
    family = summary["harnesses"][0]
    assert (family["traces"], family["primary_traces"], family["nested_traces"]) == (2, 1, 1)
    assert (family["cumulative_tools"], family["recorded_calls"], family["static_call_sites"]) == (
        2,
        6,
        4,
    )
    assert len(summary["harnesses"]) == 1  # Versions do not split a harness.
    shell = summary["shell"][0]
    assert shell["output_records"] == shell["recognized_truncations"] == 4
    assert shell["mean_returned_chars"] == 12.5
    tools = {row["tool"]: row for row in _csv(output, "tools")}
    assert tools["shell"]["recorded_calls"] == "5"
    assert float(tools["shell"]["recorded_call_share"]) == pytest.approx(5 / 6)
    assert tools["reader"]["static_call_sites"] == "4"


def test_unknown_and_zero_declarations_full_run_grid(database: Path, tmp_path: Path) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "unknown", [_tool("shell", 1)], run="run-unknown")
        _trace(
            db,
            "known",
            [_tool("shell", 0, declared=True, unused=True), _tool("reader", 1, declared=False)],
            signature="synthetic-known",
            run="run-known",
        )
        _trace(db, "empty", [], signature="synthetic-empty", run=None)
    output = tmp_path / "report"
    report.export(database, output)
    traces = {row["trace_id"]: row for row in _csv(output, "traces")}
    assert traces["unknown"]["declared_tools"] == ""
    assert traces["empty"]["declared_tools"] == "0"
    rows = {(r["run_id"], r["tool"]): r for r in _csv(output, "run_tools")}
    assert set(rows) == {
        (run, tool)
        for run in ("run-unknown", "run-known", "unassigned")
        for tool in ("shell", "reader")
    }
    assert rows["run-unknown", "shell"]["declared_offered_traces"] == ""
    assert rows["run-unknown", "shell"]["confirmed_offered_unused_traces"] == ""
    assert rows["run-known", "reader"]["declared_offered_traces"] == "0"
    assert rows["run-known", "shell"]["confirmed_offered_unused_traces"] == "1"
    absent = rows["run-unknown", "reader"]
    assert absent["recorded_calls"] == absent["static_call_sites"] == "0"
    assert absent["declared_offered_traces"] == absent["confirmed_offered_unused_traces"] == ""
    assert rows["unassigned", "shell"]["declared_offered_traces"] == ""


def test_variants_use_schema_hash_not_version(database: Path, tmp_path: Path) -> None:
    signatures: list[str] = []
    with sqlite3.connect(database) as db:
        for index, (version, kind) in enumerate(
            (("1.0", "string"), ("2.0", "string"), ("2.0", "integer"))
        ):
            evidence = trace.analyze_trace(
                {
                    "agent": {
                        "name": "fast-agent",
                        "version": version,
                        "tool_definitions": [
                            {
                                "name": "shell",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"command": {"type": kind}},
                                },
                            }
                        ],
                    },
                    "steps": [],
                }
            )
            signature = evidence.declarations.signature
            assert signature is not None and len(signature) == 64
            signatures.append(signature)
            audit._insert_metrics(db, f"t{index}", measure(evidence), "primary", None, "run")
        _trace(db, "unknown", [_tool("shell", 1)])
    assert signatures[0] == signatures[1] != signatures[2]
    output = tmp_path / "report"
    report.export(database, output)
    variants = {row["signature"]: row for row in _csv(output, "variants")}
    assert set(variants) == {signatures[0], signatures[2], "unknown"}
    assert variants[signatures[0]]["traces"] == "2"
    assert set(variants[signatures[0]]["versions"].split(",")) == {"1.0", "2.0"}
    assert variants[signatures[2]]["traces"] == "1"
    assert len(_csv(output, "harnesses")) == 1


def test_followup_rates_exclude_unknown_eligibility(database: Path, tmp_path: Path) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "t", [_tool("shell", 8)])
        for step, (retention, followed) in enumerate(
            [
                ("file", True),
                ("file", False),
                ("file", None),
                ("process_handle", True),
                ("process_handle", None),
                ("not_indicated", None),
                ("unresolved_retention", None),
            ]
        ):
            _output(db, "t", step, retention, followed)
        _output(db, "t", 8, "file", None, context="shared_terminal")
    summary = report.export(database, tmp_path / "report")
    shell = {row["context"]: row for row in summary["shell"]}
    row = shell["shell"]
    assert (
        row["file_spool_events"],
        row["file_followup_eligible"],
        row["file_followed_events"],
    ) == (3, 2, 1)
    assert row["file_followup_rate"] == 0.5
    assert (
        row["handle_events"],
        row["handle_followup_eligible"],
        row["handle_followed_events"],
    ) == (2, 1, 1)
    assert row["handle_followup_rate"] == 1.0
    assert row["truncated_without_retention_notice"] == row["unresolved_retention_events"] == 1
    assert shell["shared_terminal"]["file_followup_rate"] is None
    assert shell["shared_terminal"]["handle_followup_rate"] is None


@pytest.mark.parametrize("length", [None, 0])
def test_empty_quantiles_are_null_not_zero(
    database: Path, tmp_path: Path, length: int | None
) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "t", [_tool("shell", 2)])
        _output(db, "t", 0, length=length)
        _output(db, "t", 1, length=0, truncated=False, context="shared_terminal")
    output = tmp_path / "report"
    summary = report.export(database, output)
    shell = {row["context"]: row for row in summary["shell"]}
    for field in (
        "p50_returned_chars",
        "p95_returned_chars",
        "truncated_p50_chars",
        "truncated_p95_chars",
        "truncated_p99_chars",
    ):
        assert shell["shell"][field] == length
    assert shell["shared_terminal"]["p50_returned_chars"] == 0
    assert shell["shared_terminal"]["truncated_p50_chars"] is None
    boundary = _csv(output, "boundaries")[0]
    assert (
        boundary["p50_returned_chars"]
        == boundary["p95_returned_chars"]
        == ("" if length is None else "0")
    )


def test_quantiles_nearest_rank_and_unknown_lengths() -> None:
    with sqlite3.connect(":memory:") as db:
        for name, aggregate in (("p50", report.Quantile), ("p95", report.P95), ("p99", report.P99)):
            db.create_aggregate(name, 1, aggregate)
        db.execute("CREATE TABLE lengths(value)")
        db.executemany("INSERT INTO lengths VALUES(?)", [(None,), (0,), (10,), (10,), (100,)])
        assert db.execute("SELECT p50(value),p95(value),p99(value) FROM lengths").fetchone() == (
            10,
            100,
            100,
        )
        db.execute("DELETE FROM lengths")
        assert db.execute("SELECT NULLIF(p50(value),-1) FROM lengths").fetchone() == (None,)


def test_exports_are_allowlisted_private_and_input_unchanged(
    database: Path, tmp_path: Path
) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "t", [_tool("shell", 1)])
        _output(db, "t", 0, length=10)
        db.execute("ALTER TABLE copies ADD COLUMN path TEXT")
        db.execute(
            "INSERT INTO copies VALUES(?,?,?,?,?,?)", ("t", 1, "hf", "run", "a" * 64, _PRIVATE)
        )
        db.execute("INSERT INTO native_sources VALUES(?,?,?)", ("t", "b" * 64, "matched_session"))
    before = database.read_bytes()
    output = tmp_path / "report"
    summary = report.export(database, output)
    assert database.read_bytes() == before
    assert {p.name for p in output.iterdir()} == {
        *(f"{name}.csv" for name in report.TABLES),
        "summary.json",
        "truncations.jsonl",
        "REPORT.md",
    }
    assert len(report.TABLES) == 14
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for file in output.iterdir():
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
        text = file.read_text()
        for secret in (
            _PRIVATE,
            "synthetic-private-snippet",
            "/never/export",
            str(database),
            str(output),
        ):
            assert secret not in text
    assert as_object(json.loads((output / "summary.json").read_text())) == summary
    event = as_object(json.loads((output / "truncations.jsonl").read_text()))
    assert set(event) == {"trace_id", "family", "signature", *report.EVENT_FIELDS}
    assert event["payload_chars"] is None
    methods = (output / "REPORT.md").read_text()
    assert "NOT token counts" in methods
    assert "no advertised spool" in methods
    assert "unknown" in methods


@pytest.mark.parametrize("states", [[], [0], [1, 0], [1, 1]])
def test_refuses_incomplete_or_ambiguous_analysis(
    database: Path, tmp_path: Path, states: list[int]
) -> None:
    with sqlite3.connect(database) as db:
        db.execute("DELETE FROM state")
        db.executemany("INSERT INTO state VALUES(?)", [(value,) for value in states])
    output = tmp_path / "report"
    with pytest.raises(report.CorpusError, match=r"^analysis_incomplete$"):
        report.export(database, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "kind", ["file_symlink", "parent_symlink", "directory", "fifo", "invalid", "missing"]
)
def test_refuses_unsafe_inputs(database: Path, tmp_path: Path, kind: str) -> None:
    unsafe = tmp_path / "unsafe"
    if kind == "file_symlink":
        unsafe.symlink_to(database)
    elif kind == "parent_symlink":
        unsafe.symlink_to(tmp_path, target_is_directory=True)
        unsafe /= database.name
    elif kind == "directory":
        unsafe.mkdir()
    elif kind == "fifo":
        os.mkfifo(unsafe)
    elif kind == "invalid":
        unsafe.write_text("synthetic non-database")
    with pytest.raises(report.CorpusError, match=r"^invalid_analysis$"):
        report.export(unsafe, tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("kind", ["existing", "file_symlink", "parent_symlink", "traversal"])
def test_refuses_unsafe_outputs(database: Path, tmp_path: Path, kind: str) -> None:
    output = tmp_path / "report"
    if kind == "existing":
        output.mkdir()
    elif kind == "file_symlink":
        output.symlink_to(database)
    elif kind == "parent_symlink":
        link = tmp_path / "linked"
        link.symlink_to(tmp_path, target_is_directory=True)
        output = link / "report"
    else:
        output = tmp_path / ".." / "report"
    before = database.read_bytes()
    with pytest.raises(report.CorpusError, match=r"^(private_output_unavailable|unsafe_path)$"):
        report.export(database, output)
    assert database.read_bytes() == before


def test_refuses_repository_paths_without_creating_files(database: Path, tmp_path: Path) -> None:
    repo = Path(report.__file__).resolve().parents[2]
    with pytest.raises(report.CorpusError, match=r"^output_inside_repository$"):
        report.export(database, repo / "synthetic-report-must-not-exist")
    with pytest.raises(report.CorpusError, match=r"^input_inside_repository$"):
        report.export(repo / "synthetic-database-must-not-exist", tmp_path / "report")
    assert not (repo / "synthetic-report-must-not-exist").exists()


def test_pinned_output_survives_directory_replacement(
    database: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report"
    retained = tmp_path / "retained"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    original = report._export

    def replace(db: sqlite3.Connection, folder: int) -> Doc:
        output.rename(retained)
        output.symlink_to(elsewhere, target_is_directory=True)
        return original(db, folder)

    monkeypatch.setattr(report, "_export", replace)
    report.export(database, output)
    assert (retained / "summary.json").is_file()
    assert list(elsewhere.iterdir()) == []


def test_exclusive_files_refuse_planted_symlink(
    database: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report"
    original = report._export
    before = database.read_bytes()

    def replace(db: sqlite3.Connection, folder: int) -> Doc:
        (output / "harnesses.csv").symlink_to(database)
        return original(db, folder)

    monkeypatch.setattr(report, "_export", replace)
    with pytest.raises(report.CorpusError, match=r"^private_output_unavailable$"):
        report.export(database, output)
    assert database.read_bytes() == before


def test_pinned_input_survives_filename_replacement(
    database: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "retained-trace", [_tool("shell", 2)])
    replacement = tmp_path / "replacement.sqlite3"
    with sqlite3.connect(replacement) as db:
        db.executescript(audit._SCHEMA)
        db.execute("UPDATE state SET complete=1")
    retained = tmp_path / "retained.sqlite3"
    original = sqlite3.connect

    def replace(location: str, *, uri: bool) -> sqlite3.Connection:
        # The exporter has already pinned the regular file before SQLite opens it.
        assert location.startswith("file:/proc/self/fd/") and location.endswith("?mode=ro")
        assert uri is True
        database.rename(retained)
        replacement.rename(database)
        return original(location, uri=uri)

    monkeypatch.setattr(report.sqlite3, "connect", replace)
    summary = report.export(database, tmp_path / "report")
    assert summary["harnesses"][0]["recorded_calls"] == 2


def test_followup_zero_is_distinct_from_ineligible(database: Path, tmp_path: Path) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "t", [_tool("shell", 4)])
        for step, retention in enumerate(("file", "process_handle")):
            _output(db, "t", step, retention, False, context="shell")
            _output(db, "t", step, retention, None, context="shared_terminal")
    summary = report.export(database, tmp_path / "report")
    shell = {row["context"]: row for row in summary["shell"]}
    for prefix in ("file", "handle"):
        assert shell["shell"][f"{prefix}_followup_rate"] == 0
        assert shell["shared_terminal"][f"{prefix}_followup_rate"] is None


def test_events_include_possible_and_terminal_but_not_clean_outputs(
    database: Path, tmp_path: Path
) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "t", [_tool("shell", 4)])
        for step in range(4):
            _output(db, "t", step, truncated=step == 0)
        for step, kind in ((1, "possible_signal"), (2, "terminal_screen")):
            db.execute(
                "UPDATE outputs SET data=json_set(data,'$.kind',?) WHERE step=?", (kind, step)
            )
    output = tmp_path / "report"
    summary = report.export(database, output)
    events = [
        as_object(json.loads(line))
        for line in (output / "truncations.jsonl").read_text().splitlines()
    ]
    assert {event["step"] for event in events} == {0, 1, 2}
    assert summary["tables"]["truncations"] == 3
    assert summary["shell"][0]["recognized_truncations"] == 1
    assert summary["shell"][0]["possible_signals"] == 1
    assert summary["shell"][0]["terminal_screen_records"] == 1


def test_source_connection_is_read_only(database: Path) -> None:
    with report._source(database) as db, pytest.raises(sqlite3.OperationalError, match="readonly"):
        db.execute("UPDATE state SET complete=0")


def test_visibility_keeps_exporter_text_separate_from_native_block(
    database: Path, tmp_path: Path
) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "native", [_tool("Bash", 1)], family="claude-code")
        _output(db, "native", 0, length=2200)
        db.execute(
            """UPDATE outputs SET data=json_set(data,
               '$.recorded_chars',32000,'$.recorded_bytes',33000,
               '$.model_visible_chars',2200,'$.model_visible_bytes',2300,
               '$.visibility_basis','claude_native_tool_result',
               '$.returned_basis','model_visible')"""
        )
        _trace(db, "proxy", [_tool("shell", 1)])
        _output(db, "proxy", 0, length=16000)
        db.execute(
            """UPDATE outputs SET data=json_set(data,
               '$.recorded_chars',16000,'$.recorded_bytes',16000,
               '$.visibility_basis','unknown','$.returned_basis','recorded')
               WHERE trace_id='proxy'"""
        )
    output = tmp_path / "report"
    report.export(database, output)
    visibility = {r["family"]: r for r in _csv(output, "visibility")}
    native = visibility["claude-code"]
    assert native["model_visible_records"] == "1"
    assert float(native["mean_model_visible_chars"]) == 2200
    assert float(native["mean_recorded_chars"]) == 32000
    assert visibility["fast-agent"]["model_visible_records"] == "0"
    assert visibility["fast-agent"]["mean_model_visible_chars"] == ""
    events = [
        as_object(json.loads(line))
        for line in (output / "truncations.jsonl").read_text().splitlines()
    ]
    event = next(e for e in events if e["trace_id"] == "native")
    assert event["recorded_chars"] == 32000
    assert event["model_visible_chars"] == 2200
    assert event["returned_basis"] == "model_visible"


def test_rejected_only_tool_is_not_exported_in_harness_inventory(
    database: Path, tmp_path: Path
) -> None:
    doc: dict[str, object] = {
        "agent": {"name": "claude-code", "model_name": "synthetic-model"},
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {"tool_call_id": "a", "function_name": "ghost_tool", "arguments": {}}
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "a",
                            "content": (
                                "<tool_use_error>Error: No such tool available: "
                                "ghost_tool</tool_use_error>"
                            ),
                        }
                    ]
                },
            }
        ],
    }
    with sqlite3.connect(database) as db:
        audit._analyze_document(db, doc, "synthetic-id", None, "synthetic-run")
        assert db.execute("SELECT COUNT(*) FROM rejected_tools").fetchone() == (1,)
    output = tmp_path / "report"
    report.export(database, output)
    assert _csv(output, "tools") == []
    assert _csv(output, "trace_tools") == []
    assert all("ghost_tool" not in p.read_text() for p in output.iterdir())


def test_unsupported_truncation_has_unknown_rate_and_separate_denominator(
    database: Path, tmp_path: Path
) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "u", [_tool("shell", 2)], family="synthetic-unsupported")
        _output(db, "u", 0, length=10, truncated=False)
        db.execute(
            """UPDATE outputs SET truncation=NULL,
               data=json_set(data,'$.kind','unsupported','$.truncation',NULL)"""
        )
        _trace(db, "mixed", [_tool("shell", 2)])
        _output(db, "mixed", 0, length=10)
        _output(db, "mixed", 1, length=10, truncated=False)
        db.execute(
            """UPDATE outputs SET truncation=NULL,
               data=json_set(data,'$.kind','possible_signal','$.truncation',NULL)
               WHERE trace_id='mixed' AND step=1"""
        )
    summary = report.export(database, tmp_path / "report")
    values = {r["family"]: r for r in summary["shell"]}
    unknown = values["synthetic-unsupported"]
    assert unknown["assessed_output_records"] == 0
    assert unknown["recognized_truncations"] is None
    assert unknown["recognized_truncation_rate"] is None
    mixed = values["fast-agent"]
    assert mixed["output_records"] == 2
    assert mixed["assessed_output_records"] == 1
    assert mixed["recognized_truncation_rate"] == 1
