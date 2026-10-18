"""Synthetic-only regressions for the separate private MODEL report."""

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
    from atif_scan.data.jsonval import Doc, JsonObject

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools" / "research"))
import harness_audit as audit
import harness_model_report as report
from harness_corpus import CorpusError

_PRIVATE = "synthetic-private-content /never/export/synthetic-path https://synthetic.invalid/"
_FAMILY = "synthetic-family-marker"
_MODEL = "provider/exact-model?reasoning=high"
_EVENT_FIELDS = {
    "trace_id",
    "step",
    "call",
    "tool",
    "model",
    "model_source",
    "observation",
    "reason",
    "declaration_conflict",
    "pairing",
}


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "analysis.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript(audit._SCHEMA)
        db.execute("UPDATE state SET complete=1")
    return path


def _trace(db: sqlite3.Connection, key: str, dropped: int = 0) -> None:
    db.execute(
        "INSERT INTO traces VALUES(?,?,?,?,?,?,?,?)",
        (
            key,
            _FAMILY,
            "synthetic-variant",
            "synthetic-version",
            "primary",
            None,
            "run",
            json.dumps({"dropped_steps": dropped, "raw_text": _PRIVATE, "path": _PRIVATE}),
        ),
    )


def _usage(
    db: sqlite3.Connection,
    key: str,
    model: str | None,
    source: str,
    recorded: int,
    rejected: int = 0,
    unknown: int = 0,
) -> None:
    db.execute(
        "INSERT INTO model_usage VALUES(?,?,?,?,?,?)",
        (key, model, source, recorded, rejected, unknown),
    )


def _event(
    db: sqlite3.Connection,
    key: str,
    index: int,
    model: str | None = _MODEL,
    source: str = "step",
    tool: str | None = "missing_tool",
    reason: str = "not_found",
    conflict: bool | None = None,
) -> None:
    db.execute(
        "INSERT INTO rejected_tools VALUES(?,?,?,?,?,?,?,?,?,?)",
        (key, index, 0, tool, model, source, 0, reason, conflict, "recorded_id"),
    )


def _csv(output: Path, name: str) -> list[dict[str, str]]:
    with (output / f"{name}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_model_aggregation_sources_unique_traces_and_nonmultiplicative_join(
    database: Path, tmp_path: Path
) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "t1")
        _trace(db, "t2")
        db.execute(
            "UPDATE traces SET family='another-synthetic-interface',scope='nested' "
            "WHERE trace_id='t2'"
        )
        db.executemany(
            "INSERT INTO copies VALUES(?,?,?,?,?)",
            [("t1", index, "hf", "run", "a" * 64) for index in range(4)],
        )
        _usage(db, "t1", _MODEL, "step", 5, 3, 1)
        _usage(db, "t1", _MODEL, "root", 7, 1)
        _usage(db, "t2", _MODEL, "root", 11, 1, 2)
        for index in range(3):
            _event(db, "t1", index)
        _event(db, "t1", 3, source="root", tool="other_tool", reason="unavailable")
        _event(db, "t2", 0, source="root")
    output = tmp_path / "report"
    summary = report.export(database, output)
    model = summary["models"][0]
    assert model["model"] == _MODEL
    assert (model["traces"], model["affected_traces"]) == (2, 2)
    assert (model["recorded_requests"], model["rejected_requests"]) == (23, 5)
    assert model["rejection_rate_denominator"] == 23
    assert model["rejection_rate"] == pytest.approx(5 / 23)
    assert model["unknown_tool_label_requests"] == 3
    assert (model["step_requests"], model["root_requests"], model["unknown_source_requests"]) == (
        5,
        18,
        0,
    )
    assert (model["step_traces"], model["root_traces"]) == (1, 2)
    tools = {r["tool"]: r for r in _csv(output, "model_tools")}
    missing = tools["missing_tool"]
    assert missing["rejected_requests"] == "4"
    assert missing["affected_traces"] == missing["model_traces"] == "2"
    assert missing["model_recorded_requests"] == missing["rejection_rate_denominator"] == "23"
    assert missing["model_rejected_requests"] == "5"
    assert float(missing["rejection_rate"]) == pytest.approx(4 / 23)
    assert tools["other_tool"]["rejected_requests"] == "1"
    attribution = {r["model_source"]: r for r in _csv(output, "attribution")}
    assert attribution["root"]["recorded_requests"] == "18"
    assert attribution["root"]["affected_traces"] == "2"
    assert float(attribution["step"]["rejection_rate"]) == pytest.approx(3 / 5)


def test_two_models_aliases_and_unknown_model_remain_separate(
    database: Path, tmp_path: Path
) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "t")
        for model, source, total in (
            (_MODEL, "step", 3),
            ("alias", "root", 4),
            (None, "unknown", 2),
            ("unknown", "root", 1),
        ):
            _usage(db, "t", model, source, total, 1)
            _event(
                db,
                "t",
                total,
                model=model,
                source=source,
                tool=None if model is None else "missing_tool",
            )
    output = tmp_path / "report"
    summary = report.export(database, output)
    models = {r["model"]: r for r in summary["models"]}
    assert set(models) == {_MODEL, "alias", None, "unknown"}
    assert models[None]["unknown_source_requests"] == 2
    assert models["unknown"]["root_requests"] == 1
    assert models[_MODEL]["step_requests"] == 3
    assert models["alias"]["recorded_requests"] == 4
    unknown = next(r for r in _csv(output, "model_tools") if r["model"] == "")
    assert unknown["tool"] == ""
    assert unknown["rejection_rate_denominator"] == "2"
    assert float(unknown["rejection_rate"]) == 0.5
    assert summary["corpus"]["unknown_model_requests"] == 2


def test_zero_rejections_positive_denominator_and_zero_requests_null(
    database: Path, tmp_path: Path
) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "recorded")
        _trace(db, "no-requests")
        _trace(db, "not-enumerable")
        _usage(db, "recorded", "observed", "root", 8)
        _usage(db, "no-requests", "zero-denominator", "step", 0)
    output = tmp_path / "report"
    summary = report.export(database, output)
    models = {r["model"]: r for r in summary["models"]}
    assert models["observed"]["rejection_rate"] == 0.0
    assert models["observed"]["rejection_rate_denominator"] == 8
    assert models["observed"]["affected_traces"] == 0
    assert models["zero-denominator"]["rejection_rate"] is None
    assert models["zero-denominator"]["rejection_rate_denominator"] == 0
    assert len(models) == 2
    assert summary["corpus"]["traces"] == 3
    assert _csv(output, "model_tools") == _csv(output, "rejected_requests") == []
    for name in ("model_usage", "attribution"):
        values = {r["model"]: r for r in _csv(output, name)}
        assert values["observed"]["rejection_rate"] == "0.0"
        assert values["zero-denominator"]["rejection_rate"] == ""


def test_empty_corpus_does_not_invent_models(database: Path, tmp_path: Path) -> None:
    summary = report.export(database, tmp_path / "report")
    assert summary["models"] == summary["attribution"] == []
    assert summary["corpus"]["recorded_requests"] == 0
    assert summary["tables"] == dict.fromkeys(report.TABLES, 0)


def test_exports_allowlisted_no_family_private_and_readonly(database: Path, tmp_path: Path) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "t", dropped=2)
        _usage(db, "t", _MODEL, "step", 1, 1)
        _event(db, "t", 0, conflict=True)
        db.execute("ALTER TABLE model_usage ADD COLUMN raw_args TEXT")
        db.execute("ALTER TABLE rejected_tools ADD COLUMN raw_args TEXT")
        db.execute("UPDATE model_usage SET raw_args=?", (_PRIVATE,))
        db.execute("UPDATE rejected_tools SET raw_args=?", (_PRIVATE,))
        db.execute("INSERT INTO failures VALUES(?,?)", ("failed", "invalid_json"))
    before = database.read_bytes()
    output = tmp_path / "report"
    summary = report.export(database, output)
    assert database.read_bytes() == before
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert {p.name for p in output.iterdir()} == {
        *(f"{name}.csv" for name in report.TABLES),
        "summary.json",
        "REPORT.md",
    }
    for path in output.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        text = path.read_text()
        for withheld in (
            _PRIVATE,
            _FAMILY,
            "family",
            "harness",
            "variant",
            "synthetic-version",
            "raw_args",
            str(database),
            str(output),
        ):
            assert withheld not in text
    event = _csv(output, "rejected_requests")[0]
    assert set(event) == _EVENT_FIELDS
    assert event["declaration_conflict"] == "1"
    assert (event["step"], event["call"], event["observation"], event["pairing"]) == (
        "0",
        "0",
        "0",
        "recorded_id",
    )
    assert summary["corpus"]["dropped_steps"] == 2
    assert summary["corpus"]["failures"] == 1
    assert as_object(json.loads((output / "summary.json").read_text())) == summary
    methods = (output / "REPORT.md").read_text()
    assert "NOT a causal model ranking" in methods
    assert "Task-mix differences" in methods
    assert "NOT clean absence" in methods


def _step(name: str, text: str | None = None) -> JsonObject:
    return {
        "source": "agent",
        "tool_calls": [
            {"function_name": name, "tool_call_id": "a", "arguments": {"input": _PRIVATE}}
        ],
        "observation": {
            "results": [] if text is None else [{"source_call_id": "a", "content": text}]
        },
    }


def test_audited_dispatch_only_step_authority_missing_results_and_derived_exclusion(
    database: Path, tmp_path: Path
) -> None:
    explicit = _step("missing_tool", "Unknown tool: missing_tool")
    explicit["model_name"] = "provider/step-model"
    unknown = _step(
        "/unsafe/tool",
        "<tool_use_error>Error: No such tool available: /unsafe/tool</tool_use_error>",
    )
    unknown["model_name"] = None
    code = _step("exec", "synthetic capture")
    code["tool_calls"] = [
        {
            "function_name": "exec",
            "tool_call_id": "a",
            "arguments": {"input": 'await tools.exec_command({cmd: "echo synthetic"});'},
        }
    ]
    doc: JsonObject = {
        "agent": {"name": "fast-agent", "model_name": "provider/root-model"},
        "steps": [
            explicit,
            unknown,
            code,
            _step("shell"),
            _step("Read", "Error executing tool Read: Invalid arguments: path not found"),
            _step("shell", "bash: synthetic: command not found"),
        ],
    }
    with sqlite3.connect(database) as db:
        audit._analyze_document(db, doc, "synthetic-trace", None, None)
    output = tmp_path / "report"
    summary = report.export(database, output)
    models = {r["model"]: r for r in summary["models"]}
    assert models["provider/root-model"]["recorded_requests"] == 4
    assert models["provider/root-model"]["rejected_requests"] == 0
    assert models["provider/step-model"]["recorded_requests"] == 1
    assert models["provider/step-model"]["rejected_requests"] == 1
    assert models[None]["recorded_requests"] == models[None]["rejected_requests"] == 1
    assert models[None]["unknown_tool_label_requests"] == 1
    assert summary["corpus"]["recorded_requests"] == 6  # Derived exec_command is not a request.
    events = _csv(output, "rejected_requests")
    assert [(r["step"], r["model_source"], r["tool"]) for r in events] == [
        ("0", "step", "missing_tool"),
        ("1", "unknown", ""),
    ]


@pytest.mark.parametrize("states", [[], [0], [1, 0], [1, 1]])
def test_refuses_incomplete_analysis(database: Path, tmp_path: Path, states: list[int]) -> None:
    with sqlite3.connect(database) as db:
        db.execute("DELETE FROM state")
        db.executemany("INSERT INTO state VALUES(?)", [(value,) for value in states])
    output = tmp_path / "report"
    with pytest.raises(CorpusError, match=r"^analysis_incomplete$"):
        report.export(database, output)
    assert not output.exists()


@pytest.mark.parametrize("table", ["model_usage", "rejected_tools"])
def test_old_schema_fails_before_output(database: Path, tmp_path: Path, table: str) -> None:
    with sqlite3.connect(database) as db:
        db.execute(f"DROP TABLE {table}")
    output = tmp_path / "report"
    with pytest.raises(CorpusError, match=r"^invalid_analysis$"):
        report.export(database, output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["file_symlink", "parent_symlink", "directory", "fifo", "missing"])
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
    with pytest.raises(CorpusError, match=r"^invalid_analysis$"):
        report.export(unsafe, tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("kind", ["existing", "file_symlink", "parent_symlink"])
def test_refuses_unsafe_outputs(database: Path, tmp_path: Path, kind: str) -> None:
    output = tmp_path / "report"
    if kind == "existing":
        output.mkdir()
    elif kind == "file_symlink":
        output.symlink_to(database)
    else:
        output.symlink_to(tmp_path, target_is_directory=True)
        output /= "nested"
    with pytest.raises(CorpusError, match=r"^private_output_unavailable$"):
        report.export(database, output)
    assert not (tmp_path / "nested").exists()


def test_refuses_repository_destinations(database: Path, tmp_path: Path) -> None:
    repo = Path(report.__file__).resolve().parents[2]
    with pytest.raises(CorpusError, match=r"^output_inside_repository$"):
        report.export(database, repo / "synthetic-model-report-must-not-exist")
    with pytest.raises(CorpusError, match=r"^input_inside_repository$"):
        report.export(repo / "synthetic-model-db-must-not-exist", tmp_path / "report")


def test_pinned_output_and_exclusive_files(
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


def test_planted_output_symlink_cannot_overwrite_database(
    database: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report"
    original = report._export
    before = database.read_bytes()

    def plant(db: sqlite3.Connection, folder: int) -> Doc:
        (output / "model_usage.csv").symlink_to(database)
        return original(db, folder)

    monkeypatch.setattr(report, "_export", plant)
    with pytest.raises(CorpusError, match=r"^private_output_unavailable$"):
        report.export(database, output)
    assert database.read_bytes() == before


def test_cli(
    database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "report"
    monkeypatch.setattr(
        sys, "argv", ["harness_model_report", "--database", str(database), "--output", str(output)]
    )
    report.main()
    assert as_object(json.loads(capsys.readouterr().out)) == dict.fromkeys(report.TABLES, 0)
    assert (output / "REPORT.md").is_file()


def test_same_tool_different_reasons_share_model_denominator(
    database: Path, tmp_path: Path
) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "t")
        _usage(db, "t", _MODEL, "step", 10, 2)
        _event(db, "t", 0, reason="not_found", conflict=False)
        _event(db, "t", 1, reason="unavailable")
    output = tmp_path / "report"
    report.export(database, output)
    tools = _csv(output, "model_tools")
    assert {(r["tool"], r["reason"]) for r in tools} == {
        ("missing_tool", "not_found"),
        ("missing_tool", "unavailable"),
    }
    for row in tools:
        assert row["rejected_requests"] == row["affected_traces"] == "1"
        assert row["rejection_rate_denominator"] == "10"
        assert float(row["rejection_rate"]) == 0.1
    events = _csv(output, "rejected_requests")
    assert [r["declaration_conflict"] for r in events] == ["0", ""]


def test_missing_dropped_step_coverage_is_not_zero(database: Path, tmp_path: Path) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "t")
        db.execute("UPDATE traces SET data=?", (json.dumps({"raw_text": _PRIVATE}),))
        _usage(db, "t", None, "unknown", 1)
    summary = report.export(database, tmp_path / "report")
    assert summary["corpus"]["unknown_dropped_steps_traces"] == 1
    assert summary["corpus"]["dropped_steps"] is None
    assert summary["corpus"]["unknown_model_requests"] == 1


def test_pinned_input_survives_path_replacement(
    database: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with sqlite3.connect(database) as db:
        _trace(db, "t")
        _usage(db, "t", _MODEL, "root", 7)
    replacement = tmp_path / "replacement.sqlite3"
    with sqlite3.connect(replacement) as db:
        db.executescript(audit._SCHEMA)
        db.execute("UPDATE state SET complete=1")
    retained = tmp_path / "retained.sqlite3"
    original = sqlite3.connect

    def replace(location: str, *, uri: bool) -> sqlite3.Connection:
        assert location.startswith("file:/proc/self/fd/") and location.endswith("?mode=ro")
        assert uri is True
        database.rename(retained)
        replacement.rename(database)
        return original(location, uri=uri)

    monkeypatch.setattr(sqlite3, "connect", replace)
    summary = report.export(database, tmp_path / "report")
    assert summary["models"][0]["recorded_requests"] == 7


def test_export_connection_is_readonly(
    database: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = report._export

    def check(db: sqlite3.Connection, folder: int) -> Doc:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("UPDATE state SET complete=0")
        return original(db, folder)

    monkeypatch.setattr(report, "_export", check)
    report.export(database, tmp_path / "report")
