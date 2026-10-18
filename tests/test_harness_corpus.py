"""Synthetic-only regression tests for private, offline corpus discovery."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools" / "research"))
import harness_corpus as corpus


def _trace(*, message: str = "synthetic task", calls: list[object] | None = None) -> Doc:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "synthetic-session",
        "steps": [
            {"source": "user", "message": "Synthetic instructions only."},
            {"source": "agent", "message": message, "tool_calls": calls or []},
        ],
    }


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _inspect(path: Path) -> corpus.Inspection:
    return corpus.inspect_candidate(corpus.Candidate("synthetic-root", path))


def test_default_roots_are_only_explicit_six(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    roots = corpus.default_roots()
    assert tuple(roots) == corpus.ROOT_NAMES
    assert roots["archive"] == tmp_path / ".cache" / "atif-scan" / "archive"
    assert not roots["archive"].exists()


def test_discovery_includes_renamed_json_in_every_selected_root(tmp_path: Path):
    roots = {label: tmp_path / label for label in corpus.ROOT_NAMES}
    for root in roots.values():
        _write(root / "deep" / "renamed-task.json", _trace())
    _write(roots["hf"] / "00000000-0000-0000-0000-000000000001.json", {"source": "receipt"})
    _write(roots["hf"] / "trajectory.json", _trace())
    _write(roots["archive"] / "trial.ATIF.JSON", _trace())
    _write(roots["archive"] / "config.json", _trace())  # filenames cannot prove nontrace
    _write(roots["archive"] / "ignore.jsonl", _trace())
    _write(tmp_path / "not-selected" / "task.json", _trace())
    records = list(corpus.iter_inspections(roots))
    assert len(records) == 10
    assert sum(r.classification == "trace" for r in records) == 9
    assert {r.candidate.root_label for r in records} == set(roots)
    assert all(r.candidate.path.is_absolute() for r in records)


@pytest.mark.parametrize("name", ["review.atif.json", "hack_hunt.review.atif.json"])
def test_review_filenames_excluded_without_opening(name: str, tmp_path: Path):
    record = _inspect(tmp_path / name)  # deliberately absent: filename exclusion first
    assert record.classification == "judge"
    assert record.raw_sha256 is None


@pytest.mark.parametrize("tool", sorted(corpus.INSPECTION_TOOLS))
def test_actual_inspection_tool_calls_excluded(tool: str, tmp_path: Path):
    path = _write(tmp_path / "renamed.json", _trace(calls=[{"function_name": tool}]))
    record = _inspect(path)
    assert record.classification == "judge"
    assert record.raw_sha256 is not None


def test_inspection_mentions_arguments_outputs_and_user_calls_are_not_judges(tmp_path: Path):
    doc = _trace(
        message="Implement uv__read_steps, uv__search_trace and uv__trace_outline.",
        calls=[{"function_name": "Shell", "arguments": {"command": "uv__read_steps"}}],
    )
    steps = doc["steps"]
    steps[0]["tool_calls"] = [{"function_name": "uv__read_steps"}]
    steps[1]["observation"] = {"results": [{"content": "uv__trace_outline"}]}
    assert _inspect(_write(tmp_path / "task.json", doc)).classification == "trace"


def test_mixed_tools_and_copied_calls_do_not_establish_inspection_session(tmp_path: Path):
    doc = _trace(calls=[{"function_name": "uv__read_steps"}, {"function_name": "Shell"}])
    assert _inspect(_write(tmp_path / "mixed.json", doc)).classification == "trace"
    doc = _trace(calls=[{"function_name": "uv__read_steps"}])
    doc["steps"][1]["is_copied_context"] = True
    assert _inspect(_write(tmp_path / "copied.json", doc)).classification == "trace"


@pytest.mark.parametrize("calls", [None, "unknown", [{"arguments": {}}]])
def test_unknown_calls_do_not_prove_judge(calls: object, tmp_path: Path):
    doc = _trace(calls=[{"function_name": "uv__read_steps"}])
    doc["steps"].append({"source": "agent", "tool_calls": calls})
    # null/absent tool_calls are legal no-call steps; malformed shapes are not evidence.
    expected = "judge" if calls is None else "trace"
    assert _inspect(_write(tmp_path / "task.json", doc)).classification == expected


@pytest.mark.parametrize(
    "doc",
    [
        {"task": "synthetic", "reward": 1},
        {"config": {"agent": "synthetic"}},
        {"kind": "source", "run_id": "synthetic", "plan_path": "/never/read"},
        {"steps": []},
        {"steps": ["not ATIF"]},
        [{"source": "agent"}],
        None,
    ],
)
def test_metadata_and_uuid_receipts_are_nontraces(doc: object, tmp_path: Path):
    assert _inspect(_write(tmp_path / "receipt.json", doc)).classification == "non_trace"


@pytest.mark.parametrize("name", ["result.json", "config.json", "auth.json", "task.json"])
def test_filenames_do_not_override_structure(name: str, tmp_path: Path):
    assert _inspect(_write(tmp_path / name, _trace())).classification == "trace"


def test_legacy_and_empty_declared_atif(tmp_path: Path):
    doc = _trace()
    del doc["schema_version"]
    assert _inspect(_write(tmp_path / "legacy.json", doc)).classification == "trace"
    record = _inspect(_write(tmp_path / "empty.json", {"schema_version": "ATIF-v2.0", "steps": []}))
    assert record.classification == "trace"
    assert record.schema_version == "ATIF-v2.0"


@pytest.mark.parametrize("steps", [None, "bad", [None], [{"source": "unknown"}]])
def test_declared_but_broken_atif_is_unknown_not_nontrace(steps: object, tmp_path: Path):
    doc = {"schema_version": "ATIF-v1.7", "steps": steps}
    assert _inspect(_write(tmp_path / "broken.json", doc)).classification == "invalid_structure"


@pytest.mark.parametrize(
    "data",
    [
        b"{",
        b"\xff",
        b'{"a":1,"a":2}',
        b'{"n":NaN}',
        b'{"n":Infinity}',
        b'{"n":-Infinity}',
        b'{"n":1e999}',
        b'{"n":-1e999}',
        b'{"nested":{"a":1,"a":2}}',
        b'{"n":[REDACTED],"n":0}',
        b'{"n":[REDACTED],"bad":NaN}',
    ],
)
def test_invalid_json_is_explicit(data: bytes, tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_bytes(data)
    record = _inspect(path)
    assert record.classification == "invalid_json"
    assert record.semantic_sha256 is None
    assert record.raw_sha256 == hashlib.sha256(data).hexdigest()
    assert record.redacted_values is None
    with pytest.raises(corpus.CorpusError, match=r"^invalid_json$"):
        corpus.read_document(record.candidate)


def test_size_limit_and_missing_file(tmp_path: Path):
    path = _write(tmp_path / "task.json", _trace())
    record = corpus.inspect_candidate(corpus.Candidate("test", path), max_bytes=8)
    assert record.classification == "too_large"
    assert record.byte_count == path.stat().st_size
    assert record.raw_sha256 is None
    assert _inspect(tmp_path / "missing.json").classification == "unreadable"
    with pytest.raises(corpus.CorpusError, match=r"^unreadable$"):
        corpus.read_document(corpus.Candidate("test", tmp_path / "missing.json"))
    with pytest.raises(corpus.CorpusError, match="invalid_size_cap"):
        corpus.inspect_candidate(corpus.Candidate("test", path), max_bytes=corpus.MAX_BYTES + 1)


def test_oversize_sparse_file_not_read(tmp_path: Path):
    path = tmp_path / "huge.json"
    with path.open("wb") as stream:
        stream.truncate(corpus.MAX_BYTES + 1)
    assert _inspect(path).classification == "too_large"
    with pytest.raises(corpus.CorpusError, match=r"^too_large$"):
        corpus.read_document(corpus.Candidate("test", path))


def test_symlink_files_directories_roots_and_ancestors_refused(tmp_path: Path):
    root = tmp_path / "root"
    target = _write(tmp_path / "outside" / "task.json", _trace())
    own = _write(root / "task.json", _trace())
    (root / "linked.json").symlink_to(target)
    (root / "linked-dir").symlink_to(target.parent, target_is_directory=True)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    assert list(corpus.discover({"test": root})) == [corpus.Candidate("test", own)]
    for path in (root / "linked.json", root / "linked-dir" / "task.json", alias / "task.json"):
        assert _inspect(path).classification == "unsafe_path"
        with pytest.raises(corpus.CorpusError, match=r"^unsafe_path$"):
            corpus.read_document(corpus.Candidate("test", path))
    with pytest.raises(corpus.CorpusError, match="discovery_unavailable"):
        list(corpus.discover({"test": alias}))
    with pytest.raises(corpus.CorpusError, match="discovery_unavailable"):
        list(corpus.discover({"test": alias / "deeper"}))


def test_missing_roots_and_parent_traversal_not_silent(tmp_path: Path):
    with pytest.raises(corpus.CorpusError, match="discovery_unavailable"):
        list(corpus.discover({"missing": tmp_path / "absent"}))
    with pytest.raises(corpus.CorpusError, match="unsafe_path"):
        list(corpus.discover({"bad": tmp_path / ".." / "elsewhere"}))
    assert _inspect(tmp_path / ".." / "task.json").classification == "unsafe_path"
    with pytest.raises(corpus.CorpusError, match=r"^unsafe_path$"):
        corpus.read_document(corpus.Candidate("test", tmp_path / ".." / "task.json"))


def test_nonregular_input_not_read(tmp_path: Path):
    path = tmp_path / "pipe.json"
    os.mkfifo(path)
    assert not list(corpus.discover({"test": tmp_path}))
    assert _inspect(path).classification == "unsafe_path"
    assert _inspect(tmp_path).classification == "unsafe_path"
    for unsafe in (path, tmp_path):
        with pytest.raises(corpus.CorpusError, match=r"^unsafe_path$"):
            corpus.read_document(corpus.Candidate("test", unsafe))


def test_file_change_is_explicit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    path = _write(tmp_path / "task.json", _trace())
    original = corpus._snapshot
    calls = 0

    def changed(info: os.stat_result) -> tuple[int, ...]:
        nonlocal calls
        calls += 1
        # The second snapshot differs, simulating an in-read source mutation.
        result = original(info)
        return (*result, calls)

    monkeypatch.setattr(corpus, "_snapshot", changed)
    assert _inspect(path).classification == "changed_file"
    with pytest.raises(corpus.CorpusError, match=r"^changed_file$"):
        corpus.read_document(corpus.Candidate("test", path))


def test_raw_semantic_and_session_digests_are_separate(tmp_path: Path):
    doc = _trace(message="synthetic café")
    first = _write(tmp_path / "first.json", doc)
    exact = tmp_path / "exact.json"
    exact.write_bytes(first.read_bytes())
    reformatted = tmp_path / "formatted.json"
    reformatted.write_bytes(
        json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-16")
    )
    different = _write(tmp_path / "different.json", _trace(message="another execution"))
    a, b, c, d = map(_inspect, (first, exact, reformatted, different))
    assert a.raw_sha256 == b.raw_sha256
    assert a.raw_sha256 != c.raw_sha256
    assert a.semantic_sha256 == b.semantic_sha256 == c.semantic_sha256
    assert a.session_sha256 == b.session_sha256 == c.session_sha256 == d.session_sha256
    assert a.semantic_sha256 != d.semantic_sha256


def test_canonical_digest_preserves_array_order_and_number_types(tmp_path: Path):
    records = [
        _inspect(_write(tmp_path / f"{i}.json", value))
        for i, value in enumerate(([1, 2], [2, 1], {"n": 1}, {"n": 1.0}))
    ]
    assert len({r.semantic_sha256 for r in records}) == 4


def test_private_inventory_exact_dedup_preserves_divergent_executions(tmp_path: Path):
    root = tmp_path / "inputs"
    first = _write(root / "first.json", _trace(message="synthetic private marker"))
    (root / "exact.json").write_bytes(first.read_bytes())
    (root / "formatted.json").write_text(
        json.dumps(_trace(message="synthetic private marker"), indent=2)
    )
    _write(root / "divergent.json", _trace(message="different execution"))
    _write(root / "review.atif.json", _trace())
    _write(root / "receipt.json", {"token": "synthetic-secret-marker"})
    (root / "broken.json").write_bytes(b"{")
    output = tmp_path / "private"
    inventory = corpus.build_inventory({"sensitive-root-label": root}, output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE(inventory.database_path.stat().st_mode) == 0o600
    summary = inventory.summary
    assert summary["candidate_count"] == 7
    assert summary["unique_byte_traces"] == 3
    assert summary["unique_semantic_traces"] == 2
    assert summary["session_groups"] == 1
    assert summary["divergent_sessions"] == 1
    assert summary["classifications"]["invalid_json"] == 1
    unique = list(corpus.iter_unique_inventory(inventory))
    assert len(unique) == 3  # no semantic/session collapse
    assert {r.candidate.path for r in unique} <= set(root.iterdir())
    assert all(r.candidate.root_label == "sensitive-root-label" for r in unique)
    encoded = json.dumps(summary)
    for private in (
        str(tmp_path),
        "sensitive-root-label",
        "synthetic private marker",
        "synthetic-secret-marker",
    ):
        assert private not in encoded
    db_content = inventory.database_path.read_bytes()
    assert b"synthetic private marker" not in db_content
    assert b"synthetic-secret-marker" not in db_content
    with sqlite3.connect(inventory.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM copies").fetchone() == (7,)


def test_overlapping_explicit_roots_keep_copies_but_dedup_bytes(tmp_path: Path):
    root = tmp_path / "root"
    _write(root / "nested" / "task.json", _trace())
    inventory = corpus.build_inventory({"whole": root, "subset": root / "nested"}, tmp_path / "out")
    assert inventory.summary["candidate_count"] == 2
    assert inventory.summary["unique_byte_traces"] == 1
    assert len(list(corpus.iter_unique_inventory(inventory))) == 1


def test_inventory_refuses_existing_repo_and_symlink_output(tmp_path: Path):
    root = tmp_path / "inputs"
    root.mkdir()
    with pytest.raises(corpus.CorpusError, match="output_inside_repository"):
        corpus.build_inventory(
            {"test": root}, Path(__file__).absolute().parents[1] / "never-created"
        )
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(corpus.CorpusError, match="private_output_unavailable"):
        corpus.build_inventory({"test": root}, output)
    alias = tmp_path / "alias"
    alias.symlink_to(output, target_is_directory=True)
    with pytest.raises(corpus.CorpusError, match="private_output_unavailable"):
        corpus.build_inventory({"test": root}, alias / "new")
    assert not (output / "new").exists()


def test_failed_build_is_not_a_completed_inventory(tmp_path: Path):
    output = tmp_path / "private"
    with pytest.raises(corpus.CorpusError, match="discovery_unavailable"):
        corpus.build_inventory({"missing": tmp_path / "absent"}, output)
    incomplete = corpus.Inventory(output / "inventory.sqlite3", {})
    with pytest.raises(corpus.CorpusError, match="inventory_incomplete"):
        list(corpus.iter_unique_inventory(incomplete))


def test_empty_selection_is_explicit_and_successful(tmp_path: Path):
    inventory = corpus.build_inventory({}, tmp_path / "private")
    assert inventory.summary["candidate_count"] == 0
    assert list(corpus.iter_unique_inventory(inventory)) == []


def test_unrecorded_copy_status_does_not_establish_judge(tmp_path: Path):
    doc = _trace(calls=[{"function_name": "uv__read_steps"}])
    doc["steps"][1]["is_copied_context"] = "unknown"
    assert _inspect(_write(tmp_path / "task.json", doc)).classification == "trace"


def test_referenced_paths_are_never_opened(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    outside = _write(tmp_path / "outside" / "auth.json", {"token": "synthetic-secret"})
    root = tmp_path / "inputs"
    receipt = _write(
        root / "receipt.json",
        {"path": str(outside), "trajectory_path": str(outside), "url": "https://example.invalid"},
    )
    original = corpus._read
    opened: list[Path] = []

    def read(candidate: corpus.Candidate, cap: int):
        opened.append(candidate.path)
        assert candidate.path == receipt
        return original(candidate, cap)

    monkeypatch.setattr(corpus, "_read", read)
    records = list(corpus.iter_inspections({"selected": root}))
    assert [r.classification for r in records] == ["non_trace"]
    assert opened == [receipt]


def test_inspections_are_lazy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for i in range(3):
        _write(tmp_path / f"{i}.json", _trace())
    inspected: list[corpus.Candidate] = []
    original = corpus.inspect_candidate

    def inspect(candidate: corpus.Candidate) -> corpus.Inspection:
        inspected.append(candidate)
        return original(candidate)

    monkeypatch.setattr(corpus, "inspect_candidate", inspect)
    stream = corpus.iter_inspections({"selected": tmp_path})
    assert inspected == []
    first = next(stream)
    assert inspected == [first.candidate]
    assert len(list(stream)) == 2


def test_deep_json_is_an_explicit_failure(tmp_path: Path):
    path = tmp_path / "nested.json"
    path.write_bytes(b"[" * 10_000 + b"0" + b"]" * 10_000)
    assert _inspect(path).classification == "invalid_json"


def _redacted_data() -> bytes:
    doc = _trace(message='Literal [REDACTED], escaped "quote", slash \\ and café.')
    doc["final_metrics"] = {
        "token_count": "synthetic-bare-marker",
        "extra": ["synthetic-bare-marker", None, {"literal": "[REDACTED]"}],
    }
    return (
        json.dumps(doc, ensure_ascii=False)
        .replace('"synthetic-bare-marker"', "[REDACTED]")
        .encode("utf-8")
    )


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-32"])
def test_bare_redactions_normalize_with_original_raw_and_normalized_semantic_hashes(
    encoding: str, tmp_path: Path
):
    data = _redacted_data().decode("utf-8").encode(encoding)
    path = tmp_path / "renamed.json"
    path.write_bytes(data)
    record = _inspect(path)
    loaded = corpus.read_document(record.candidate, record.raw_sha256)
    assert record.classification == "trace"
    assert record.redacted_values == loaded.redacted_values == 2
    assert record.raw_sha256 == loaded.raw_sha256 == hashlib.sha256(data).hexdigest()
    expected = _trace(message='Literal [REDACTED], escaped "quote", slash \\ and café.')
    expected["final_metrics"] = {
        "token_count": None,
        "extra": [None, None, {"literal": "[REDACTED]"}],
    }
    assert loaded.doc == expected
    normalized = _inspect(_write(tmp_path / "normalized.json", expected))
    assert normalized.redacted_values == 0
    assert normalized.semantic_sha256 == record.semantic_sha256
    assert normalized.raw_sha256 != record.raw_sha256


@pytest.mark.parametrize(
    "literal",
    [
        "[REDACTED]",
        '"[REDACTED]"',
        "\\[REDACTED]",
        'Escaped \\" and repeated [REDACTED] [REDACTED].',
        "\n[REDACTED]\t",
    ],
)
def test_redaction_markers_inside_strings_and_keys_are_unchanged(literal: str, tmp_path: Path):
    doc = {literal: literal, "other": {"literal": literal}}
    path = _write(tmp_path / "literal.json", doc)
    data = path.read_text()
    assert corpus._unredact(data) == (data, 0)
    loaded = corpus.read_document(corpus.Candidate("test", path))
    assert loaded.doc == doc
    assert loaded.redacted_values == _inspect(path).redacted_values == 0


@pytest.mark.parametrize(
    "data",
    [
        b'{"a":[REDACTED],}',
        b'{"a":[REDACTED] "missing_comma":0}',
        b'{"a":undefined}',
        b"{[REDACTED]:0}",
        b'{"a":true[REDACTED]}',
        b'{"a":"unterminated [REDACTED]}',
        b'{"a":"bad\\q", "b":[REDACTED]}',
        b"not JSON: synthetic deepswe log",
    ],
)
def test_other_bad_syntax_stays_invalid_even_with_redactions(data: bytes, tmp_path: Path):
    path = tmp_path / "deepswe" / "not-a-trace.json"
    path.parent.mkdir()
    path.write_bytes(data)
    record = _inspect(path)
    assert record.classification == "invalid_json"
    assert record.raw_sha256 == hashlib.sha256(data).hexdigest()
    assert record.semantic_sha256 is None
    assert record.redacted_values is None
    with pytest.raises(corpus.CorpusError, match=r"^invalid_json$"):
        corpus.read_document(record.candidate)


@pytest.mark.parametrize("ending", ["", "\\"])
def test_unterminated_long_string_preserves_markers_without_backtracking(ending: str):
    text = '"' + ('synthetic\\"[REDACTED]' * 20_000) + ending
    assert corpus._unredact(text) == (text, 0)


def test_pinned_digest_mismatch_refused_before_parsing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    path = _write(tmp_path / "task.json", _trace())
    record = _inspect(path)
    path.write_bytes(b"changed synthetic invalid JSON [REDACTED]")

    def never_parse(data: bytes) -> tuple[object, int]:
        pytest.fail("mismatched raw bytes must not be parsed")

    monkeypatch.setattr(corpus, "_parse", never_parse)
    with pytest.raises(corpus.CorpusError, match=r"^digest_mismatch$"):
        corpus.read_document(record.candidate, record.raw_sha256)


def test_read_document_reads_and_parses_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    path = tmp_path / "task.json"
    data = _redacted_data()
    path.write_bytes(data)
    original_read, original_parse = corpus._read, corpus.json.loads
    reads = parses = 0

    def read(candidate: corpus.Candidate, cap: int):
        nonlocal reads
        reads += 1
        assert cap == corpus.MAX_BYTES
        return original_read(candidate, cap)

    def parse(*args, **kwargs) -> object:
        nonlocal parses
        parses += 1
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(corpus, "_read", read)
    monkeypatch.setattr(corpus.json, "loads", parse)
    loaded = corpus.read_document(corpus.Candidate("test", path), hashlib.sha256(data).hexdigest())
    assert loaded.redacted_values == 2
    assert reads == parses == 1


@pytest.mark.parametrize("doc", [None, [], "synthetic", 0])
def test_read_document_refuses_nonobject_but_inspection_stays_nontrace(doc: object, tmp_path: Path):
    path = _write(tmp_path / "nonobject.json", doc)
    assert _inspect(path).classification == "non_trace"
    with pytest.raises(corpus.CorpusError, match=r"^invalid_structure$"):
        corpus.read_document(corpus.Candidate("test", path))


def test_read_document_does_not_impose_trace_or_review_classification(tmp_path: Path):
    doc = {"synthetic_metadata": None}
    path = _write(tmp_path / "review.atif.json", doc)
    assert _inspect(path).classification == "judge"
    loaded = corpus.read_document(corpus.Candidate("test", path))
    assert loaded.doc == doc
    assert loaded.redacted_values == 0


def test_inventory_roundtrips_redaction_counts_and_unknown_coverage(tmp_path: Path):
    root = tmp_path / "inputs"
    root.mkdir()
    data = _redacted_data()
    (root / "redacted.json").write_bytes(data)
    (root / "duplicate.json").write_bytes(data)
    _write(root / "literal.json", _trace(message="[REDACTED]"))
    (root / "bad.json").write_bytes(b"{")
    _write(root / "review.atif.json", _trace())
    inventory = corpus.build_inventory({"synthetic": root}, tmp_path / "private")
    assert inventory.summary["redacted_documents"] == 2
    assert inventory.summary["redacted_values"] == 4
    assert inventory.summary["redaction_unknown_documents"] == 2
    assert inventory.summary["classifications"]["trace"] == 3
    unique = list(corpus.iter_unique_inventory(inventory))
    assert sorted(
        record.redacted_values for record in unique if record.redacted_values is not None
    ) == [
        0,
        2,
    ]
    with sqlite3.connect(inventory.database_path) as connection:
        assert connection.execute(
            "SELECT redacted_values FROM copies ORDER BY redacted_values"
        ).fetchall() == [(None,), (None,), (0,), (2,), (2,)]
    for record in unique:
        loaded = corpus.read_document(record.candidate, record.raw_sha256)
        assert loaded.redacted_values == record.redacted_values


def test_empty_inventory_redaction_counts_are_zero(tmp_path: Path):
    inventory = corpus.build_inventory({}, tmp_path / "private")
    assert inventory.summary["redacted_documents"] == 0
    assert inventory.summary["redacted_values"] == 0
    assert inventory.summary["redaction_unknown_documents"] == 0


def test_read_document_maps_parser_recursion_failure_to_fixed_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    path = _write(tmp_path / "task.json", _trace())

    def fail(*args, **kwargs) -> object:
        raise RecursionError("synthetic private exception text")

    monkeypatch.setattr(corpus.json, "loads", fail)
    with pytest.raises(corpus.CorpusError, match=r"^invalid_json$"):
        corpus.read_document(corpus.Candidate("test", path))
