"""Synthetic journal durability and filesystem safety regressions."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from atif_scan.browser.feedback import JOURNAL, MAX_ROW, FeedbackStore, digest

if TYPE_CHECKING:
    from pathlib import Path

    from atif_scan.data.jsonval import Doc


def test_private_permissions_and_append_reload(tmp_path: Path):
    directory = tmp_path / "feedback"
    directory.mkdir(mode=0o755)
    path = directory / JOURNAL
    path.write_text("")
    path.chmod(0o644)
    store = FeedbackStore(directory)
    binding = digest({"input": "synthetic"})
    store.append(binding, "valid_signal", "first")
    store.append(binding, "unclear", "second")
    assert len(path.read_text().splitlines()) == 2
    assert FeedbackStore(directory).load()[binding] == {"verdict": "unclear", "note": "second"}
    assert directory.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600
    store.append(binding, "unreviewed", "")
    assert store.load()[binding]["verdict"] == "unreviewed"


@pytest.mark.parametrize("verdict,note", [("clean", ""), ("unclear", "n" * 2001)])
def test_invalid_feedback_not_appended(tmp_path: Path, verdict: str, note: str):
    store = FeedbackStore(tmp_path / "feedback")
    with pytest.raises(ValueError, match="invalid_feedback"):
        store.append(digest("synthetic"), verdict, note)
    assert store.load() == {}


def test_long_unicode_note(tmp_path: Path):
    store = FeedbackStore(tmp_path / "feedback")
    note = "\U0001f642" * 2000
    store.append(digest("synthetic"), "unclear", note)
    assert store.load()[digest("synthetic")]["note"] == note


@pytest.mark.parametrize(
    "data",
    [
        b"not json\n",
        b"{}\n",
        b"[]\n",
        b'{"schema": 1, "binding": false, "verdict": "unclear", "note": ""}\n',
        b'{"schema": 1, "binding": "x", "verdict": "unclear", "note": ""}\n',
        b"\xff\n",
        b"x" * 32769,
    ],
)
def test_malformed_journal_fails_closed(tmp_path: Path, data: bytes):
    store = FeedbackStore(tmp_path / "feedback")
    path = tmp_path / "feedback" / JOURNAL
    path.write_bytes(data)
    with pytest.raises(ValueError, match="malformed_feedback"):
        store.load()
    with pytest.raises(ValueError, match="malformed_feedback"):
        store.append(digest("synthetic"), "unclear", "")
    assert path.read_bytes() == data


def test_partial_final_row_is_not_silently_discarded(tmp_path: Path):
    store = FeedbackStore(tmp_path / "feedback")
    store.append(digest("synthetic"), "unclear", "")
    path = tmp_path / "feedback" / JOURNAL
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    with pytest.raises(ValueError, match="malformed_feedback"):
        store.load()


@pytest.mark.parametrize("target", ["directory", "ancestor", "journal"])
def test_symlinks_rejected(tmp_path: Path, target: str):
    real = tmp_path / "real"
    real.mkdir()
    directory = tmp_path / "feedback"
    if target in ("directory", "ancestor"):
        directory.symlink_to(real, target_is_directory=True)
        if target == "ancestor":
            directory /= "child"
    else:
        directory.mkdir()
        outside = real / "outside"
        outside.write_text("unchanged")
        (directory / JOURNAL).symlink_to(outside)
    with pytest.raises((OSError, ValueError)):
        FeedbackStore(directory)
    if target == "journal":
        assert (real / "outside").read_text() == "unchanged"


def test_hardlinked_journal_rejected(tmp_path: Path):
    directory = tmp_path / "feedback"
    directory.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("unchanged")
    os.link(outside, directory / JOURNAL)
    with pytest.raises(ValueError, match="unsafe_feedback_file"):
        FeedbackStore(directory)
    assert outside.read_text() == "unchanged"


def test_metadata_persisted_without_changing_annotation(tmp_path: Path):
    store = FeedbackStore(tmp_path / "feedback")
    binding = digest("synthetic")
    context = {
        "input_id": "synthetic-input",
        "task": "synthetic-task",
        "trace_sha256": digest("synthetic-trace"),
        "check_id": "synthetic-check",
        "check_version": "1",
        "finding_sha256": digest({"assessment": "synthetic"}),
    }
    before = datetime.now(UTC)
    assert store.append(binding, "unclear", "review", context=context) == {
        "verdict": "unclear",
        "note": "review",
    }
    row = json.loads((store.directory / JOURNAL).read_text())
    assert row["context"] == context
    saved_at = datetime.fromisoformat(row["saved_at"])
    assert saved_at.utcoffset() == timedelta(0)
    assert before <= saved_at <= datetime.now(UTC)
    assert set(row) == {"schema", "binding", "verdict", "note", "saved_at", "context"}
    assert FeedbackStore(store.directory).load()[binding] == {
        "verdict": "unclear",
        "note": "review",
    }


@pytest.mark.parametrize(
    "context",
    [
        {"source_path": "/synthetic/trace.json"},
        {"unknown": ""},
        {"input_id": None},
        {"input_id": 1},
        {"task": False},
        {"trace_sha256": "x" * 64},
        {"trace_sha256": "a" * 63},
        {"trace_sha256": 12},
        {"check_id": []},
        {"check_version": {}},
        {"finding_sha256": None},
        {"finding_sha256": "a" * 65},
    ],
)
def test_invalid_context_not_appended(tmp_path: Path, context: Doc):
    store = FeedbackStore(tmp_path / "feedback")
    with pytest.raises(ValueError, match="invalid_feedback_context"):
        store.append(digest("synthetic"), "unclear", "", context=context)
    assert (store.directory / JOURNAL).read_bytes() == b""


@pytest.mark.parametrize(
    "metadata",
    [
        {"context": None},
        {"context": []},
        {"context": {"source_path": "/synthetic"}},
        {"context": {"finding_sha256": "not-hex"}},
        {"context": {"task": 123}},
        {"saved_at": None},
        {"saved_at": 123},
        {"saved_at": True},
        {"saved_at": "not-a-date"},
        {"saved_at": "2026-10-05"},
        {"saved_at": "2026-10-05T12:00:00"},
        {"saved_at": "2026-99-05T12:00:00+00:00"},
    ],
)
def test_malformed_metadata_fails_closed(tmp_path: Path, metadata: Doc):
    store = FeedbackStore(tmp_path / "feedback")
    row = {
        "schema": 1,
        "binding": digest("synthetic"),
        "verdict": "unclear",
        "note": "",
        **metadata,
    }
    data = (json.dumps(row) + "\n").encode()
    path = store.directory / JOURNAL
    path.write_bytes(data)
    with pytest.raises(ValueError, match="malformed_feedback"):
        store.load()
    with pytest.raises(ValueError, match="malformed_feedback"):
        store.append(digest("synthetic"), "unclear", "")
    assert path.read_bytes() == data


def test_old_schema_and_nullable_context_remain_readable(tmp_path: Path):
    store = FeedbackStore(tmp_path / "feedback")
    binding = digest("synthetic")
    row = {"schema": 1, "binding": binding, "verdict": "unclear", "note": "old"}
    (store.directory / JOURNAL).write_text(json.dumps(row) + "\n")
    assert store.load()[binding] == {"verdict": "unclear", "note": "old"}
    store.append(binding, "valid_signal", "new", context={"task": None, "trace_sha256": None})
    assert FeedbackStore(store.directory).load()[binding] == {
        "verdict": "valid_signal",
        "note": "new",
    }


def test_serialized_metadata_row_cap(tmp_path: Path):
    store = FeedbackStore(tmp_path / "feedback")
    store.append(digest("synthetic"), "unclear", "")
    path = store.directory / JOURNAL
    before = path.read_bytes()
    with pytest.raises(ValueError, match="malformed_feedback"):
        store.append(
            digest("synthetic"),
            "unclear",
            "",
            context={"input_id": "\U0001f642" * (MAX_ROW // 12)},
        )
    assert path.read_bytes() == before
