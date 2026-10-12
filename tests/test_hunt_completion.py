"""Completion is ingestion-valid, not merely a successful runner exit (synthetic only)."""

import json
import subprocess

import pytest

from atif_scan.cli import hunt
from atif_scan.review.answers import MAX_READ, Answers
from atif_scan.review.catalogue import BY_ID
from atif_scan.review.prompts import schema

REPLY = {"answer": "used", "confidence": "high", "steps": [2], "reason": "synthetic"}


@pytest.fixture
def bundle(tmp_path):
    (tmp_path / "index.jsonl").write_text("")
    folder = tmp_path / "synthetic"
    folder.mkdir()
    meta = folder / "lookup_used.json"
    meta.write_text(
        json.dumps(
            {
                "question": "lookup_used",
                "version": BY_ID["lookup_used"].version,
                "input_id": "synthetic",
            }
        )
    )
    meta.with_suffix(".md").write_text("Synthetic prompt")
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / meta.name).write_text(json.dumps(schema(BY_ID["lookup_used"])))
    return tmp_path, meta


def run(bundle, monkeypatch, stdout, *, code=0, extra=()):
    root, _ = bundle
    calls = []

    def runner(executable, command):
        calls.append(command)
        return subprocess.CompletedProcess(command, code, stdout, "")

    monkeypatch.setattr(hunt, "run_go", runner)
    monkeypatch.setattr(hunt.shutil, "which", lambda _: "synthetic-runner")
    status = hunt.main(["--model", "synthetic", "--questions", str(root), *extra])
    return status, calls


@pytest.mark.parametrize(
    "text",
    [
        "",
        "passthrough prompt text",
        "{}",
        "[]",
        json.dumps({**REPLY, "steps": [True]}),
        json.dumps({**REPLY, "answer": "bogus"}),
        json.dumps({key: value for key, value in REPLY.items() if key != "reason"}),
        "x" * (MAX_READ + 1),
        "é" * MAX_READ,
    ],
)
def test_invalid_output_fails_and_is_retried(bundle, monkeypatch, text):
    root, meta = bundle
    status, calls = run(bundle, monkeypatch, text)
    assert status != 0 and len(calls) == 1
    assert not meta.with_suffix(".answer.json").exists()
    assert hunt.pending(root, [], False) == [meta]
    assert (root / "ask-errors.log").exists()
    status, calls = run(bundle, monkeypatch, json.dumps(REPLY))
    assert status == 0 and len(calls) == 1


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "status line\n{}", "Sure! {}"])
@pytest.mark.parametrize("indent", [None, 2])
def test_valid_wrapped_answers_are_usable_and_resume(bundle, monkeypatch, wrapper, indent):
    root, meta = bundle
    text = wrapper.format(json.dumps(REPLY, indent=indent))
    assert run(bundle, monkeypatch, text)[0] == 0
    assert Answers.load(root).by_input["synthetic"][0][2] == "answered"
    assert run(bundle, monkeypatch, "must not run")[1] == []
    assert hunt.pending(root, [], True) == [meta]


@pytest.mark.parametrize("cached", [b"", b"{}", b"not JSON", b"\xff", b"x" * (MAX_READ + 1)])
def test_invalid_cache_is_replaced(bundle, monkeypatch, cached):
    _, meta = bundle
    target = meta.with_suffix(".answer.json")
    target.write_bytes(cached)
    status, calls = run(bundle, monkeypatch, json.dumps(REPLY))
    assert status == 0 and len(calls) == 1
    assert json.loads(target.read_text()) == REPLY


@pytest.mark.parametrize("metadata", ["{}", "not JSON", '{"question":"custom"}'])
def test_missing_contract_fails_without_runner(bundle, monkeypatch, metadata):
    root, meta = bundle
    meta.write_text(metadata)
    meta.with_suffix(".answer.json").write_text(json.dumps(REPLY))
    assert hunt.pending(root, [], False) == [meta]
    status, calls = run(bundle, monkeypatch, json.dumps(REPLY))
    assert status != 0 and calls == []


@pytest.mark.parametrize("code,text", [(1, json.dumps(REPLY)), (0, "Error: synthetic failure")])
def test_runner_failures_preserve_prior_answer(bundle, monkeypatch, code, text):
    _, meta = bundle
    target = meta.with_suffix(".answer.json")
    original = json.dumps(REPLY)
    target.write_text(original)
    assert run(bundle, monkeypatch, text, code=code, extra=("--force",))[0] != 0
    assert target.read_text() == original


def test_mixed_batch_nonzero_and_only_failure_retried(bundle, monkeypatch):
    root, meta = bundle
    second = meta.parent / "fabrication_hunt.json"
    second.write_text(
        json.dumps(
            {
                "question": "fabrication_hunt",
                "version": BY_ID["fabrication_hunt"].version,
                "input_id": "synthetic",
            }
        )
    )
    # "used" is valid only for lookup_used.
    assert run(bundle, monkeypatch, json.dumps(REPLY))[0] != 0
    assert hunt.pending(root, [], False) == [second]
    assert not list(root.glob("*/*.tmp.*"))


def test_invalid_forced_reply_preserves_valid_cache(bundle, monkeypatch):
    _, meta = bundle
    target = meta.with_suffix(".answer.json")
    original = json.dumps(REPLY)
    target.write_text(original)
    assert run(bundle, monkeypatch, "not an answer", extra=("--force",))[0] != 0
    assert target.read_text() == original


@pytest.mark.parametrize(
    "reply",
    [
        {**REPLY, "answer": "clean"},
        {**REPLY, "answer": "clean", "mechanism": "special_casing"},
        {**REPLY, "answer": "hack", "mechanism": "none"},
    ],
)
def test_mechanism_contract_applies_to_new_and_cached_answers(bundle, monkeypatch, reply):
    root, meta = bundle
    meta.write_text(
        json.dumps(
            {
                "question": "hack_hunt",
                "version": BY_ID["hack_hunt"].version,
                "input_id": "synthetic",
            }
        )
    )
    target = meta.with_suffix(".answer.json")
    target.write_text(json.dumps(reply))
    assert hunt.pending(root, [], False) == [meta]
    assert run(bundle, monkeypatch, json.dumps(reply))[0] != 0


def test_stale_metadata_requires_regenerated_bundle(bundle, monkeypatch):
    root, meta = bundle
    fields = json.loads(meta.read_text())
    fields["version"] = "obsolete-synthetic"
    meta.write_text(json.dumps(fields))
    meta.with_suffix(".answer.json").write_text(json.dumps(REPLY))
    assert hunt.pending(root, [], False) == [meta]
    assert run(bundle, monkeypatch, json.dumps(REPLY)) == (1, [])


@pytest.mark.parametrize(
    "fields",
    [
        {"input_id": None},
        {"input_id": 1},
        {"version": None},
        {"version": 1},
        {"question": "custom"},
        {"question": ["lookup_used"]},
    ],
)
def test_invalid_metadata_fields_never_call_runner(bundle, monkeypatch, fields):
    root, meta = bundle
    metadata = json.loads(meta.read_text())
    metadata.update(fields)
    meta.write_text(json.dumps(metadata))
    assert run(bundle, monkeypatch, json.dumps(REPLY)) == (1, [])
    assert hunt.pending(root, [], False) == [meta]


@pytest.mark.parametrize("content", [b"\xff", b" " * (MAX_READ + 1), b"[" * 2000])
def test_unreadable_metadata_never_calls_runner(bundle, monkeypatch, content):
    _, meta = bundle
    meta.write_bytes(content)
    assert run(bundle, monkeypatch, json.dumps(REPLY)) == (1, [])


def test_failed_retry_keeps_invalid_cache_pending(bundle, monkeypatch):
    root, meta = bundle
    target = meta.with_suffix(".answer.json")
    target.write_text("{}")
    status, calls = run(bundle, monkeypatch, "invalid")
    assert status == 1 and len(calls) == 1
    assert target.read_text() == "{}"
    assert hunt.pending(root, [], False) == [meta]
    status, calls = run(bundle, monkeypatch, json.dumps(REPLY))
    assert status == 0 and len(calls) == 1
    assert hunt.pending(root, [], False) == []
