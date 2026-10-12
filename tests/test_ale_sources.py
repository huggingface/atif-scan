"""ALE recognition uses synthetic receipts, never benchmark traces or credentials."""

import json

import pytest
from test_sync import FS, ROOT, trajectory
from typing_extensions import override

from atif_scan.cli import main
from atif_scan.data.ale import LIMIT, facts, recognized
from atif_scan.data.loader import TraceError
from atif_scan.sources.inputs import resolve
from atif_scan.sources.sync import sync_remote


def receipt():
    return {
        "schema_version": 2,
        "run_id": "synthetic-attempt",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "agent": {
            "id": "demo",
            "model": "example/model",
            "config_repr": {"secret": "DO-NOT-EXPORT"},
        },
        "task": {
            "slug": "synthetic__widgets",
            "path": "tasks/synthetic/widgets",
            "variant_index": 0,
        },
        "status": "completed",
        "score": 1.0,
        "termination": {"reason": "completed", "error": "DO-NOT-EXPORT"},
        "timings": {"duration_s": 12.0},
    }


def attempt(path, *, export=True, run=None, evaluation=True):
    path.mkdir(parents=True, exist_ok=True)
    (path / "run.json").write_text(json.dumps(receipt() if run is None else run))
    (path / "trajectory.json").write_text(json.dumps({"summary": "NOT AN ATIF"}))
    if export:
        (path / "atif.json").write_bytes(
            trajectory("cat /mnt/ale-task-data/synthetic/widgets/base/reference/answer.csv")
        )
    if evaluation:
        (path / "eval_result.json").write_text(json.dumps({"eval_status": "success", "score": 1.0}))
    return path


def scan(capsys, *args):
    main([*map(str, args), "--format", "json", "--no-cache"])
    return json.loads(capsys.readouterr().out)


def test_default_directory_selects_export_once_and_autoloads_pack(tmp_path, capsys):
    attempt(tmp_path / "attempt")
    doc = scan(capsys, tmp_path)
    assert doc["packs"] == [{"pack": "ale", "reason": "dataset"}]
    assert len(doc["inputs"]) == 1
    row = doc["inputs"][0]
    assert row["task"] == "synthetic__widgets" and row["reward"] == 1
    assert row["configured_agent"] == "demo"
    assert row["source_metadata"]["variant_index"] == 0
    assert row["source_metadata"]["trajectory_source"] == "atif_export"
    assert (
        next(a for a in row["assessments"] if a["id"] == "ale.mounted_reference_access")["status"]
        == "match"
    )
    assert "DO-NOT-EXPORT" not in json.dumps(doc)
    assert str(tmp_path) not in json.dumps(doc)


def test_missing_export_is_not_replaced_by_summary(tmp_path, capsys):
    attempt(tmp_path, export=False)
    doc = scan(capsys, tmp_path)
    row = doc["inputs"][0]
    assert row["input_status"] == "unavailable_or_invalid"
    assert row["input_error"] == "no_trajectory_downloaded"
    assert row["reward"] == 1
    assert "trajectory_missing" in row["source_metadata"]["source_issues"]


def test_invalid_export_does_not_fall_back_to_another_trace(tmp_path):
    attempt(tmp_path)
    (tmp_path / "atif.json").write_text("not JSON")
    (tmp_path / "trajectory.json").write_bytes(trajectory())
    (source,) = resolve([str(tmp_path)])
    with pytest.raises(TraceError):
        source.load()


def test_fast_agent_native_atif_is_preferred_over_root_summary(tmp_path):
    run = receipt()
    run["agent"]["id"] = "fast_agent"
    attempt(tmp_path, run=run, export=False)
    native = tmp_path / "origin_log/fast_agent/trajectory.json"
    native.parent.mkdir(parents=True)
    native.write_bytes(trajectory())
    (source,) = resolve([str(tmp_path)])
    assert source.local == native
    assert source.meta["trajectory_source"] == "native_atif"
    assert source.load().agent_steps == 1


def test_paths_in_metadata_are_never_followed(tmp_path):
    run = receipt()
    run["trajectory_path"] = "../../outside.json"
    attempt(tmp_path, run=run, export=False)
    (source,) = resolve([str(tmp_path)])
    with pytest.raises(TraceError, match="no_trajectory_downloaded"):
        source.load()


def test_symlink_export_is_not_followed(tmp_path):
    root = attempt(tmp_path / "attempt", export=False)
    outside = tmp_path / "outside.json"
    outside.write_bytes(trajectory())
    (root / "atif.json").symlink_to(outside)
    (source,) = resolve([str(root)])
    with pytest.raises(TraceError):
        source.load()


def test_task_and_score_conflicts_remain_unknown(tmp_path, capsys):
    run = receipt()
    run["task"]["path"] = "tasks/synthetic/different"
    run["score"] = 0.5
    attempt(tmp_path, run=run)
    row = scan(capsys, tmp_path)["inputs"][0]
    assert row["task"] is None and row["reward"] is None
    assert set(row["source_metadata"]["source_issues"]) == {
        "task_identity_conflict",
        "score_conflict",
    }


@pytest.mark.parametrize("score", [True, "1", float("nan"), -1, 2])
def test_invalid_evaluation_score_is_unknown(score):
    doc = facts(receipt(), {"eval_status": "success", "score": score})
    assert doc["reward"] is None
    assert "evaluation_unresolved" in doc["source_issues"]


def test_failed_evaluation_is_not_a_zero_score():
    doc = facts(receipt(), {"eval_status": "error", "score": 0})
    assert doc["reward"] is None


def test_missing_evaluation_qualifies_recorded_run_score():
    doc = facts(receipt(), {})
    assert doc["reward"] == 1 and doc["reward_source"] == "run"
    assert doc["source_issues"] == ["evaluation_not_recorded"]


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 3},
        {"schema_version": True},
        {"task": "synthetic__widgets"},
        {"agent": {}},
    ],
)
def test_similar_or_unknown_schema_is_not_recognized(change):
    assert not recognized({**receipt(), **change})


def test_oversized_metadata_is_not_used(tmp_path):
    attempt(tmp_path)
    (tmp_path / "run.json").write_text(" " * LIMIT + json.dumps(receipt()))
    (source,) = resolve([str(tmp_path)])
    assert not source.meta.get("benchmark")


def test_mixed_tree_keeps_non_ale_inputs(tmp_path):
    attempt(tmp_path / "ale")
    generic = tmp_path / "other"
    generic.mkdir()
    (generic / "trajectory.json").write_bytes(trajectory())
    sources = resolve([str(tmp_path)])
    assert len(sources) == 2
    assert sum(s.meta.get("benchmark") == "agents-last-exam" for s in sources) == 1


def test_packs_none_keeps_recognition_without_activating_checks(tmp_path, capsys):
    attempt(tmp_path)
    doc = scan(capsys, tmp_path, "--packs", "none")
    assert not doc.get("packs")
    row = doc["inputs"][0]
    assert row["source_metadata"]["benchmark"] == "agents-last-exam"
    assert not any(a["id"].startswith("ale.") for a in row["assessments"])


def test_manifest_preserves_explicit_task_reward_and_ale_identity(tmp_path, capsys):
    root = attempt(tmp_path / "attempt")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "id": "chosen",
                        "path": str(root / "atif.json"),
                        "task": "synthetic__override",
                        "reward": 0.25,
                    }
                ]
            }
        )
    )
    row = scan(capsys, "--manifest", manifest)["inputs"][0]
    assert row["task"] == "synthetic__override" and row["reward"] == 0.25
    assert row["source_metadata"]["reward_source"] == "manifest"


def test_direct_file_keeps_explicit_selection_and_receipt_facts(tmp_path, capsys):
    attempt(tmp_path)
    row = scan(capsys, tmp_path / "atif.json")["inputs"][0]
    assert row["task"] == "synthetic__widgets"
    assert row["source_metadata"]["trajectory_source"] == "explicit_file"


def test_remote_sync_collects_export_and_eval_without_native_secrets(tmp_path):
    files = {
        f"{ROOT}/run.json": json.dumps(receipt()).encode(),
        f"{ROOT}/atif.json": trajectory(),
        f"{ROOT}/trajectory.json": b'{"summary": "not atif"}',
        f"{ROOT}/eval_result.json": b'{"eval_status":"success","score":1}',
        f"{ROOT}/origin_log/demo/transcript.jsonl": b"DO-NOT-DOWNLOAD",
    }
    fs = FS(files)
    dest, _ = sync_remote("hf://" + ROOT, tmp_path / "mirror", fs=fs)
    assert (dest / "atif.json").exists()
    assert (dest / "eval_result.json").exists()
    assert not (dest / "origin_log/demo/transcript.jsonl").exists()
    (source,) = resolve([str(dest)])
    assert source.meta["reward"] == 1 and source.load().agent_steps == 1


def test_remote_without_sync_uses_same_selection():
    fs = FS({f"{ROOT}/run.json": json.dumps(receipt()).encode(), f"{ROOT}/atif.json": trajectory()})
    (source,) = resolve(["hf://" + ROOT], fs=fs)
    assert source.meta["task"] == "synthetic__widgets"
    assert source.load().agent_steps == 1


def test_explicit_export_pattern_keeps_metadata(tmp_path, capsys):
    attempt(tmp_path / "attempt")
    doc = scan(capsys, tmp_path, "--pattern", "atif.json")
    assert doc["inputs"][0]["task"] == "synthetic__widgets"
    assert doc["packs"] == [{"pack": "ale", "reason": "dataset"}]


def test_symlink_evaluation_is_not_opened(tmp_path):
    root = attempt(tmp_path / "attempt", evaluation=False)
    outside = tmp_path / "outside.json"
    outside.write_text('{"eval_status":"success","score":0}')
    (root / "eval_result.json").symlink_to(outside)
    (source,) = resolve([str(root)])
    assert source.meta["reward"] == 1
    assert source.meta["reward_source"] == "run"
    issues = source.meta["source_issues"]
    assert isinstance(issues, list)
    assert "evaluation_not_recorded" in issues


@pytest.mark.parametrize("content", ["not JSON", "{}", " " * LIMIT + "{}"])
def test_unreadable_evaluation_is_unknown_not_run_score_fallback(tmp_path, content):
    attempt(tmp_path)
    (tmp_path / "eval_result.json").write_text(content)
    (source,) = resolve([str(tmp_path)])
    assert source.meta["reward"] is None
    issues = source.meta["source_issues"]
    assert isinstance(issues, list)
    assert "evaluation_unreadable" in issues


@pytest.mark.parametrize(
    ("harness", "native_folder"), [("demo", "demo"), ("openclaw_cli", "openclaw-cli")]
)
def test_native_and_output_availability_are_inventory_only(tmp_path, harness, native_folder):
    attempt(tmp_path)
    run = receipt()
    run["agent"]["id"] = harness
    (tmp_path / "run.json").write_text(json.dumps(run))
    native = tmp_path / f"origin_log/{native_folder}/transcript.jsonl"
    native.parent.mkdir(parents=True)
    native.write_text("not JSON: do not parse")
    output = tmp_path / "output/submission.txt"
    output.parent.mkdir()
    output.write_text("private synthetic answer")
    (source,) = resolve([str(tmp_path)])
    assert source.meta["native_transcript_available"] is True
    assert source.meta["submission_artifacts_available"] is True
    assert "private synthetic answer" not in json.dumps(dict(source.meta))


def test_metadata_is_reread_between_resolutions(tmp_path):
    attempt(tmp_path)
    (before,) = resolve([str(tmp_path)])
    assert before.meta["reward"] == 1
    (tmp_path / "eval_result.json").write_text('{"eval_status":"success","score":0.2}')
    (after,) = resolve([str(tmp_path)])
    assert after.meta["reward"] is None
    issues = after.meta["source_issues"]
    assert isinstance(issues, list)
    assert "score_conflict" in issues


def test_receipt_provenance_and_termination_are_allowlisted(tmp_path, capsys):
    attempt(tmp_path)
    doc = scan(capsys, tmp_path)
    encoded = json.dumps(doc)
    assert "receipt_schema_version" in encoded
    assert "listed_files_only" in encoded
    run = receipt()
    run["termination"]["reason"] = "private synthetic termination"
    assert facts(run, {})["termination_reason"] is None
    run["termination"]["reason"] = "timeout"
    assert facts(run, {})["termination_reason"] == "timeout"


@pytest.mark.parametrize("path", ["tasks//widgets", "tasks/../widgets", "tasks/demo/."])
def test_invalid_task_path_is_not_an_ale_receipt(path):
    run = receipt()
    run["task"]["path"] = path
    assert not recognized(run)


def test_failed_sync_never_uses_stale_ale_export(tmp_path):
    class FailingExport(FS):
        @override
        def get_file(self, path, local):
            if path.endswith("/atif.json"):
                raise OSError("synthetic failure")
            super().get_file(path, local)

    files = {
        f"{ROOT}/run.json": json.dumps(receipt()).encode(),
        f"{ROOT}/atif.json": trajectory(),
        f"{ROOT}/trajectory.json": trajectory(),
    }
    dest, _ = sync_remote("hf://" + ROOT, tmp_path / "mirror", fs=FS(files))
    assert (dest / "atif.json").exists()
    files[f"{ROOT}/atif.json"] = trajectory("echo changed")
    sync_remote("hf://" + ROOT, dest, fs=FailingExport(files))
    (source,) = resolve([str(dest)])
    assert source.meta["benchmark"] == "agents-last-exam"
    with pytest.raises(TraceError, match="sync_failed"):
        source.load()


def test_inspection_never_reads_ale_receipts(tmp_path):
    from dataclasses import replace

    from atif_scan.sources.inputs import list_input
    from atif_scan.sources.layout import inspect_listing

    attempt(tmp_path)

    def forbidden_read(relative, limit):
        pytest.fail("inspection must not read receipts")

    listing = replace(list_input(str(tmp_path)), reader=forbidden_read)
    doc = inspect_listing(listing, "trajectory.json", 1)
    assert doc["selection_basis"] == "filename_pattern_before_receipt_recognition"
    assert doc["would_scan"]["total"] == 1
