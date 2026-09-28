"""Bundled task packs load themselves for runs they recognise (synthetic jobs only)."""

from __future__ import annotations

import json

import pytest

from atif_scan.cli import main

TB21 = "terminal-bench/terminal-bench-2-1"


def job(tmp_path, tasks, dataset=TB21, result_json=True):
    """A Harbor job folder: config.json (with `dataset`, if any) and one trial per task."""
    root = tmp_path / "job"
    root.mkdir()
    config = {"n_attempts": 1, "agents": [{"name": "demo"}]}
    if dataset:
        config["datasets"] = [{"name": dataset}]
    (root / "config.json").write_text(json.dumps(config))
    (root / "result.json").write_text(json.dumps({"n_total_trials": len(tasks)}))
    for i, task in enumerate(tasks):
        trial = root / f"{task}__t{i}"
        (trial / "agent").mkdir(parents=True)
        steps = [
            {"source": "user", "message": "Do the task."},
            {"source": "agent", "message": "Only 3 mismatched pixels remain; I'll hard-code them."},
        ]
        (trial / "agent" / "trajectory.json").write_text(json.dumps({"steps": steps}))
        if result_json:
            (trial / "result.json").write_text(
                json.dumps({"task_name": f"terminal-bench/{task}", "verifier_result": {}})
            )
    return root


def scan(capsys, *argv):
    main([*map(str, argv), "--format", "json", "--no-cache"])
    doc = json.loads(capsys.readouterr().out)
    ids = {a["id"] for i in doc["inputs"] for a in i["assessments"]}
    return doc, ids


PIXEL = "tb21.path_tracing_reverse.pixel_special_case"


def test_a_recorded_tb21_dataset_loads_the_pack(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ATIF_SCAN_REFERENCE", raising=False)
    root = job(tmp_path, ["path-tracing-reverse", "fix-git"])
    doc, ids = scan(capsys, root)
    assert doc["packs"] == [{"pack": "tb21", "reason": "dataset"}]
    item = next(i for i in doc["inputs"] if i["input_id"].startswith("path-tracing-reverse"))
    assert next(a for a in item["assessments"] if a["id"] == PIXEL)["status"] == "match"

    doc, ids = scan(capsys, root, "--packs", "none")
    assert "packs" not in doc and PIXEL not in ids
    main([str(root), "--packs", "none", "--brief", "--format", "json", "--no-cache"])
    assert json.loads(capsys.readouterr().out)["suggested_packs"] == ["atif_scan.packs.tb21:checks"]


def test_an_explicit_plugin_is_not_loaded_twice(tmp_path, capsys):
    root = job(tmp_path, ["path-tracing-reverse"])
    doc, ids = scan(capsys, root, "--plugin", "atif_scan.packs.tb21:checks")
    assert "packs" not in doc and PIXEL in ids  # loaded once, as named


@pytest.mark.parametrize(
    ("tasks", "dataset", "loaded"),
    [
        (["path-tracing-reverse", "fix-git"], None, [{"pack": "tb21", "reason": "tasks"}]),
        (["my-own-task", "another-one"], None, None),  # not TB2.1's tasks
        (["path-tracing-reverse", "fix-git"], "terminal-bench/terminal-bench-4", None),
    ],
)
def test_task_names_count_only_when_no_dataset_was_recorded(
    tmp_path, capsys, tasks, dataset, loaded
):
    doc, _ = scan(capsys, job(tmp_path, tasks, dataset))
    assert doc.get("packs") == loaded


def test_reference_pack_needs_sources_for_the_tasks(tmp_path, capsys, monkeypatch):
    root = job(tmp_path, ["path-tracing-reverse"])
    refs = tmp_path / "refs"
    refs.mkdir()
    monkeypatch.setenv("ATIF_SCAN_REFERENCE", str(refs))
    doc, _ = scan(capsys, root)
    assert [p["pack"] for p in doc["packs"]] == ["tb21"]  # no sources for this task
    (refs / "path-tracing-reverse").mkdir()
    doc, _ = scan(capsys, root)
    assert {"pack": "reference", "reason": "reference_sources"} in doc["packs"]


TB4 = "terminal-bench/terminal-bench"


@pytest.mark.parametrize(
    ("tasks", "dataset", "loaded"),
    [
        (["rs-archive-clone", "wdm-design"], TB4, [{"pack": "tb4", "reason": "dataset"}]),
        # The same package name for a later version with other tasks: not TB4's pack.
        (["a-tb5-task", "another-tb5-task"], TB4, None),
        (["rs-archive-clone", "wdm-design"], None, [{"pack": "tb4", "reason": "tasks"}]),
    ],
)
def test_tb4_needs_its_dataset_and_its_tasks(tmp_path, capsys, monkeypatch, tasks, dataset, loaded):
    monkeypatch.delenv("ATIF_SCAN_REFERENCE", raising=False)
    doc, _ = scan(capsys, job(tmp_path, tasks, dataset))
    assert doc.get("packs") == loaded
