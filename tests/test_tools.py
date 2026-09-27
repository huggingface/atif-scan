"""tools/task_diff.py: infrastructure-only forks vs content changes. Synthetic tasks only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "task_diff.py"
spec = importlib.util.spec_from_file_location("task_diff", TOOL)
task_diff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task_diff)

TOML = """[metadata]
author = "a"
[agent]
timeout_sec = 600.0
[verifier]
timeout_sec = 300.0
[environment]
docker_image = "example/demo:1"
cpus = 1
"""


def task(root: Path, name: str, toml: str = TOML, dockerfile: str = "FROM debian:bullseye\n"):
    t = root / name
    (t / "tests").mkdir(parents=True)
    (t / "solution").mkdir()
    (t / "environment").mkdir()
    (t / "instruction.md").write_text("Do the thing.")
    (t / "tests" / "test_outputs.py").write_text("def test_x(): pass\n")
    (t / "solution" / "solve.sh").write_text("echo x\n")
    (t / "environment" / "Dockerfile").write_text(dockerfile)
    (t / "task.toml").write_text(toml)
    (t / "README.md").write_text("docs")


def test_image_swap_is_infrastructure_only(tmp_path, capsys):
    task(tmp_path / "a", "demo")
    task(
        tmp_path / "b",
        "demo",
        TOML.replace("example/demo:1", "ghcr.io/example/worker@sha256:abc"),
        "FROM debian:bookworm\n",
    )
    (tmp_path / "b" / "demo" / "README.md").write_text("docs, updated")
    assert task_diff.main([str(tmp_path / "a"), str(tmp_path / "b")]) == 0
    out = capsys.readouterr().out
    assert "infrastructure 2" in out and "docs 1" in out and "content" not in out


def test_content_and_budget_changes_fail(tmp_path, capsys):
    task(tmp_path / "a", "demo")
    task(tmp_path / "b", "demo", TOML.replace("timeout_sec = 600.0", "timeout_sec = 6000.0"))
    (tmp_path / "b" / "demo" / "tests" / "test_outputs.py").write_text("def test_x(): assert 1\n")
    assert task_diff.main([str(tmp_path / "a"), str(tmp_path / "b"), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    kinds = {what: kind for kind, what in report["tasks"]["demo"]}
    assert kinds["[agent].timeout_sec"] == "content"
    assert kinds["tests/test_outputs.py"] == "content"


def test_timeout_overruns_from_job_results(tmp_path, capsys):
    task(tmp_path / "a", "demo")
    task(tmp_path / "b", "demo")
    job = tmp_path / "job"
    for name, minutes, reward in (("demo__1", 5, 1.0), ("demo__2", 30, 1.0), ("demo__3", 30, 0.0)):
        (job / name).mkdir(parents=True)
        (job / name / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "demo",
                    "verifier_result": {"rewards": {"reward": reward}},
                    "agent_execution": {
                        "started_at": "2026-09-11T10:00:00Z",
                        "finished_at": f"2026-09-11T10:{minutes:02d}:00Z",
                    },
                }
            )
        )
    assert (
        task_diff.main([str(tmp_path / "a"), str(tmp_path / "b"), "--job", str(job), "--json"]) == 0
    )
    t = json.loads(capsys.readouterr().out)["timeouts"]
    assert (t["over_timeout"], t["over_timeout_rewarded"]) == (2, 1)
    assert t["accuracy"] == 66.67 and t["accuracy_if_overruns_failed"] == 33.33
