"""tools/regrade.py: counterfactual re-grades of a DeepSWE-layout trial. Synthetic tasks and
patches only; docker is a fake recording its calls (no engine, nothing is executed)."""

from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "regrade.py"
spec = importlib.util.spec_from_file_location("regrade", TOOL)
assert spec and spec.loader
regrade = importlib.util.module_from_spec(spec)
sys.modules["regrade"] = regrade
spec.loader.exec_module(regrade)

PATCH = (
    "diff --git a/src/app.py b/src/app.py\nindex 1..2 100644\n--- a/src/app.py\n"
    "+++ b/src/app.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
    "diff --git a/tests/conftest.py b/tests/conftest.py\nnew file mode 100644\n"
    "--- /dev/null\n+++ b/tests/conftest.py\n@@ -0,0 +1 @@\n+import os\n"
    "diff --git a/old.py b/new.py\nsimilarity index 90%\nrename from old.py\nrename to new.py\n"
)
TOML = """[environment]
docker_image = "example/task:1"
cpus = 4
memory_mb = 4096
[verifier.environment]
cpus = 2
memory_mb = 8192
"""


def task_dir(root: Path, toml: str = TOML) -> Path:
    t = root / "task"
    (t / "tests").mkdir(parents=True)
    for name in ("test.sh", "grader.py", "config.json", "test.patch", "Dockerfile"):
        (t / "tests" / name).write_text(name)
    (t / "task.toml").write_text(toml)
    return t


def test_drop_removes_whole_file_sections_by_old_or_new_path():
    kept, removed = regrade.drop_files(PATCH, {"tests/conftest.py", "old.py"})
    assert removed == ["tests/conftest.py", "old.py"]
    assert kept.startswith("diff --git a/src/app.py") and "conftest" not in kept
    assert "rename" not in kept and kept.endswith("+x = 2\n")


def test_unknown_drop_path_is_an_error_not_a_silent_control():
    with pytest.raises(SystemExit, match=r"not in the patch: tests/typo\.py"):
        regrade._variants(PATCH, ["tests/typo.py"])
    assert list(regrade._variants(PATCH, [])) == ["control"]


def test_task_settings_prefer_verifier_limits(tmp_path):
    task = regrade.task_settings(task_dir(tmp_path))
    assert (task.image, task.cpus, task.memory_mb) == ("example/task:1", 2.0, 8192)


@pytest.mark.parametrize(
    ("toml", "missing", "message"),
    [
        ("[environment]\ncpus = 1\n", None, "docker_image"),
        (TOML, "grader.py", "lacks grader.py"),
    ],
)
def test_task_settings_reject_other_layouts(tmp_path, toml, missing, message):
    t = task_dir(tmp_path, toml)
    if missing:
        (t / "tests" / missing).unlink()
    with pytest.raises(SystemExit, match=message):
        regrade.task_settings(t)


def test_container_is_isolated():
    task = regrade.Task("example/task:1", 2.0, 8192, Path("tests"))
    args = regrade.create_args(task, "c")
    assert args[args.index("--network") + 1] == "none"
    assert "no-new-privileges" in args and "--pids-limit" in args
    assert not {"-v", "--volume", "--mount", "--privileged"} & set(args)
    assert args[args.index("--cpus") + 1] == "2" and args[args.index("--memory") + 1] == "8192m"


def junit(cases: dict[str, str]) -> str:
    body = {"passed": "", "failed": "<failure/>", "skipped": "<skipped/>"}
    rows = "".join(
        f'<testcase classname="t" name="{n}">{body[o]}</testcase>' for n, o in cases.items()
    )
    return f"<testsuites><testsuite>{rows}</testsuite></testsuites>"


def test_outcomes_and_changes(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "new.xml").write_text(junit({"a": "passed", "b": "failed"}))
    (tmp_path / "reports" / "broken.xml").write_text("<testsuite")
    found = regrade.junit_outcomes(tmp_path)
    assert found == {"new:t.a": "passed", "new:t.b": "failed"}
    other = {"new:t.a": "failed", "new:t.c": "skipped"}
    assert regrade.changed(found, other) == [
        ("new:t.a", "passed", "failed"),
        ("new:t.b", "failed", "missing"),
        ("new:t.c", "missing", "skipped"),
    ]


# A fake docker: records argv, keeps the staged model.patch per container, and on
# `cp NAME:/logs/verifier DEST` writes a verifier result that depends on whether the
# staged patch still adds tests/conftest.py (the counterfactual under test).
FAKE = """#!{python}
import json, os, shutil, sys
from pathlib import Path
state = Path(os.environ["FAKE_STATE"]); state.mkdir(exist_ok=True)
args = sys.argv[1:]
with (state / "calls.jsonl").open("a") as f:
    f.write(json.dumps(args) + "\\n")
if args[0] == "cp" and ":" in args[2] and args[2].endswith(":/logs"):
    name = args[2].split(":")[0]
    shutil.copy(Path(args[1].rstrip(".")) / "artifacts" / "model.patch", state / name)
elif args[0] == "cp" and args[1].endswith(":/logs/verifier"):
    name = args[1].split(":")[0]
    ok = "tests/conftest.py" in (state / name).read_text()
    dest = Path(args[2]); (dest / "reports").mkdir(parents=True)
    reward = {{"reward": int(ok), "f2p_passed": int(ok), "f2p_total": 1}}
    (dest / "reward.json").write_text(json.dumps(reward))
    body = "" if ok else "<failure/>"
    (dest / "reports" / "new.xml").write_text(
        f'<testsuite><testcase classname="t" name="x">{{body}}</testcase></testsuite>')
elif args[0] == "start":
    print("[verifier] fake run")
"""


def fake_docker(tmp_path: Path) -> Path:
    exe = tmp_path / "docker"
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


def test_counterfactual_run_reports_the_changed_outcome(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_STATE", str(tmp_path / "state"))
    trial = tmp_path / "demo__abc"
    (trial / "artifacts").mkdir(parents=True)
    (trial / "artifacts" / "model.patch").write_text(PATCH)
    out = tmp_path / "out"
    argv = [str(task_dir(tmp_path)), str(trial), "--drop", "tests/conftest.py"]
    argv += ["--docker", str(fake_docker(tmp_path)), "--out", str(out), "--yes"]
    assert regrade.main(argv) == 0
    printed = capsys.readouterr().out
    assert 'control-1: {"reward": 1' in printed and 'drop-1: {"reward": 0' in printed
    assert "drop-1 vs control-1: 1 test outcome(s) changed" in printed
    assert "new:t.x: passed -> failed" in printed
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    assert (out / "control-1" / "verifier-stdout.txt").read_text().strip() == "[verifier] fake run"
    calls = [json.loads(line) for line in (tmp_path / "state" / "calls.jsonl").open()]
    creates = [c for c in calls if c[0] == "create"]
    assert len(creates) == 2 and all(c[c.index("--network") + 1] == "none" for c in creates)
    assert sum(c[0] == "rm" for c in calls) == 2  # every container removed
    staged_tests = [c for c in calls if c[0] == "cp" and c[2].endswith(":/tests")]
    assert all(Path(c[1].rstrip(".")).name == "tests" for c in staged_tests)
    assert not list(out.glob("*/tmp*"))  # staging removed


def test_refuses_without_consent(tmp_path, capsys):
    trial = tmp_path / "demo__abc"
    (trial / "artifacts").mkdir(parents=True)
    (trial / "artifacts" / "model.patch").write_text(PATCH)
    with pytest.raises(SystemExit):
        regrade.main([str(task_dir(tmp_path)), str(trial)])
    assert "pass --yes" in capsys.readouterr().err
