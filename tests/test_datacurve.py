"""Datacurve's public DeepSWE mirrors: the trial index, and Harbor's cut trial-folder names.

Regressions from scanning a mirror of Datacurve's public DeepSWE v1.1 trials:

- Harbor names a trial folder `{task[:32].rstrip("_-")}__<7>`, so 41 of DeepSWE's 113
  tasks have a cut name in the folder. `--task-from trial-dir` took the cut name as the
  task: the DeepSWE pack wasn't recognised (63% of folders named a known task, below
  the 90% share) and, loaded by hand, its checks said `not_applicable`, which reads as
  clean, instead of applying.
- Datacurve's `trials.json` index (task, reward, error, cost per trial) wasn't read, so
  every trial was "task unknown", "reward unknown" and `integrity.cost_missing`.

Synthetic fixtures only.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from test_sync import FS, trajectory

from atif_scan import packs
from atif_scan.cli import main
from atif_scan.packs import Pack, harbor_folder_prefix, untruncated_task
from atif_scan.packs.deepswe import TASK_NAMES
from atif_scan.sources.datacurve import trial_index
from atif_scan.sources.sync import sync_remote

if TYPE_CHECKING:
    from pathlib import Path

    from atif_scan.data.jsonval import Doc

LONG = "csstree-shorthand-expansion-compression"  # 39 characters
SHORT = "abs-module-cache-flags"
SECRET = "SENTINEL_FREE_TEXT"


def test_harbor_folder_prefix_matches_harbor():
    assert harbor_folder_prefix(LONG) == "csstree-shorthand-expansion-comp"
    assert harbor_folder_prefix("a" * 31 + "-tail") == "a" * 31  # trailing "-" stripped
    assert harbor_folder_prefix(SHORT) == SHORT


def test_every_cut_deepswe_task_is_restored():
    cut = {t for t in TASK_NAMES if harbor_folder_prefix(t) != t}
    assert len(cut) == 41
    for task in TASK_NAMES:
        assert untruncated_task(harbor_folder_prefix(task)) == task


def test_unknown_or_ambiguous_cut_names(monkeypatch):
    assert untruncated_task("some-other-benchmark-task") == "some-other-benchmark-task"
    assert untruncated_task("x" * 40) == "x" * 40  # longer than a cut name: as it is
    twins = frozenset({"p" * 32 + "-alpha", "p" * 32 + "-beta"})
    monkeypatch.setattr(packs, "BUNDLED", (Pack("twins", "m:f", tasks=lambda: twins),))
    assert untruncated_task("p" * 32) is None  # two tasks cut to it: unknown, not a guess


def trial_dir(root: Path, name: str, command: str = "ls") -> None:
    (root / name / "agent").mkdir(parents=True)
    (root / name / "agent" / "trajectory.json").write_bytes(trajectory(command))


def scan(capsys, *argv: str) -> Doc:
    main([*argv, "--format", "json", "--no-cache"])
    return json.loads(capsys.readouterr().out)


def test_task_from_trial_dir_restores_cut_names_and_loads_the_pack(tmp_path, capsys):
    trial_dir(tmp_path, f"{harbor_folder_prefix(LONG)}__AbCdEf1", "git log --oneline")
    trial_dir(tmp_path, f"{SHORT}__AbCdEf2", "git log --oneline")
    doc = scan(capsys, str(tmp_path), "--task-from", "trial-dir")
    assert sorted(i["task"] for i in doc["inputs"]) == sorted([LONG, SHORT])
    assert {"pack": "deepswe", "reason": "tasks"} in doc["packs"]
    for item in doc["inputs"]:
        found = {a["id"]: a["status"] for a in item["assessments"]}
        assert found["deepswe.upstream_lookup"] != "not_applicable"
        assert found["expected.deepswe.git_history_scrubbed"] == "match"


def row(trial: str, task: str, reward: float | None, **extra: object) -> Doc:
    return {
        "trial_name": trial,
        "task_name": task,
        "source": "deep-swe",
        "harness": "mini-swe-agent",
        "config": "mini_swe_agent_demo_high",
        "reward": reward,
        "errored": False,
        "error_category": None,
        "exception": f"Traceback {SECRET}",
        "critique": SECRET,
        "note": SECRET,
        "cost_usd": 0.25,
        "n_input_tokens": 1000,
        "n_cache_tokens": 800,
        "n_output_tokens": 50,
        "started_at": "2026-06-12T11:33:53.691464Z",
        "finished_at": "2026-06-12T11:35:33.691464Z",
        "agent_duration_seconds": 90.5,
        **extra,
    }


def index(*rows: Doc) -> bytes:
    return json.dumps({"scope": "demo", "n_trials": len(rows), "rows": list(rows)}).encode()


def test_trial_index_keeps_allowlisted_facts_only():
    cut = f"{harbor_folder_prefix(LONG)}__AbCdEf1"
    data = index(
        row(cut, LONG, 1.0),
        row("t__2", SHORT, None, errored=True, error_category="agent_timeout"),
        row("t__3", SHORT, None, errored=True),
        row("dup__4", SHORT, 1.0),
        row("dup__4", SHORT, 0.0),
        row("../escape", SHORT, 1.0),
    )
    run, trials = trial_index(data)
    assert set(trials) == {cut, "t__2", "t__3"}  # repeated and unsafe names dropped
    assert trials[cut] == {
        "task": LONG,
        "reward": 1.0,
        "cost_usd": 0.25,
        "input_tokens": 1000,
        "cache_tokens": 800,
        "output_tokens": 50,
        "duration_sec": 100.0,
        "agent_duration_sec": 90.5,
        "configured_agent": "mini-swe-agent",
        "configured_model": "mini_swe_agent_demo_high",
    }
    assert trials["t__2"]["error_type"] == "agent_timeout"
    assert trials["t__3"]["error_type"] == "other"
    assert run is not None
    assert run["source"] == "datacurve_index" and run["datasets"] == ["deep-swe"]
    assert run["job_name"] == "mini_swe_agent_demo_high" and run["listed_trials"] == 3
    assert SECRET not in json.dumps([run, trials])


def test_only_mirrored_configs_and_only_indexes():
    data = index(
        row("a__1", SHORT, 1.0),
        row("a__2", SHORT, 1.0),  # same config, not mirrored: still part of the run
        row("b__3", SHORT, 0.0, config="other_config"),
    )
    run, trials = trial_index(data, {"a__1"})
    assert set(trials) == {"a__1", "a__2"} and run and run["listed_trials"] == 2
    assert trial_index(data, set()) == (None, {})
    for junk in (b"", b"\xff", b"[]", b'{"rows": []}', json.dumps({"trials": []}).encode()):
        assert trial_index(junk) == (None, {})


def mirror(root: Path) -> None:
    """`<config>/trials.json` beside `<config>/<trial>/agent/trajectory.json`, as a mirror
    of Datacurve's artifacts lays them out."""
    cut = f"{harbor_folder_prefix(LONG)}__AbCdEf1"
    rows = [row(cut, LONG, 1.0), row(f"{SHORT}__AbCdEf2", SHORT, 0.0, cost_usd=0.75)]
    # Published without a trajectory (Datacurve has such rows), and another config's.
    rows.append(row(f"{SHORT}__AbCdEf3", SHORT, 1.0, has_trajectory=False))
    rows.append(row(f"{SHORT}__AbCdEf4", SHORT, 1.0, config="other_config"))
    root.mkdir(parents=True)
    (root / "trials.json").write_bytes(index(*rows))
    trial_dir(root, cut, "git log --oneline")
    trial_dir(root, f"{SHORT}__AbCdEf2")


def test_scan_of_a_mirror_uses_the_index(tmp_path, capsys):
    mirror(tmp_path / "run")
    doc = scan(capsys, str(tmp_path / "run"))
    available = [i for i in doc["inputs"] if i["input_status"] == "available"]
    items = {i["task"]: i for i in available}
    assert set(items) == {LONG, SHORT}  # the index's full names, no --task-from
    # The run's trial without a trajectory is reported unavailable, not dropped.
    (gone,) = [i for i in doc["inputs"] if i["input_status"] != "available"]
    assert gone["input_id"] == f"{SHORT}__AbCdEf3/agent"
    assert (gone["task"], gone["reward"]) == (SHORT, 1.0)
    assert len(doc["inputs"]) == 3  # other_config's row isn't this run's
    assert (items[LONG]["reward"], items[LONG]["cost_usd"]) == (1.0, 0.25)
    assert (items[SHORT]["reward"], items[SHORT]["cost_usd"]) == (0.0, 0.75)
    assert items[LONG]["agent_duration_sec"] == 90.5
    assert {"pack": "deepswe", "reason": "dataset"} in doc["packs"]
    (run,) = doc["runs"]
    assert run["source"] == "datacurve_index" and run["listed_trials"] == 3
    for item in available:
        found = {a["id"]: a["status"] for a in item["assessments"]}
        assert found["context.rewarded"] in ("match", "no_match")  # reward known
    out = json.dumps(doc)
    assert SECRET not in out

    main([str(tmp_path / "run"), "--format", "text", "--no-cache"])
    text = capsys.readouterr().out
    assert "tasks unknown" not in text and "rewards unknown" not in text
    assert "Datacurve index · mini_swe_agent_demo_high" in text
    assert "COST       $1.25 recorded" in text  # the index's cost, not "missing"


def test_a_recorded_task_beats_the_folder_name(tmp_path, capsys):
    mirror(tmp_path / "run")
    doc = scan(capsys, str(tmp_path / "run"), "--task-from", "trial-dir")
    assert sorted(i["task"] for i in doc["inputs"]) == sorted([LONG, SHORT, SHORT])


def test_remote_mirror_syncs_the_index(tmp_path, capsys):
    root = "buckets/o/b/deepswe"
    fs = FS(
        {
            f"{root}/trials.json": index(row("t__1", SHORT, 1.0)),
            f"{root}/t__1/agent/trajectory.json": trajectory(),
        }
    )
    dest = tmp_path / "copy"
    _, counts = sync_remote(f"hf://{root}", dest, fs=fs)
    assert counts["downloaded"] == 2 and (dest / "trials.json").is_file()
    (item,) = scan(capsys, str(dest))["inputs"]
    assert (item["task"], item["reward"]) == (SHORT, 1.0)
