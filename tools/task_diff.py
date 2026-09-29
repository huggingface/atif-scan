"""Check that a forked task set only makes infrastructure replacements, and what the
run's timeout overrides bought.

    uv run python tools/task_diff.py CANONICAL_TASKS RUN_TASKS [--job JOB_DIR] [--json]

CANONICAL_TASKS and RUN_TASKS are task folders (`<task>/instruction.md`, `task.toml`,
`environment/`, `tests/`, `solution/`), e.g. checkouts of the benchmark repo and of the
fork the run used (the job's `config.json` names the repo and commit). Every differing file
is classed as:

- **content**: `instruction.md`, `tests/`, `solution/`, and `task.toml` `[verifier]` and
  `[agent]` settings. It changes what's asked or how it's graded. Exit status 1.
- **resources**: `task.toml` `[environment]` cpus/memory/storage/gpus.
- **infrastructure**: `environment/` files (Dockerfile, build context) and `task.toml`
  `[environment]` image and build settings. It changes how the environment is built.
  Review it: a Dockerfile can also add files to `/app`.
- **docs**: `README.md` and `task.toml` `[metadata]`.

With `--job`, the Harbor job folder's trials are checked against RUN_TASKS' own agent
timeouts. The report says how many ran past them (possible only with an agent timeout
override), how many of those were rewarded, and the accuracy if they had counted as
failures. Only counts and task names are printed, never file contents.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atif_scan.jsonval import Doc

RESOURCES = {"cpus", "memory_mb", "storage_mb", "gpus", "memory", "storage"}
Change = tuple[str, str]  # (class, file or task.toml key)


def files(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".git" not in p.parts:
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def key_kind(section: str, key: str) -> str:
    """The class of a changed `[section].key` in task.toml."""
    if section == "metadata":
        return "docs"
    if section == "environment":
        return "resources" if key in RESOURCES else "infrastructure"
    return "content"  # verifier, agent, anything else: grading or the agent's budget


def section_changes(section: str, x: object, y: object) -> list[Change]:
    """(class, key) for each differing key of one task.toml section (or top-level value)."""
    if not isinstance(x, dict) or not isinstance(y, dict):
        return [("content", section)] if x != y else []
    return [
        (key_kind(section, key), f"[{section}].{key}")
        for key in sorted(set(x) | set(y))
        if x.get(key) != y.get(key)
    ]


def toml_changes(a: Path, b: Path) -> list[Change]:
    """(class, key) for each differing task.toml key."""
    try:
        old, new = tomllib.loads(a.read_text()), tomllib.loads(b.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return [("content", "task.toml (unparseable)")]
    out: list[Change] = []
    for section in sorted(set(old) | set(new)):
        out += section_changes(section, old.get(section, {}), new.get(section, {}))
    return out


def classify(rel: str) -> str:
    top = rel.split("/", 1)[0]
    if rel == "README.md" or (rel.endswith("/README.md") and top not in ("tests", "solution")):
        return "docs"
    if top == "environment":
        return "infrastructure"
    return "content"  # instruction.md, tests/, solution/, anything unexpected


def presence(rel: str, fa: dict[str, str], fb: dict[str, str]) -> str:
    """ "" for a file in both task folders, else " (added)" or " (removed)"."""
    if rel in fa and rel in fb:
        return ""
    return " (added)" if rel in fb else " (removed)"


def task_changes(a: Path, b: Path) -> list[Change]:
    """(class, what) for each file (or task.toml key) that differs between two tasks."""
    fa, fb = files(a), files(b)
    changes: list[Change] = []
    for rel in sorted(set(fa) | set(fb)):
        if fa.get(rel) == fb.get(rel):
            continue
        if rel == "task.toml" and rel in fa and rel in fb:
            changes += toml_changes(a / rel, b / rel)
        else:
            changes.append((classify(rel), rel + presence(rel, fa, fb)))
    return changes


def diff_tasks(canonical: Path, run: Path) -> Doc:
    tasks = sorted(
        {p.name for p in canonical.iterdir() if p.is_dir()}
        | {p.name for p in run.iterdir() if p.is_dir()}
    )
    report: Doc = {"tasks": {}, "missing": [], "added": []}
    for task in tasks:
        a, b = canonical / task, run / task
        if not a.is_dir():
            report["added"].append(task)
        elif not b.is_dir():
            report["missing"].append(task)
        elif changes := task_changes(a, b):
            report["tasks"][task] = changes
    return report


def _time(value: str | None) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None
    except ValueError:
        return None


def overruns(job: Path, run: Path) -> Doc:
    """Trials whose agent phase outlasted the (run's own) task agent timeout."""
    limits: dict[str, float] = {}
    for toml in run.glob("*/task.toml"):
        with contextlib.suppress(KeyError, ValueError, TypeError, tomllib.TOMLDecodeError):
            limits[toml.parent.name] = float(
                tomllib.loads(toml.read_text())["agent"]["timeout_sec"]
            )
    trials = rewarded = over = over_rewarded = 0
    per_task: Counter[str] = Counter()
    for result in job.glob("*/result.json"):
        try:
            r = json.loads(result.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        task = (r.get("task_name") or result.parent.name.rsplit("__", 1)[0]).rsplit("/", 1)[-1]
        rewards = (r.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward") if isinstance(rewards, dict) else None
        trials += 1
        ok = isinstance(reward, (int, float)) and reward > 0
        rewarded += ok
        ae = r.get("agent_execution") or {}
        start, end = _time(ae.get("started_at")), _time(ae.get("finished_at"))
        if task in limits and start and end and (end - start).total_seconds() > limits[task]:
            over += 1
            over_rewarded += ok
            per_task[task] += 1
    return {
        "trials": trials,
        "rewarded": rewarded,
        "over_timeout": over,
        "over_timeout_rewarded": over_rewarded,
        "accuracy": round(100 * rewarded / trials, 2) if trials else None,
        "accuracy_if_overruns_failed": (
            round(100 * (rewarded - over_rewarded) / trials, 2) if trials else None
        ),
        "tasks_over": dict(per_task.most_common()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("canonical", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--job", type=Path, help="Harbor job folder that used RUN_TASKS")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = diff_tasks(args.canonical, args.run)
    if args.job:
        report["timeouts"] = overruns(args.job, args.run)
    kinds = Counter(k for changes in report["tasks"].values() for k, _ in changes)
    report["summary"] = dict(kinds)
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        print_text(report, kinds)
    return 1 if kinds.get("content") or report["missing"] else 0


def print_text(report: Doc, kinds: Counter[str]) -> None:
    shown = " · ".join(f"{k} {v}" for k, v in sorted(kinds.items())) or "no differences"
    print(f"{len(report['tasks'])} task(s) differ · {shown}")
    for task, changes in report["tasks"].items():
        print(f"  {task}")
        for kind, what in changes:
            print(f"    {kind:<14} {what}")
    for key in ("missing", "added"):
        if report[key]:
            print(f"  {key}: {', '.join(report[key])}")
    t = report.get("timeouts")
    if t:
        print(
            f"timeouts: {t['over_timeout']} of {t['trials']} trials ran past their task's "
            f"agent timeout ({t['over_timeout_rewarded']} rewarded) · accuracy "
            f"{t['accuracy']}% → {t['accuracy_if_overruns_failed']}% if those had failed"
        )


if __name__ == "__main__":
    sys.exit(main())
