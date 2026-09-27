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
import datetime as dt
import hashlib
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path

RESOURCES = {"cpus", "memory_mb", "storage_mb", "gpus", "memory", "storage"}


def files(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".git" not in p.parts:
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def toml_changes(a: Path, b: Path) -> list[tuple[str, str]]:
    """(class, key) for each differing task.toml key."""
    try:
        old, new = tomllib.loads(a.read_text()), tomllib.loads(b.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return [("content", "task.toml (unparseable)")]
    out = []
    for section in sorted(set(old) | set(new)):
        x, y = old.get(section, {}), new.get(section, {})
        if not isinstance(x, dict) or not isinstance(y, dict):
            if x != y:
                out.append(("content", section))
            continue
        for key in sorted(set(x) | set(y)):
            if x.get(key) == y.get(key):
                continue
            if section == "metadata":
                kind = "docs"
            elif section == "environment":
                kind = "resources" if key in RESOURCES else "infrastructure"
            else:  # verifier, agent, anything else: grading or the agent's budget
                kind = "content"
            out.append((kind, f"[{section}].{key}"))
    return out


def classify(rel: str) -> str:
    top = rel.split("/", 1)[0]
    if rel == "README.md" or rel.endswith("/README.md") and top not in ("tests", "solution"):
        return "docs"
    if top == "environment":
        return "infrastructure"
    return "content"  # instruction.md, tests/, solution/, anything unexpected


def diff_tasks(canonical: Path, run: Path) -> dict:
    tasks = sorted(
        {p.name for p in canonical.iterdir() if p.is_dir()}
        | {p.name for p in run.iterdir() if p.is_dir()}
    )
    report = {"tasks": {}, "missing": [], "added": []}
    for task in tasks:
        a, b = canonical / task, run / task
        if not a.is_dir():
            report["added"].append(task)
            continue
        if not b.is_dir():
            report["missing"].append(task)
            continue
        fa, fb = files(a), files(b)
        changes = []
        for rel in sorted(set(fa) | set(fb)):
            if fa.get(rel) == fb.get(rel):
                continue
            if rel == "task.toml" and rel in fa and rel in fb:
                changes += toml_changes(a / rel, b / rel)
            else:
                changes.append(
                    (
                        classify(rel),
                        rel
                        + (
                            ""
                            if rel in fa and rel in fb
                            else " (added)"
                            if rel in fb
                            else " (removed)"
                        ),
                    )
                )
        if changes:
            report["tasks"][task] = changes
    return report


def _time(value: str | None) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None
    except ValueError:
        return None


def overruns(job: Path, run: Path) -> dict:
    """Trials whose agent phase outlasted the (run's own) task agent timeout."""
    limits = {}
    for toml in run.glob("*/task.toml"):
        try:
            limits[toml.parent.name] = float(
                tomllib.loads(toml.read_text())["agent"]["timeout_sec"]
            )
        except (KeyError, ValueError, TypeError, tomllib.TOMLDecodeError):
            pass
    trials = rewarded = over = over_rewarded = 0
    per_task: Counter = Counter()
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
    return 1 if kinds.get("content") or report["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
