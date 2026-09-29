"""Turn the TB2.1 judge's rulings into labels on public trials (the atif-scan eval set).

    uv run python tools/tb21_labels.py INVENTORY_DIR      # after tools/tb21_inventory.py

The judge (CI) reviews every rewarded trial of a submission and lists the suspicious ones
in a PR comment; maintainers tick those they disqualify. Those rows point at CI-owned
clones, usually private. The public copies are the Hub leaderboard row's trials (or a
visible clone job), so rulings are mapped onto them:

- a visible clone job: by trial id (exact);
- a leaderboard row: by task. The judge only reviews rewarded trials, so when a task's
  judged count equals its rewarded trials in the row and the rulings agree, every one of
  them carries that ruling (exact); otherwise the task carries counts only.

Writes INVENTORY_DIR/labels.json: one record per run with its public job ids and, per
task, the rewarded trial names, the disqualified/cleared counts and exact labels.
Reads the Hub through the user's `harbor` CLI; prints counts only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from atif_scan.harbor_hub import HarborCLI, listing, row_listing

if TYPE_CHECKING:
    from atif_scan.harbor_hub import RowJobs
    from atif_scan.jsonval import Doc, JsonObject


def slug(task: object) -> str:
    return str(task or "").rsplit("/", 1)[-1]


def rewarded(rows: list[JsonObject]) -> dict[str, list[JsonObject]]:
    by_task: dict[str, list[JsonObject]] = defaultdict(list)
    for row in rows:
        reward = row.get("reward")
        if isinstance(reward, int | float) and reward > 0:
            by_task[slug(row.get("task_name"))].append(row)
    return by_task


def label_run(judged: list[Doc], jobs: RowJobs, exact_ids: bool) -> dict[str, Doc]:
    by_id = {str(r["id"]).lower(): r for _, rows in jobs for r in rows}
    wins = rewarded([r for _, rows in jobs for r in rows])
    tasks: dict[str, Doc] = {}
    for task in sorted({slug(j["task"]) for j in judged} | set(wins)):
        rulings = [j for j in judged if slug(j["task"]) == task]
        dq, cleared = (
            sum(j["disqualified"] for j in rulings),
            sum(not j["disqualified"] for j in rulings),
        )
        exact: dict[str, str] = {}
        if exact_ids:
            for j in rulings:
                if (row := by_id.get(j["clone_trial"])) is not None:
                    exact[str(row.get("name"))] = "dq" if j["disqualified"] else "cleared"
        elif rulings and len(rulings) == len(wins.get(task, [])) and (dq == 0 or cleared == 0):
            exact = {str(r.get("name")): "dq" if dq else "cleared" for r in wins[task]}
        tasks[task] = {
            "rewarded": sorted(str(r.get("name")) for r in wins.get(task, [])),
            "dq": dq,
            "cleared": cleared,
            "exact": exact,
        }
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inventory", type=Path)
    args = parser.parse_args()
    inv = json.loads((args.inventory / "inventory.json").read_text())
    rows_by_pr = {r["pr"]: r for r in inv["rows"] if r["pr"] is not None}
    cli = HarborCLI.find()
    runs: list[Doc] = []
    for pr in inv["prs"]:
        judged = pr["judged"]
        if not judged:
            continue
        clones = sorted({j["clone_job"] for j in judged})
        visible = [c for c in clones if inv["jobs"].get(c, {}).get("name")]
        if visible:
            jobs = [(job, listing(cli, job)[1]) for job in visible]
            source, exact_ids = "clone_job", True
        elif pr["number"] in rows_by_pr:
            _, jobs = row_listing(cli, rows_by_pr[pr["number"]]["id"])
            source, exact_ids = "leaderboard_row", False
        else:
            runs.append(
                {
                    "pr": pr["number"],
                    "state": pr["state"],
                    "source": None,
                    "judged": len(judged),
                    "dq": sum(j["disqualified"] for j in judged),
                }
            )
            continue
        tasks = label_run(judged, jobs, exact_ids)
        runs.append(
            {
                "pr": pr["number"],
                "state": pr["state"],
                "source": source,
                "row": rows_by_pr.get(pr["number"], {}).get("id"),
                "jobs": sorted({job for job, _ in jobs}),
                "judged": len(judged),
                "dq": sum(j["disqualified"] for j in judged),
                "tasks": tasks,
            }
        )
        print(
            f"#{pr['number']} {source}: {len(judged)} judged, "
            f"{sum(len(t['exact']) for t in tasks.values())} trials labelled exactly",
            file=sys.stderr,
        )
    (args.inventory / "labels.json").write_text(json.dumps(runs, indent=1))

    mapped = [r for r in runs if r["source"]]
    exact = Counter(v for r in mapped for t in r["tasks"].values() for v in t["exact"].values())
    print(
        f"runs with rulings {len(runs)} · mapped to public trials {len(mapped)} "
        f"({sum(r['source'] == 'clone_job' for r in mapped)} clone jobs, "
        f"{sum(r['source'] == 'leaderboard_row' for r in mapped)} rows)"
    )
    print(
        f"rulings: {sum(r['judged'] for r in runs)} ({sum(r['dq'] for r in runs)} DQ) · on mapped "
        f"runs {sum(r['judged'] for r in mapped)} ({sum(r['dq'] for r in mapped)} DQ)"
    )
    print(f"exact trial labels: {exact.get('dq', 0)} dq, {exact.get('cleared', 0)} cleared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
