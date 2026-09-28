"""Compare two directories of `atif-scan --format json` reports, check by check.

Scan the same inputs before and after a change (same flags, `--no-cache`), one report per
run in each directory, with matching file names, then:

    python tools/report_diff.py BEFORE_DIR AFTER_DIR [--totals]

Prints check IDs and counts only (status transitions, evidence-count changes, match
totals), never input labels or trace text, so the output is safe to paste into a review.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

Key = tuple[str, str, str]  # (report file, input id, check id)


def load(directory: Path) -> dict[Key, tuple]:
    out = {}
    for path in sorted(directory.glob("*.json")):
        doc = json.loads(path.read_text())
        if "inputs" not in doc:  # a --brief/--overview report
            continue
        for item in doc["inputs"]:
            for a in item["assessments"]:
                key = (path.name, item["input_id"], a["id"])
                out[key] = (a["status"], a["complete"], len(a["evidence"]), bool(a["expected_by"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--totals", action="store_true", help="also print match totals")
    args = parser.parse_args()
    before, after = load(args.before), load(args.after)
    print(f"assessments: {len(before)} before, {len(after)} after")
    changes: Counter[tuple] = Counter()
    for key in before.keys() | after.keys():
        old, new = before.get(key), after.get(key)
        if old != new:
            what = "evidence/expected" if old and new and old[0] == new[0] else ""
            changes[(key[2], old and old[0], new and new[0], what)] += 1
    for (check, old, new, what), n in sorted(changes.items()):
        print(f"{n:7d}  {check:55s} {old} -> {new} {what}")
    if not changes:
        print("identical")
    if args.totals:
        totals = [Counter(k[2] for k, v in d.items() if v[0] == "match") for d in (before, after)]
        for check in sorted(totals[0].keys() | totals[1].keys()):
            print(f"{check:55s} {totals[0][check]:7d} {totals[1][check]:7d}")


if __name__ == "__main__":
    main()
