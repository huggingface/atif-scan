"""Cheap layout inspection: what's under an input, from file names and sizes only.

No trace is read. The same listing and selection code as a scan is used, so "would
scan" is exactly what a scan would read. Harbor markers follow harbor 0.23:

    job/   job.log config.json lock.json result.json <trial>/...
    trial/ trial.log config.json lock.json result.json [exception.txt]
           agent/trajectory.json   user-agent/trajectory.json (simulated user)
           verifier/reward.{txt,json}   steps/<name>/{agent,verifier}/  (multi-step)

Recognition is heuristic ("looks like"), and output is limited to counts, relative
paths (the same strings a scan uses as labels) and file names.
"""

from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath

from .checks import identifier
from .loader import MAX_BYTES
from .sources import Entry, Listing, label_for, selected

EXAMPLES = 20


def _children(entries: tuple[Entry, ...]) -> dict[str, set[str]]:
    """Directory ("" is the root) -> names of its files and subdirectories."""
    tree: dict[str, set[str]] = {"": set()}
    for entry in entries:
        parts = entry.path.split("/")
        for depth in range(len(parts)):
            parent = "/".join(parts[:depth])
            tree.setdefault(parent, set()).add(parts[depth])
    return tree


def _is_trial(names: set[str]) -> bool:
    return "trial.log" in names or (
        {"config.json", "result.json"} <= names and bool(names & {"agent", "verifier"})
    )


def _parents(path: str):
    parts = path.split("/")[:-1]
    for depth in range(len(parts), -1, -1):
        yield "/".join(parts[:depth])


def _within(path: str, folder: str) -> str:
    return path[len(folder) + 1 :] if folder else path


def _role(relative: str) -> str:
    parts = relative.split("/")
    if parts[0] == "agent":
        return "agent"
    if parts[0] == "user-agent":
        return "simulated_user"
    if parts[0] == "steps" and len(parts) >= 3:
        return {"agent": "step_agent", "user-agent": "simulated_user"}.get(
            parts[2], "other_in_trial"
        )
    return "other_in_trial"


def _name(value: str) -> str:
    try:
        return identifier(value)
    except ValueError:
        return "<unlisted-name>"


def _folder(path: str) -> str:
    return _name(path) if path else "."


def inspect_listing(listing: Listing, pattern: str, number: int) -> dict:
    entries = listing.entries
    chosen = selected(listing, pattern)
    chosen_paths = {e.path for e in chosen}
    tree = _children(entries) if listing.directory else {"": set()}
    trials = {d for d, names in tree.items() if _is_trial(names)}
    trial_parents = {t.rpartition("/")[0] for t in trials if t}
    jobs = {
        d
        for d, names in tree.items()
        if "job.log" in names or ({"config.json", "result.json"} <= names and d in trial_parents)
    } - trials

    def label(entry: Entry) -> str:
        return label_for(entry, pattern) or (_name(entry.path) if entry.path else "<file>")

    roles: Counter[str] = Counter()
    per_trial: dict[str, list[Entry]] = {t: [] for t in trials}
    by_role: dict[str, list[str]] = {}
    for entry in chosen:
        trial = next((d for d in _parents(entry.path) if d in trials), None)
        if trial is None:
            role = "trajectory"
        else:
            role = _role(_within(entry.path, trial))
            per_trial[trial].append(entry)
        roles[role] += 1
        by_role.setdefault(role, []).append(label(entry))

    anomalies: list[dict] = []

    def flag(code: str, examples: list[str]) -> None:
        if examples:
            anomalies.append(
                {"code": code, "count": len(examples), "examples": examples[:EXAMPLES]}
            )

    if not chosen:
        flag("no_files_match_pattern", ["<none>"])
    flag("simulated_user_trajectories_would_be_scanned", by_role.get("simulated_user", []))
    flag(
        "trials_without_agent_trajectory",
        sorted(
            _folder(t)
            for t, found in per_trial.items()
            if all(_role(_within(e.path, t)) == "simulated_user" for e in found)
        ),
    )
    flag(
        "trials_with_multiple_scanned_trajectories",
        sorted(_folder(t) for t, found in per_trial.items() if len(found) > 1),
    )
    loose = [e for e in chosen if not any(d in trials for d in _parents(e.path))]
    depths = Counter(e.path.count("/") for e in loose)
    if len(depths) > 1:
        usual = depths.most_common(1)[0][0]
        flag("mixed_depths", [label(e) for e in loose if e.path.count("/") != usual])
    flag("oversize_files", [label(e) for e in chosen if (e.size or 0) > MAX_BYTES])
    if listing.directory:
        flag("positional_labels", [_name(e.path) for e in chosen if label_for(e, pattern) is None])
    flag(
        "unscanned_trajectory_like_files",
        [
            _name(e.path)
            for e in entries
            if e.path not in chosen_paths
            and "trajectory" in PurePosixPath(e.path).name
            and e.path.endswith(".json")
        ],
    )
    flag("trials_with_exception", sorted(_folder(t) for t in trials if "exception.txt" in tree[t]))
    other = Counter(
        _name(PurePosixPath(e.path).name) for e in entries if e.path not in chosen_paths
    )
    # Files that sit next to scanned trajectories, and in how many of those folders.
    folders = {PurePosixPath(e.path).parent.as_posix() for e in chosen if e.path}
    scanned_names = {PurePosixPath(e.path).name for e in chosen}
    companions: Counter[str] = Counter()
    for folder in folders:
        folder = "" if folder == "." else folder
        for name in tree.get(folder, set()) - scanned_names:
            companions[name + "/" if _join(folder, name) in tree else name] += 1
    sizes = [e.size for e in entries]
    if not listing.directory:
        layout = "single_file"
    elif jobs:
        layout = "harbor_jobs"
    elif trials and loose:
        layout = "mixed"
    elif trials:
        layout = "harbor_trials"
    elif chosen:
        layout = "trajectory_folders"
    else:
        layout = "no_trajectories"
    return {
        "input": number,
        "remote": listing.remote,
        "directory": listing.directory,
        "files": len(entries),
        "bytes": None if None in sizes else sum(sizes),
        "layout": layout,
        "harbor": {
            "jobs": len(jobs),
            "trials": len(trials),
            "trials_with_reward": sum(
                bool({"reward.txt", "reward.json"} & tree.get(_join(t, "verifier"), set()))
                for t in trials
            ),
            "multi_step_trials": sum("steps" in tree[t] for t in trials),
        }
        if trials
        else None,
        "would_scan": {"total": len(chosen), "by_role": dict(sorted(roles.items()))},
        "anomalies": anomalies,
        "folders": len(folders),
        "alongside": {_name(k): v for k, v in companions.most_common(10)},
        "other_files": dict(other.most_common(10)),
        "other_file_count": sum(other.values()),
    }


def _join(folder: str, name: str) -> str:
    return f"{folder}/{name}" if folder else name


def document(listings: list[Listing], pattern: str) -> dict:
    return {
        "schema_version": 1,
        "kind": "inspection",
        "pattern": pattern,
        "inputs": [inspect_listing(x, pattern, i) for i, x in enumerate(listings, 1)],
    }
