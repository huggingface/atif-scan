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
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ..data.jsonval import identifier
from ..data.loader import MAX_BYTES
from .inputs import Entry, Listing, label_for, selected

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..data.jsonval import Doc

EXAMPLES = 20
TOP = 10  # most common alongside/other file names listed
OTHER_ROLE = "other_in_trial"
# Trajectory roles by the trial subfolder, and in multi-step trials steps/<name>/<subfolder>.
TRIAL_ROLES = {"agent": "agent", "user-agent": "simulated_user"}
STEP_ROLES = {"agent": "step_agent", "user-agent": "simulated_user"}


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


def _parents(path: str) -> Iterator[str]:
    parts = path.split("/")[:-1]
    for depth in range(len(parts), -1, -1):
        yield "/".join(parts[:depth])


def _within(path: str, folder: str) -> str:
    return path[len(folder) + 1 :] if folder else path


def _role(relative: str) -> str:
    parts = relative.split("/")
    if parts[0] == "steps":
        return STEP_ROLES.get("/".join(parts[2:3]), OTHER_ROLE)
    return TRIAL_ROLES.get(parts[0], OTHER_ROLE)


def _name(value: str) -> str:
    try:
        return identifier(value)
    except ValueError:
        return "<unlisted-name>"


def _folder(path: str) -> str:
    return _name(path) if path else "."


@dataclass
class _View:
    """One listing as a scan sees it: the entries it would read and the Harbor folders."""

    listing: Listing
    pattern: str
    chosen: list[Entry] = field(init=False)
    tree: dict[str, set[str]] = field(init=False)
    trials: set[str] = field(init=False)
    jobs: set[str] = field(init=False)
    # Scanned entries outside any trial folder.
    loose: list[Entry] = field(init=False)

    def __post_init__(self) -> None:
        self.chosen = selected(self.listing, self.pattern)
        self.tree = _children(self.listing.entries) if self.listing.directory else {"": set()}
        self.trials = {d for d, names in self.tree.items() if _is_trial(names)}
        trial_parents = {t.rpartition("/")[0] for t in self.trials if t}
        self.jobs = {
            d
            for d, names in self.tree.items()
            if "job.log" in names
            or ({"config.json", "result.json"} <= names and d in trial_parents)
        } - self.trials
        self.loose = [e for e in self.chosen if not any(d in self.trials for d in _parents(e.path))]

    def label(self, entry: Entry) -> str:
        return label_for(entry, self.pattern) or (_name(entry.path) if entry.path else "<file>")

    def trial_of(self, entry: Entry) -> str | None:
        return next((d for d in _parents(entry.path) if d in self.trials), None)


def _roles(view: _View) -> tuple[Counter[str], dict[str, list[Entry]], dict[str, list[str]]]:
    """Scanned trajectories per role, per trial folder, and role -> labels."""
    roles: Counter[str] = Counter()
    per_trial: dict[str, list[Entry]] = {t: [] for t in view.trials}
    by_role: dict[str, list[str]] = {}
    for entry in view.chosen:
        trial = view.trial_of(entry)
        if trial is None:
            role = "trajectory"
        else:
            role = _role(_within(entry.path, trial))
            per_trial[trial].append(entry)
        roles[role] += 1
        by_role.setdefault(role, []).append(view.label(entry))
    return roles, per_trial, by_role


def _mixed_depths(view: _View) -> list[str]:
    depths = Counter(e.path.count("/") for e in view.loose)
    if len(depths) <= 1:
        return []
    usual = depths.most_common(1)[0][0]
    return [view.label(e) for e in view.loose if e.path.count("/") != usual]


def _anomalies(
    view: _View, per_trial: dict[str, list[Entry]], by_role: dict[str, list[str]]
) -> list[Doc]:
    chosen, pattern = view.chosen, view.pattern
    chosen_paths = {e.path for e in chosen}
    found = [
        ("no_files_match_pattern", [] if chosen else ["<none>"]),
        ("simulated_user_trajectories_would_be_scanned", by_role.get("simulated_user", [])),
        (
            "trials_without_agent_trajectory",
            sorted(
                _folder(t)
                for t, entries in per_trial.items()
                if all(_role(_within(e.path, t)) == "simulated_user" for e in entries)
            ),
        ),
        (
            "trials_with_multiple_scanned_trajectories",
            sorted(_folder(t) for t, entries in per_trial.items() if len(entries) > 1),
        ),
        ("mixed_depths", _mixed_depths(view)),
        ("oversize_files", [view.label(e) for e in chosen if (e.size or 0) > MAX_BYTES]),
        (
            "positional_labels",
            [_name(e.path) for e in chosen if label_for(e, pattern) is None]
            if view.listing.directory
            else [],
        ),
        (
            "unscanned_trajectory_like_files",
            [
                _name(e.path)
                for e in view.listing.entries
                if e.path not in chosen_paths
                and "trajectory" in PurePosixPath(e.path).name
                and e.path.endswith(".json")
            ],
        ),
        (
            "trials_with_exception",
            sorted(_folder(t) for t in view.trials if "exception.txt" in view.tree[t]),
        ),
    ]
    return [
        {"code": code, "count": len(examples), "examples": examples[:EXAMPLES]}
        for code, examples in found
        if examples
    ]


def _companions(view: _View) -> tuple[int, Counter[str]]:
    """Folders holding scanned trajectories, and the names sitting next to them (with a
    trailing / for folders) counted by how many of those folders hold them."""
    folders = {PurePosixPath(e.path).parent.as_posix() for e in view.chosen if e.path}
    scanned_names = {PurePosixPath(e.path).name for e in view.chosen}
    companions: Counter[str] = Counter()
    for parent in folders:
        folder = "" if parent == "." else parent
        for name in view.tree.get(folder, set()) - scanned_names:
            companions[name + "/" if _join(folder, name) in view.tree else name] += 1
    return len(folders), companions


def _layout(view: _View) -> str:
    """The first layout that fits, most specific first."""
    fits = (
        ("single_file", not view.listing.directory),
        ("harbor_jobs", bool(view.jobs)),
        ("mixed", bool(view.trials and view.loose)),
        ("harbor_trials", bool(view.trials)),
        ("trajectory_folders", bool(view.chosen)),
    )
    return next((layout for layout, fit in fits if fit), "no_trajectories")


def _harbor(view: _View) -> Doc | None:
    if not view.trials:
        return None
    return {
        "jobs": len(view.jobs),
        "trials": len(view.trials),
        "trials_with_reward": sum(
            bool({"reward.txt", "reward.json"} & view.tree.get(_join(t, "verifier"), set()))
            for t in view.trials
        ),
        "multi_step_trials": sum("steps" in view.tree[t] for t in view.trials),
    }


def inspect_listing(listing: Listing, pattern: str, number: int) -> Doc:
    view = _View(listing, pattern)
    entries = listing.entries
    roles, per_trial, by_role = _roles(view)
    chosen_paths = {e.path for e in view.chosen}
    other = Counter(
        _name(PurePosixPath(e.path).name) for e in entries if e.path not in chosen_paths
    )
    folders, companions = _companions(view)
    sizes = [e.size for e in entries if e.size is not None]
    return {
        "input": number,
        "remote": listing.remote,
        "directory": listing.directory,
        "files": len(entries),
        "bytes": sum(sizes) if len(sizes) == len(entries) else None,
        "layout": _layout(view),
        "harbor": _harbor(view),
        "would_scan": {"total": len(view.chosen), "by_role": dict(sorted(roles.items()))},
        "anomalies": _anomalies(view, per_trial, by_role),
        "folders": folders,
        "alongside": {_name(k): v for k, v in companions.most_common(TOP)},
        "other_files": dict(other.most_common(TOP)),
        "other_file_count": sum(other.values()),
    }


def _join(folder: str, name: str) -> str:
    return f"{folder}/{name}" if folder else name


def document(listings: list[Listing], pattern: str) -> Doc:
    return {
        "schema_version": 1,
        "kind": "inspection",
        "pattern": pattern,
        "inputs": [inspect_listing(x, pattern, i) for i, x in enumerate(listings, 1)],
    }
