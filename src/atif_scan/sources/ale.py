"""ALE directory receipts and fixed companion locations, using capped listing reads.

No paths from receipt contents are followed. Native histories and submissions are
associated by availability only; neither is opened or executed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ..data.ale import LIMIT, facts, recognized
from ..data.jsonval import Doc, load_object

if TYPE_CHECKING:
    from .inputs import Listing


@dataclass(frozen=True)
class Attempt:
    folder: str
    trajectory: str
    metadata: Doc

    def contains(self, path: str) -> bool:
        return path.startswith(self.folder + "/") if self.folder else True


def _path(folder: str, name: str) -> str:
    return f"{folder}/{name}" if folder else name


def _trajectory(listing: Listing, folder: str, harness: object) -> tuple[str, str]:
    native = _path(folder, "origin_log/fast_agent/trajectory.json")
    exported = _path(folder, "atif.json")
    if harness == "fast_agent" and native in listing.paths:
        return native, "native_atif"
    # Never substitute a root trajectory.json summary for an absent/failed export.
    return exported, "atif_export"


def _native_transcript(listing: Listing, folder: str, harness: object) -> bool:
    # OpenClaw receipts use underscores, but its native-log folder uses a hyphen.
    names = (harness, "openclaw-cli") if harness == "openclaw_cli" else (harness,)
    return any(
        _path(folder, f"origin_log/{name}/transcript.jsonl") in listing.paths for name in names
    )


def attempts(listing: Listing) -> list[Attempt]:
    if not listing.directory:
        return []
    found = []
    for entry in listing.entries:
        if PurePosixPath(entry.path).name != "run.json" or entry.failed:
            continue
        run = load_object(listing.read(entry.path, LIMIT + 1), LIMIT)
        if not recognized(run):
            continue
        folder = entry.path.rpartition("/")[0]
        eval_path = _path(folder, "eval_result.json")
        evaluation = load_object(
            listing.read(eval_path, LIMIT + 1) if eval_path in listing.paths else None, LIMIT
        )
        meta = facts(run, evaluation, evaluation_present=eval_path in listing.paths)
        trajectory, origin = _trajectory(listing, folder, meta["configured_agent"])
        meta.update(
            trajectory_source=origin,
            native_transcript_available=_native_transcript(
                listing, folder, meta["configured_agent"]
            ),
            submission_artifacts_available=any(
                p.startswith(_path(folder, "output/")) for p in listing.paths
            ),
        )
        if trajectory not in listing.paths:
            meta["source_issues"].append("trajectory_missing")
        found.append(Attempt(folder, trajectory, meta))
    return found
