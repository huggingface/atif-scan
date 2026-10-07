"""Harbor run files found in a listing, next to the trajectories a scan reads.

Given a `sources.Listing` (names and sizes only), this finds and reads the small files
Harbor and harbor-hf write beside each trial: the verifier's reward file, the trial's
`result.json` (plus harbor-hf's `attempt-costs/`), Harbor's `exception.txt` marker, each
job folder's `config.json`/`result.json` (run facts and which trials the job accounts
for), and saved Harbor Hub listings / `trials.jsonl` ledgers. Parsing is harbor_files'
and harbor_hub's; which record wins for a trial's facts is decided in `facts`.

Every read is capped and best effort: an unreadable file is unknown, never an error,
a zero or a negative result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePath, PurePosixPath
from typing import TYPE_CHECKING, TypeVar

from ...data.jsonval import Doc, as_object, number
from ...data.submission import MAX_BYTES as PATCH_BYTES
from ...data.submission import Submission, parse_patch
from .files import (
    ATTEMPT_COST_BYTES,
    ATTEMPT_COSTS,
    FAST_AGENT_RESULTS,
    FAST_AGENT_RESULTS_BYTES,
    LEDGER_BYTES,
    RUN_MANIFEST,
    RUN_MANIFEST_BYTES,
    TRIAL_LEDGER,
    attempt_cost,
    declared_prices,
    job_listed_trials,
    job_meta,
    safety_details,
    trial_ledger,
    trial_result,
)
from .listing import SAVED_LISTING, saved_listing

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from ..inputs import Entry, Listing

REWARD_FILES = ("reward.json", "reward.txt")  # Harbor reads reward.json first
REWARD_NAMES = tuple(f"verifier/{name}" for name in REWARD_FILES)
REWARD_BYTES = 4096
RESULT_BYTES = 1024 * 1024  # Harbor result.json / config.json
LISTING_BYTES = 16 * 1024 * 1024  # a saved Hub listing: ~500 bytes per trial
EXCEPTION_MARKER = "exception.txt"
# Relative paths inside a job folder that make a folder one of its trials.
TRAJECTORY_PARTS = (("agent", "trajectory.json"), ("trajectory.json",))

P = TypeVar("P", bound=PurePath)
# A saved Hub listing or ledger: (run facts or None, {trial folder: trial facts}).
SavedRun = tuple[Doc | None, dict[str, Doc]]


def _no_reward() -> float | None:
    return None


def _no_details() -> Doc:
    return {}


def parse_reward(data: bytes, name: str) -> float | None:
    """A finite number from Harbor's reward file; anything else is unknown (None)."""
    try:
        text = data[:REWARD_BYTES].decode("utf-8").strip()
        value = json.loads(text) if name.endswith(".json") else float(text)
    except (UnicodeError, ValueError):
        return None
    if isinstance(value, dict):
        value = as_object(value).get("reward")
    return number(value)


def _near(path: P, names: Iterable[str]) -> list[P]:
    """`<dir>/<name>` for the file's folder, then one level up (Harbor's trial folder:
    `<trial>/agent/trajectory.json` -> `<trial>/verifier/reward.*`). Path or PurePath."""
    parent = path.parent
    folders = [parent] + ([parent.parent] if parent != parent.parent else [])
    return [folder / name for folder in folders for name in names]


def reward_candidates(path: str) -> list[str]:
    """Relative reward-file paths for a listed trajectory, in lookup order."""
    return [p.as_posix() for p in _near(PurePosixPath(path), REWARD_NAMES)]


def _patch_path(trajectory: Path | None) -> Path | None:
    """`<trial>/artifacts/model.patch` for `<trial>/agent/trajectory.json`, if a plain
    file (not a symlink) is there."""
    if trajectory is None or trajectory.parent.name != "agent":
        return None
    path = trajectory.parent.parent / "artifacts" / "model.patch"
    try:
        return None if path.is_symlink() or not path.is_file() else path
    except OSError:
        return None


def submission_near(trajectory: Path | None) -> Submission | None:
    """The trial's submitted patch beside its trajectory (DeepSWE/Pier); None when
    there is none to read (no local trajectory, another layout, missing, a symlink or
    unreadable): unknown, never an empty submission. A patch over the size cap is kept
    as not understood."""
    path = _patch_path(trajectory)
    data = _read_local(path, PATCH_BYTES + 1) if path is not None else None
    if data is None:
        return None
    if len(data) > PATCH_BYTES:
        return Submission((), hashlib.sha256(data).hexdigest(), understood=False)
    return parse_patch(data)


def _read_local(path: Path, limit: int) -> bytes | None:
    """Up to `limit` bytes of a local metadata file; None when missing or unreadable."""
    try:
        with path.open("rb") as handle:
            return handle.read(limit)
    except OSError:
        return None


def reward_lookup(listing: Listing, entry: Entry, value: str) -> Callable[[], float | None]:
    """Find the reward next to a selected trajectory without any extra listing call.

    `value` is the (normalized) input the listing was made from."""
    if listing.directory:
        return _listed_reward(listing, entry)
    if listing.remote:
        return _no_reward
    candidates = _near(Path(value), REWARD_NAMES)

    def local() -> float | None:
        for candidate in candidates:
            if candidate.is_file():
                data = _read_local(candidate, REWARD_BYTES)
                return None if data is None else parse_reward(data, candidate.name)
        return None

    return local


def _listed_reward(listing: Listing, entry: Entry) -> Callable[[], float | None]:
    found = next((c for c in reward_candidates(entry.path) if c in listing.paths), None)
    if found is None or listing.reader is None:
        return _no_reward

    def listed() -> float | None:
        data = listing.read(found, REWARD_BYTES)
        # An unreadable reward is unknown, never an error or a zero.
        return None if data is None else parse_reward(data, found)

    return listed


def trial_details(listing: Listing, entry: Entry, value: str) -> Callable[[], Doc]:
    """Lazy reader for the trial's Harbor result.json (trajectory folder or one up),
    with harbor-hf's attempt cost for listed trials. `value` as for reward_lookup."""
    if listing.directory:
        return _listed_details(listing, entry)
    if listing.remote:
        return _no_details
    candidates = _near(Path(value), ["result.json"])
    # fast-agent's results file sits beside the trajectory (<trial>/agent/).
    companion = Path(value).parent / FAST_AGENT_RESULTS

    def local() -> Doc:
        for candidate in candidates:
            if candidate.is_file() and (
                facts := trial_result(_read_local(candidate, RESULT_BYTES))
            ):
                if facts.get("error_type") and companion.is_file():
                    facts |= safety_details(_read_local(companion, FAST_AGENT_RESULTS_BYTES))
                return facts
        return {}

    return local


def _listed_details(listing: Listing, entry: Entry) -> Callable[[], Doc]:
    near = _near(PurePosixPath(entry.path), ["result.json"])
    found = [p for p in (f.as_posix() for f in near) if p in listing.paths]
    if not found or listing.reader is None:
        return _no_details
    # fast-agent's results file sits beside the trajectory (<trial>/agent/).
    companion = (PurePosixPath(entry.path).parent / FAST_AGENT_RESULTS).as_posix()

    def listed() -> Doc:
        for path in found:
            if facts := trial_result(listing.read(path, RESULT_BYTES)):
                if facts.get("error_type") and companion in listing.paths:
                    facts |= safety_details(listing.read(companion, FAST_AGENT_RESULTS_BYTES))
                return with_attempt_cost(listing, path, facts)
        return {}

    return listed


def with_attempt_cost(listing: Listing, result_path: str, facts: Doc) -> Doc:
    """Trial facts plus the cost harbor-hf recorded beside the trial
    (`<run>/attempt-costs/<attempt id>.json`, the job folder being `<run>/job/`) as
    `attempt_cost_usd`. The attempt id is a lookup key only and is dropped."""
    facts = dict(facts)
    attempt = facts.pop("attempt_id", None)
    if attempt is None or listing.reader is None:
        return facts
    trial = PurePosixPath(result_path).parent
    for run in (trial.parent.parent, trial.parent):
        path = (run / ATTEMPT_COSTS / f"{attempt}.json").as_posix()
        if path in listing.paths:
            # Unreadable: unknown, never a zero.
            data = listing.read(path, ATTEMPT_COST_BYTES)
            if (cost := attempt_cost(data, attempt, trial.name)) is not None:
                facts["attempt_cost_usd"] = cost
            return facts
    return facts


def errored(listing: Listing, entry: Entry) -> bool:
    """Harbor writes `<trial>/exception.txt` when a trial raised; seen in the listing."""
    if not listing.directory:
        return False
    near = _near(PurePosixPath(entry.path), [EXCEPTION_MARKER])
    return any(p.as_posix() in listing.paths for p in near)


def run_prices(listing: Listing, folder: str) -> Doc | None:
    """Prices declared by harbor-hf's `run.json` beside (or inside) a job folder."""
    job = PurePosixPath(folder)
    for run in (job.parent, job) if folder != "." else (job,):
        path = (run / RUN_MANIFEST).as_posix()
        if path in listing.paths and listing.reader is not None:
            return declared_prices(listing.read(path, RUN_MANIFEST_BYTES))
    return None


def _job_candidates(present: Iterable[str]) -> list[str]:
    """Folders with config.json and at least one subfolder with a result.json."""
    paths = [PurePosixPath(p) for p in present]
    has_trial_child = {
        str(p.parent.parent) for p in paths if p.name == "result.json" and len(p.parts) > 1
    }
    configured = {str(p.parent) for p in paths if p.name == "config.json"}
    return sorted(configured & has_trial_child)


def _job_trials(present: Iterable[str], prefix: str) -> tuple[set[str], set[str]]:
    """(trial folders, those with a trajectory) directly inside the job folder `prefix`."""
    trials: set[str] = set()
    traced: set[str] = set()
    for p in present:
        if not p.startswith(prefix):
            continue
        parts = PurePosixPath(p[len(prefix) :]).parts
        if parts[1:] == ("result.json",):
            trials.add(parts[0])
        elif parts[1:] in TRAJECTORY_PARTS:
            trials.add(parts[0])
            traced.add(parts[0])
    return trials, traced


def job_folders(listing: Listing) -> tuple[list[Doc], dict[str, bool]]:
    """(run facts per Harbor job folder, {trial folder: listed in its job's result.json}).

    A job folder has config.json and at least one subfolder with a result.json (its
    trials); trial folders' own agent/ and verifier/ subfolders never do, so trial
    configs aren't read. job_meta then rejects anything that isn't a job config.

    Trial folders the job's result.json doesn't account for usually come from another
    execution of the same job (a rerun or resume writing into the folder, possibly while
    the first one still ran). They are still scanned; the run facts count them. When the
    job's own listing is missing or incomplete, membership is unknown (not recorded).
    """
    if not listing.directory or listing.reader is None:
        return [], {}
    present = listing.paths
    runs: list[Doc] = []
    membership: dict[str, bool] = {}
    for folder in _job_candidates(present):
        prefix = "" if folder == "." else folder + "/"
        config = listing.read(prefix + "config.json", RESULT_BYTES)
        result_path = prefix + "result.json"
        result = listing.read(result_path, RESULT_BYTES) if result_path in present else None
        if config is None or (result is None and result_path in present):
            continue  # unreadable: not a known job
        if (run := job_meta(config, result)) is None:
            continue
        if (prices := run_prices(listing, folder)) is not None:
            run["declared_prices"] = prices
        listed = job_listed_trials(result)
        if listed is not None:
            trials, traced = _job_trials(present, prefix)
            unlisted = trials - listed
            run["trial_folders"] = len(trials)
            run["unlisted_trials"] = len(unlisted)
            run["unlisted_with_trajectory"] = len(unlisted & traced)
            membership.update({prefix + t: t in listed for t in trials})
        runs.append(run)
    return runs, membership


def trial_folder(entry: Entry) -> str:
    """The trial folder of a trajectory: `<trial>/trajectory.json` or `<trial>/agent/...`."""
    trial = PurePosixPath(entry.path).parent
    return str(trial.parent if trial.name == "agent" else trial)


def saved_hub_listings(listing: Listing) -> dict[str, SavedRun]:
    """Saved Harbor Hub listings (harbor_hub.save_listing) and `trials.jsonl` run ledgers
    by the folder holding them: (run facts, {trial folder: trial facts}), so a scan of a
    synced job keeps its recorded facts (reward, task, error, cost). A ledger lists the
    trials of the folder it sits in or of its `trials/` subfolder (see hub_trial).
    Unreadable files are skipped: those trials simply lack the facts."""
    # harbor_hub imports sources, which imports this module.

    if not listing.directory or listing.reader is None:
        return {}
    out: dict[str, SavedRun] = {}
    for path in sorted(listing.paths):
        name = PurePosixPath(path).name
        folder = str(PurePosixPath(path).parent)
        if name == TRIAL_LEDGER:
            if (data := listing.read(path, LEDGER_BYTES)) is None:
                continue
            run, known = out.get(folder, (None, {}))
            out[folder] = (run, {**trial_ledger(data), **known})  # a saved Hub listing wins
        elif name == SAVED_LISTING:
            if (data := listing.read(path, LISTING_BYTES)) is None:
                continue
            run, trials = saved_listing(data)
            out[folder] = (run, {**out.get(folder, (None, {}))[1], **trials})
    return out


def hub_trial(saved: dict[str, SavedRun], entry: Entry) -> Doc:
    """The saved Hub facts of the trial whose trajectory is `entry` (`<job>/<trial>/
    trajectory.json`, or `<job>/job/<trial>/agent/trajectory.json` for a full archive)."""
    trial = PurePosixPath(entry.path).parent
    if trial.name == "agent":
        trial = trial.parent
    for folder in (trial.parent, trial.parent.parent):
        found = saved.get(str(folder))
        if found and trial.name in found[1]:
            return found[1][trial.name]
    return {}
