"""Inputs: manifests, paths and remote sources resolved to (source, context) records,
syncing, tasks and the checks to run."""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from ..access import access_rules
from ..checks import Context, identifier
from ..data.facts import trial_task
from ..data.loader import TraceError
from ..detectors import builtin_detectors
from ..detectors.priming import builtin_allowances
from ..packs import recognise, tasks_needed, untruncated_task
from ..packs.reference import ENV as PACK_ENV
from ..policy import load_rules
from ..sources.harbor.files import trial_result
from ..sources.harbor.hub import harbor_sources, is_harbor
from ..sources.harbor.runs import RESULT_BYTES
from ..sources.inputs import (
    HF_PREFIX,
    Source,
    file_source,
    normalize,
    resolve,
)
from ..sources.sync import sync_remote, sync_target

if TYPE_CHECKING:
    import argparse

    from ..checks import Detector
    from ..data.jsonval import Doc
    from ..data.model import Trace
    from ..rules import Allowance, Rule
    from ..sources.selection import Selection

    Check = Detector | Rule | Allowance


Record = tuple[Source, Context]


MANIFEST_RECORD_KEYS = {"id", "path", "task", "partial", "reward"}


def _manifest_record(raw: object, manifest: Path) -> Record:
    if not isinstance(raw, dict) or set(raw) - MANIFEST_RECORD_KEYS:
        raise ValueError("invalid_input_record")
    if not {"id", "path"} <= set(raw) or not isinstance(raw["path"], str):
        raise ValueError("invalid_input_record")
    location = raw["path"]
    if "://" not in location:
        location = str(manifest.parent / location)
    source = file_source(identifier(raw["id"]), location)
    if "reward" in raw:
        reward = Context(reward=raw["reward"]).reward  # validated number or None
        source = Source(source.label, source.load, lambda: reward, local=source.local)
    return source, Context(raw.get("task"), raw.get("partial", False))


def manifest_inputs(path: Path) -> list[Record]:
    """Explicit file list: `{"inputs": [{"id", "path", "task"?, "partial"?}]}`.

    Local paths resolve relative to the manifest; `hf://` paths are used as given.
    """
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or set(value) != {"inputs"}
        or not isinstance(value["inputs"], list)
    ):
        raise ValueError("invalid_input_manifest")
    return [_manifest_record(raw, path) for raw in value["inputs"]]


def _quiet(message: str) -> None:
    pass


def inputs(args: argparse.Namespace) -> list[Record]:
    args.sync_failures = []
    if args.manifest:
        if args.paths or args.task or args.task_from or args.partial:
            raise ValueError("manifest_cannot_be_combined_with_direct_inputs")
        result = manifest_inputs(args.manifest)
    else:
        # Local/hf:// inputs resolve together so positional labels stay unique.
        local = [sync_if_remote(v, args) for v in args.paths if not is_harbor(v)]
        found = resolve_local(local, args) if local else []
        for value in (v for v in args.paths if is_harbor(v)):
            hub, run = harbor_sources(
                value,
                args.download_dir,
                full=args.full,
                workers=args.jobs,
                refresh=args.refresh,
                progress=status_line if sys.stderr.isatty() else _quiet,
            )
            args.runs.append(run)
            found += hub
        if getattr(args, "selection", None) is not None:
            found = select_sources(found, args.selection, [Path(p) for p in local])
            jobs = args.selection.replacement_jobs
            args.runs[:] = [r for r in args.runs if r.get("job_name") not in jobs]
        result = [(s, Context(task_for(s, args), args.partial)) for s in found]
    if not result or len({s.label for s, _ in result}) != len(result):
        raise ValueError("empty_or_duplicate_inputs")
    return result


def resolve_local(values: list[str], args: argparse.Namespace) -> list[Source]:
    """Selected bench parts get disjoint labels, even when trial names repeat."""
    if not getattr(args, "run", None) and getattr(args, "selection", None) is None:
        return resolve(values, args.pattern, runs=args.runs, sync_failures=args.sync_failures)
    found = []
    for index, value in enumerate(values, 1):
        part = resolve([value], args.pattern, runs=args.runs, sync_failures=args.sync_failures)
        found.extend(replace(s, label=f"part-{index}/{s.label}") for s in part)
    return found


def _trial_key(source: Source, jobs: set[Path]) -> tuple[str, str] | None:
    """(job folder, trial folder) of a local trajectory under one of `jobs`."""
    if source.local is None:
        return None
    for folder in source.local.parents:
        if folder.parent in jobs:
            return folder.parent.name, folder.name
    return None


def _missing_trial(label: str, trial: Path, lineage: Doc) -> Source:
    """A replaced trial that left no trajectory (it failed before the agent ran): an
    item all the same, with its result.json facts, so its lineage can be inspected."""

    def load() -> Trace:
        raise TraceError("trajectory_missing")

    def details() -> Doc:
        path = trial / "result.json"
        try:
            with path.open("rb") as stream:
                return trial_result(stream.read(RESULT_BYTES))
        except OSError:
            return {}

    return Source(label, load, details=details, selection=lineage)


def select_sources(found: list[Source], selection: Selection, jobs: list[Path]) -> list[Source]:
    """Each source stamped with its role and lineage; replaced trials without a
    trajectory added as sources of their own, labelled like their job's other trials."""
    folders = {job.resolve() for job in jobs} | set(jobs)
    parts = {job.name: (job, f"part-{i}/") for i, job in enumerate(jobs, 1)}
    out: list[Source] = []
    seen: set[tuple[str, str]] = set()
    for source in found:
        key = _trial_key(source, folders)
        if key is None:
            out.append(source)
            continue
        seen.add(key)
        stamp = {"role": selection.role(key), **selection.lineage(key)}
        out.append(replace(source, selection=stamp))
    for key, lineage in selection.replaced.items():
        if key in seen or key[0] not in parts:
            continue
        job, prefix = parts[key[0]]
        stamp = {**lineage, "role": "replaced"}
        out.append(_missing_trial(f"{prefix}{key[1]}/agent", job / key[1], stamp))
    return out


def status_line(message: str) -> None:
    """One rewritten stderr line (terminals only): fixed text and counts, no names."""
    print(f"\r\033[Katif-scan: {message}", end="", file=sys.stderr, flush=True)


def sync_if_remote(value: str, args: argparse.Namespace) -> str:
    """With --sync (default), mirror an hf:// input locally and scan the copy."""
    if not args.sync or not normalize(value).startswith(HF_PREFIX):
        return value
    dest = sync_target(value, args.sync_root)
    tty = sys.stderr.isatty()
    if tty:
        status_line("listing remote files")

    def progress(done: int, total: int) -> None:
        status_line(f"syncing {done}/{total}")

    path, counts = sync_remote(
        value,
        dest,
        args.pattern,
        workers=args.jobs,
        refresh=args.refresh,
        progress=progress if tty else None,
    )
    if tty:
        print(
            f"\r\033[Katif-scan: local copy · {counts['downloaded']} downloaded,"
            f" {counts['up_to_date']} up to date"
            + (f", {counts['failed']} failed" if counts["failed"] else ""),
            file=sys.stderr,
        )
    return str(path)


def folder_task(source: Source) -> str | None:
    """The task in a Harbor trial folder name (`<task>__<suffix>`). Job folders can
    contain `__` too (`2026-08-19__20-38-18`), so the part closest to the file wins.
    Harbor cuts a task name to 32 characters in the folder name; a bundled pack's task
    is restored from its cut name (None when two tasks share it)."""
    for part in reversed(re.split(r"[\\/]", source.hint or source.label)):
        task, separator, _ = part.partition("__")
        if separator and task:
            try:
                return untruncated_task(identifier(task))
            except ValueError:
                return None
    return None


def task_for(source: Source, args: argparse.Namespace) -> str | None:
    """--task for every input, else the task its run listing recorded (e.g. the Harbor
    Hub), else (explicitly, --task-from) derived from the trial folder name."""
    inferred = folder_task(source) if args.task_from == "trial-dir" else None
    return trial_task(args.task, source.meta, inferred)


def load_checks(args: argparse.Namespace, records: list[Record] | None = None) -> list[Check]:
    checks: list[Check] = [*builtin_detectors(), *access_rules(), *builtin_allowances()]
    args.packs_loaded = []
    if args.packs == "auto" and records is not None:
        # The recorded task (Hub, result.json) when it wasn't given: small capped reads,
        # needed only when no dataset was recorded, the dataset needs the tasks to pin a
        # pack, or a pack needs task data.
        needs_tasks = PACK_ENV in os.environ or tasks_needed(args.runs)
        tasks = [
            context.task or (source.details().get("task") if needs_tasks else None)
            for source, context in records
        ]
        for pack, reason in recognise(args.runs, tasks):
            if pack.plugin not in args.plugin:  # named explicitly: loaded below
                checks.extend(pack.load())
                args.packs_loaded.append({"pack": pack.name, "reason": reason})
    for plugin in args.plugin:
        module, separator, factory = plugin.partition(":")
        if not separator or not module or not factory:
            raise ValueError("invalid_plugin_reference")
        checks.extend(getattr(importlib.import_module(module), factory)())
    for rules in args.rules:
        checks.extend(load_rules(json.loads(rules.read_text())))
    return checks
