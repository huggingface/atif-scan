"""Inputs: manifests, paths and remote sources resolved to (source, context) records,
syncing, tasks and the checks to run."""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from typing import TYPE_CHECKING

from ..access import access_rules
from ..checks import Context, identifier
from ..data.facts import trial_task
from ..detectors import builtin_detectors
from ..packs import recognise, tasks_needed
from ..packs.reference import ENV as PACK_ENV
from ..policy import load_rules
from ..sources.harbor.hub import harbor_sources, is_harbor
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
    from pathlib import Path

    from ..checks import Detector
    from ..rules import Allowance, Rule

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
        found = (
            resolve(local, args.pattern, runs=args.runs, sync_failures=args.sync_failures)
            if local
            else []
        )
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
        result = [(s, Context(task_for(s, args), args.partial)) for s in found]
    if not result or len({s.label for s, _ in result}) != len(result):
        raise ValueError("empty_or_duplicate_inputs")
    return result


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
    contain `__` too (`2026-08-19__20-38-18`), so the part closest to the file wins."""
    for part in reversed(re.split(r"[\\/]", source.hint or source.label)):
        task, separator, _ = part.partition("__")
        if separator and task:
            try:
                return identifier(task)
            except ValueError:
                return None
    return None


def task_for(source: Source, args: argparse.Namespace) -> str | None:
    """--task for every input, else the task its run listing recorded (e.g. the Harbor
    Hub), else (explicitly, --task-from) derived from the trial folder name."""
    inferred = folder_task(source) if args.task_from == "trial-dir" else None
    return trial_task(args.task, source.meta, inferred)


def load_checks(args: argparse.Namespace, records: list[Record] | None = None) -> list[Check]:
    checks: list[Check] = [*builtin_detectors(), *access_rules()]
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
