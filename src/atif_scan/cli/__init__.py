"""Offline CLI. Reports never contain trace text, source paths, URLs or exception bodies."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ..data.jsonval import identifier
from ..sources.harbor.files import SUBMISSION_BYTES, submission, text_label
from ..sources.harbor.hub import is_harbor
from ..sources.sync import default_sync_root
from . import bench, hunt, labels
from .args import _check_combinations, _check_input_source, build_parser, price
from .inspect import inspect
from .scan import scan

if TYPE_CHECKING:
    import argparse

    from ..data.jsonval import Doc


def _submission(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Doc | None:
    """--submission FILE: its source jobs join the inputs as harbor://jobs/<id>; the file's
    name (a label) and source filter go into the report."""
    if args.submission is None:
        return None
    if args.manifest:
        parser.error("--submission can't be combined with --manifest")
    try:
        found = submission(args.submission.read_bytes()[: SUBMISSION_BYTES + 1])
    except OSError:
        found = None
    if found is None:
        parser.error("--submission: not a readable leaderboard submission (no source_jobs)")
    args.paths = [*args.paths, *(f"harbor://jobs/{job}" for job in found["jobs"])]
    name = text_label(args.submission.stem, 120)
    return {"name": name, **found}


# A task subset file: a JSON list of task names (a benchmark's subset is tens of tasks).
TASK_SUBSET_BYTES = 256 * 1024


def _task_subset(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> frozenset[str] | None:
    """--task-subset FILE: the task names to keep, for Harbor Hub inputs only (their
    listings name each trial's task before anything is downloaded)."""
    if args.task_subset is None:
        return None
    if args.manifest or args.run or args.release or not args.paths:
        parser.error("--task-subset applies to harbor:// jobs and leaderboard rows only")
    if not all(is_harbor(p) for p in args.paths):
        parser.error("--task-subset applies to harbor:// jobs and leaderboard rows only")
    try:
        data = args.task_subset.read_bytes()[: TASK_SUBSET_BYTES + 1]
        names = json.loads(data.decode("utf-8")) if len(data) <= TASK_SUBSET_BYTES else None
        if not isinstance(names, list):
            raise TypeError("not_a_list")
        # Names as the Hub records them (`org/task`) or bare: compared without the prefix.
        tasks = frozenset(identifier(n).rsplit("/", 1)[-1] for n in names)
    except (OSError, UnicodeDecodeError, ValueError, TypeError, RecursionError):
        tasks = frozenset[str]()
    if not tasks:
        parser.error("--task-subset: not a readable JSON list of task names")
    return tasks


# Subcommands come first; scan an input with one of these names as ./labels or ./hunt.
COMMANDS = {"labels": labels.main, "hunt": hunt.main, "runs": bench.main}


def _scan_mode(args: argparse.Namespace) -> int:
    """A normal scan, or one followed by the private desk or the static viewer export."""
    if args.browse:
        from .browse import browse  # noqa: PLC0415 - POSIX support is opt-in.

        return browse(args)
    if args.viewer is not None:
        from .viewer import export_viewer  # noqa: PLC0415 - export is opt-in.

        return export_viewer(args)
    if args.highlights is not None:
        from .highlights import export_highlights  # noqa: PLC0415 - export is opt-in.

        return export_highlights(args)
    return scan(args)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in COMMANDS:
        return COMMANDS[argv[0]](argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    bench.select(parser, args)
    _check_combinations(parser, args)
    _check_input_source(parser, args)
    args.price_rates = price(args.price)  # validate early, whatever the output format
    args.submission_doc = _submission(parser, args)
    args.task_names = _task_subset(parser, args)
    if args.inspect:
        return inspect(args)
    args.sync_root = args.sync_dir or default_sync_root()
    with tempfile.TemporaryDirectory(prefix="atif-scan-") as scratch:
        # Harbor downloads: kept under the sync root with --sync, else a temp dir.
        args.download_dir = args.sync_root / "harbor" if args.sync else Path(scratch)
        args.runs = []
        return _scan_mode(args)
