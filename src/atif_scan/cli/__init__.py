"""Offline CLI. Reports never contain trace text, source paths, URLs or exception bodies."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ..sources.harbor.files import SUBMISSION_BYTES, submission, text_label
from ..sources.sync import default_sync_root
from . import hunt, labels
from .args import _check_combinations, build_parser, price
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


# Subcommands come first; scan an input with one of these names as ./labels or ./hunt.
COMMANDS = {"labels": labels.main, "hunt": hunt.main}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in COMMANDS:
        return COMMANDS[argv[0]](argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    _check_combinations(parser, args)
    args.price_rates = price(args.price)  # validate early, whatever the output format
    args.submission_doc = _submission(parser, args)
    if args.inspect:
        return inspect(args)
    args.sync_root = args.sync_dir or default_sync_root()
    with tempfile.TemporaryDirectory(prefix="atif-scan-") as scratch:
        # Harbor downloads: kept under the sync root with --sync, else a temp dir.
        args.download_dir = args.sync_root / "harbor" if args.sync else Path(scratch)
        args.runs = []
        return scan(args)
