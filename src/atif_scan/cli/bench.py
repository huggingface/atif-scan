"""Bench-run listing, and --run/--release selection of the trials that count."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ..data.paths import bench_root
from ..sources.bench import catalogue, load_run
from ..sources.selection import release, replacements

if TYPE_CHECKING:
    from ..sources.selection import Selection


def _check(parser: argparse.ArgumentParser, args: argparse.Namespace) -> bool:
    """Whether a selection was asked for; rejects flag combinations that can't apply."""
    if args.run is None and args.release is None:
        if args.bench_root is not None or args.as_run:
            parser.error("--bench-root and --as-run require --run (or use 'atif-scan runs')")
        return False
    if args.run is not None and args.release is not None:
        parser.error("--run and --release cannot be combined")
    if args.release is not None and args.as_run:
        parser.error("--as-run applies to --run only")
    if args.paths or args.manifest or args.submission:
        parser.error("--run/--release cannot be combined with paths, --manifest or --submission")
    return True


def _selection(args: argparse.Namespace) -> Selection | None:
    root = args.bench_root or bench_root()
    if args.release is not None:
        return release(root, args.release)
    run = load_run(root, args.run)
    args.paths = run.scan_paths()
    return None if args.as_run else replacements(root, run)


def select(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """--run (replacements resolved unless --as-run) or --release: the job folders to
    scan, and the selection saying which of their trials count (`args.selection`)."""
    args.selection = None
    if not _check(parser, args):
        return
    try:
        args.selection = _selection(args)
    except OSError:
        parser.error("--run/--release: selection rejected (unreadable_or_missing)")
    except ValueError as exc:
        code = str(exc) if re.fullmatch(r"[a-z_]{1,64}", str(exc)) else "invalid"
        parser.error(f"--run/--release: selection rejected ({code})")
    if args.selection is not None:
        args.paths = [str(job) for job in args.selection.jobs]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="atif-scan runs")
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=None,
        help="bench-run checkout (default: $ATIF_SCAN_BENCH_ROOT, else ~/source/bench-run)",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    try:
        rows = catalogue(args.bench_root or bench_root())
    except (OSError, ValueError):
        parser.error("unreadable or invalid bench-run catalogue")
    if args.format == "json":
        print(json.dumps({"runs": rows, "selection": "original_job_parts"}, indent=2))
    else:
        print("RUN  STATUS  LOCAL/PARTS (original jobs; --run resolves replacements)")
        for row in rows:
            parts = f"{row['local_parts']}/{row['parts']}" if row["parts"] else "unknown"
            print(f"{row['run']}  {row['status']}  {parts}")
    return 0
