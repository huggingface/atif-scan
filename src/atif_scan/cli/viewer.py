"""Scan normally, then write a static trajectory viewer for individual publication."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ..browser import export, session
from ..checks import Severity
from ..output.brief import brief
from ..review.catalogue import BY_ID
from .scan import Scanner, _document, _prepare, _scan_all

if TYPE_CHECKING:
    import argparse

    from ..data.jsonval import Doc
    from ..engine import Engine
    from ..review.answers import Answers
    from .inputs import Record


def _pinned(records: list[Record]) -> list[str | None] | None:
    """Digests of local traces before scanning, or None (reported) when unavailable."""
    # Replaced trials that left no trajectory are evidence-only items, not missing inputs.
    if any(source.local is None and not source.selection for source, _ in records):
        print("atif-scan: --viewer needs local traces (details withheld)", file=sys.stderr)
        return None
    try:
        return session.source_digests(records)
    except (OSError, ValueError):
        print("atif-scan: viewer sources unavailable (details withheld)", file=sys.stderr)
        return None


def _scanned(
    args: argparse.Namespace, records: list[Record], engine: Engine
) -> tuple[list[Doc], Doc, int, Answers | None]:
    """Scan items, the report document, the normal scan exit status and the answers the
    scan read (the same load, so reasons match the rows)."""
    sync_failed = sum(getattr(args, "sync_failures", []))
    threshold = Severity[args.fail_on.upper()] if args.fail_on else None
    scanner = Scanner.for_args(args, engine)
    items, invalid, failed = _scan_all(scanner, records, threshold)
    if sync_failed:
        print(
            f"atif-scan: {sync_failed} sync file(s) unavailable; export incomplete", file=sys.stderr
        )
    status = 2 if invalid or sync_failed else 1 if failed else 0
    return items, _document(items, args, sync_failed, engine), status, scanner.answers


def export_viewer(args: argparse.Namespace) -> int:
    """Pin sources, scan, then export masked fields; keeps the normal scan exit status."""
    prepared = _prepare(args)
    digests = _pinned(prepared[0]) if prepared is not None else None
    if prepared is None or digests is None:
        return 2
    records, engine = prepared
    items, doc, status, answers = _scanned(args, records, engine)
    try:
        run = export.run_facts(
            brief(doc, args.dq_on, args.min_trials, args.expect_tasks, args.price_rates)
        )
        review = BY_ID[args.review] if args.review else None
        document = export.bundle(
            records, items, digests, doc, run, review, answers, full=args.viewer_full
        )
        export.write(args.viewer, document)
    except (OSError, ValueError):
        print("atif-scan: viewer export failed (details withheld)", file=sys.stderr)
        return 2
    print(
        f"atif-scan: wrote a static viewer of {len(items)} trajectory(ies) to {args.viewer}\n"
        "atif-scan: it contains masked trace text (best effort): review it before publishing",
        file=sys.stderr,
    )
    return status
