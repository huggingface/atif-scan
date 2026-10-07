"""Scan normally, then serve a private, explicitly requested loopback evidence desk."""

from __future__ import annotations

import sys
from contextlib import suppress
from typing import TYPE_CHECKING

from ..browser import server, session
from ..checks import Severity
from ..data.paths import home
from .scan import Scanner, _document, _prepare, _scan_all

if TYPE_CHECKING:
    import argparse

    from .inputs import Record


def _serve(desk: session.Session, port: int) -> None:
    with server.create_server(desk, port=port) as httpd:
        print(
            "atif-scan: private trace browser and feedback; keep the URL and data out of Git",
            file=sys.stderr,
        )
        print(httpd.url, file=sys.stderr, flush=True)
        with suppress(KeyboardInterrupt):
            httpd.serve_forever()


def _source_digests(records: list[Record]) -> list[str | None]:
    # Replaced trials that left no trajectory are evidence-only items, not missing inputs.
    if any(source.local is None and not source.selection for source, _ in records):
        raise ValueError("local_traces_required")
    return session.source_digests(records)


def browse(args: argparse.Namespace) -> int:
    """Bind findings to pre-scan content; retain normal scan exit status after serving."""
    prepared = _prepare(args)
    if prepared is None:
        return 2
    records, engine = prepared
    try:
        before = _source_digests(records)
    except (OSError, ValueError):
        print("atif-scan: browser sources unavailable (details withheld)", file=sys.stderr)
        return 2
    sync_failed = sum(getattr(args, "sync_failures", []))
    if sync_failed:
        print(
            f"atif-scan: {sync_failed} sync file(s) unavailable; report incomplete",
            file=sys.stderr,
        )
    threshold = Severity[args.fail_on.upper()] if args.fail_on else None
    scanner = Scanner.for_args(args, engine)
    items, invalid, failed = _scan_all(scanner, records, threshold)
    doc = _document(items, args, sync_failed, engine)
    print(f"atif-scan: {len(items)} input(s) scanned", file=sys.stderr)
    try:
        desk = session.Session(
            records,
            items,
            args.feedback_dir or home() / "feedback",
            expected_digests=before,
            report=doc,
        )
        _serve(desk, args.browse_port if args.browse_port is not None else 0)
    except (OSError, ValueError):
        print("atif-scan: browser startup failed (details withheld)", file=sys.stderr)
        return 2
    return 2 if invalid or sync_failed else 1 if failed else 0
