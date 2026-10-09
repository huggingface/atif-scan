"""Scan normally with excerpts of interesting moments, then write the highlight report."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ..browser import export, highlights
from ..output.brief import brief
from .scan import _prepare
from .viewer import _pinned, _scanned

if TYPE_CHECKING:
    import argparse

ASSETS = ("index.html", "highlights.css", "highlights.js")


def export_highlights(args: argparse.Namespace) -> int:
    """Pin sources, scan (excerpts on), then export; keeps the normal scan exit status."""
    prepared = _prepare(args)
    digests = _pinned(prepared[0]) if prepared is not None else None
    if prepared is None or digests is None:
        return 2
    records, engine = prepared
    items, doc, status, _ = _scanned(args, records, engine)  # highlights never show reasons
    try:
        summary = brief(doc, args.dq_on, args.min_trials, args.expect_tasks, args.price_rates)
        document = highlights.document(items, doc, summary)
        export.write(args.highlights, document, "highlights", ASSETS, "ATIF_HIGHLIGHTS")
    except (OSError, ValueError):
        print("atif-scan: highlight export failed (details withheld)", file=sys.stderr)
        return 2
    moments = sum(len(rows) for rows in document["moments"].values())
    print(
        f"atif-scan: wrote a highlight report of {moments} excerpt(s) from "
        f"{len(document['trials'])} trial(s) to {args.highlights}\n"
        "atif-scan: it contains masked trace text (best effort): keep it private",
        file=sys.stderr,
    )
    return status
