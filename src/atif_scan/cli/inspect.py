"""`--inspect`: classify a listing or a Harbor job without reading traces."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ..output.document import (
    inspection_text,
    overview,
    overview_text,
    to_json,
)
from ..sources.harbor.hub import inspect_job, is_harbor
from ..sources.inputs import (
    SourceError,
    list_input,
)
from ..sources.layout import document as inspection
from .args import output_format

if TYPE_CHECKING:
    import argparse

    from ..data.jsonval import Doc


def _harbor_cards(args: argparse.Namespace, jobs: list[Doc]) -> list[Doc]:
    """Scorecards for Harbor Hub jobs from their listings only: nothing is downloaded."""
    cards = []
    for job in jobs:
        items = [
            dict(t, input_status="listed", incomplete=False, assessments=[]) for t in job["trials"]
        ]
        cards.append(
            overview(
                {"inputs": items, "runs": [job["run"]]},
                min_trials=args.min_trials,
                scanned=False,
                expect_tasks=args.expect_tasks,
            )
        )
    return cards


def inspect(args: argparse.Namespace) -> int:
    if args.manifest or not args.paths:
        print("atif-scan: --inspect takes paths, not a manifest", file=sys.stderr)
        return 2
    try:
        local = [p for p in args.paths if not is_harbor(p)]
        doc = inspection([list_input(p) for p in local], args.pattern)
        jobs = [inspect_job(p) for p in args.paths if is_harbor(p)]
    except SourceError as error:
        print(f"atif-scan: {error}", file=sys.stderr)
        return 2
    cards = _harbor_cards(args, jobs)
    if cards:
        doc["harbor_jobs"] = cards
    if output_format(args) == "json":
        print(to_json(doc))
    else:
        text = inspection_text(doc) if local else ""
        for card in cards:
            text += "\n".join(overview_text(card)) + "\n"
        print(text, end="")
    return 0
