"""Output views: brief, overview, summary and detail, as text or JSON."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ..checks import Severity
from ..output.brief import brief, brief_text, print_brief
from ..output.document import filter_findings, to_json
from ..output.overview import overview, overview_text
from ..output.summary import summary, summary_text
from ..output.text import to_text
from ..output.views import render_rich, render_summary_rich
from ..review.questions import tally as answer_tally
from .args import output_format

if TYPE_CHECKING:
    import argparse

    from ..data.jsonval import Doc


def _brief(doc: Doc, args: argparse.Namespace, fmt: str) -> None:
    report_ = brief(doc, args.dq_on, args.min_trials, args.expect_tasks, args.price_rates)
    if fmt == "json":
        print(to_json(report_))
    else:
        print_brief(brief_text(report_))


def _scorecard(doc: Doc, args: argparse.Namespace) -> Doc:
    return overview(doc, args.dq_on, args.min_trials, expect_tasks=args.expect_tasks)


def _overview(doc: Doc, args: argparse.Namespace, fmt: str) -> None:
    scorecard = _scorecard(doc, args)
    out = {"kind": "overview", "scanner_version": doc["scanner_version"], **scorecard}
    print(to_json(out) if fmt == "json" else "\n".join(overview_text(scorecard)))


def _cite_filter(doc: Doc, args: argparse.Namespace, *, hide_empty: bool = False) -> Doc:
    minimum = Severity[args.cite.upper()]
    return filter_findings(doc, minimum, hide_empty=hide_empty, checks=tuple(args.cite_check))


def _summary(doc: Doc, args: argparse.Namespace, fmt: str) -> None:
    shown = _cite_filter(doc, args) if args.cite else doc
    rolled = dict(
        summary(shown),
        overview=_scorecard(doc, args),
        dq_threshold=args.dq_on,
        answers=answer_tally(doc["inputs"]),
    )
    if fmt == "json":
        print(to_json(rolled))
    elif not render_summary_rich(rolled):
        print(summary_text(rolled), end="")


def _detail(doc: Doc, args: argparse.Namespace, fmt: str) -> None:
    if args.cite:
        doc = _cite_filter(doc, args, hide_empty=True)
    if fmt == "json":
        print(to_json(doc))
        return
    try:
        render_rich(doc)
    except ImportError:
        sys.stdout.write(to_text(doc))


VIEWS = {"brief": _brief, "overview": _overview, "summary": _summary, "detail": _detail}


def emit(doc: Doc, args: argparse.Namespace) -> None:
    fmt = output_format(args)
    view = args.view
    if view is None:
        # Default: the brief for a run's text view (unless citing), else the detail.
        many = len(doc["inputs"]) > 1
        view = (
            "brief"
            if fmt == "text" and (many or args.judge_prompts) and not args.cite
            else "detail"
        )
    VIEWS[view](doc, args, fmt)
