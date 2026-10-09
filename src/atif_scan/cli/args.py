"""Command-line arguments: the parser, option groups and combination checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..checks import Severity, check_pattern
from ..output.bundle import SCOPES
from ..review.catalogue import BY_ID, OPEN, QUESTIONS, RETIRED
from ..sources.inputs import (
    DEFAULT_PATTERN,
)

# --price: $/M tokens for uncached input, cached input and output.
PRICE_PARTS = 3
MAX_PORT = 65535


PRICE_USAGE = "atif-scan: --price takes three numbers: U,C,O ($/M tokens)"


def price(value: str | None) -> tuple[float, float, float] | None:
    if not value:
        return None
    try:
        parts = tuple(float(p) for p in value.split(","))
    except ValueError:
        raise SystemExit(PRICE_USAGE) from None
    if len(parts) != PRICE_PARTS or any(p < 0 for p in parts):
        raise SystemExit(PRICE_USAGE)
    uncached, cached, output = parts
    return uncached, cached, output


def output_format(args: argparse.Namespace) -> str:
    """--format, with auto resolved: text on a terminal, JSON when piped."""
    if args.format == "auto":
        return "text" if sys.stdout.isatty() else "json"
    return str(args.format)


def _input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "paths",
        nargs="*",
        help="trace files, directories, hf:// paths or huggingface.co URLs "
        "(e.g. hf://buckets/org/name/run1)",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"filename glob when expanding a directory/prefix (default: {DEFAULT_PATTERN})",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="list what's under each path (layout, Harbor markers, what would be scanned) "
        "without reading any trace",
    )
    parser.add_argument(
        "--run",
        help="bench-run ID: scan its receipt-pinned job parts, each replaced trial swapped for "
        "its replacement (replaced originals kept as evidence; --as-run for the original jobs)",
    )
    parser.add_argument(
        "--as-run",
        action="store_true",
        help="with --run: scan the original job parts only, ignoring replacements",
    )
    parser.add_argument(
        "--release",
        type=Path,
        metavar="FILE",
        help="bench-run release manifest: scan exactly its reported trials (Harbor Hub's "
        "trial_ids), with replaced trials kept as evidence",
    )
    parser.add_argument(
        "--bench-root",
        type=Path,
        help="bench-run checkout (default: $ATIF_SCAN_BENCH_ROOT, else ~/source/bench-run)",
    )
    parser.add_argument("--manifest", type=Path, help="explicit input manifest (JSON)")
    parser.add_argument(
        "--submission",
        type=Path,
        metavar="FILE",
        help="a leaderboard submission file (e.g. terminal-bench leaderboard/submissions/"
        "*.json): scan its source_jobs from the Harbor Hub as one run, and check every trial "
        "against its source_filter",
    )
    harbor = parser.add_argument_group("remote inputs (hf://, harbor://jobs/<id> or a hub URL)")
    harbor.add_argument(
        "--sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mirror remote (hf://, harbor://) inputs locally and scan the copy; re-runs only "
        "fetch what changed (default: on; --no-sync streams without keeping files)",
    )
    harbor.add_argument(
        "--sync-dir",
        "--sync-to",
        dest="sync_dir",
        type=Path,
        metavar="DIR",
        help="where local copies live (default: $ATIF_SCAN_SYNC_DIR, else $ATIF_SCAN_HOME, "
        "$XDG_CACHE_HOME/atif-scan or ~/.cache/atif-scan)",
    )
    harbor.add_argument(
        "--refresh", action="store_true", help="re-download synced files even if present"
    )
    harbor.add_argument(
        "--full",
        action="store_true",
        help="download full Harbor archives, including companion session history; "
        "detectors still scan ATIF only",
    )
    harbor.add_argument("--jobs", type=int, default=16, help="parallel downloads (default 16)")
    tasks = parser.add_mutually_exclusive_group()
    tasks.add_argument("--task", help="task identity for task-scoped checks")
    tasks.add_argument(
        "--task-from",
        choices=["trial-dir"],
        help="derive each input's task from its Harbor trial folder (<task>__<id>)",
    )
    parser.add_argument(
        "--partial", action="store_true", help="live/incomplete trace; negatives unknown"
    )


def _check_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rules",
        type=Path,
        action="append",
        default=[],
        help="JSON policy with rules and/or allowances (repeatable)",
    )
    parser.add_argument(
        "--packs",
        choices=["auto", "none"],
        default="auto",
        help="auto: load the bundled task packs (e.g. tb21) for runs recognised by their "
        "recorded dataset or tasks; none: built-ins and --plugin only",
    )
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE:FACTORY",
        help="explicitly trust/import a Python factory returning detectors/rules/allowances",
    )


def _output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=["auto", "json", "text"],
        default="auto",
        help="auto: text on a terminal, JSON when piped",
    )
    views = parser.add_mutually_exclusive_group()
    for flag, help_ in (
        ("summary", "one rollup across inputs: counts for info/low, per-trace details for medium+"),
        ("brief", "one-screen run integrity report (default text view for several inputs)"),
        ("detail", "per-trace listing (default text view for a single input)"),
        ("overview", "only the run scorecard: coverage, accuracy, DQ candidates, cost"),
    ):
        views.add_argument(f"--{flag}", dest="view", action="store_const", const=flag, help=help_)
    parser.add_argument(
        "--cache",
        type=Path,
        metavar="DIR",
        help="per-trace result cache (default: <sync dir>/results; --no-cache disables)",
    )
    parser.add_argument("--no-cache", action="store_true", help="disable the result cache")
    parser.add_argument(
        "--dq-on",
        choices=[s.name.lower() for s in Severity if s >= Severity.LOW],
        default="high",
        help="severity at which a rewarded trial's unexcused finding is a DQ candidate",
    )
    parser.add_argument(
        "--min-trials",
        type=int,
        help="expected trials per task (default: the job's n_attempts when known)",
    )
    parser.add_argument(
        "--price",
        metavar="U,C,O",
        help="$ per million uncached-input,cached-input,output tokens: estimate unpriced trials",
    )
    parser.add_argument(
        "--expect-tasks",
        type=int,
        help="tasks the dataset should cover (e.g. 89 for Terminal-Bench 2.1)",
    )


def _question_id(value: str) -> str:
    if value in BY_ID:
        return value
    if value in RETIRED:
        message = f"{value} was retired: ask {RETIRED[value]} (with --blind for a blind review)"
        raise argparse.ArgumentTypeError(message)
    raise argparse.ArgumentTypeError(f"unknown question; choose from {', '.join(BY_ID)}")


def _review_arguments(parser: argparse.ArgumentParser) -> None:
    for flags, help_ in (
        (
            ("--questions",),
            "write a private, MCP-ready review bundle (prompts with masked trace text) to a "
            "new/empty DIR, for a person, `atif-scan hunt` or any LLM to answer; never calls "
            "a model. Default: the DQ candidates, asked hack_hunt plus each finding-specific "
            "question that applies",
        ),
        (("--judge-prompts", "--judge"), argparse.SUPPRESS),  # the former names
    ):
        parser.add_argument(*flags, dest="questions", type=Path, metavar="DIR", help=help_)
    for flags, help_ in (
        (
            ("--question-scope",),
            "trials the bundle covers: dq-candidates (default, uses --dq-on), rewarded, or all "
            "trials including failed and unknown-reward controls",
        ),
        (("--judge-scope",), argparse.SUPPRESS),  # the former name
    ):
        parser.add_argument(*flags, dest="question_scope", choices=SCOPES, help=help_)
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        type=_question_id,
        metavar="ID",
        help="only these questions (repeatable): the finding-specific "
        f"{', '.join(q.id for q in QUESTIONS)}; the open {', '.join(q.id for q in OPEN)}; "
        "or web_provenance",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="ask the open questions without showing scanner findings, so the answers can "
        "measure the scanner (default question: hack_hunt)",
    )
    parser.add_argument(
        "--answers",
        type=Path,
        metavar="DIR",
        help="read answers (<input>/<question>.answer.json) from a --questions DIR and "
        "annotate the report; answers never change findings or DQ candidates",
    )
    parser.add_argument(
        "--image-model",
        metavar="MODEL",
        help="have this fast-agent model transcribe the images that leave a check unknown "
        "(via `fast-agent go --attach`), and re-check with the transcripts. Sends those "
        "images to the model's provider; transcripts are kept in <atif-scan home>/images/",
    )
    parser.add_argument(
        "--cite",
        nargs="?",
        const="medium",
        choices=[s.name.lower() for s in Severity],
        metavar="SEVERITY",
        help="show finding rows at/above SEVERITY (default: medium), with masked trace "
        "excerpts and context. Scores and coverage still use the full scan. "
        "Output then contains trace text.",
    )
    parser.add_argument(
        "--cite-check",
        action="append",
        default=[],
        type=_check_glob,
        metavar="CHECK",
        help="cite only checks matching this ID or glob (repeatable, e.g. 'awareness.*'); "
        "implies --cite info unless --cite sets a level. Output then contains trace text.",
    )
    parser.add_argument(
        "--fail-on",
        choices=[s.name.lower() for s in Severity],
        help="exit 1 for an unexcused finding at/above this review severity",
    )


def _check_glob(value: str) -> str:
    try:
        return check_pattern(value)
    except ValueError:
        message = "expected a check ID or glob, e.g. 'awareness.*'"
        raise argparse.ArgumentTypeError(message) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan ATIF trajectories with evidence-scoped behavioural checks. Never "
        "runs trace commands or calls a model (unless --image-model); reports carry no "
        "trace text unless you ask.",
        epilog="commands: `atif-scan labels …` (the label store) and `atif-scan hunt …` "
        "(answer a question bundle with fast-agent); see their --help. To scan an input "
        "named labels or hunt, write ./labels or ./hunt.",
    )
    _input_arguments(parser)
    _check_arguments(parser)
    _output_arguments(parser)
    _review_arguments(parser)
    _browse_arguments(parser)
    return parser


def _check_review_dir(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """--questions needs durable local traces and a new or empty directory."""
    if args.answers or args.inspect:
        parser.error("--questions cannot be combined with --answers or --inspect")
    if not args.sync:
        parser.error("--questions requires durable local traces; omit --no-sync")
    directory = Path(args.questions)
    try:
        if directory.is_symlink() or (
            directory.exists() and (not directory.is_dir() or any(directory.iterdir()))
        ):
            parser.error("--questions requires a new or empty directory")
    except OSError:
        parser.error("review directory unavailable (details withheld)")


def _check_questions(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.questions:
        if args.question_scope or args.blind or args.question:
            parser.error("--question, --question-scope and --blind require --questions DIR")
        return
    _check_review_dir(parser, args)
    open_ids = {q.id for q in OPEN}
    if args.blind and (closed := [q for q in args.question if q not in open_ids]):
        parser.error(f"--blind asks open questions only, not {', '.join(closed)}")


def _check_combinations(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    _check_browse(parser, args)
    _check_questions(parser, args)
    if args.cite_check and not args.cite:
        args.cite = "info"  # checks picked by name: severity shouldn't hide them
    if args.cite and (args.view in ("brief", "overview") or args.inspect):
        parser.error(
            "--cite/--cite-check require detail or summary output, not brief/overview/inspect"
        )


def _browse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("port must be an integer from 0 to 65535") from None
    if not 0 <= port <= MAX_PORT:
        raise argparse.ArgumentTypeError("port must be an integer from 0 to 65535")
    return port


def _browse_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--browse",
        action="store_true",
        help="serve a private local evidence browser after scanning; never opens a browser",
    )
    parser.add_argument(
        "--browse-port",
        type=_browse_port,
        metavar="N",
        help="loopback port for --browse (default: 0, choose a free port)",
    )
    parser.add_argument(
        "--feedback-dir",
        type=Path,
        metavar="DIR",
        help="private browser feedback (default: <atif-scan home>/feedback); requires --browse",
    )
    parser.add_argument(
        "--viewer",
        type=Path,
        metavar="DIR",
        help="write a static viewer of finding-related steps (masked text) to a new or empty DIR",
    )
    parser.add_argument(
        "--viewer-full",
        action="store_true",
        help="with --viewer: include informational matches and all recorded steps "
        "(blind --review exports always include all steps)",
    )
    parser.add_argument(
        "--highlights",
        type=Path,
        metavar="DIR",
        help="write a static highlight report to a new or empty DIR: masked excerpts of "
        "where findings, benchmark-awareness stages and judge concerns (--answers) happened",
    )
    parser.add_argument(
        "--review",
        metavar="QUESTION",
        type=_question_id,
        help="with --viewer: a blind human-review export for this question (no findings, "
        "scores or answers; its candidate steps to jump between; verdicts saved in the "
        "browser and exported as a file for `atif-scan labels import-review`)",
    )


def _check_viewer_dir(
    parser: argparse.ArgumentParser, directory: Path, flag: str = "--viewer"
) -> None:
    try:
        if directory.is_symlink() or (
            directory.exists() and (not directory.is_dir() or any(directory.iterdir()))
        ):
            parser.error(f"{flag} requires a new or empty directory")
    except OSError:
        parser.error("export directory unavailable (details withheld)")


def _check_browse(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.browse and (args.browse_port is not None or args.feedback_dir is not None):
        parser.error("--browse-port and --feedback-dir require --browse")
    _check_viewer_options(parser, args)
    if sum((args.browse, args.viewer is not None, args.highlights is not None)) > 1:
        parser.error("--browse, --viewer and --highlights are separate modes: pick one")
    if args.viewer is not None:
        _check_viewer_dir(parser, args.viewer)
    if args.highlights is not None:
        _check_viewer_dir(parser, args.highlights, "--highlights")
    if args.browse or args.viewer is not None or args.highlights is not None:
        _check_desk_conflicts(parser, args)


def _check_viewer_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.review is not None and args.viewer is None:
        parser.error("--review requires --viewer DIR")
    if args.viewer_full and args.viewer is None:
        parser.error("--viewer-full requires --viewer DIR")


def _check_desk_conflicts(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """--browse and --viewer show a scan: no bundle writing, citations or inspection."""
    incompatible = (
        args.inspect,
        not args.sync,
        args.cite,
        args.cite_check,
        args.questions,
        args.question,
        args.question_scope,
        args.blind,
    )
    if any(incompatible):
        flag = "--browse" if args.browse else "--viewer" if args.viewer else "--highlights"
        parser.error(
            f"{flag} cannot be combined with --inspect, --no-sync, --cite/--cite-check, "
            "--questions, --question, --question-scope or --blind"
        )
    # Answers are shown as annotations; a blind review export must not show a judge's.
    if args.answers and args.review is not None:
        parser.error("--review is a blind export: it cannot be combined with --answers")
