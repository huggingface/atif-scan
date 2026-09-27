"""Offline CLI. Reports never contain trace text, source paths, URLs or exception bodies."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

from .brief import brief, brief_text
from .cache import ResultCache, checks_signature
from .checks import Context, Severity, Status, identifier
from .cite import citations
from .detectors import builtin_detectors
from .engine import Engine
from .harbor_hub import harbor_sources, inspect_job, is_harbor
from .layout import document as inspection
from .loader import TraceError
from .policy import load_rules
from .report import (
    document,
    inspection_text,
    overview,
    overview_text,
    render_rich,
    report,
    summary,
    summary_text,
    to_json,
    to_text,
)
from .sources import (
    DEFAULT_PATTERN,
    HF_PREFIX,
    Source,
    SourceError,
    default_sync_root,
    file_source,
    list_input,
    normalize,
    resolve,
    sync_remote,
    sync_target,
)


def manifest_inputs(path: Path) -> list[tuple[Source, Context]]:
    """Explicit file list: `{"inputs": [{"id", "path", "task"?, "partial"?}]}`.

    Local paths resolve relative to the manifest; `hf://` paths are used as given.
    """
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) != {"inputs"}:
        raise ValueError("invalid_input_manifest")
    if not isinstance(value["inputs"], list):
        raise ValueError("invalid_input_manifest")
    result = []
    for raw in value["inputs"]:
        if not isinstance(raw, dict) or set(raw) - {"id", "path", "task", "partial", "reward"}:
            raise ValueError("invalid_input_record")
        if not {"id", "path"} <= set(raw) or not isinstance(raw["path"], str):
            raise ValueError("invalid_input_record")
        location = raw["path"]
        if "://" not in location:
            location = str(path.parent / location)
        source = file_source(identifier(raw["id"]), location)
        if "reward" in raw:
            reward = Context(reward=raw["reward"]).reward  # validated number or None
            source = Source(source.label, source.load, lambda reward=reward: reward)
        result.append((source, Context(raw.get("task"), raw.get("partial", False))))
    return result


def inputs(args: argparse.Namespace) -> list[tuple[Source, Context]]:
    if args.manifest:
        if args.paths or args.task or args.task_from or args.partial:
            raise ValueError("manifest_cannot_be_combined_with_direct_inputs")
        result = manifest_inputs(args.manifest)
    else:
        # Local/hf:// inputs resolve together so positional labels stay unique.
        local = [sync_if_remote(v, args) for v in args.paths if not is_harbor(v)]
        found = resolve(local, args.pattern, runs=args.runs) if local else []
        for value in (v for v in args.paths if is_harbor(v)):
            hub, run = harbor_sources(
                value, args.download_dir, full=args.full, workers=args.jobs, refresh=args.refresh
            )
            args.runs.append(run)
            found += hub
        result = [(s, Context(task_for(s, args), args.partial)) for s in found]
    if not result or len({s.label for s, _ in result}) != len(result):
        raise ValueError("empty_or_duplicate_inputs")
    return result


def sync_if_remote(value: str, args: argparse.Namespace) -> str:
    """With --sync (default), mirror an hf:// input locally and scan the copy."""
    if not args.sync or not normalize(value).startswith(HF_PREFIX):
        return value
    dest = sync_target(value, args.sync_root)
    tty = sys.stderr.isatty()

    def progress(done: int, total: int) -> None:
        print(f"\ratif-scan: syncing {done}/{total}", end="", file=sys.stderr)

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
            f"\r\033[Katif-scan: local copy {dest} · {counts['downloaded']} downloaded,"
            f" {counts['up_to_date']} up to date"
            + (f", {counts['failed']} failed" if counts["failed"] else ""),
            file=sys.stderr,
        )
    return str(path)


def task_for(source: Source, args: argparse.Namespace) -> str | None:
    """--task for every input, or (explicitly) derived from the trial folder name."""
    if args.task:
        return args.task
    if source.meta.get("task"):
        return str(source.meta["task"])  # recorded by the source (e.g. the Harbor Hub)
    if args.task_from != "trial-dir":
        return None
    # Harbor trial folders are `<task>__<suffix>`. Job folders can contain `__` too
    # (`2026-08-19__20-38-18`), so the part closest to the file wins.
    for part in reversed(re.split(r"[\\/]", source.hint or source.label)):
        task, separator, _ = part.partition("__")
        if separator and task:
            try:
                return identifier(task)
            except ValueError:
                return None
    return None


def price(value: str | None) -> tuple[float, float, float] | None:
    if not value:
        return None
    try:
        parts = tuple(float(p) for p in value.split(","))
    except ValueError:
        raise SystemExit("atif-scan: --price takes three numbers: U,C,O ($/M tokens)") from None
    if len(parts) != 3 or any(p < 0 for p in parts):
        raise SystemExit("atif-scan: --price takes three numbers: U,C,O ($/M tokens)")
    return parts


def run_facts(meta: dict, trace) -> dict:
    """Per-trial run facts: recorded metadata (Hub, Harbor result.json), else final_metrics."""
    usage = trace.usage if trace is not None else None
    facts = {
        "error_type": meta.get("error_type"),
        "status": meta.get("status"),
        "hub_trial_id": meta.get("hub_trial_id"),
        "duration_sec": meta.get("duration_sec"),
        "overrides": meta.get("overrides") or [],
        "cost_usd": meta.get("cost_usd"),
        "input_tokens": meta.get("input_tokens"),
        "cache_tokens": meta.get("cache_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "agent_name": trace.agent[0] if trace is not None else None,
        "agent_version": trace.agent[1] if trace is not None else None,
        "model_name": trace.agent[2] if trace is not None else None,
        "llm_calls": (trace.llm_calls or trace.agent_steps) if trace is not None else None,
    }
    if meta.get("input_tokens") is None and usage is not None:
        facts.update(
            cost_usd=usage.cost_usd,
            input_tokens=usage.prompt_tokens,
            cache_tokens=usage.cached_tokens,
            output_tokens=usage.completion_tokens,
        )
    return facts


def load_checks(args: argparse.Namespace) -> list:
    checks = builtin_detectors()
    for plugin in args.plugin:
        module, separator, factory = plugin.partition(":")
        if not separator or not module or not factory:
            raise ValueError("invalid_plugin_reference")
        checks.extend(getattr(importlib.import_module(module), factory)())
    for rules in args.rules:
        checks.extend(load_rules(json.loads(rules.read_text())))
    return checks


def emit(doc: dict, args: argparse.Namespace) -> None:
    fmt = args.format
    if fmt == "auto":
        fmt = "text" if sys.stdout.isatty() else "json"
    many = len(doc["inputs"]) > 1
    default_brief = fmt == "text" and many and not (args.detail or args.summary or args.overview)
    if args.brief or default_brief:
        report_ = brief(doc, args.dq_on, args.min_trials, args.expect_tasks, args.price_rates)
        print(to_json(report_) if fmt == "json" else brief_text(report_), end="")
        if fmt == "json":
            print()
        return
    if args.overview or args.summary:
        scorecard = overview(doc, args.dq_on, args.min_trials, expect_tasks=args.expect_tasks)
        if args.overview:
            out = {"kind": "overview", "scanner_version": doc["scanner_version"], **scorecard}
            print(to_json(out) if fmt == "json" else "\n".join(overview_text(scorecard)))
            return
        rolled = dict(summary(doc), overview=scorecard)
        print(to_json(rolled) if fmt == "json" else summary_text(rolled), end="")
        if fmt == "json":
            print()
        return
    if fmt == "json":
        print(to_json(doc))
        return
    try:
        render_rich(doc)
    except ImportError:
        sys.stdout.write(to_text(doc))


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
    cards = []
    for job in jobs:  # listing only: no trajectory is downloaded
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
    if cards:
        doc["harbor_jobs"] = cards
    fmt = args.format
    if fmt == "auto":
        fmt = "text" if sys.stdout.isatty() else "json"
    if fmt == "json":
        print(to_json(doc))
    else:
        text = inspection_text(doc) if local else ""
        for card in cards:
            text += "\n".join(overview_text(card)) + "\n"
        print(text, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--manifest", type=Path, help="explicit input manifest (JSON)")
    harbor = parser.add_argument_group("Harbor Hub jobs (harbor://jobs/<id> or a hub URL)")
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
        help="where local copies live (default: $ATIF_SCAN_SYNC_DIR, $XDG_CACHE_HOME/atif-scan "
        "or ~/.cache/atif-scan)",
    )
    harbor.add_argument(
        "--refresh", action="store_true", help="re-download synced files even if present"
    )
    harbor.add_argument(
        "--full", action="store_true", help="download the full job archive, not only trajectories"
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
    parser.add_argument(
        "--rules",
        type=Path,
        action="append",
        default=[],
        help="JSON policy with rules and/or allowances (repeatable)",
    )
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE:FACTORY",
        help="explicitly trust/import a Python factory returning detectors/rules/allowances",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "json", "text"],
        default="auto",
        help="auto: text on a terminal, JSON when piped",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="one rollup across inputs: counts for info/low, per-trace details for medium+",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="one-screen run integrity report (default text view for several inputs)",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="per-trace listing (default text view for a single input)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        metavar="DIR",
        help="per-trace result cache (default: <sync dir>/results; --no-cache disables)",
    )
    parser.add_argument("--no-cache", action="store_true", help="disable the result cache")
    parser.add_argument(
        "--overview",
        action="store_true",
        help="only the run scorecard: coverage, accuracy, DQ candidates, cost",
    )
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
    parser.add_argument(
        "--cite",
        nargs="?",
        const="medium",
        choices=[s.name.lower() for s in Severity],
        metavar="SEVERITY",
        help="include the trace text (masked excerpt + before/after context) behind "
        "findings at/above SEVERITY (default: medium). Output then contains trace text.",
    )
    parser.add_argument(
        "--fail-on",
        choices=[s.name.lower() for s in Severity],
        help="exit 1 for an unexcused finding at/above this review severity",
    )
    args = parser.parse_args(argv)
    args.price_rates = price(args.price)  # validate early, whatever the output format
    if args.inspect:
        return inspect(args)
    args.sync_root = args.sync_dir or default_sync_root()
    with tempfile.TemporaryDirectory(prefix="atif-scan-") as scratch:
        # Harbor downloads: kept under the sync root with --sync, else a temp dir.
        args.download_dir = args.sync_root / "harbor" if args.sync else Path(scratch)
        args.runs = []
        return scan(args)


def scan(args: argparse.Namespace) -> int:
    try:
        records = inputs(args)
        engine = Engine(load_checks(args))
    except SourceError as error:
        # Fixed codes only (e.g. a missing optional extra); never paths or remote messages.
        print(f"atif-scan: {error}", file=sys.stderr)
        return 2
    except Exception:
        print(
            "atif-scan: invalid input, policy or plugin configuration (details withheld)",
            file=sys.stderr,
        )
        return 2
    output = []
    invalid = failed = False
    threshold = Severity[args.fail_on.upper()] if args.fail_on else None
    cache = None
    if not args.no_cache and not args.cite:  # citations carry trace text: never cached
        directory = args.cache or args.sync_root / "results"
        if directory is not None:
            cache = ResultCache(directory, version("atif-scan"), checks_signature(engine))
    progress = sys.stderr.isatty() and len(records) > 1
    for number, (source, context) in enumerate(records, 1):
        if progress:
            print(f"\ratif-scan: scanning {number}/{len(records)}", end="", file=sys.stderr)
        # Recorded run facts (Hub listing, Harbor result.json) beat folder-name inference.
        meta = {**source.meta, **source.details()}
        task = context.task if args.task else (meta.get("task") or context.task)
        reward = source.reward()
        context = Context(
            task, context.partial, reward if reward is not None else meta.get("reward")
        )
        fingerprint = source.fingerprint() if cache is not None else None
        key = cache.key(fingerprint, context) if fingerprint else None
        cached = cache.get(key) if key else None
        if cached is not None and cached.get("input_id") == source.label:
            output.append(cached)
            rows = cached["assessments"]
            invalid = (
                invalid
                or any(a["status"] == "error" for a in rows)
                or (
                    cached["input_status"] != "available"
                    and cached.get("input_error") != "no_trajectory_downloaded"
                )
            )
            failed = failed or (
                threshold is not None
                and any(a.get("score") is not None and a["score"] >= int(threshold) for a in rows)
            )
            continue
        error = None
        try:
            trace = source.load()
        except TraceError as exc:
            trace = None
            # TraceError carries fixed codes only (e.g. trace_too_large); re-check anyway.
            error = str(exc) if re.fullmatch(r"[a-z_]{1,64}", str(exc)) else "unreadable_trace"
            # A Hub trial without a trajectory (e.g. it errored first) is a reported run
            # fact, not bad input; unreadable local/hf files still exit 2.
            invalid = invalid or error != "no_trajectory_downloaded"
        if trace is not None and trace.compacted and not context.partial:
            # Earlier history isn't recorded: negatives must not read as clean.
            context = Context(context.task, True, context.reward)
        assessments = engine.evaluate(trace, context)
        item = report(assessments)
        item.update(
            input_id=source.label,
            input_status="available" if trace is not None else "unavailable_or_invalid",
            input_error=error,
            compacted=bool(trace is not None and trace.compacted),
            task=context.task,
            reward=context.reward,
            **run_facts(meta, trace),
            partial=context.partial,
            agent_steps=trace.agent_steps if trace is not None else None,
            # Counts only: tool names and arguments never enter the report.
            tool_calls=trace.tool_calls if trace is not None else None,
            unrecognized_tool_calls=trace.unrecognized_tool_calls if trace is not None else None,
        )
        if key is not None:
            cache.put(key, item)
        if args.cite and trace is not None:
            # Opt-in trace text; the only report field that isn't allowlisted metadata.
            item["citations"] = citations(trace, assessments, Severity[args.cite.upper()])
        output.append(item)
        invalid = invalid or any(a.result.status == Status.ERROR for a in assessments)
        failed = failed or (
            threshold is not None
            and any(a.counts and a.spec.severity >= threshold for a in assessments)
        )
    if progress:
        print("\r\033[K", end="", file=sys.stderr)
    doc = document(output, version("atif-scan"))
    if args.runs:
        doc["runs"] = args.runs
    emit(doc, args)
    return 2 if invalid else 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
