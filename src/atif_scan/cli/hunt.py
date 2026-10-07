"""atif-scan hunt: answer a question bundle with fast-agent.

    atif-scan JOB --questions DIR [--question hack_hunt]       # write the bundle
    atif-scan hunt --model MODEL --questions DIR [--inspect-tool] [--jobs 4]
    atif-scan JOB --answers DIR --brief                        # read the answers back

Each prompt (<input>/<question>.md) is sent once with no shell or subagents and the
question's JSON Schema for structured output. With --inspect-tool the model also gets
read-only tools over that one trace and its local companion archives
(atif_scan.review.inspect_server): they mask secrets and run nothing. The reply goes to
<input>/<question>.answer.json and the answering run's own ATIF trajectory to
<input>/<question>.review.atif.json. Existing answers are kept unless --force; atif-scan
validates replies when it reads them (--answers). fast-agent runs `--isolated`
(atif_scan.cli.fast_agent): prompts aren't saved in its home and no plugins load.

Prompts contain masked trace text and are sent to the model's provider: only run this
with a provider you're allowed to send the traces to. Keep the bundle out of Git.
Failures are logged (first error line only) to DIR/ask-errors.log.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..data.jsonval import as_object, as_str
from .fast_agent import run_go

MCP = "mcp>=1.2,<2"
SERVER = "atif_scan.review.inspect_server"
REASON_CHARS = 300
_LOG_LOCK = threading.Lock()


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="atif-scan hunt",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="the fast-agent model to answer with")
    p.add_argument("--questions", type=Path, required=True, metavar="DIR", help="the bundle")
    p.add_argument("--question", action="append", default=[], metavar="ID", help="only these")
    p.add_argument("--jobs", type=int, default=4, help="parallel answers (default 4)")
    p.add_argument("--timeout", type=int, default=300, help="seconds per answer (default 300)")
    p.add_argument("--force", action="store_true", help="re-ask answered questions")
    p.add_argument("--dry-run", action="store_true", help="list what would be asked")
    p.add_argument(
        "--inspect-tool",
        action="store_true",
        help="let the model read more of the trace through a read-only MCP server bound to "
        f"that one trajectory (needs atif-scan[mcp], else uv to fetch {MCP})",
    )
    p.add_argument("--fast-agent", default="fast-agent", metavar="CMD", help="default: fast-agent")
    return p


def pending(bundle: Path, questions: list[str], force: bool) -> list[Path]:
    """Question metadata files still to ask: not answers, the answering runs' own
    trajectories, temp files or schemas."""
    todo = []
    for meta in sorted(bundle.glob("*/*.json")):
        name = meta.name
        if name.endswith((".answer.json", ".review.atif.json")) or ".tmp." in name:
            continue
        if meta.parent.name == "schemas" or (questions and meta.stem not in questions):
            continue
        if not force and _answered(meta):
            continue
        todo.append(meta)
    return todo


def _answered(meta: Path) -> bool:
    answer = meta.with_suffix(".answer.json")
    return answer.is_file() and answer.stat().st_size > 0


def server_command(trace: str) -> str:
    """The read-only MCP server for one trajectory, as a shell command for `--stdio`."""
    if importlib.util.find_spec("mcp") is not None:
        return shlex.join([sys.executable, "-m", SERVER, trace])
    # Layer mcp over this interpreter's environment (which provides atif_scan).
    uv = ["uv", "run", "--no-project", "--python", sys.executable, "--with", MCP]
    return shlex.join([*uv, "python", "-m", SERVER, trace])


def _command(args: argparse.Namespace, meta: Path) -> list[str] | str:
    """`fast-agent go`'s arguments for one question, or why it can't be asked."""
    extra: list[str] = []
    if args.inspect_tool:
        trace = as_str(as_object(json.loads(meta.read_text())).get("trace_path"))
        if not trace or not Path(trace).is_file():
            return "no local trajectory for --inspect-tool"
        # Keep the inspection tools available beyond the first tool round.
        extra = ["--structured-tool-policy", "always", "--stdio", server_command(trace)]
    return [
        *("--model", args.model, "--no-shell", "--no-subagents", "--quiet"),
        *extra,
        *("--trajectory-output", str(meta.with_suffix(".review.atif.json"))),
        *("--timeout", str(args.timeout)),
        *("--prompt-file", str(meta.with_suffix(".md"))),
        *("--json-schema", str(args.questions / "schemas" / f"{meta.stem}.json")),
    ]


def _reply(out: str) -> str:
    """fast-agent can print a tool-status line before the JSON even with --quiet: keep the
    last line that is a JSON object (--answers validates it either way)."""
    objects = [line for line in out.splitlines() if line.startswith("{")]
    return objects[-1] + "\n" if objects else out


def _reason(text: str) -> str:
    """The first error line only: fast-agent may otherwise echo prompt text."""
    lines = [line for line in text.splitlines() if line.strip()]
    errors = [line for line in lines if "error" in line.lower()]
    return ((errors or lines or ["no output"])[0])[:REASON_CHARS]


def answer(args: argparse.Namespace, meta: Path) -> bool:
    command = _command(args, meta)
    if isinstance(command, str):
        return _failed(args.questions, meta, command)
    run = run_go(shlex.split(args.fast_agent), command)
    failed = any(line.startswith("Error:") for line in run.stdout.splitlines())
    if run.returncode != 0 or not run.stdout.strip() or failed:
        return _failed(args.questions, meta, _reason(run.stdout + "\n" + run.stderr))
    target = meta.with_suffix(".answer.json")
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(_reply(run.stdout))
    tmp.replace(target)
    print(f"answered  {meta}", file=sys.stderr)
    return True


def _failed(bundle: Path, meta: Path, why: str) -> bool:
    with _LOG_LOCK, (bundle / "ask-errors.log").open("a") as log:
        log.write(f"{meta}\t{why}\n")
    print(f"failed    {meta}", file=sys.stderr)
    return False


def _check(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not (args.questions / "index.jsonl").is_file():
        parser.error(f"no questions in {args.questions} (run atif-scan --questions DIR first)")
    if not args.dry_run and shutil.which(shlex.split(args.fast_agent)[0]) is None:
        parser.error("fast-agent not found (--fast-agent CMD)")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")


def main(argv: list[str]) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _check(parser, args)
    todo = pending(args.questions, args.question, args.force)
    tool = ", read-only trace tool" if args.inspect_tool else ""
    print(
        f"{len(todo)} question(s) to ask with model '{args.model}' (jobs {args.jobs}{tool})",
        file=sys.stderr,
    )
    if args.dry_run:
        print("\n".join(str(meta) for meta in todo))
        return 0
    if not todo:
        return 0
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda meta: answer(args, meta), todo))
    print(f"answered {sum(results)} · failed {results.count(False)}", file=sys.stderr)
    print(f"next: atif-scan <same inputs> --answers {args.questions} --brief", file=sys.stderr)
    return 0
