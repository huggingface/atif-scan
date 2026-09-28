"""`atif-inspect`: read specific parts of one ATIF trajectory without writing JSON parsers.

    atif-inspect TRACE                      outline: one line per step (no trace text)
    atif-inspect TRACE --step 12            that step in full (message, reasoning, calls, results)
    atif-inspect TRACE --around 12 -w 2     steps 10..14
    atif-inspect TRACE --steps 5-9 --part reasoning,calls
    atif-inspect TRACE --grep 'solve\\.sh'   steps and parts matching, with windows
    atif-inspect TRACE --step 12 --json     the same as JSON

TRACE is a trajectory file or a trial folder holding `trajectory.json` (or
`agent/trajectory.json`). Step numbers are ATIF `step_id`s, as in atif-scan reports and
question prompts. Output other than the outline is trace text: credential shapes and
secret values are masked (best effort), but treat it like the trace itself. Nothing in the
trace is ever run or fetched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .cite import mask, trace_secrets
from .loader import TraceError, load_trace
from .model import Channel, Trace

PARTS = ("message", "reasoning", "calls", "results")


def resolve(path: Path) -> Path:
    if path.is_dir():
        for candidate in (path / "trajectory.json", path / "agent" / "trajectory.json"):
            if candidate.is_file():
                return candidate
        raise SystemExit("atif-inspect: no trajectory.json in that folder")
    return path


def _cut(text: str, limit: int) -> str:
    if limit and len(text) > limit:
        return text[:limit] + f"… [{len(text) - limit} more chars; raise --max-chars]"
    return text


def outline(trace: Trace) -> list[str]:
    lines = []
    for step in trace.steps:
        sid = trace.step_numbers[step.index]
        tools = ",".join(c.name for c in step.calls)
        sizes = []
        for name, text in (("msg", step.message.text), ("reasoning", step.reasoning.text)):
            if text:
                sizes.append(f"{name} {len(text)}")
        if step.observations:
            total = sum(len(o.content.text) for o in step.observations)
            sizes.append(f"results {len(step.observations)}×/{total}")
        flags = []
        if step.index in trace.compacted:
            flags.append("COMPACTION-SUMMARY")
        if any(o.content.media for o in step.observations) or step.message.media:
            flags.append("media")
        if step.copied:
            flags.append("copied")
        if any(o.pairing_reconstructed for o in step.observations):
            flags.append("PAIRING-RECONSTRUCTED")
        lines.append(
            f"{sid:>5} {step.source:<6} {tools[:60]:<60} {' · '.join(sizes)}"
            + (f"  [{', '.join(flags)}]" if flags else "")
        )
    return lines


def step_record(trace: Trace, index: int, parts, limit: int, known) -> dict:
    step = trace.steps[index]
    out: dict = {"step": trace.step_numbers[index], "source": step.source}
    if "message" in parts and step.message.text:
        out["message"] = _cut(mask(step.message.text, known), limit)
    if "reasoning" in parts and step.reasoning.text:
        out["reasoning"] = _cut(mask(step.reasoning.text, known), limit)
    if "calls" in parts and step.calls:
        out["calls"] = [
            {
                "tool": call.name,
                "category": call.tool,
                "id": call.id,
                "arguments": {
                    f"{channel.value}{i}": _cut(mask(c.text, known), limit)
                    for i, (channel, c) in enumerate(call.fields)
                    if c.text
                },
            }
            for call in step.calls
        ]
    if "results" in parts and step.observations:
        out["results"] = [
            {
                "call_id": o.source_call_id,
                "text": _cut(mask(o.content.text, known), limit),
                **({"media": True} if o.content.media else {}),
                **({"pairing_reconstructed": True} if o.pairing_reconstructed else {}),
            }
            for o in step.observations
        ]
    return out


def render(record: dict) -> str:
    lines = [f"===== step {record['step']} ({record['source']}) ====="]
    for key in ("message", "reasoning"):
        if key in record:
            lines += [f"--- {key}", record[key]]
    for call in record.get("calls", []):
        lines.append(f"--- call {call['tool']} [{call['category']}] id={call['id']}")
        lines += [f"  {k}: {v}" for k, v in call["arguments"].items()]
    for r in record.get("results", []):
        lines.append(
            f"--- result for {r['call_id']}"
            + (" [media]" if r.get("media") else "")
            + (
                " [WARNING: positional pairing reconstructed]"
                if r.get("pairing_reconstructed")
                else ""
            )
        )
        lines.append(r["text"])
    return "\n".join(lines)


def grep(trace: Trace, pattern: re.Pattern[str], known, window: int = 160) -> list[dict]:
    hits = []
    for step in trace.steps:
        texts = [("message", step.message.text), ("reasoning", step.reasoning.text)]
        texts += [
            (f"call {c.name} {ch.value}", t.text)
            for c in step.calls
            for ch, t in c.fields
            if ch != Channel.METADATA
        ]
        texts += [
            (
                f"result {o.source_call_id}"
                + (" [PAIRING-RECONSTRUCTED]" if o.pairing_reconstructed else ""),
                o.content.text,
            )
            for o in step.observations
        ]
        for part, text in texts:
            if not text:
                continue
            masked = mask(text, known)
            for m in pattern.finditer(masked):
                a, b = max(0, m.start() - window), min(len(masked), m.end() + window)
                hits.append(
                    {
                        "step": trace.step_numbers[step.index],
                        "part": part,
                        "text": ("…" if a else "") + masked[a:b] + ("…" if b < len(masked) else ""),
                    }
                )
                break  # one window per part
    return hits


def select(trace: Trace, args) -> list[int]:
    numbers = trace.step_numbers
    by_number = {n: i for i, n in enumerate(numbers)}
    wanted: set[int] = set(args.step)
    for span in args.steps:
        a, _, b = span.partition("-")
        wanted.update(range(int(a), int(b or a) + 1))
    for n in args.around:
        wanted.update(range(n - args.window, n + args.window + 1))
    return [by_number[n] for n in sorted(wanted) if n in by_number]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atif-inspect",
        description="Read parts of one ATIF trajectory (masked). Never runs trace content.",
    )
    parser.add_argument("trace", type=Path, help="trajectory file or trial folder")
    parser.add_argument("--step", type=int, action="append", default=[], help="step_id")
    parser.add_argument("--steps", action="append", default=[], help="range, e.g. 5-9")
    parser.add_argument("--around", type=int, action="append", default=[], help="step_id")
    parser.add_argument("-w", "--window", type=int, default=2, help="steps each side")
    parser.add_argument("--part", default=",".join(PARTS), help="comma list of " + "/".join(PARTS))
    parser.add_argument("--grep", help="regex over all parts (case-insensitive)")
    parser.add_argument("--max-chars", type=int, default=4000, help="per field; 0 = no limit")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    parts = {p.strip() for p in args.part.split(",") if p.strip()}
    if parts - set(PARTS):
        parser.error(f"--part takes {', '.join(PARTS)}")
    try:
        trace = load_trace(resolve(args.trace))
    except (TraceError, OSError, ValueError):
        print("atif-inspect: unreadable trajectory", file=sys.stderr)
        return 2
    known = trace_secrets(trace)
    if args.grep:
        hits = grep(trace, re.compile(args.grep, re.I), known)
        if args.json:
            print(json.dumps(hits, indent=1))
        else:
            for h in hits:
                print(f"{h['step']:>5} {h['part']}: {h['text'].replace(chr(10), ' ⏎ ')}")
        return 0 if hits else 1
    indices = select(trace, args)
    if not indices:
        if args.step or args.steps or args.around:
            print("atif-inspect: no such step(s)", file=sys.stderr)
            return 1
        lines = outline(trace)
        if args.json:
            print(json.dumps(lines, indent=1))
        else:
            print(
                f"{len(trace.steps)} steps · {trace.agent_steps} agent · "
                f"{trace.tool_calls} tool calls" + (" · compacted" if trace.compacted else "")
            )
            print("\n".join(lines))
        return 0
    records = [step_record(trace, i, parts, args.max_chars, known) for i in indices]
    if args.json:
        print(json.dumps(records, indent=1))
    else:
        print("\n\n".join(render(r) for r in records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
