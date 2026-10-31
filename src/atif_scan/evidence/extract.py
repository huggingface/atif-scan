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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..data.loader import TraceError, load_trace
from ..data.model import Channel, Content, Observation, Step, Trace
from .cite import mask, trace_secrets
from .navigation import LEGEND, TAGS, all_tags, ranges, tag_index

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence, Set

    from ..data.jsonval import Doc

PARTS = ("message", "reasoning", "calls", "results")
SegmentPart = Literal["message", "reasoning", "call", "call_info", "result"]
SegmentLocator = tuple[SegmentPart, int, int]
MAX_SEGMENT_CHARS = 6000


def _pairing_provenance(observation: Observation) -> Doc:
    """Inspection-only allowlist; never serialize the observation itself."""
    return {
        "source_call_index": observation.source_call_index,
        "pairing_method": observation.pairing_method,
        **({"pairing_reconstructed": True} if observation.pairing_reconstructed else {}),
    }


def _pairing_warning(result: Doc) -> str:
    if not result.get("pairing_reconstructed"):
        return ""
    method = result.get("pairing_method")
    if method == "unique_remainder":
        return " [WARNING: pairing reconstructed by unique remainder]"
    return " [WARNING: positional pairing reconstructed]"


def _segment_content(step: Step, part: SegmentPart, index: int, field: int) -> tuple[Content, Doc]:
    if part in ("message", "reasoning"):
        if index or field:
            raise ValueError("message/reasoning require index=0 and field=0")
        return (step.message if part == "message" else step.reasoning), {}
    if part == "result":
        if field:
            raise ValueError("result requires field=0")
        if index >= len(step.observations):
            raise ValueError("no such observation index")
        observation = step.observations[index]
        return observation.content, _pairing_provenance(observation)
    if part == "call_info":
        return _call_metadata(step, index, field)
    return _call_segment_content(step, index, field)


def _call_metadata(step: Step, index: int, field: int) -> tuple[Content, Doc]:
    """Counts-only call anchor, including calls with no inspectable argument fields."""
    if field or index >= len(step.calls):
        raise ValueError("invalid call metadata index")
    call = step.calls[index]
    inspectable = sum(bool(c.understood or c.text or c.media) for _, c in call.fields)
    text = (
        f"Recorded tool call {index}.\n"
        f"Inspectable argument fields: {inspectable}.\n"
        f"Linked recorded results: {len(step.results_for(call))}.\n"
        "Call metadata only; missing arguments or results have not been recovered."
    )
    return Content(text), {"metadata_only": True}


def _call_segment_content(step: Step, index: int, field: int) -> tuple[Content, Doc]:
    if index >= len(step.calls):
        raise ValueError("no such call index")
    call = step.calls[index]
    if field >= len(call.fields):
        raise ValueError("no such call field index")
    channel, content = call.fields[field]
    return content, {"channel": channel.value}


def _segment_status(content: Content, part: SegmentPart) -> str:
    if content.media:
        return "media_partial" if content.text else "media"
    if not content.understood:
        return "unreadable_partial" if content.text else "unreadable"
    if content.text:
        return "text"
    # The parsed model does not retain presence bits for message/reasoning.
    return "empty_or_absent" if part in ("message", "reasoning") else "empty"


def read_segment(
    trace: Trace,
    step_number: int,
    part: SegmentPart,
    index: int = 0,
    field: int = 0,
    offset: int = 0,
    limit: int = 3000,
    known: frozenset[str] | None = None,
) -> Doc:
    """Private inspection of ONE field, masked before slicing.

    Indices are zero-based (call fields use ToolCall.fields order). Offsets and
    total_length count masked characters, not raw characters. No raw metadata,
    tool arguments, paths or sibling fields are returned. Empty message/reasoning
    cannot be distinguished from absence in the parsed model.
    """
    if part not in ("message", "reasoning", "call", "call_info", "result"):
        raise ValueError("invalid part")
    if any(type(n) is not int or n < 0 for n in (index, field, offset)):
        raise ValueError("indices and offset must be non-negative integers")
    if type(limit) is not int or not 1 <= limit <= MAX_SEGMENT_CHARS:
        raise ValueError(f"limit must be 1-{MAX_SEGMENT_CHARS}")
    if type(step_number) is not int or step_number not in trace.step_numbers:
        raise ValueError("no such step number")
    step = trace.steps[trace.step_numbers.index(step_number)]
    content, provenance = _segment_content(step, part, index, field)
    masked = mask(content.text, trace_secrets(trace) if known is None else known)
    if offset > len(masked):
        raise ValueError("offset exceeds masked field length")
    end = min(offset + limit, len(masked))
    return {
        "step": step_number,
        "part": part,
        "index": index,
        "field": field,
        "status": _segment_status(content, part),
        "media": content.media,
        "understood": content.understood,
        "offset": offset,
        "end_offset": end,
        "limit": limit,
        "max_limit": MAX_SEGMENT_CHARS,
        "total_length": len(masked),
        "next_offset": end if end < len(masked) else None,
        "truncated": end < len(masked),
        "text": masked[offset:end],
        **provenance,
    }


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


def outline(
    trace: Trace,
    span: tuple[int | None, int | None] = (None, None),
    nav: Sequence[Sequence[str]] | None = None,
) -> list[str]:
    """One line per step (step numbers `span[0]`..`span[1]`, inclusive): number, source,
    tool names, text sizes, flags and navigation tags (`nav`, by step index; computed
    when not given). No trace text."""
    nav = all_tags(trace) if nav is None else nav
    lines = []
    for step in in_span(trace, span):
        sid = trace.step_numbers[step.index]
        tools = ",".join(c.name for c in step.calls)
        flags = _flags(trace, step)
        marks = " ".join(f"#{t}" for t in nav[step.index])
        lines.append(
            f"{sid:>5} {step.source:<6} {tools[:60]:<60} {' · '.join(_sizes(step))}"
            + (f"  [{', '.join(flags)}]" if flags else "")
            + (f"  {marks}" if marks else "")
        )
    return lines


def tag_summary(trace: Trace, nav: Sequence[Sequence[str]] | None = None) -> list[str]:
    """Where each navigation tag occurs across the whole trace, with the legend."""
    index = tag_index(trace, all_tags(trace) if nav is None else nav)
    if not index:
        return [f"tags: none ({LEGEND})"]
    return [f"tags ({LEGEND}):"] + [
        f"  #{tag} ({TAGS[tag]}): steps {ranges(steps)}" for tag, steps in index.items()
    ]


def in_span(trace: Trace, span: tuple[int | None, int | None]) -> list[Step]:
    """The steps whose ATIF step numbers lie in `span` (inclusive; None is open)."""
    first, last = span
    numbers = trace.step_numbers
    return [
        step
        for step in trace.steps
        if (first is None or numbers[step.index] >= first)
        and (last is None or numbers[step.index] <= last)
    ]


def _sizes(step: Step) -> list[str]:
    sizes = [
        f"{name} {len(text)}"
        for name, text in (("msg", step.message.text), ("reasoning", step.reasoning.text))
        if text
    ]
    if step.observations:
        total = sum(len(o.content.text) for o in step.observations)
        sizes.append(f"results {len(step.observations)}×/{total}")
    return sizes


def _flags(trace: Trace, step: Step) -> list[str]:
    flags = {
        "COMPACTION-SUMMARY": step.index in trace.compacted,
        "media": any(o.content.media for o in step.observations) or step.message.media,
        "copied": step.copied,
        "PAIRING-RECONSTRUCTED": any(o.pairing_reconstructed for o in step.observations),
    }
    return [flag for flag, on in flags.items() if on]


def step_record(
    trace: Trace, index: int, parts: Set[str], limit: int, known: frozenset[str]
) -> Doc:
    step = trace.steps[index]
    out: Doc = {"step": trace.step_numbers[index], "source": step.source}
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
                **_pairing_provenance(o),
            }
            for o in step.observations
        ]
    return out


def render(record: Doc) -> str:
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
            + _pairing_warning(r)
            + (
                f" [source_call_index={r['source_call_index']}; "
                f"pairing_method={r.get('pairing_method')}]"
                if r.get("source_call_index") is not None
                else ""
            )
        )
        lines.append(r["text"])
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Query:
    """One search request: a pattern, the windows to show, a step-number range."""

    pattern: str
    max_hits: int = 20
    span: tuple[int | None, int | None] = (None, None)
    windows: int = 1


# Matches counted per part before the count is reported as a lower bound.
MAX_COUNT = 1000


def segments(step: Step) -> Iterator[tuple[str, SegmentLocator, Content]]:
    """Every searchable part of a step: (label, read_segment locator, content)."""
    yield "message", ("message", 0, 0), step.message
    yield "reasoning", ("reasoning", 0, 0), step.reasoning
    for i, call in enumerate(step.calls):
        for f, (ch, content) in enumerate(call.fields):
            if ch != Channel.METADATA:
                yield f"call {call.name} {ch.value}", ("call", i, f), content
    for i, o in enumerate(step.observations):
        label = f"result {o.source_call_id}" + (
            " [PAIRING-RECONSTRUCTED]" if o.pairing_reconstructed else ""
        )
        yield label, ("result", i, 0), o.content


def grep(
    trace: Trace,
    pattern: re.Pattern[str],
    known: frozenset[str],
    window: int = 160,
    *,
    span: tuple[int | None, int | None] = (None, None),
    windows: int = 1,
) -> list[Doc]:
    """Masked windows around matches, in step order: up to `windows` per part, each with
    the part's match count (`matches`, at least that many when `matches_capped`), the
    masked offset of its match and the `read_segment` locator of the part."""
    hits: list[Doc] = []
    for step in in_span(trace, span):
        sid = trace.step_numbers[step.index]
        for label, (part, index, field), content in segments(step):
            if not content.text:
                continue
            masked = mask(content.text, known)
            found = _matches(pattern, masked)
            for m in found[: max(1, windows)]:
                a, b = max(0, m.start() - window), min(len(masked), m.end() + window)
                hits.append(
                    {
                        "step": sid,
                        "part": label,
                        "segment": {"part": part, "index": index, "field": field},
                        "matches": len(found),
                        **({"matches_capped": True} if len(found) >= MAX_COUNT else {}),
                        "offset": m.start(),
                        "text": ("…" if a else "") + masked[a:b] + ("…" if b < len(masked) else ""),
                    }
                )
    return hits


def _matches(pattern: re.Pattern[str], text: str) -> list[re.Match[str]]:
    found: list[re.Match[str]] = []
    for m in pattern.finditer(text):
        found.append(m)
        if len(found) >= MAX_COUNT:
            break
    return found


def unsearchable(trace: Trace, span: tuple[int | None, int | None]) -> dict[str, list[int]]:
    """Step numbers in `span` whose parts hold content a text search can't see: media
    (images and the like) and unreadable parts (e.g. arguments built at runtime)."""
    gaps: dict[str, list[int]] = {"media": [], "unreadable": []}
    for step in in_span(trace, span):
        contents = [c for _, _, c in segments(step)]
        sid = trace.step_numbers[step.index]
        if any(c.media for c in contents):
            gaps["media"].append(sid)
        if any(not c.understood for c in contents):
            gaps["unreadable"].append(sid)
    return gaps


def select(trace: Trace, args: argparse.Namespace) -> list[int]:
    numbers = trace.step_numbers
    by_number = {n: i for i, n in enumerate(numbers)}
    wanted: set[int] = set(args.step)
    for span in args.steps:
        a, _, b = span.partition("-")
        wanted.update(range(int(a), int(b or a) + 1))
    for n in args.around:
        wanted.update(range(n - args.window, n + args.window + 1))
    return [by_number[n] for n in sorted(wanted) if n in by_number]


def _parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
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
        return _print_grep(trace, args, known)
    indices = select(trace, args)
    if not indices:
        return _print_outline(trace, args)
    records = [step_record(trace, i, parts, args.max_chars, known) for i in indices]
    if args.json:
        print(json.dumps(records, indent=1))
    else:
        print("\n\n".join(render(r) for r in records))
    return 0


def _print_grep(trace: Trace, args: argparse.Namespace, known: frozenset[str]) -> int:
    hits = grep(trace, re.compile(args.grep, re.I), known)
    if args.json:
        print(json.dumps(hits, indent=1))
    else:
        for h in hits:
            count = f" ({_count(h)} matches)" if h["matches"] > 1 else ""
            print(f"{h['step']:>5} {h['part']}{count}: {h['text'].replace(chr(10), ' ⏎ ')}")
    return 0 if hits else 1


def _count(hit: Doc) -> str:
    """A part's match count; `1000+` when counting stopped."""
    return f"{hit['matches']}{'+' if hit.get('matches_capped') else ''}"


def _print_outline(trace: Trace, args: argparse.Namespace) -> int:
    """The outline when no step was asked for; exit 1 if the asked-for steps don't exist."""
    if args.step or args.steps or args.around:
        print("atif-inspect: no such step(s)", file=sys.stderr)
        return 1
    nav = all_tags(trace)
    lines = outline(trace, nav=nav)
    if args.json:
        print(json.dumps(lines, indent=1))
    else:
        print(
            f"{len(trace.steps)} steps · {trace.agent_steps} agent · "
            f"{trace.tool_calls} tool calls" + (" · compacted" if trace.compacted else "")
        )
        print("\n".join(tag_summary(trace, nav)))
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
