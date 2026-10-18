"""Offline shell-output envelope evidence; no I/O, execution, models or tokenizer.

Call ``parse_output(text, family)`` on ONE decoded tool-result text, not a whole
trajectory. The parent selects benchmark shell calls, excludes judges, unwraps
ATIF observation.results[].content block lists (including serialized JSON lists),
counts calls/features and correlates later spool-path/process-handle accesses.
Some historical Codex observations are Python repr block lists, not JSON: those
also require safe parent-side content decoding, never eval. This module does not
unwrap lists, concatenate observations or inspect files named in text.

``truncation=False`` means no explicit *recognized* truncation, NOT completeness.
``kind`` distinguishes screen views, possible damaged/new envelopes, unsupported
families and no marker. Command output can imitate any harness template; signatures
are evidence, not authenticated provenance. Version strings alone prove nothing.
Only ``to_doc()`` is an export: fixed codes/counts, no paths, snippets or family input.
``spool_paths`` is PRIVATE in-memory correlation data (also hidden from repr).
Never serialize the dataclass with asdict or persist it as a report.

Authoritative source evidence (checked-out source, NOT historical config):
* fast-agent: src/fast_agent/tools/output_truncation.py:30-79
  emits byte head/tail notice; ~token figures are estimates, NOT token boundaries.
* fast-agent: src/fast_agent/tools/shell_output.py:18-30,106-195
  emits previews, retained-file guidance and managed-process-handle guidance.
* fast-agent: src/fast_agent/tools/transient_artifacts.py:62-82
  defines complete/partial retained artifact locator templates.
* mini-swe-agent: src/minisweagent/config/mini.yaml:112-128
  emits JSON head/tail at >=10000 Python characters, including zero elision.
* mini-swe-agent: src/minisweagent/config/default.yaml:114-141
  emits the XML variant, also characters (5000+5000), NOT bytes or tokens.
* harbor (a59ebaac): src/harbor/agents/terminus_2/terminus_2.py:531-567
  emits byte head/tail omission; UTF-8 edge decoding can retain fewer than limit.
* harbor (a59ebaac): src/harbor/agents/terminus_2/tmux_session.py:841-876
  emits Current Terminal Screen / New Terminal Output. A screen is NOT an explicit
  byte truncation, and neither label proves complete command output.

Representative first-per-job samples from hf/harbor/harbor-full also established
Codex warning/line-count, Claude persisted-output, Grok middle-marker and Muse JSON
signatures. Their emitting upstream implementation/config is NOT locally verified.
Pinned local samples additionally establish the older fast-agent quota notice,
Codex Original token count headers, dext byte notices and indusagi structured
truncation metadata. Their emitting sources are not locally verified. Cursor-cli
outputLocation alone does not establish truncation and remains unsupported. Ordinary
Claude '... (truncated)' lines can be program output and are deliberately ignored.
Do not infer global frequency/absence, default limits or historical configurations
from these samples. Boundaries are reported units, never interchangeable estimates.
Payload lengths are unknown unless wrapper separation is reliable. Returned lengths
include wrappers; invalid Unicode has unknown UTF-8 bytes, not a guessed encoding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from atif_scan.data.jsonval import as_object, as_str, count

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc, JsonObject

Kind = Literal[
    "explicit_truncation", "terminal_screen", "possible_signal", "no_marker", "unsupported"
]
Unit = Literal["chars", "bytes", "lines", "tokens", "KB"]
Marker = Literal[
    "fast_agent.head_tail_bytes",
    "fast_agent.preview_bytes",
    "codex.token_warning",
    "claude.persisted_output",
    "mini_swe.json_elision",
    "mini_swe.xml_elision",
    "terminus.byte_omission",
    "terminus.current_screen",
    "grok.middle_omission",
    "muse.json_truncated",
    "dext.byte_notice",
    "indusagi.truncation_metadata",
]
_SUPPORTED = frozenset(
    {
        "fast-agent",
        "codex",
        "claude-code",
        "mini-swe-agent",
        "terminus-2",
        "grok-build",
        "muse-code",
        "dext",
        "indusagi",
    }
)
_JSON_LIMIT = 2 * 1024 * 1024
_MINI_HALF = 5000
_MINI_LIMIT = 2 * _MINI_HALF
_DIGITS = r"[0-9]{1,18}"
_PATH = r"/[A-Za-z0-9._/-]+"
_READ_FILE = (
    r" Use read_text_file for selected line ranges or run a targeted search against that file; "
    r"avoid reading the entire file unless necessary\."
)
_ARTIFACT = (
    rf"(?:The complete output is available during this session at (?P<complete_path>{_PATH})\."
    rf"|The first (?P<retained>{_DIGITS}) bytes of the output are available during "
    rf"this session at (?P<partial_path>{_PATH}); the temporary-file quota was reached\.)"
    + _READ_FILE
)
_LEGACY_ARTIFACT = (
    rf"The first (?P<legacy_retained>{_DIGITS}) bytes retained before the temporary-file "
    rf"quota was reached is available during this session at (?P<legacy_path>{_PATH})\."
    + _READ_FILE
)
_PROCESS = (
    rf"(?:The complete output|The first (?P<process_retained>{_DIGITS}) bytes retained "
    r"before the temporary-output quota was reached) can be read through the managed process "
    r"handle\. Use the process tool with action='read_output' and the process_id from this result\."
)
_GUIDANCE = (
    r"(?:Increase shell_execution\.output_byte_limit to retain more\."
    r"|Use action='read_output' for retained output\."
    + "|"
    + _ARTIFACT
    + "|"
    + _LEGACY_ARTIFACT
    + "|"
    + _PROCESS
    + ")"
    + r"(?: Also, before drawing conclusions from truncated output, inspect the relevant "
    r"retained content\.)?"
)
_FAST_HEAD = re.compile(
    rf"^\[Output truncated: showing first (?P<head>{_DIGITS}) bytes and last "
    rf"(?P<tail>{_DIGITS}) bytes of (?P<total>{_DIGITS}) bytes "
    rf"\(~(?P<shown_tokens>{_DIGITS}) of ~(?P<total_tokens>{_DIGITS}) tokens\); "
    rf"omitted (?P<omitted>{_DIGITS}) middle bytes\. (?P<guidance>{_GUIDANCE})\]$",
    re.M,
)
_FAST_PREVIEW = re.compile(
    rf"^\[Output truncated: showing (?P<shown>{_DIGITS}) of (?P<total>{_DIGITS}) bytes\. "
    rf"(?P<guidance>{_GUIDANCE})\]$",
    re.M,
)
_CODEX_HEADER = re.compile(
    r"\A(?:Script completed\nWall time [0-9.]+ seconds\nOutput:\n"
    r"|Chunk ID: [A-Za-z0-9_-]+\nWall time: [0-9.]+ seconds\n"
    r"(?:Process exited with code -?[0-9]+|Process running with session ID [0-9]+)\n"
    rf"(?:Original token count: (?P<original_tokens>{_DIGITS})\n)?"
    r"(?:Final output|Output):\n)?\n?"
)
_CODEX_WARNING = re.compile(
    rf"Warning: truncated output \(original token count: (?P<tokens>{_DIGITS})\)\n"
    rf"(?:Total output lines: (?P<lines>{_DIGITS})\n)?"
)
_CLAUDE = re.compile(
    rf"\A<persisted-output>\nOutput too large \([0-9]+(?:\.[0-9]+)?[KMG]?B\)\. "
    rf"Full output saved to: (?P<path>{_PATH})\n\n"
    rf"Preview \(first (?P<preview>{_DIGITS})(?P<unit>KB| bytes| characters)\):\n"
    r"(?P<body>[\s\S]*?)\n</persisted-output>(?:\n|\Z)"
)
_TERMINUS = re.compile(
    rf"\n\[\.\.\. output limited to (?P<limit>{_DIGITS}) bytes; "
    rf"(?P<omitted>{_DIGITS}) interior bytes omitted \.\.\.\]\n"
)
_MINI_WARNING = (
    "<warning>\nThe output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can try use head, tail or sed to view a smaller number "
    "of lines selectively.\n"
    "If you're using grep or find and it produced too much output, you can use a more "
    "selective search pattern.\n"
    "If you really need to see something from the full command's output, you can redirect "
    "output to a file and then search in that file.\n</warning>"
)
_MINI_XML = re.compile(
    r"\A<returncode>-?[0-9]+</returncode>\n"
    + re.escape(_MINI_WARNING)
    + r"\s*<output_head>\n(?P<head>[\s\S]*?)\n</output_head>\s*"
    + rf"<elided_chars>\n(?P<elided>{_DIGITS}) characters elided\n</elided_chars>\s*"
    + r"<output_tail>\n(?P<tail>[\s\S]*?)\n</output_tail>\n?\Z"
)


@dataclass(frozen=True)
class OutputEvidence:
    """Typed evidence; use to_doc for export, never the private spool locators."""

    truncation: bool = False
    marker: Marker | None = None
    kind: Kind = "no_marker"
    spool_paths: tuple[str, ...] = field(default=(), repr=False)
    spool_indicated: bool = False
    spool_complete: bool | None = None
    reported_original_chars: int | None = None
    reported_original_bytes: int | None = None
    reported_original_lines: int | None = None
    reported_original_tokens: int | None = None
    estimated_original_tokens: int | None = None
    omitted_chars: int | None = None
    omitted_bytes: int | None = None
    limit: int | None = None
    limit_unit: Unit | None = None
    payload_chars: int | None = None
    payload_bytes: int | None = None
    returned_chars: int | None = None
    returned_bytes: int | None = None

    def to_doc(self) -> Doc:
        """Explicit allowlist; no dataclass dumps, raw text, family or paths."""
        return {
            "truncation": (
                None if self.kind in {"unsupported", "possible_signal"} else self.truncation
            ),
            "marker": self.marker,
            "kind": self.kind,
            "spool_indicated": self.spool_indicated,
            "spool_complete": self.spool_complete,
            "reported_original_chars": self.reported_original_chars,
            "reported_original_bytes": self.reported_original_bytes,
            "reported_original_lines": self.reported_original_lines,
            "reported_original_tokens": self.reported_original_tokens,
            "estimated_original_tokens": self.estimated_original_tokens,
            "omitted_chars": self.omitted_chars,
            "omitted_bytes": self.omitted_bytes,
            "limit": self.limit,
            "limit_unit": self.limit_unit,
            "payload_chars": self.payload_chars,
            "payload_bytes": self.payload_bytes,
            "returned_chars": self.returned_chars,
            "returned_bytes": self.returned_bytes,
        }


def _bytes(text: str) -> int | None:
    try:
        return len(text.encode("utf-8"))
    except UnicodeEncodeError:
        return None


def _unique_object(pairs: list[tuple[str, object]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(_: str) -> object:
    raise ValueError("nonfinite_json")


def _object(text: str) -> JsonObject:
    if len(text) > _JSON_LIMIT:
        return {}
    try:
        return as_object(
            json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        )
    except (ValueError, RecursionError):
        return {}


def _spool(match: re.Match[str]) -> tuple[tuple[str, ...], bool, bool | None]:
    groups = match.groupdict()
    path = (
        as_str(groups.get("complete_path"))
        or as_str(groups.get("partial_path"))
        or as_str(groups.get("legacy_path"))
    )
    if path:
        return (path,), True, groups.get("complete_path") is not None
    guidance = as_str(groups["guidance"]) or ""
    if "managed process handle" in guidance:
        return (), True, guidance.startswith("The complete output")
    if guidance == "Use action='read_output' for retained output.":
        return (), True, None
    return (), False, None


def _fast(text: str) -> OutputEvidence | None:
    match = _FAST_HEAD.search(text) or _FAST_PREVIEW.search(text)
    if match is None:
        return None
    groups = match.groupdict()
    total = int(groups["total"])
    shown = int(groups["head"]) + int(groups["tail"]) if "head" in groups else int(groups["shown"])
    omitted = int(groups["omitted"]) if "omitted" in groups else total - shown
    if total <= shown or omitted != total - shown:
        return OutputEvidence(kind="possible_signal")
    paths, indicated, complete = _spool(match)
    return OutputEvidence(
        truncation=True,
        kind="explicit_truncation",
        marker="fast_agent.head_tail_bytes" if "head" in groups else "fast_agent.preview_bytes",
        spool_paths=paths,
        spool_indicated=indicated,
        spool_complete=complete,
        reported_original_bytes=total,
        estimated_original_tokens=int(groups["total_tokens"]) if "total_tokens" in groups else None,
        omitted_bytes=omitted,
        limit=shown,
        limit_unit="bytes",
    )


def _codex(text: str) -> OutputEvidence | None:
    header = _CODEX_HEADER.match(text)
    match = _CODEX_WARNING.match(text, header.end() if header else 0)
    if match is None:
        return None
    return OutputEvidence(
        truncation=True,
        kind="explicit_truncation",
        marker="codex.token_warning",
        reported_original_tokens=int(match["tokens"]),
        reported_original_lines=int(match["lines"]) if match["lines"] else None,
    )


def _claude_unit(unit: str) -> Unit:
    # Preserve the observed KB spelling; decimal vs binary scale is unverified.
    units: dict[str, Unit] = {"KB": "KB", " bytes": "bytes", " characters": "chars"}
    return units[unit]


def _claude(text: str) -> OutputEvidence | None:
    match = _CLAUDE.match(text)
    if match is None:
        return None
    return OutputEvidence(
        truncation=True,
        kind="explicit_truncation",
        marker="claude.persisted_output",
        spool_paths=(match["path"],),
        spool_indicated=True,
        spool_complete=None,  # "Full output saved" is a notice, not file verification.
        limit=int(match["preview"]),
        limit_unit=_claude_unit(match["unit"]),
    )


def _mini_elision(head: str, tail: str, elided: int, marker: Marker) -> OutputEvidence:
    consistent = len(head) == _MINI_HALF and len(tail) == _MINI_HALF
    payload = head + tail
    return OutputEvidence(
        truncation=elided > 0,
        kind="explicit_truncation" if elided else "no_marker",
        marker=marker,
        # The notice survives transport/redaction; original lengths do not.
        reported_original_chars=len(payload) + elided if consistent else None,
        omitted_chars=elided,
        limit=_MINI_LIMIT if consistent else None,
        limit_unit="chars" if consistent else None,
        payload_chars=len(payload),
        payload_bytes=_bytes(payload),
    )


def _mini_json(obj: JsonObject) -> OutputEvidence | None:
    head, tail = as_str(obj.get("output_head")), as_str(obj.get("output_tail"))
    elided = count(obj.get("elided_chars"))
    if head is not None and tail is not None and elided is not None:
        if type(obj.get("returncode")) is int and obj.get("warning") == "Output too long.":
            return _mini_elision(head, tail, elided, "mini_swe.json_elision")
        return OutputEvidence(kind="possible_signal")
    if any(key in obj for key in ("output_head", "output_tail", "elided_chars")):
        return OutputEvidence(kind="possible_signal")
    return None


def _mini(text: str) -> OutputEvidence | None:
    result = _mini_json(_object(text))
    if result is not None:
        return result
    match = _MINI_XML.match(text)
    if match:
        return _mini_elision(
            match["head"], match["tail"], int(match["elided"]), "mini_swe.xml_elision"
        )
    return None


def _terminus(text: str) -> OutputEvidence | None:
    match = _TERMINUS.search(text)
    if match:
        payload = text[: match.start()] + text[match.end() :]
        size = _bytes(payload)
        omitted = int(match["omitted"])
        if omitted == 0:
            return OutputEvidence(kind="possible_signal")
        return OutputEvidence(
            truncation=True,
            kind="explicit_truncation",
            marker="terminus.byte_omission",
            reported_original_bytes=size + omitted if size is not None else None,
            omitted_bytes=omitted,
            limit=int(match["limit"]),
            limit_unit="bytes",
            payload_chars=len(payload),
            payload_bytes=size,
        )
    if text.startswith("Current Terminal Screen:\n"):
        return OutputEvidence(kind="terminal_screen", marker="terminus.current_screen")
    return None


def _grok(text: str) -> OutputEvidence | None:
    if re.search(r"\n\.\.\. \(output truncated\) \.\.\.\n", text):
        return OutputEvidence(
            truncation=True, kind="explicit_truncation", marker="grok.middle_omission"
        )
    return None


def _muse(text: str) -> OutputEvidence | None:
    obj = _object(text)
    required = {"chunk_id", "command", "description", "exit_code", "terminal_status", "output"}
    if not required.issubset(obj):
        return None
    output = as_str(obj.get("output"))
    if output is None or type(obj.get("truncated")) is not bool:
        return OutputEvidence(kind="possible_signal")
    truncated = obj["truncated"] is True
    return OutputEvidence(
        truncation=truncated,
        kind="explicit_truncation" if truncated else "no_marker",
        marker="muse.json_truncated" if truncated else None,
        reported_original_bytes=count(obj.get("original_output_bytes")),
        reported_original_tokens=count(obj.get("original_output_tokens")),
        payload_chars=len(output),
        payload_bytes=_bytes(output),
    )


_DEXT = re.compile(
    rf"^…\[truncated (?P<total>{_DIGITS}) bytes total; kept first "
    rf"(?P<head>{_DIGITS}) and last (?P<tail>{_DIGITS})\. Narrow/paginate; "
    r"prefer symbol-scoped reads or targeted diff/stat before retrying\.\]$",
    re.M,
)


def _dext(text: str) -> OutputEvidence | None:
    match = _DEXT.search(text)
    if match is None:
        return None
    total, head, tail = (int(match[key]) for key in ("total", "head", "tail"))
    if total <= head + tail:
        return OutputEvidence(kind="possible_signal")
    return OutputEvidence(
        truncation=True,
        kind="explicit_truncation",
        marker="dext.byte_notice",
        reported_original_bytes=total,
        omitted_bytes=total - head - tail,
        limit=head + tail,
        limit_unit="bytes",
        # UTF-8/transport differences leave the payload boundary unverified.
    )


def _indusagi(text: str) -> OutputEvidence | None:
    obj = _object(text)
    details = as_object(obj.get("details"))
    meta = as_object(details.get("truncation"))
    if not meta:
        return None
    body = as_str(meta.get("content"))
    keys = ("maxBytes", "maxLines", "outputBytes", "outputLines", "totalBytes", "totalLines")
    counts = {key: count(meta.get(key)) for key in keys}
    if (
        not isinstance(obj.get("content"), list)
        or body is None
        or type(meta.get("truncated")) is not bool
        or any(value is None for value in counts.values())
        or meta.get("truncatedBy") not in ("bytes", "lines", None)
        or any(
            type(meta.get(key)) is not bool for key in ("firstLineExceedsLimit", "lastLinePartial")
        )
    ):
        return OutputEvidence(kind="possible_signal")
    return _indusagi_evidence(details, meta, body, counts)


def _indusagi_evidence(
    details: JsonObject, meta: JsonObject, body: str, counts: dict[str, int | None]
) -> OutputEvidence:
    truncated = meta["truncated"] is True
    by = meta.get("truncatedBy")
    path = as_str(details.get("fullOutputPath"))
    paths = (path,) if path and re.fullmatch(_PATH, path) else ()
    total, shown = counts["totalBytes"], counts["outputBytes"]
    omitted = total - shown if total is not None and shown is not None and total >= shown else None
    return OutputEvidence(
        truncation=truncated,
        kind="explicit_truncation" if truncated else "no_marker",
        marker="indusagi.truncation_metadata" if truncated else None,
        spool_paths=paths,
        spool_indicated=bool(path),
        spool_complete=None,
        reported_original_bytes=total,
        reported_original_lines=counts["totalLines"],
        omitted_bytes=omitted if truncated else None,
        limit=counts["maxBytes"]
        if by == "bytes"
        else counts["maxLines"]
        if by == "lines"
        else None,
        limit_unit="bytes" if by == "bytes" else "lines" if by == "lines" else None,
        payload_chars=len(body),
        payload_bytes=_bytes(body),
    )


_PARSERS = {
    "fast-agent": _fast,
    "codex": _codex,
    "claude-code": _claude,
    "mini-swe-agent": _mini,
    "terminus-2": _terminus,
    "grok-build": _grok,
    "muse-code": _muse,
    "dext": _dext,
    "indusagi": _indusagi,
}
_POSSIBLE = {
    "fast-agent": re.compile(r"^\[Output truncated:", re.M),
    "codex": re.compile(r"\A(?:Warning: truncated output \(|Total output lines:)"),
    "claude-code": re.compile(r"\A(?:<persisted-output>|Output too large \()"),
    "mini-swe-agent": re.compile(r"^<(?:elided_chars|output_head)>|^\s*\"elided_chars\":", re.M),
    "terminus-2": re.compile(r"^\[\.\.\. output limited to ", re.M),
    "dext": re.compile(r"^…\[truncated ", re.M),
}


def parse_output(text: str, family: str) -> OutputEvidence:
    """Parse one decoded shell result. No evidence implies unknown completeness.

    Family is an explicit harness name (e.g. codex), never a version/config guess.
    Returned lengths include all input text. Payload lengths exclude only proven
    wrappers; limits are observed boundaries, not inferred defaults. Later access
    and frequency aggregation belong to the parent, not this stateless parser.
    """
    name = family.strip().lower()
    parser = _PARSERS.get(name)
    result = parser(text) if parser else None
    if result is None:
        possible = _POSSIBLE.get(name)
        kind: Kind = "no_marker" if name in _SUPPORTED else "unsupported"
        if possible and possible.search(text):
            kind = "possible_signal"
        result = OutputEvidence(kind=kind)
    return replace(result, returned_chars=len(text), returned_bytes=_bytes(text))
