"""Bounded, static parsing of fast-agent shell result ``Observation.content.text``.

Supported formats are a whole JSON object (only its top-level fields), a contiguous
prefix of unindented ``key: value`` metadata lines, or one exact fast-agent durable
launch banner followed immediately by that prefix. Banners must start at the first
character; the five supported lines are enumerated in ``_DURABLE_LAUNCH_BANNERS``.
Installed fast-agent emits the banner, effective_lifecycle, process_id, os_pid, an
optional output_spool_path, then explanation. The spool path is not parsed or resolved.
The text prefix ends at the first non-metadata line, including output_spool_path, a
blank line, explanation or stdout/stderr label; parsing never resumes. No arbitrary
prose, fences, nested messages, output bodies or trailing metadata blocks are searched.
JSON may have surrounding whitespace.

Recorded metadata is a claim, not an execution or persistence guarantee. In particular,
arbitrary program output can imitate the plain-text prefix (or a JSON object). Callers
must establish tool/result provenance separately. Unknown is not a negative result.
Identifiers and PIDs are private, in-memory evidence only, never report export fields.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .jsonval import JsonObject, as_object, as_str, count

# Reject oversized input in full, rather than accepting a truncated metadata prefix.
MAX_TEXT_CHARS = 64 * 1024
MAX_JSON_DEPTH = 32
MAX_PROCESS_ID_CHARS = 128
MAX_OS_PID = 2**31 - 1
_FIELDS = frozenset({"effective_lifecycle", "process_id", "os_pid"})
_LIFECYCLES = frozenset({"persistent", "foreground", "background"})
_DURABLE_LAUNCH_BANNERS = frozenset(
    {
        "Managed background process is still running.",
        "Managed background process completed before launch acknowledgement.",
        "Managed background process failed before launch acknowledgement.",
        "Managed background process was stopped during launch.",
        "Managed background process supervisor is unavailable.",
    }
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
# Skip strings while counting containers, including escaped quotes/brackets in stdout.
_JSON_TOKENS = re.compile(r'"(?:[^"\\]|\\[\s\S])*"|[{}\[\]]')


@dataclass(frozen=True)
class ProcessResult:
    """Recorded claims only; ``None`` means missing, unsupported or invalid evidence.

    This is not a report document. Private identifiers are also omitted from repr.
    """

    lifecycle: str | None = None
    process_id: str | None = field(default=None, repr=False)
    os_pid: int | None = field(default=None, repr=False)


def _unique_object(pairs: list[tuple[str, object]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError("non_json_constant")


def _within_json_depth(text: str) -> bool:
    depth = 0
    for match in _JSON_TOKENS.finditer(text):
        token = match.group()
        if token in ("{", "["):
            depth += 1
            if depth > MAX_JSON_DEPTH:
                return False
        elif token in ("}", "]"):
            depth -= 1
    return True


def _json_fields(text: str) -> JsonObject:
    if not _within_json_depth(text):
        return {}
    try:
        value: object = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (ValueError, RecursionError):
        return {}
    return as_object(value)


def _prefix_fields(text: str) -> JsonObject:
    result: JsonObject = {}
    # At most three unique fields plus one line to detect a duplicate. Never scan body.
    for raw_line in text.split("\n", len(_FIELDS) + 1)[: len(_FIELDS) + 1]:
        line = raw_line.removesuffix("\r")
        key, separator, value = line.partition(": ")
        if key not in _FIELDS:
            # A recognized key with malformed spacing is invalid, not a body boundary.
            if line.partition(":")[0] in _FIELDS:
                return {}
            break
        if not separator or key in result:
            return {}
        if key == "os_pid":
            if re.fullmatch(r"[1-9][0-9]{0,9}", value) is None:
                return {}
            result[key] = int(value)
        else:
            result[key] = value
    return result


def _text_fields(text: str) -> JsonObject:
    first_line, separator, remainder = text.partition("\n")
    if separator and first_line.removesuffix("\r") in _DURABLE_LAUNCH_BANNERS:
        return _prefix_fields(remainder)
    return _prefix_fields(text)


def _process_id(value: object) -> str | None:
    text = as_str(value)
    if text and len(text) <= MAX_PROCESS_ID_CHARS and _IDENTIFIER.fullmatch(text):
        return text
    return None


def _os_pid(value: object) -> int | None:
    pid = count(value)
    return pid if pid is not None and 0 < pid <= MAX_OS_PID else None


def _validated_result(data: JsonObject) -> ProcessResult:
    lifecycle = as_str(data.get("effective_lifecycle"))
    process_id = _process_id(data.get("process_id"))
    os_pid = _os_pid(data.get("os_pid"))
    validated = {"effective_lifecycle": lifecycle, "process_id": process_id, "os_pid": os_pid}
    if any(key in data and value is None for key, value in validated.items()):
        return ProcessResult()
    return ProcessResult(lifecycle if lifecycle in _LIFECYCLES else None, process_id, os_pid)


def parse_process_result(text: str) -> ProcessResult:
    """Parse supported recorded metadata without executing or resolving anything.

    Limits: 64 Ki characters total, JSON container depth 32, ASCII identifier length
    1..128 (alphanumeric first, then alphanumeric/underscore/dot/hyphen), PID 1..2**31-1.
    Text uses exact ``key: value`` lines with LF or CRLF, in any order, no indentation,
    optionally immediately after one exact durable launch banner. Banners do not infer
    lifecycle or running state; persistent is configuration, even for failed launches.
    Missing fields stay unknown; only persistent/foreground/background are lifecycles.
    Other lifecycle strings (including session) stay unknown without discarding IDs.

    Malformed present fields (including JSON null, bool, float or string PIDs, and text
    ``os_pid: None`` from a launch without a child PID) conservatively invalidate all
    metadata. Duplicate JSON keys anywhere invalidate the object; duplicate text
    fields invalidate the prefix even when values agree. Unrecognized JSON fields are
    ignored. Unsupported, oversized or malformed input returns all fields unknown.
    """
    if not text or len(text) > MAX_TEXT_CHARS:
        return ProcessResult()
    data = _json_fields(text) if text.lstrip().startswith("{") else _text_fields(text)
    return _validated_result(data)
