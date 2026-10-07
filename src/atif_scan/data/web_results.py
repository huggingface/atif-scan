"""What a recorded web call returned: content, a runner status only, nothing, or unknown.

A classification of the record, shared by the trial facts, detectors, browser and report;
deciding what counts as a finding is the detectors' job. Never retrieval or execution.
"""

from __future__ import annotations

import ast
import re
from enum import StrEnum
from typing import TYPE_CHECKING

from .content import content as parse_content
from .jsonval import as_object, as_str

if TYPE_CHECKING:
    from .model import Content, Step, ToolCall

# A provider acknowledgement is status metadata, not the page/search body.
STATUS_ONLY = re.compile(
    r"\s*(?:status\s*[:=]\s*)?(?:ok|success|successful|done|completed|"
    r"pending|running)\s*[.!]?\s*",
    re.I,
)
RUNNER = re.compile(
    r"\AScript[ \t]+completed[.!]?[ \t]*(?:\r?\n"
    r"[ \t]*Wall[ \t]+time[ \t]*:?[ \t]*\d+(?:\.\d+)?[ \t]+seconds[ \t]*)?"
    r"(?:\r?\n[ \t]*Output:[ \t]*(?P<body>[\s\S]*))?\s*\Z",
    re.I,
)
# Deliberately anchored: prose mentioning a failed fetch is still page content.
RETRIEVAL_ERROR = re.compile(
    r"(?:"
    r"(?:status\s*[:=]\s*)?(?:error|failed)[.!]?"
    r"|(?:Error|Web(?:Fetch|Search)Error|HTTPError)\s*:\s*\S[^\r\n]*"
    r"|(?:failed|unable) to (?:fetch|retrieve|open|access|download)\b[^\r\n]*"
    r"|(?:fetch|retrieval|web search|request) failed\b[^\r\n]*"
    r"|HTTP(?: error)?\s+[45]\d\d(?:\s+[^\r\n]+)?"
    r")",
    re.I,
)
# Provider error envelopes require a whole header and citation/source metadata.
# A page merely discussing internal errors or unresolved clicks is not an error.
PROVIDER_METADATA = re.compile(
    r"\ue200cite\ue202turn\d+[a-z]+\d+\ue201"
    r"[ \t]+\[wordlim: \d+\][ \t]+Source:[ \t]+"
    r"(?:find|open|click|search)\(\{[^\r\n]*\}\)[^\r\n]*"
)
CLICK_ERROR = re.compile(r"Unable to resolve click call\b[^\r\n]*", re.I)
# A serialized content-block envelope ([{"type": "text", "text": ...}, ...], as JSON or a
# Python repr), or an empty one. Only a known block type opens it: a JSON API body that
# is an array of records (GitHub contents, Hub trees with "type": "file") is page text.
BLOCK_TYPES = (
    "text|input_text|output_text|image|input_image|image_url|refusal|resource|audio|input_audio"
)
SERIALIZED_BLOCKS = re.compile(
    r"""\A\[\s*(?:]\s*\Z|\{\s*['"]type['"]\s*:\s*['"](?:""" + BLOCK_TYPES + r""")['"])"""
)
# A client's HTTP error status line before the body (`HTTP 403 Forbidden` + a challenge
# page): a recorded failed retrieval, even when the body embeds media.
HTTP_ERROR_LINE = re.compile(r"\AHTTP(?:/\d(?:\.\d)?)?[ \t]+[45]\d\d\b[^\r\n]*(?:\r?\n|\Z)")
MAX_WRAPPER_BYTES = 1024 * 1024


class WebResultState(StrEnum):
    """Evidence available in one result, not a claim of successful retrieval."""

    UNAVAILABLE = "unavailable"
    STATUS_ONLY = "status_only"
    ERROR = "error"
    CONTENT = "content"


def _provider_error(text: str) -> bool:
    lines = text.splitlines()
    if not lines[1:]:
        return False
    if lines[0] == "Internal Error ()":
        return PROVIDER_METADATA.fullmatch(lines[1]) is not None
    # Some providers place the explicit click failure after citation metadata.
    return (
        PROVIDER_METADATA.fullmatch(lines[0]) is not None
        and CLICK_ERROR.fullmatch(lines[1]) is not None
    )


def _text_state(text: str) -> WebResultState:
    text = text.strip()
    wrapper = RUNNER.fullmatch(text)
    if wrapper:
        text = (as_str(wrapper.group("body")) or "").strip()
        if not text:
            return WebResultState.STATUS_ONLY
    if not text:
        return WebResultState.UNAVAILABLE
    if STATUS_ONLY.fullmatch(text):
        return WebResultState.STATUS_ONLY
    error = RETRIEVAL_ERROR.fullmatch(text) or _provider_error(text)
    return WebResultState.ERROR if error else WebResultState.CONTENT


def _literal_block(node: ast.expr) -> dict[str, object]:
    """Read only a flat dict of constants; no evaluation, calls or expressions."""
    if not isinstance(node, ast.Dict):
        return {}
    block: dict[str, object] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if not isinstance(key, ast.Constant) or not isinstance(value, ast.Constant):
            return {}
        if not isinstance(key.value, str) or key.value in block:
            return {}
        block[key.value] = value.value
    return block


def _block_state(block: object) -> WebResultState:
    data = as_object(block)
    text = as_str(data.get("text"))
    if data.get("type") not in ("text", "output_text", "input_text") or text is None:
        return WebResultState.UNAVAILABLE
    parsed = parse_content(text)
    if parsed.media:
        return WebResultState.UNAVAILABLE
    return _text_state(text)


def _combined_state(states: list[WebResultState]) -> WebResultState:
    # Never let one body (or error) hide an unavailable/status-only sibling.
    for state in (WebResultState.UNAVAILABLE, WebResultState.STATUS_ONLY, WebResultState.ERROR):
        if state in states:
            return state
    return WebResultState.CONTENT if states else WebResultState.UNAVAILABLE


def _runner_preface(block: object) -> bool:
    data = as_object(block)
    if data.get("type") not in ("text", "output_text", "input_text"):
        return False
    text = as_str(data.get("text"))
    wrapper = RUNNER.fullmatch(text.strip()) if text is not None else None
    return wrapper is not None and not (wrapper.group("body") or "").strip()


def _serialized_blocks_state(blocks: list[dict[str, object]]) -> WebResultState:
    states = [_block_state(block) for block in blocks]
    if any(state in (WebResultState.CONTENT, WebResultState.ERROR) for state in states):
        # Only exact runner acknowledgements within this transport envelope can
        # be discarded. Separate Observations and other status/unknown blocks
        # remain required evidence.
        states = [
            state for block, state in zip(blocks, states, strict=True) if not _runner_preface(block)
        ]
    return _combined_state(states)


def _serialized_blocks(text: str) -> list[dict[str, object]]:
    """Supported JSON/Python repr text-block lists, parsed as syntax only."""
    if len(text) > MAX_WRAPPER_BYTES:
        return []
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return []
    if not isinstance(tree.body, ast.List):
        return []
    return [_literal_block(node) for node in tree.body.elts]


def web_result_state(content: Content) -> WebResultState:
    """Classify readable result evidence, retaining unknown/media and wrapper gaps."""
    if content.understood and content.media and HTTP_ERROR_LINE.match(content.text.strip()):
        return WebResultState.ERROR
    if not content.understood or content.media:
        return WebResultState.UNAVAILABLE
    text = content.text.strip()
    if SERIALIZED_BLOCKS.match(text):
        return _serialized_blocks_state(_serialized_blocks(text))
    return _text_state(text)


def recorded_web_content(content: Content) -> bool:
    """Only meaningful, understood text can clear a returned-content predicate."""
    return web_result_state(content) is WebResultState.CONTENT


def web_results_complete(step: Step, call: ToolCall) -> bool:
    """Every result chunk must contain content; explicit errors do not clear this."""
    results = step.results_for(call)
    return bool(results) and all(recorded_web_content(o.content) for _, o in results)


def web_outcomes_recorded(step: Step, call: ToolCall) -> bool:
    """Every linked chunk records content or an explicit error, not just runner status.

    This says nothing about input validity, reference provenance or retrieval success.
    """
    results = step.results_for(call)
    recorded = (WebResultState.CONTENT, WebResultState.ERROR)
    return bool(results) and all(web_result_state(o.content) in recorded for _, o in results)


SCRIPT_FAILURE = re.compile(
    r"Script failed[.!]?[ \t]*\r?\n"
    r"[ \t]*Wall time[ \t]*:?[ \t]*\d+(?:\.\d+)? seconds[ \t]*\r?\n"
    r"[ \t]*Output:[ \t]*\r?\n\s*"
    r"Script error:[ \t]*\r?\n\S[^\r\n]*(?:\r?\n[^\r\n]*)*",
    re.I,
)


def _recorded_texts(content: Content) -> list[str]:
    """Readable transport text only; malformed or media siblings invalidate it."""
    if not content.understood or content.media:
        return []
    text = content.text.strip()
    if not SERIALIZED_BLOCKS.match(text):
        return [text]
    blocks = _serialized_blocks(text)
    if any(_block_state(block) is WebResultState.UNAVAILABLE for block in blocks):
        return []
    return [as_str(block.get("text")) or "" for block in blocks]


def recorded_script_failure(content: Content) -> bool:
    """Exact runner failure envelope, not arbitrary error mentions in page text."""
    texts = _recorded_texts(content)
    return bool(texts) and SCRIPT_FAILURE.fullmatch("\n".join(texts).strip()) is not None


def runner_status_only(content: Content) -> bool:
    """Known empty runner acknowledgements, not arbitrary statuses or errors."""
    texts = _recorded_texts(content)
    return bool(texts) and all(_runner_preface({"type": "text", "text": text}) for text in texts)
