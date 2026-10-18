"""Private, offline ATIF adapter for corpus research (stdlib only).

This is not a report export: steps contain untrusted trace text. Declarations are
read from agent.tool_definitions (or agent.tools / root.tool_definitions). Native
OpenAI/Anthropic-style definition lists are used only when ATIF declarations are
absent. Explicit null is malformed; an explicit empty list is known.

Step indexes remain original document positions. Only authored agent steps are
retained; dropped_steps counts parse failures, not intentionally excluded steps.
Use Step.results_for(call) for pairing and call.result_id to identify derived
code-mode calls. canonical_tool never changes the loader's names or categories.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from harness_visibility import Visibility, visible_result

from atif_scan.data.content import content
from atif_scan.data.jsonval import as_list, as_object, as_str, compact_json, is_object
from atif_scan.data.loader import parse_step
from atif_scan.data.model import Content

if TYPE_CHECKING:
    from atif_scan.data.jsonval import JsonObject
    from atif_scan.data.model import Step

MAX_SERIALIZED_CHARS = 1024 * 1024
MAX_REPR_CHARS = 64 * 1024
MAX_REPR_NODES = 4096
_DESCRIPTIVE = frozenset({"description", "title", "examples"})
_SCHEMA_MAPS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies"}
)
_SCHEMA_CHILDREN = frozenset(
    {
        "items",
        "prefixItems",
        "additionalItems",
        "contains",
        "additionalProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
        "propertyNames",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
    }
)
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,99}")
_MISSING = object()
# Local fast-agent tool_call_planning.py resolves a unique case-insensitive tool name.
# Only known built-ins are normalized here; arbitrary plugin namespaces stay distinct.
_FAST_BUILTINS = frozenset(
    {
        "shell",
        "bash",
        "process",
        "edit_file",
        "read_text_file",
        "write_text_file",
        "attach_media",
        "apply_patch",
        "exec",
        "execute",
        "poll_process",
        "terminate_process",
    }
)


@dataclass(frozen=True)
class Inventory:
    tools: tuple[str, ...]
    source: Literal["atif", "native", "unknown"]
    status: Literal["known", "missing", "malformed"]
    signature: str | None


@dataclass(frozen=True)
class ResultVisibility:
    """Private per-result surfaces at original step/observation positions."""

    step: int
    index: int
    visibility: Visibility


@dataclass(frozen=True)
class TraceEvidence:
    family: str | None
    version: str | None
    declarations: Inventory
    steps: tuple[Step, ...]
    issues: tuple[str, ...]
    dropped_steps: int = 0
    result_visibility: tuple[ResultVisibility, ...] = ()


def canonical_tool(name: str, family: str | None) -> str:
    """Normalize namespaces and fast-agent built-ins, not semantic categories."""
    while name.startswith(("tools.", "functions.")):
        name = name.split(".", 1)[1]
    if family == "fast-agent" and name.lower() in _FAST_BUILTINS:
        return name.lower()
    return name


def _safe_label(value: object) -> str | None:
    text = as_str(value)
    return text if text is not None and _LABEL.fullmatch(text) else None


def _family(value: object) -> str | None:
    if isinstance(value, str) and value.lower() in {"claude code", "claude-code"}:
        return "claude-code"
    label = _safe_label(value)
    return label.lower() if label else None


def _declarations(doc: JsonObject, native: object, family: str | None) -> Inventory:
    agent = as_object(doc.get("agent"))
    raw = _MISSING
    for owner, key in ((agent, "tool_definitions"), (agent, "tools"), (doc, "tool_definitions")):
        if key in owner:
            raw = owner[key]
            break
    source: Literal["atif", "native", "unknown"] = "atif"
    if raw is _MISSING:
        raw, source = (native, "native") if native is not None else (_MISSING, "unknown")
    if raw is _MISSING:
        return Inventory((), "unknown", "missing", None)
    return _inventory(raw, source, family)


def _definition(value: object, family: str | None) -> tuple[str, object]:
    if not is_object(value):
        raise ValueError("invalid_definition")
    definition = as_object(value["function"]) if "function" in value else value
    name = as_str(definition.get("name")) or as_str(definition.get("function_name"))
    if not name:
        raise ValueError("invalid_definition")
    name = canonical_tool(name, family)
    if _safe_label(name) is None:
        raise ValueError("invalid_definition")
    schema = definition.get("parameters", definition.get("input_schema"))
    if not is_object(schema):
        raise ValueError("invalid_schema")
    return name, _schema(schema)


def _schema(value: object) -> object:
    """Drop schema annotations, never property names or values of enum/const."""
    if isinstance(value, list):
        return [_schema(item) for item in as_list(value)]
    if not is_object(value):
        return value
    normalized: JsonObject = {}
    for key, item in value.items():
        if key in _DESCRIPTIVE:
            continue
        if key in _SCHEMA_MAPS and is_object(item):
            normalized[key] = {name: _schema(child) for name, child in item.items()}
        elif key in _SCHEMA_CHILDREN:
            normalized[key] = _schema(item)
        else:
            normalized[key] = item
    return normalized


def _inventory(
    raw: object, source: Literal["atif", "native", "unknown"], family: str | None
) -> Inventory:
    if not isinstance(raw, list):
        return Inventory((), source, "malformed", None)
    definitions: dict[str, object] = {}
    malformed = False
    for item in as_list(raw):
        try:
            name, schema = _definition(item, family)
            if name in definitions:
                malformed = True
            definitions[name] = schema
        except (ValueError, TypeError, RecursionError):
            malformed = True
    serialized = compact_json(definitions)
    signature = None if malformed or serialized is None else _digest(serialized)
    status: Literal["known", "malformed"] = "known" if signature else "malformed"
    return Inventory(tuple(sorted(definitions)), source, status, signature)


def _digest(text: str) -> str:
    # Backslash encoding keeps even unpaired surrogate input deterministic.
    return hashlib.sha256(text.encode("utf-8", errors="backslashreplace")).hexdigest()


def _parse_agent_step(
    index: int, raw: object, family: str | None
) -> tuple[Step, tuple[ResultVisibility, ...]]:
    step = parse_step(index, raw)
    results = as_list(as_object(as_object(raw).get("observation")).get("results"))
    observations = []
    visibility = []
    for result_index, (observation, result) in enumerate(
        zip(step.observations, results, strict=True)
    ):
        obj = as_object(result)
        surfaces = visible_result(obj, family, result_text(obj.get("content")))
        selected = (
            surfaces.model_visible if surfaces.model_visible is not None else surfaces.recorded
        )
        observations.append(replace(observation, content=selected))
        visibility.append(ResultVisibility(index, result_index, surfaces))
    return replace(step, observations=tuple(observations)), tuple(visibility)


def _steps(
    doc: JsonObject, family: str | None
) -> tuple[tuple[Step, ...], tuple[ResultVisibility, ...], int, list[str]]:
    raw_steps = doc.get("steps")
    if not isinstance(raw_steps, list):
        return (), (), 0, ["invalid_steps"]
    steps: list[Step] = []
    visibility: list[ResultVisibility] = []
    dropped = 0
    for index, raw in enumerate(as_list(raw_steps)):
        obj = as_object(raw)
        if obj.get("source") in ("system", "user") or obj.get("is_copied_context") is True:
            continue
        try:
            step, surfaces = _parse_agent_step(index, raw, family)
        except (ValueError, TypeError, RecursionError, OverflowError):
            dropped += 1
            continue
        if step.authored:
            steps.append(step)
            visibility.extend(surfaces)
    return tuple(steps), tuple(visibility), dropped, ["step_parse_failed"] if dropped else []


def analyze_trace(doc: JsonObject, native_definitions: object = None) -> TraceEvidence:
    """Analyze one in-memory document; never descend into subagent trajectories."""
    agent = as_object(doc.get("agent"))
    family, version = _family(agent.get("name")), _safe_label(agent.get("version"))
    declarations = _declarations(doc, native_definitions, family)
    steps, visibility, dropped, issues = _steps(doc, family)
    for key, label in (("name", family), ("version", version)):
        if key in agent and label is None:
            issues.append(f"invalid_agent_{key}")
    if declarations.status != "known":
        issues.append(f"declarations_{declarations.status}")
    if "subagent_trajectories" in doc:
        issues.append("subagent_trajectories_present")
    return TraceEvidence(family, version, declarations, steps, tuple(issues), dropped, visibility)


def _block_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(is_object(item) and isinstance(item.get("type"), str) for item in as_list(value))
    )


def _serialized_blocks(text: str) -> object:
    stripped = text.strip()
    if len(text) > MAX_SERIALIZED_CHARS or not stripped.startswith("["):
        return None
    try:
        parsed: object = json.loads(stripped)
    except (ValueError, RecursionError):
        parsed = _repr_blocks(stripped)
    return parsed if _block_list(parsed) else None


def _repr_blocks(text: str) -> object:
    if (
        len(text) > MAX_REPR_CHARS
        or re.match(r"\[\s*\{", text) is None
        or re.search(r"['\"]type['\"]\s*:", text[:256]) is None
    ):
        return None
    try:
        tree = ast.parse(text, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > MAX_REPR_NODES:
            return None
        return ast.literal_eval(tree)
    except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
        return None


def unwrap_result(value: Content) -> Content:
    """Decode one serialized block list, preserving unknown/media/computed flags."""
    blocks = _serialized_blocks(value.text)
    if blocks is None:
        return value
    decoded = content(blocks)
    media = value.media or decoded.media
    media_parts = [part for part in (value, decoded) if part.media]
    ids = (
        tuple(dict.fromkeys(i for part in media_parts for i in part.media_ids))
        if all(part.media_ids for part in media_parts)
        else ()
    )
    return replace(
        decoded,
        understood=value.understood and decoded.understood,
        media=media,
        media_ids=ids,
        computed=value.computed,
    )


def result_text(value: object) -> Content:
    """Decode result blocks; structured shell-output objects stay JSON text."""
    if is_object(value) and "type" not in value:
        serialized = compact_json(value)
        return Content(serialized) if serialized is not None else Content(understood=False)
    return unwrap_result(content(value))
