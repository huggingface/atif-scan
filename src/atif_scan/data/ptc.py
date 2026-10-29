"""Validated fast-agent dispatch ledgers, never nested output or authored tokens.

Only an explicitly linked, uniquely bound execute_python result can override static
dispatch inference. Even a complete ledger says nothing about direct Python I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from .jsonval import as_list, as_object, as_str, compact_json, is_object
from .model import ToolCall
from .tools import call_fields, computed_fields, program_fields, program_tool
from .web_activity import activity
from .web_inputs import parse_web_input

if TYPE_CHECKING:
    from collections.abc import Mapping

SCHEMA = "fast-agent.ptc-calls/v1"
Status = Literal["completed", "raised", "in_flight", "cancelled"]


@dataclass(frozen=True)
class Dispatch:
    id: str = field(repr=False)
    tool: str = field(repr=False)
    arguments: object = field(repr=False)
    status: Status
    is_error: bool | None


def _status(value: object) -> Status | None:
    if value == "completed":
        return "completed"
    if value == "raised":
        return "raised"
    if value == "in_flight":
        return "in_flight"
    return "cancelled" if value == "cancelled" else None


def _arguments_valid(entry: Mapping[str, object]) -> bool:
    args = entry.get("arguments")
    if "arguments" not in entry:
        return False  # Legacy id/tool/is_error exports are not complete ledgers.
    if args is None:
        return entry.get("arguments_error") == "not_json_serializable"
    return is_object(args) and compact_json(args) is not None and "arguments_error" not in entry


def _dispatch(value: object, parent: str) -> Dispatch | None:
    entry = as_object(value)
    call_id, tool = as_str(entry.get("id")), as_str(entry.get("tool"))
    prefix = f"{parent}::ptc"
    valid_id = call_id is not None and call_id.startswith(prefix) and len(call_id) > len(prefix)
    error = entry.get("is_error")
    status = _status(entry.get("status"))
    if not (
        valid_id
        and tool
        and tool.strip()
        and status is not None
        and "is_error" in entry
        and (error is None or type(error) is bool)
        and _arguments_valid(entry)
    ):
        return None
    return Dispatch(
        call_id, tool, entry["arguments"], status, error if isinstance(error, bool) else None
    )


def _ledger(observation: object, parent: str, reserved: set[str]) -> list[Dispatch] | None:
    results = [
        as_object(r)
        for r in as_list(as_object(observation).get("results"))
        if as_object(r).get("source_call_id") == parent
    ]
    if len(results) != 1:
        return None  # No positional inference; multipart/duplicate bindings are ambiguous.
    extra = as_object(results[0].get("extra"))
    entries = extra.get("ptc_calls")
    if extra.get("ptc_calls_schema") != SCHEMA or not isinstance(entries, list):
        return None
    found: list[Dispatch] = []
    seen = set(reserved)
    for entry in entries:
        dispatch = _dispatch(entry, parent)
        if dispatch is None or dispatch.id in seen:
            return None  # Validate the entire ledger before using any of it.
        seen.add(dispatch.id)
        found.append(dispatch)
    return found


def _call(dispatch: Dispatch, parent: ToolCall) -> ToolCall:
    args, name = dispatch.arguments, dispatch.tool
    tool = program_tool(name, args)
    return ToolCall(
        0,  # Reindexed after replacing static calls, preserving outer call positions.
        tool,
        computed_fields(tool) if args is None else call_fields(tool, args, name=name),
        dispatch.id,
        f"{parent.name}>{name}",
        None,
        result_id=parent.id,
        web_input=parse_web_input(name, args),
        web_activity=activity(name, tool, args),
        provenance="runtime_recorded",
        ptc_status=dispatch.status,
        ptc_is_error=dispatch.is_error,
    )


def _replace_program(
    calls: list[ToolCall], parent: ToolCall, args: object, ledger: list[Dispatch]
) -> list[ToolCall]:
    complete = all(d.status == "completed" for d in ledger)
    # Keep source literal scanning, including arguments of calls the ledger replaces,
    # under the existing literal-reader rules. General Python operations stay unresolved:
    # open()/Path()/os calls and persisted helpers can bypass the dispatch ledger.
    fields = program_fields(args, parent.name) + computed_fields("other")
    kept = [
        replace(c, fields=fields) if c.index == parent.index else c
        for c in calls
        if not (complete and c.provenance == "static_inferred" and c.result_id == parent.id)
    ]
    return [*kept, *(_call(d, parent) for d in ledger)]


def apply_ledger(
    calls: list[ToolCall], observation: object, sources: Mapping[str, object]
) -> list[ToolCall]:
    """Replace inferred dispatches only for valid completed ledgers; partial ones add
    runtime evidence alongside static fallback. Absent/invalid metadata changes nothing.
    """
    parents = [c for c in calls if c.result_id is None]
    reserved = {c.id for c in parents}
    for parent in parents:
        if (
            parent.name != "execute_python"
            or not parent.id
            or sum(c.id == parent.id for c in parents) != 1
        ):
            continue
        ledger = _ledger(observation, parent.id, reserved)
        if ledger is not None:
            calls = _replace_program(calls, parent, sources[parent.id], ledger)
            reserved.update(d.id for d in ledger)
    return [replace(c, index=i) for i, c in enumerate(calls)]
