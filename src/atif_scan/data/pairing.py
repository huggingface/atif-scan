"""Observation pairing: which call a result belongs to when the export doesn't say, and
rejected-input results that ran nothing. Reconstructed links are marked, never presented
as recorded ones.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Observation, ToolCall

# The harness refused the call's input before running it (arguments that weren't valid
# JSON, required parameters missing), so nothing ran: Claude Code, WorkHarness, OrcaTerm,
# fast-agent's filesystem tools ("Error: 'path' and 'new_string' are required strings",
# "Error: 'input' argument is required and must be a string").
INPUT_REJECTED = re.compile(
    r"^\s*(?:<tool_use_error>\s*InputValidationError\b|Invalid input for [\w.-]+:|"
    r"Validation failed for tool \"?[\w.-]+\"?:|"
    r"Error: '[\w.-]+'(?: and '[\w.-]+')* (?:argument is required and must be|"
    r"are required strings)\b)"
)


def drop_rejected_unknowns(
    calls: list[ToolCall], observations: list[Observation]
) -> list[ToolCall]:
    """A rejected call's unreadable input can't hide an action: drop its unreadable fields
    (readable ones, e.g. an unparsed command's text, are still scanned)."""
    rejected = {
        o.source_call_id
        for o in observations
        if o.source_call_id and INPUT_REJECTED.match(o.content.text)
    }
    return [
        replace(c, fields=tuple(f for f in c.fields if f[1].understood))
        if c.result_key in rejected
        else c
        for c in calls
    ]


# A lone call's results need no pairing by order.
MIN_PAIRED_CALLS = 2


def _positional_links_valid(calls: list[ToolCall], observations: list[Observation]) -> bool:
    ids = [c.id for c in calls]
    return (
        len(calls) == len(observations)
        and all(ids)
        and len(set(ids)) == len(ids)
        and all(o.source_call_id in (None, c.id) for c, o in zip(calls, observations, strict=True))
    )


def _unique_remainder(calls: list[ToolCall], observations: list[Observation]) -> ToolCall | None:
    """Only valid, unambiguous explicit IDs can eliminate recorded calls.

    Multipart results may eliminate the same call repeatedly. Foreign IDs or duplicate
    nonempty recorded IDs block inference altogether; empty IDs eliminate nothing.
    """
    ids = [c.id for c in calls if c.id]
    if len(set(ids)) != len(ids):
        return None
    explicit = {o.source_call_id for o in observations if o.source_call_id is not None}
    if not explicit <= set(ids):
        return None
    if sum(o.source_call_id is None for o in observations) != 1:
        return None
    remaining = [c for c in calls if not c.id or c.id not in explicit]
    return remaining[0] if len(remaining) == 1 else None


def reconstruct_observation_links(
    calls: list[ToolCall], observations: list[Observation]
) -> list[Observation]:
    """Infer links with warnings, without altering explicit links or source data.

    Preserve valid positional reconstruction first. Otherwise a sole unlinked result
    and sole unmatched recorded call can be associated by index, regardless of order.
    This intentionally resolves formerly contradictory positional examples when their
    explicit links prove a unique remainder. Derived code-mode calls are excluded.
    """
    if len(calls) < MIN_PAIRED_CALLS or not any(o.source_call_id is None for o in observations):
        return observations
    if _positional_links_valid(calls, observations):
        return [
            replace(
                o,
                source_call_id=c.id,
                source_call_index=c.index,
                pairing_method="position",
                pairing_reconstructed=True,
            )
            if o.source_call_id is None
            else o
            for c, o in zip(calls, observations, strict=True)
        ]
    remainder = _unique_remainder(calls, observations)
    return [
        replace(
            o,
            source_call_index=remainder.index,
            pairing_method="unique_remainder",
            pairing_reconstructed=True,
        )
        if remainder is not None and o.source_call_id is None
        else replace(o, pairing_unresolved=True)
        if o.source_call_id is None
        else o
        for o in observations
    ]
