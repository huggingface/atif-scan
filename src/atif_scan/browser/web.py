"""Call-level web-gap navigation over the bound trace; no text or targets exported."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..data.jsonval import as_str, count
from ..data.web_results import WebResultState, web_result_state

if TYPE_CHECKING:
    from ..data.jsonval import Doc
    from ..data.model import Step, ToolCall, Trace


def _call_gaps(step: Step, call: ToolCall, number: int) -> list[Doc]:
    field = next(
        (i for i, (_, c) in enumerate(call.fields) if c.understood or c.text or c.media), None
    )
    target = {
        "step": number,
        "part": "call" if field is not None else "call_info",
        "index": call.index,
        "field": field if field is not None else 0,
    }
    base = {"step": number, "call": call.index, "call_location": target}
    results = step.results_for(call)
    if not results:
        return [{**base, "reason": "no_linked_result", "result_location": None}]
    gaps: list[Doc] = []
    for index, observation in results:
        state = web_result_state(observation.content)
        if state not in (WebResultState.UNAVAILABLE, WebResultState.STATUS_ONLY):
            continue
        gaps.append(
            {
                **base,
                "reason": "status_only"
                if state is WebResultState.STATUS_ONLY
                else "unusable_content",
                "result_location": {
                    "step": number,
                    "part": "result",
                    "index": index,
                    "field": 0,
                },
                "pairing_reconstructed": observation.pairing_reconstructed,
            }
        )
    return gaps


def web_gap_details(trace: Trace, locations: list[Doc]) -> list[Doc]:
    """Resolve only recorded call locators. Old step-only gaps stay unlocated."""
    steps = dict(zip(trace.step_numbers, trace.steps, strict=True))
    seen: set[tuple[int, int]] = set()
    gaps: list[Doc] = []
    for location in locations:
        number, index = count(location.get("step")), count(location.get("index"))
        if (
            number is None
            or index is None
            or as_str(location.get("part"))
            not in (
                "call",
                "call_info",
            )
        ):
            continue
        step = steps.get(number)
        if step is None or index >= len(step.calls) or (number, index) in seen:
            continue
        seen.add((number, index))
        gaps.extend(_call_gaps(step, step.calls[index], number))
    return gaps
