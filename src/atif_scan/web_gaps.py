"""Counts-only web outcome gaps, in recorded call units (not backend requests)."""

from dataclasses import dataclass

from .detectors.web import runner_status_only, web_outcomes_recorded
from .jsonval import Doc
from .model import Content, Step, ToolCall, Trace


@dataclass(frozen=True)
class WebGaps:
    calls: int = 0
    missing_result_ids: int = 0
    no_emitted_contents: int = 0

    def document(self) -> Doc:
        return {
            "calls": self.calls,
            "missing_result_ids": self.missing_result_ids,
            "no_emitted_contents": self.no_emitted_contents,
        }


def _units(step: Step) -> list[ToolCall]:
    """Derived web actions sharing a parent are one recorded call/result unit."""
    parents = {c.id: c for c in step.calls if c.result_id is None and c.id}
    units: dict[int, ToolCall] = {}
    for call in step.calls:
        if call.tool not in ("web_search", "web_fetch") and call.name not in (
            "web_search_call",
            "web__run",
            "web.run",
        ):
            continue
        parent = parents.get(call.result_id or "", call)
        units.setdefault(parent.index, parent)
    return list(units.values())


def _empty(content: Content) -> bool:
    return content.understood and not content.text.strip() and not content.media


def _no_contents(step: Step, call: ToolCall) -> bool:
    results = step.results_for(call)
    if results:
        return all(_empty(o.content) or runner_status_only(o.content) for _, o in results)
    # Unlinked readable or unsupported content is not proof nothing was emitted.
    linked_elsewhere = {index for other in step.calls for index, _ in step.results_for(other)}
    return all(
        _empty(o.content)
        for index, o in enumerate(step.observations)
        if index not in linked_elsewhere
    )


def web_gaps(trace: Trace) -> WebGaps:
    calls = missing_ids = no_contents = 0
    for step in trace.steps:
        if not step.authored:
            continue
        for call in _units(step):
            if web_outcomes_recorded(step, call):
                continue
            calls += 1
            results = step.results_for(call)
            missing_ids += int(not results or any(not o.source_call_id for _, o in results))
            no_contents += int(_no_contents(step, call))
    return WebGaps(calls, missing_ids, no_contents)
