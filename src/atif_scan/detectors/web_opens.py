"""Web opens that name only a reference, read from their results.

Codex's web tool opens a page by a reference to an earlier result (`{"open": [{"ref_id":
"turn0search3"}]}`), so the call records no URL: its URL field is unreadable by design.
The opened page's URL is in the open's own result, whose header names it (`Title (URL)`,
then citation metadata naming the open). Reference IDs aren't resolved through earlier
search results: exports can omit those or reuse IDs across compactions (see
`data.web_inputs`). An open whose result names no page (an error, nothing recorded) stays
unresolved: what it opened is unknown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..data.model import Channel, Locator
from ..data.web_inputs import web_input
from ..data.web_results import recorded_text, recorded_web_content

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..data.model import Step, ToolCall, Trace

# A Codex `open` result's header: `Title (URL)` then citation metadata naming the open.
# A program that prints `JSON.stringify(r)` leaves the line break escaped (`\\n`).
OPENED_PAGE = re.compile(
    r"\((https?://[^\s)\\]+)\)\s*(?:\n|\\n)\ue200cite\ue202turn\d+view\d+\ue201[^\n]{0,200}?Source:\s*open\(",
)


def by_reference(call: ToolCall) -> bool:
    """A fetch that names only a reference (`turn0search8`), not a URL."""
    return call.tool == "web_fetch" and not web_input(call).source_known


@dataclass(frozen=True)
class Opened:
    url: str
    at: Locator  # the URL in the result (the whole result when it can't be placed)


def opened(step: Step, call: ToolCall) -> Iterator[Opened]:
    """The pages an open's results name, read from the unwrapped result (Codex records a
    Python repr of text blocks)."""
    for j, obs in step.results_for(call):
        for m in OPENED_PAGE.finditer(recorded_text(obs.content)):
            url = m.group(1)
            start = obs.content.text.find(url)
            span = (start, start + len(url)) if start >= 0 else None
            yield Opened(url, Locator(step.index, Channel.OBSERVATION, observation=j, span=span))


def unresolved(step: Step, call: ToolCall) -> bool:
    """An open by reference whose result names no page (an error, nothing recorded)."""
    if not by_reference(call):
        return False
    results = step.results_for(call)
    return not results or not all(recorded_web_content(o.content) for _, o in results)


def reference_opens(trace: Trace) -> Iterator[tuple[Step, ToolCall]]:
    """Every open by reference in the agent's calls."""
    for step, call in trace.agent_calls():
        if by_reference(call):
            yield step, call
