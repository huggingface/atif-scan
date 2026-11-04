"""Count Python tool-program evidence without interpreting or executing its source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .tools import PYTHON_PROGRAM_TOOLS

if TYPE_CHECKING:
    from .model import ToolCall, Trace


@dataclass(frozen=True)
class PtcUsage:
    """Recorded program calls and child evidence, not completed Python executions.

    Incomplete ledgers can retain both inferred and recorded children; these counts
    describe separate evidence entries, not distinct executed actions.
    """

    wrappers: int
    runtime_recorded: int
    static_inferred: int
    unrecognized_wrappers: int
    remaining_unrecognized: int


def _wrapper(call: ToolCall) -> bool:
    return (
        call.name in PYTHON_PROGRAM_TOOLS
        and call.provenance == "model_authored"
        and call.result_id is None
    )


def ptc_usage(trace: Trace | None) -> PtcUsage | None:
    """Authored parents (including category ``other``) and linked child evidence.

    Names are used only to identify parents, never retained. Non-authored steps and
    other program languages are excluded. Unknown-call accounting is unchanged.
    """
    if trace is None:
        return None
    wrappers = runtime = inferred = unrecognized = 0
    for step in trace.steps:
        if not step.authored:
            continue
        parents = [call for call in step.calls if _wrapper(call)]
        parent_ids = {call.id for call in parents}
        wrappers += len(parents)
        unrecognized += sum(call.tool == "other" for call in parents)
        children = [call for call in step.calls if call.result_id in parent_ids]
        runtime += sum(call.provenance == "runtime_recorded" for call in children)
        inferred += sum(call.provenance == "static_inferred" for call in children)
    return PtcUsage(
        wrappers, runtime, inferred, unrecognized, trace.unrecognized_tool_calls - unrecognized
    )
