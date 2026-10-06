"""OpenAI/ChatGPT account identifiers recorded in a trace: scrub before publishing.

Not cheating evidence and not credentials: these identify the account a run used
(ChatGPT-Account-Id headers, decoded OAuth claims, OpenAI organization/project ids).
Unlike the behavioural checks, every recorded step is searched, including prompts and
copied context, because publication exposes all of them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from ..checks import CheckSpec, Context, Detection, Detector, Severity
from ..data import account_ids

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..data.model import Locator, Surface, Trace

ACCOUNT_IDS_EXPOSED = CheckSpec(
    "observation.account_ids_exposed",
    Severity.MEDIUM,
    title="OpenAI account identifier in the trace",
)
ACCOUNT_ID_SHAPES = CheckSpec(
    "observation.account_id_shapes",
    Severity.LOW,
    title="OpenAI org, user or project id shape",
)
ACCOUNT_CLAIMS = CheckSpec(
    "observation.account_claims",
    Severity.LOW,
    title="OpenAI account claim or plan type in the trace",
)


def _surfaces(trace: Trace) -> Iterator[Surface]:
    for step in trace.steps:
        yield from step.authored_surfaces()
        yield from step.observation_surfaces()


@dataclass(frozen=True)
class AccountIdExposure:
    """Spans of one `account_ids` kind anywhere in the trace; never proof of misuse."""

    spec: CheckSpec
    kind: Literal["bound", "shape", "context"]

    def evaluate(self, trace: Trace, context: Context) -> Detection:
        hits: list[Locator] = []
        complete = bool(trace.steps)
        for surface in _surfaces(trace):
            complete = complete and surface.content.understood
            for found in account_ids.find(surface.content.text):
                if found.kind == self.kind:
                    hits.append(replace(surface.at, span=found.span))
        # A call without a recorded result may have printed one: not a clean negative.
        if any(not step.results_for(call) for step, call in trace.agent_calls()):
            complete = False
        return Detection.of(hits, complete)


def account_id_detectors() -> list[Detector]:
    return [
        AccountIdExposure(ACCOUNT_IDS_EXPOSED, "bound"),
        AccountIdExposure(ACCOUNT_ID_SHAPES, "shape"),
        AccountIdExposure(ACCOUNT_CLAIMS, "context"),
    ]
