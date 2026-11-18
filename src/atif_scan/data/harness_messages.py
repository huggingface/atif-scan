"""Harness messages that deliver a tool result rather than instruct the agent.

Antigravity CLI runs a command past its `WaitMsBeforeAsync` as a background task: the call's
own result only says "Tool is running as a background task with task id: …". When the task
ends, the harness tells the agent in a separate system step, exported as one wrapped
message from that task (`sender=<conversation>/task-N`): the command's exit code and output,
a `schedule` timer's prompt, or a subagent's closing report. That text is what the agent
received from its own actions, not a prompt, so it is read as an observation of that step
(on a 330-trial TB4 run, 2,030 such steps, ~1,650 carrying command output).

Exact wrapper, one message, recognized `extra.antigravity_type` only; anything else stays
the step's message as recorded. The delivery links to no call in its step.
"""

from __future__ import annotations

import re

from .content import content
from .jsonval import JsonObject, as_object
from .model import Content, Observation

ANTIGRAVITY_TYPE = "SYSTEM_MESSAGE"
_DELIVERY = re.compile(
    r"The following is a <SYSTEM_MESSAGE> not actually sent by the user\. It is provided by "
    r"the system as important information to pay attention to\.\n\n<SYSTEM_MESSAGE>\n"
    r"\[Message\] timestamp=\S+ sender=[0-9a-f-]{36}(?:/task-\d+)? priority=[A-Z_]+ "
    r"content=(?P<content>.*?)\n?</SYSTEM_MESSAGE>\s*",
    re.S,
)


def delivered_result(raw: JsonObject) -> Content | None:
    """The result an Antigravity system step delivers, or None for any other step."""
    if raw.get("source") != "system":
        return None
    if as_object(raw.get("extra")).get("antigravity_type") != ANTIGRAVITY_TYPE:
        return None
    message = raw.get("message")
    found = _DELIVERY.fullmatch(message) if isinstance(message, str) else None
    if found is None or "[Message] " in found["content"]:
        return None
    return content(found["content"])


def delivery_observations(raw: JsonObject) -> tuple[Content, tuple[Observation, ...]] | None:
    """(message, observations) for a delivery step: the text moves to one unlinked result."""
    delivered = delivered_result(raw)
    if delivered is None:
        return None
    return Content(), (Observation(None, delivered),)
