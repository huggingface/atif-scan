"""Trajectory integrity checks over recorded metadata.

These report inconsistencies in how a trace was recorded or exported (timestamps,
step ids, call links, telemetry). They are provenance/review signals: a smeared
timestamp is not misconduct, but it removes timing evidence a reviewer might rely on.
Harbor's schema validator checks timestamp syntax only, not ordering or smearing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from ..checks import CheckSpec, Context, Detection, Detector, Severity, Status
from ..data.facts import output_ratio
from ..data.model import Channel, Locator, Trace
from ..data.web_inputs import web_input
from ..data.web_results import (
    WebResultState,
    recorded_script_failure,
    web_outcomes_recorded,
    web_result_state,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# A run is "smeared" when at least this fraction of consecutive timestamps are identical.
SMEAR_FRACTION = 0.9
SMEAR_MIN_STEPS = 3


def _result(hits: list[Locator], complete: bool, matched: bool | None = None) -> Detection:
    """`matched` decides a trace-level fact that has no locator of its own."""
    if matched is None:
        return Detection.of(hits, complete)
    status = Status.MATCH if matched else Status.NO_MATCH if complete else Status.UNKNOWN
    return Detection(status, tuple(dict.fromkeys(hits)), complete)


def _meta(step) -> Locator:
    return Locator(step.index, Channel.METADATA)


def timestamp_invalid(trace: Trace) -> Detection:
    recorded = [s for s in trace.steps if s.timestamp_recorded]
    hits = [_meta(s) for s in recorded if s.timestamp is None]
    return _result(hits, complete=bool(recorded))


def timestamp_missing(trace: Trace) -> Detection:
    """Agent steps without a usable timestamp: timing evidence is absent for them."""
    hits = [_meta(s) for s in trace.steps if s.authored and s.timestamp is None]
    return _result(hits, complete=True)


MIN_ORDERED_STEPS = 2  # an order needs two steps


def timestamp_regression(trace: Trace) -> Detection:
    """Recorded timestamps that go backwards. Scope: steps that carry a timestamp;
    copied context is excluded because it may legitimately carry older times."""
    timed = [(s, t) for s in trace.steps if not s.copied and (t := s.timestamp) is not None]
    if sum(not s.copied for s in trace.steps) < MIN_ORDERED_STEPS:
        return Detection(Status.NO_MATCH)
    hits = [_meta(b) for (_, earlier), (b, later) in pairwise(timed) if later < earlier]
    return _result(hits, complete=len(timed) >= MIN_ORDERED_STEPS)


def timestamp_smearing(trace: Trace) -> Detection:
    """Recorded timestamps that are (almost) all identical, e.g. stamped at export time."""
    timed = [s for s in trace.steps if not s.copied and s.timestamp is not None]
    if sum(not s.copied for s in trace.steps) < SMEAR_MIN_STEPS:
        return Detection(Status.NO_MATCH)  # too short for smearing to be meaningful
    if len(timed) < SMEAR_MIN_STEPS:
        return Detection(Status.UNKNOWN, complete=False)
    pairs = list(pairwise(timed))
    same = sum(a.timestamp == b.timestamp for a, b in pairs)
    matched = same / len(pairs) >= SMEAR_FRACTION
    return _result([_meta(timed[0])] if matched else [], True, matched)


def step_sequence(trace: Trace) -> Detection:
    """ATIF requires step_id to run 1..n; absent ids cannot be verified (unknown)."""
    hits = [_meta(s) for s in trace.steps if s.step_id_recorded and s.step_id != s.index + 1]
    return _result(hits, complete=all(s.step_id_recorded for s in trace.steps))


def orphan_observation(trace: Trace) -> Detection:
    """An observation's source_call_id must name a tool call in the same step."""
    hits = []
    for step in trace.steps:
        ids = {call.id for call in step.calls}
        for j, observation in enumerate(step.observations):
            if observation.source_call_id is not None and observation.source_call_id not in ids:
                hits.append(Locator(step.index, Channel.OBSERVATION, observation=j))
    return _result(hits, complete=True)


def observation_pairing_reconstructed(trace: Trace) -> Detection:
    """Warning: result links were inferred by position or a unique recorded remainder."""
    return _result(
        [
            Locator(step.index, Channel.METADATA, observation=j)
            for step in trace.steps
            for j, observation in enumerate(step.observations)
            if observation.pairing_reconstructed
        ],
        complete=True,
    )


def observation_pairing_unresolved(trace: Trace) -> Detection:
    """Results without a safe positional or unique-remainder association: checks
    that depend on which call produced them are unresolved there, never cleared."""
    return _result(
        [
            Locator(step.index, Channel.METADATA, observation=j)
            for step in trace.steps
            for j, observation in enumerate(step.observations)
            if observation.pairing_unresolved
        ],
        complete=True,
    )


def call_id_reused(trace: Trace) -> Detection:
    """A tool_call_id already used by an earlier (non-copied) step: an exporter defect.
    Harmless for linking (observations link within their step), but IDs aren't unique."""
    seen: set[str] = set()
    hits = []
    for step in trace.steps:
        if step.copied:
            continue
        ids = {call.id for call in step.calls if call.id}
        if ids & seen:
            hits.append(_meta(step))
        seen |= ids
    return _result(hits, complete=True)


def agent_only_fields(trace: Trace) -> Detection:
    """System/user steps carrying tool calls, reasoning or metrics (ATIF forbids this)."""
    return _result([_meta(s) for s in trace.steps if s.agent_only_fields], complete=True)


def tool_token_telemetry(trace: Trace) -> Detection:
    """Reported zero tool-use tokens despite recorded tool calls (an exporter defect).

    Scope is *reported* telemetry: an agent that reports nothing is not implausible.
    """
    matched = trace.tool_use_tokens == 0 and trace.tool_calls > 0
    return _result([], complete=True, matched=matched)


def history_compacted(trace: Trace) -> Detection:
    """A harness notice that earlier history was compacted into a summary: steps before
    it are not recorded, so the trace is scanned as partial."""
    hits = [Locator(i, Channel.MESSAGE) for i in trace.compacted]
    return _result(hits, complete=True)


def tool_results_not_recorded(trace: Trace) -> Detection:
    """Tool results are bare status words: the trace is scanned as partial."""
    return _result([], complete=True, matched=trace.results_unrecorded)


def actions_not_recorded(trace: Trace) -> Detection:
    """The agent claims work but no tool calls were recorded: scanned as partial."""
    return _result([], complete=True, matched=trace.actions_unrecorded)


WEB_INPUT = {"web_search": Channel.QUERY, "web_fetch": Channel.URL}


def web_results_not_recorded(trace: Trace) -> Detection:
    """Only missing/unusable outcomes, not explicit retrieval errors or input gaps."""
    hits = [
        _meta(step)
        for step, call in trace.agent_calls()
        if call.tool in WEB_INPUT and not web_outcomes_recorded(step, call)
    ]
    return _result(hits, complete=True)


def web_input_unresolved(trace: Trace) -> Detection:
    """A query/target is absent, dynamic, or has unresolved reference provenance."""
    hits = [
        _meta(step)
        for step, call in trace.agent_calls()
        if call.tool in WEB_INPUT and not web_input(call).source_known
    ]
    return _result(hits, complete=True)


def agent_steps_missing(trace: Trace) -> Detection:
    """The trajectory has no agent steps: nothing the agent did (or didn't do) was recorded,
    so no behaviour check can be answered."""
    return _result([], complete=True, matched=trace.agent_steps == 0)


def redacted_values(trace: Trace) -> Detection:
    """The file carried bare `[REDACTED]` values (invalid JSON, read as unknown): a
    publisher redaction defect, e.g. token counts in TB4 trajectories on the Harbor Hub."""
    return _result([], complete=True, matched=trace.redacted_values > 0)


def trace_head_missing(trace: Trace) -> Detection:
    """No prompt before the first agent step: the start wasn't exported (partial)."""
    return _result([], complete=True, matched=trace.head_missing)


# Subagent launchers whose work happens in another context.
SUBAGENT_TOOLS = frozenset(
    {"Agent", "Task", "explore", "run_subagent", "schedule_subagent", "spawn_agent", "subagent"}
)
SUBAGENT_STUB = re.compile(r"^\s*(?:success|ok|done|launched|async agent launched\b.*)?\s*$", re.I)


def subagent_unrecorded(trace: Trace) -> Detection:
    """A subagent was launched but its result is only a status stub: its tool calls (and
    anything it fetched) aren't in this trace (ACE `explore`, async Claude Code agents)."""
    hits = []
    for step, call in trace.agent_calls():
        if call.name not in SUBAGENT_TOOLS:
            continue
        if all(SUBAGENT_STUB.match(o.content.text) for _, o in step.results_for(call)):
            hits.append(Locator(step.index, Channel.MESSAGE))
    return _result(hits, complete=True)


def cost_missing(trace: Trace) -> Detection:
    """Positive final token counts without ATIF cost; other records may carry cost."""
    usage = trace.usage
    if usage is None:
        return Detection(Status.UNKNOWN, complete=False)
    tokens = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
    return _result([], complete=True, matched=usage.cost_usd is None and tokens > 0)


# No single model call consumes more prompt tokens than this; more per recorded call means
# the totals include activity the recorded steps don't show (e.g. compacted history).
TOKENS_PER_CALL = 2_000_000


def tokens_exceed_recorded_calls(trace: Trace) -> Detection:
    usage = trace.usage
    if usage is None or usage.prompt_tokens is None:
        return Detection(Status.UNKNOWN, complete=False)
    calls = trace.llm_calls or trace.agent_steps
    if not calls:
        return Detection(Status.UNKNOWN, complete=False)
    return _result([], complete=True, matched=usage.prompt_tokens / calls > TOKENS_PER_CALL)


# Output text vs output tokens. Tokenizers encode ~2-4.5 characters of English/code per
# token. Calibrated on ~6k leaderboard traces (13 agent/model pairs), where the
# answer-only ratio ran p1 >= 1.48 and p99 <= 4.4 and the all-text ratio p99 <= 6.3.
# Outside these bounds the declared completion tokens and the recorded text disagree.
MIN_CHARS_PER_TOKEN = 1.0
MAX_CHARS_PER_TOKEN = 8.0


def output_token_ratio(trace: Trace) -> Detection:
    """Recorded agent text doesn't fit the reported completion tokens (an ATIF inspection
    issue: tokens under/over-reported, or text added/dropped after the fact)."""
    ratio = output_ratio(trace)
    if ratio is None:
        return Detection(Status.UNKNOWN, complete=False)
    if ratio.value > MAX_CHARS_PER_TOKEN:
        return _result([], complete=True, matched=True)
    if not ratio.low_verifiable:  # upper bound passed; lower bound can't be checked
        return Detection(Status.UNKNOWN, complete=False)
    # A plausible metered subset cannot clear attempts whose usage is unknown.
    complete = not trace.calls_without_usage and trace.usage_calls_complete is not False
    return _result([], complete=complete, matched=ratio.value < MIN_CHARS_PER_TOKEN)


def incomplete_tool_generation(trace: Trace) -> Detection:
    """A ratio anomaly plus an explicitly incomplete call with a linked error.

    This supports failed generation, not runaway causation or an unrecorded bill.
    A later repair in the same step cannot erase the original linked failure.
    """
    if output_token_ratio(trace).status is not Status.MATCH:
        return _result([], complete=True)
    hits = [
        Locator(step.index, Channel.METADATA, call=call.index)
        for step, call in trace.agent_calls()
        if call.status == "incomplete"
        and any(
            web_result_state(o.content) is WebResultState.ERROR
            or recorded_script_failure(o.content)
            for _, o in step.results_for(call)
        )
    ]
    return _result(hits, complete=True)


@dataclass(frozen=True)
class TraceCheck:
    spec: CheckSpec
    check: Callable[[Trace], Detection] = field(repr=False)

    def evaluate(self, trace: Trace, context: Context) -> Detection:
        return self.check(trace)


def integrity_detectors() -> list[Detector]:
    return [
        TraceCheck(
            CheckSpec(
                "integrity.incomplete_tool_generation",
                Severity.INFO,
                title="Incomplete tool generation failed",
            ),
            incomplete_tool_generation,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.timestamp_invalid",
                Severity.LOW,
                title="Timestamp not in ISO 8601 format",
            ),
            timestamp_invalid,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.timestamp_missing", Severity.INFO, title="Agent steps without timestamps"
            ),
            timestamp_missing,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.timestamp_regression",
                Severity.LOW,
                title="Timestamp earlier than the one before",
            ),
            timestamp_regression,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.timestamp_smearing",
                Severity.LOW,
                title="Nearly all timestamps identical",
            ),
            timestamp_smearing,
        ),
        TraceCheck(
            CheckSpec("integrity.step_sequence", Severity.INFO, title="Step IDs out of sequence"),
            step_sequence,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.orphan_observation",
                Severity.LOW,
                title="Tool result names a call not in its step",
            ),
            orphan_observation,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.observation_pairing_reconstructed",
                Severity.LOW,
                title="Tool result links inferred",
            ),
            observation_pairing_reconstructed,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.observation_pairing_unresolved",
                Severity.LOW,
                title="Tool results not paired to their calls",
            ),
            observation_pairing_unresolved,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.agent_only_fields",
                Severity.LOW,
                title="System or user step has agent-only fields",
            ),
            agent_only_fields,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.call_id_reused",
                Severity.INFO,
                title="Tool call ID reused by a later step",
            ),
            call_id_reused,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.tool_token_telemetry",
                Severity.INFO,
                title="Zero tool tokens reported despite tool calls",
            ),
            tool_token_telemetry,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.history_compacted",
                Severity.MEDIUM,
                "2",
                title="History compacted into a summary",
            ),
            history_compacted,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.tool_results_not_recorded",
                Severity.MEDIUM,
                title="Tool results recorded as status words only",
            ),
            tool_results_not_recorded,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.actions_not_recorded",
                Severity.MEDIUM,
                title="Work claimed with no tool calls recorded",
            ),
            actions_not_recorded,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.trace_head_missing",
                Severity.INFO,
                title="Trace starts without the prompt",
            ),
            trace_head_missing,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.subagent_unrecorded",
                Severity.LOW,
                "2",
                title="Subagent activity not recorded",
            ),
            subagent_unrecorded,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.web_results_not_recorded",
                Severity.LOW,
                version="2",
                title="Web search or fetch result not recorded",
            ),
            web_results_not_recorded,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.web_input_unresolved",
                Severity.LOW,
                title="Web input or reference provenance unresolved",
            ),
            web_input_unresolved,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.redacted_values",
                Severity.LOW,
                title="Redacted placeholder values in the file",
            ),
            redacted_values,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.agent_steps_missing", Severity.LOW, title="No agent steps recorded"
            ),
            agent_steps_missing,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.cost_missing", Severity.LOW, title="Token totals without an ATIF cost"
            ),
            cost_missing,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.tokens_exceed_recorded_calls",
                Severity.LOW,
                title="Token totals exceed the recorded calls",
            ),
            tokens_exceed_recorded_calls,
        ),
        TraceCheck(
            CheckSpec(
                "integrity.output_token_ratio",
                Severity.LOW,
                version="3",
                title="Agent text doesn't fit reported output tokens",
            ),
            output_token_ratio,
        ),
    ]
