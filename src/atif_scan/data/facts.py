"""A trial's run facts: where each one comes from, and which record wins.

The same fact can be recorded in several places. From highest precedence to lowest:

- reward: the verifier's reward file beside the trajectory (or a manifest's `reward`)
  > the run listing > the trial's result.json. (`trial_reward`)
- task: `--task` > the trial's result.json > the run listing > the Harbor trial folder
  name with `--task-from trial-dir`. (`trial_task`; `--packs auto` recognises packs from
  the task known before result.json is read, then result.json's.)
- error_type: result.json > the run listing > Harbor's `exception.txt` marker, which
  only says "exception". (`recorded_facts`, `listed_facts`)
- status, hub_trial_id, overrides: the run listing (Harbor Hub records only).
- duration_sec: result.json > the run listing (whole-trial started_at/finished_at).
- agent_duration_sec: result.json's agent_execution interval only; never inferred from
  trajectory step timestamps or substituted with whole-trial time.
- input/cache/output tokens: result.json > the run listing > the trajectory
  (final_metrics totals, else its steps' summed usage).
- cost_usd, decided separately from the tokens: result.json > the run listing >
  harbor-hf's `attempt-costs` record > the trajectory's final_metrics cost. A recorded
  cost is checked against the attempt-costs record and against the trajectory's cost
  when those exist too (`cost_records_agree`, `cost_vs_trajectory`). (`run_facts`)
- in_job_result: the job's own result.json (does it account for this trial folder?).
- agent, model, LLM calls, reasoning exposure, output ratio: the trajectory only.

"The run listing" is, per trial, the Harbor Hub listing (live, or saved beside a synced
job as hub-listing.json) or else a harbor-hf `trials.jsonl` ledger; a saved listing
replaces the ledger's record of the same trial as a whole.

Listing facts are gathered when inputs are resolved (`listed_facts`, kept on
`Source.meta`); result.json and attempt costs are read lazily (`Source.details`). Both
are read afresh on every scan, never cached. Facts derived from the trajectory alone
(`trace_facts`) are cached with the scan results and merged by `run_facts` and `report.assemble`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .accounting import apply_accounting
from .jsonval import Doc, as_object, as_str
from .model import step_reasoning_tokens
from .web_gaps import web_gaps

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .model import Trace


# Agent-authored text vs reported completion tokens: a fact about the trace, used by
# the run brief and by the integrity.output_token_ratio check.
@dataclass(frozen=True)
class OutputRatio:
    """Visible agent-message and tool-argument characters per output token.

    Reasoning text is always excluded: its completeness is not established by ATIF.

    Unrecorded output (hidden reasoning, compacted history, dropped steps) only *lowers*
    the ratio, so a high ratio is always meaningful: more text than the reported tokens
    could encode. A low ratio is only meaningful when reasoning is accounted for.
    `answer_only`: reasoning text is excluded; the denominator is visible output,
    either recorded separately or obtained by subtracting reported reasoning tokens."""

    chars: int
    tokens: int
    answer_only: bool
    low_verifiable: bool
    separate_visible: bool = False

    @property
    def value(self) -> float:
        return self.chars / self.tokens


def output_ratio(trace: Trace) -> OutputRatio | None:
    """None when no completion tokens are reported (or none remain after reasoning)."""
    agent = [s for s in trace.steps if s.authored]
    usage = trace.authored_usage
    if usage is not None and usage.completion_tokens is not None:
        steps, tokens, whole = agent, usage.completion_tokens, True
    else:  # per-step metrics: compare only the steps that report tokens
        steps = [s for s in agent if s.completion_tokens is not None]
        tokens, whole = sum(s.completion_tokens or 0 for s in steps), False
    reasoning = (
        usage.reasoning_tokens if usage is not None and whole else step_reasoning_tokens(steps)
    )
    chars = sum(
        len(s.message.text) + sum((c.raw_argument_chars or 0) for c in s.calls) for s in steps
    )
    separate = bool(whole and usage and usage.completion_basis == "separate_visible")
    answer_only = separate or reasoning is not None
    if not separate and reasoning is not None:
        tokens -= reasoning
    if tokens <= 0:
        return None
    # Compacted final totals include calls the recorded steps don't show.
    low = answer_only and not (whole and trace.compacted)
    return OutputRatio(chars, tokens, answer_only, low, separate)


# Run facts reported as recorded (None when no record has them), in report order.
RECORDED = (
    "error_type",
    "status",
    "hub_trial_id",
    "in_job_result",
    "duration_sec",
    "agent_duration_sec",
    # How a failed trial failed: Harbor's phase timings and the phase the exception fell
    # in, plus the provider's safety codes from fast-agent's results file (codes only).
    "setup_duration_sec",
    "verifier_duration_sec",
    "failed_phase",
    "safety_provider",
    "safety_reason",
    "safety_category",
)
TOKENS = ("input_tokens", "cache_tokens", "output_tokens")
# Facts derived from the trajectory alone (cacheable with its scan), in report order.
TRACE_FACTS = (
    "web_result_gaps",
    "agent_name",
    "agent_version",
    "model_name",
    "step_models",
    "llm_calls",
    "reasoning",
    "chars_per_output_token",
    "output_ratio_basis",
    "trajectory_completion_token_basis",
    "usage",
    "usage_basis",
    "calls_without_usage",
    "stream_retry_attempts",
    "termination_error",
    "final_stop_reason",
    "step_kinds_partial",
    "cache_write_tokens",
    "steps_vs_totals",
    "tokens_outside_steps",
)
# Step-level models reported per trial (most used first).
MAX_STEP_MODELS = 8
# Two recorded costs of one trial agree within a cent, or 1% of the larger.
COST_TOLERANCE_USD = 0.01
COST_TOLERANCE_RATIO = 0.01
# Trace-derived item fields reported after the run facts (the JSON layout's order).
TAIL = ("partial", "agent_steps", "tool_calls", "unrecognized_tool_calls")


def listed_facts(
    listing: Mapping[str, object], exception_marker: bool, in_job_result: bool | None
) -> Doc:
    """A resolved trial's facts from its run listing (saved Hub listing or ledger), the
    `exception.txt` marker when the listing has no error_type, and job membership
    (None: unknown, not recorded)."""
    facts = dict(listing)
    if exception_marker and not facts.get("error_type"):
        facts["error_type"] = "exception"
    if in_job_result is not None:
        facts["in_job_result"] = in_job_result
    return facts


def recorded_facts(listed: Mapping[str, object], result: Mapping[str, object]) -> Doc:
    """The trial's recorded run facts: its result.json (and attempt cost) over the
    listing's."""
    return {**listed, **result}


def trial_reward(
    given: float | None, listed: Mapping[str, object], result: Mapping[str, object]
) -> float | None:
    """The reward file's (or manifest's) value, else the listing's, else result.json's."""
    for value in (given, listed.get("reward"), result.get("reward")):
        if isinstance(value, int | float):
            return value
    return None


def trial_task(
    explicit: str | None, recorded: Mapping[str, object], inferred: str | None
) -> str | None:
    """`--task`, else the recorded task (see `recorded_facts`), else `inferred`."""
    if explicit:
        return explicit
    return as_str(recorded.get("task")) or inferred


def _usage(trace: Trace) -> tuple[Doc | None, str | None]:
    """(tokens and cost, where from): final_metrics totals, else the steps' own usage
    (e.g. the harness died before writing totals); `steps_partial` when some LLM calls
    recorded none, or some steps omit a token kind others record (a lower bound)."""
    usage = trace.usage
    basis = "final_metrics" if usage is not None else None
    if not _has_totals(trace) and trace.step_usage is not None:
        usage = replace(trace.step_usage, cost_usd=usage.cost_usd if usage else None)
        partial = trace.calls_without_usage or trace.step_kinds_partial
        basis = "steps_partial" if partial else "steps"
    if usage is None:
        return None, basis
    tokens = {
        "cost_usd": usage.cost_usd,
        "input_tokens": usage.prompt_tokens,
        "cache_tokens": usage.cached_tokens,
        "output_tokens": usage.completion_tokens,
    }
    return tokens, basis


def _has_totals(trace: Trace) -> bool:
    usage = trace.usage
    return usage is not None and (usage.prompt_tokens, usage.completion_tokens) != (None, None)


def trace_facts(trace: Trace | None) -> Doc:
    """Run facts derived from the trajectory alone (cacheable with its results)."""
    if trace is None:
        return dict.fromkeys(TRACE_FACTS)
    ratio = output_ratio(trace)
    usage, basis = _usage(trace)
    models = sorted(trace.step_models.items(), key=lambda kv: -kv[1])[:MAX_STEP_MODELS]
    return {
        "web_result_gaps": web_gaps(trace).document(),
        "agent_name": trace.agent[0],
        "agent_version": trace.agent[1],
        "model_name": trace.agent[2],
        # Agent steps per step-level model (most used first): what actually ran.
        "step_models": dict(models) or None,
        "llm_calls": trace.llm_calls or trace.agent_steps,
        # A fixed code (Trace.reasoning_exposure): presence, not completeness.
        "reasoning": trace.reasoning_exposure,
        # A number and a fixed code only: authored characters per reported completion token.
        "chars_per_output_token": round(ratio.value, 2) if ratio else None,
        "output_ratio_basis": None
        if ratio is None
        else "separate_visible"
        if ratio.separate_visible
        else "answer_only"
        if ratio.answer_only
        else "visible_only",
        # Final-metrics semantics only, not a claim about run/step totals or billing.
        "trajectory_completion_token_basis": (
            trace.usage.completion_basis
            if trace.usage and trace.usage.completion_tokens is not None
            else None
        ),
        "usage": usage,
        # A fixed code: where the trajectory's tokens come from (final_metrics totals, or
        # summed step metrics, `steps_partial` when some agent steps recorded none).
        "usage_basis": basis,
        "calls_without_usage": trace.calls_without_usage if basis == "steps_partial" else 0,
        # Explicit fast-agent provider retries are harness events, not proof of
        # missing usage or agent history, whatever the accounting basis.
        "stream_retry_attempts": trace.stream_retry_attempts,
        # How the trajectory says it ended, when in error (a code; never the message).
        "termination_error": trace.termination_error,
        # Why the last model turn stopped (a code, e.g. "safety"): how a run ended.
        "final_stop_reason": trace.final_stop_reason,
        # Token kinds only some metered steps record (their step sums are lower bounds).
        "step_kinds_partial": list(trace.step_kinds_partial) or None,
        "cache_write_tokens": trace.usage.cache_write_tokens if trace.usage else None,
        **steps_vs_totals(trace),
    }


def _step_pairs(trace: Trace) -> dict[str, tuple[int, int]]:
    """{kind: (total, sum of steps)} for the token kinds both record completely: input,
    output, cached. A kind only some steps record is left out (its step sum is a lower
    bound, not a total). Empty when they can't be compared like for like: no totals or
    no step usage, steps without usage, or compacted history (the steps cover only the
    last context). Explicitly reconciled embedded child usage is excluded from the
    totals here because the steps cover the root only."""
    totals, steps = trace.authored_usage, trace.step_usage
    if not _has_totals(trace) or totals is None or steps is None:
        return {}
    if trace.calls_without_usage or trace.compacted:
        return {}
    kinds = {
        "input": (totals.prompt_tokens, steps.prompt_tokens),
        "output": (totals.completion_tokens, steps.completion_tokens),
        "cached": (totals.cached_tokens, steps.cached_tokens),
    }
    return {
        k: (a, b)
        for k, (a, b) in kinds.items()
        if a is not None and b is not None and k not in trace.step_kinds_partial
    }


def steps_vs_totals(trace: Trace) -> Doc:
    """How the trajectory's final_metrics totals compare with the sum of its steps' own
    usage: `same`; `steps_short` when the totals are larger in some kind and smaller in
    none, with the input + output tokens outside the comparison scope (not a causal
    explanation); `differs` otherwise; None when not
    comparable (see _step_pairs)."""
    pairs = _step_pairs(trace)
    code, outside = None, None
    if pairs and all(a == b for a, b in pairs.values()):
        code, outside = "same", 0
    elif pairs and all(a >= b for a, b in pairs.values()):
        # Input and output only: cached tokens are part of the input tokens.
        code = "steps_short"
        outside = sum(a - b for k, (a, b) in pairs.items() if k in ("input", "output"))
    elif pairs:
        code = "differs"
    return {"steps_vs_totals": code, "tokens_outside_steps": outside}


def _token(values: Mapping[str, object], key: str) -> int | None:
    value = values.get(key)
    return value if isinstance(value, int) else None


def recorded_vs_trajectory(
    recorded: Mapping[str, object], traced: Mapping[str, object]
) -> str | None:
    """How the run's recorded token totals (result.json, Hub, ledger) compare with the
    trajectory's final_metrics totals: `same`; `uncached_input` when output and cached
    tokens agree and the recorded input is the trajectory's prompt tokens minus cache
    reads (and cache writes): a harness convention (Claude Code on the Harbor Hub), not a
    disagreement; `differs` otherwise. None unless both are complete totals."""
    usage = traced.get("usage")
    if not isinstance(usage, dict) or traced.get("usage_basis") not in ("final_metrics", "steps"):
        return None
    rec = {k: _token(recorded, k) for k in TOKENS}
    traj = {k: _token(usage, k) for k in TOKENS}
    known = [k for k in TOKENS if rec[k] is not None and traj[k] is not None]
    if rec["input_tokens"] is None or not known:
        return None
    if all(rec[k] == traj[k] for k in known):
        return "same"
    rec_in, traj_in = rec["input_tokens"], traj["input_tokens"]
    cached, writes = traj["cache_tokens"] or 0, _token(traced, "cache_write_tokens") or 0
    others_agree = all(rec[k] == traj[k] for k in known if k != "input_tokens")
    uncached = traj_in is not None and rec_in in (traj_in - cached, traj_in - cached - writes)
    return "uncached_input" if others_agree and uncached else "differs"


def _costs_agree(cost: float, beside: float) -> bool:
    return abs(cost - beside) <= max(COST_TOLERANCE_USD, COST_TOLERANCE_RATIO * max(cost, beside))


def run_facts(recorded: Mapping[str, object], traced: Mapping[str, object]) -> Doc:
    """A trial's reported run facts, in report order (see the module docstring).

    `recorded` is read fresh on every scan (never cached); `traced` is `trace_facts`."""
    has_tokens = recorded.get("input_tokens") is not None
    usage = traced.get("usage")
    usage = usage if isinstance(usage, dict) else None
    facts: Doc = {k: recorded.get(k) for k in RECORDED}
    # in_job_result False: the job's own result.json doesn't list this trial (rerun/resume).
    facts["overrides"] = recorded.get("overrides") or []
    facts["cost_usd"] = None  # decided below, independently of the tokens
    facts.update({k: recorded.get(k) for k in TOKENS})
    facts.update({k: v for k, v in traced.items() if k not in ("usage", "usage_basis")})
    # Fixed codes: where the tokens come from ("run" = the run's own records), and
    # whether they agree with the trajectory's (None when there's nothing to compare).
    facts["usage_basis"] = "run" if has_tokens else traced.get("usage_basis")
    facts["recorded_vs_trajectory"] = recorded_vs_trajectory(recorded, traced)
    if facts["recorded_vs_trajectory"] == "uncached_input" and usage:
        # The record counts uncached input only: report input with cache, like the rest.
        facts["input_tokens"] = usage.get("input_tokens")
    if not has_tokens and usage:
        facts.update({k: usage.get(k) for k in TOKENS})  # the trajectory's tokens
    elif not has_tokens:
        facts["usage_basis"] = None
    facts.update(_cost(recorded, usage))
    if accounting := as_object(recorded.get("accounting")):
        # The adapter qualifies its own counts; do not replace missing observed kinds
        # with canonical trajectory totals or normalise them as uncached input.
        facts.update({k: recorded.get(k) for k in TOKENS})
        apply_accounting(facts, accounting)
    return facts


def _cost(recorded: Mapping[str, object], usage: Mapping[str, object] | None) -> Doc:
    """cost_usd: the run's record > harbor-hf's attempt-costs > the trajectory's; and
    whether a recorded cost agrees with the other two where they exist (None: nothing to
    compare). A cost recorded in several places must agree."""
    costs = [
        c if isinstance(c, int | float) else None
        for c in (
            recorded.get("cost_usd"),
            recorded.get("attempt_cost_usd"),
            usage.get("cost_usd") if usage else None,
        )
    ]
    run, beside, traced = costs
    chosen = next((c for c in costs if c is not None), None)
    first = run if run is not None else beside
    return {
        "cost_usd": chosen,
        # harbor-hf's attempt-costs record, when the trial's own record has a cost too.
        "cost_records_agree": _costs_agree(run, beside)
        if run is not None and beside is not None
        else None,
        # The trajectory's own cost, when a recorded one exists too.
        "cost_vs_trajectory": ("same" if _costs_agree(first, traced) else "differs")
        if first is not None and traced is not None
        else None,
    }
