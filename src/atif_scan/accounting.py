"""Allowlisted Harbor observed accounting; never export arbitrary metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .jsonval import Doc, as_object, as_str, count, number

if TYPE_CHECKING:
    from collections.abc import Mapping

TOKEN_KEYS = {
    "input_tokens": "prompt_tokens",
    "cache_tokens": "cached_tokens",
    "output_tokens": "completion_tokens",
}
TOKEN_SOURCES = ("final_total", "observed_lower_bound", "observed_steps", "unknown")


def _boolean(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _code(value: object, allowed: tuple[str, ...]) -> str | None:
    text = as_str(value)
    return text if text in allowed else None


def harbor_accounting(agent: Mapping[str, object]) -> Doc | None:
    """Read only the versioned contract in AgentContext.metadata.

    Saved fast_agent_final_metrics remain untouched. Their accounting extension describes
    observed availability, not a replacement for canonical ATIF totals.
    """
    metadata = as_object(agent.get("metadata"))
    raw = as_object(metadata.get("fast_agent_accounting"))
    if raw.get("schema") != "harbor.fast-agent.accounting/v1" or raw.get("scope") != "observed":
        return None
    sources = as_object(raw.get("token_sources"))
    result: Doc = {
        "scope": "observed",
        "provider_usage_complete": _boolean(raw.get("provider_usage_complete")),
        "unknown_attempts": count(raw.get("unknown_attempts")),
        "token_sources": {k: _code(sources.get(k), TOKEN_SOURCES) for k in TOKEN_KEYS.values()},
        "tokens_partial": _boolean(raw.get("tokens_partial")),
        "cost_source": _code(
            raw.get("cost_source"), ("reported_final", "reported_steps", "price_derived", "unknown")
        ),
        "cost_scope": _code(raw.get("cost_scope"), ("final_total", "observed")),
        "reported_cost_usd": number(raw.get("reported_cost_usd"), 0),
        "estimated_observed_cost_usd": number(raw.get("estimated_observed_cost_usd"), 0),
        "estimate_basis": _code(raw.get("estimate_basis"), ("context_tokens", "step_tokens")),
    }
    final = as_object(metadata.get("fast_agent_final_metrics"))
    extra = as_object(final.get("extra"))
    extension = as_object(extra.get("accounting"))
    if (
        extension.get("schema") == "fast-agent.accounting/v1"
        and extension.get("scope") == "observed"
    ):
        availability = as_object(extension.get("observed_token_availability"))
        result["final_metrics_accounting"] = {
            "scope": "observed",
            "provider_usage_complete": _boolean(extension.get("provider_usage_complete")),
            "observed_token_availability": {
                k: _boolean(availability.get(k)) for k in TOKEN_KEYS.values()
            },
        }
    return result


def observed_tokens(accounting: Mapping[str, object]) -> bool:
    """Whether these run counts cannot be treated as complete final totals."""
    sources = as_object(accounting.get("token_sources"))
    return (
        accounting.get("tokens_partial") is not False
        or accounting.get("provider_usage_complete") is not True
        or bool(accounting.get("unknown_attempts"))
        or any(sources.get(k) != "final_total" for k in TOKEN_KEYS.values())
    )


def apply_accounting(facts: Doc, accounting: Doc) -> None:
    """Qualify run facts without promoting a fallback into a complete bill."""
    facts["accounting"] = accounting
    sources = as_object(accounting.get("token_sources"))
    for field, kind in TOKEN_KEYS.items():
        if sources.get(kind) in (None, "unknown"):
            facts[field] = None
    if observed_tokens(accounting):
        facts["usage_basis"] = "run_observed"
        facts["recorded_vs_trajectory"] = None
        facts["steps_vs_totals"] = None
        facts["tokens_outside_steps"] = None
    # A reported observed cost is still not a final bill; neither is a price estimate.
    facts["cost_usd"] = (
        accounting.get("reported_cost_usd")
        if accounting.get("cost_source") == "reported_final"
        and accounting.get("cost_scope") == "final_total"
        else None
    )
    facts["cost_records_agree"] = None
    facts["cost_vs_trajectory"] = None


def has_scoped_cost(item: Doc) -> bool:
    """A Harbor contract without a final bill must not be generically repriced."""
    return bool(item.get("accounting")) and item.get("cost_usd") is None
