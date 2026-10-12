"""Allowlisted facts from ALE attempt receipts; no filesystem or trajectory parsing."""

from __future__ import annotations

from .jsonval import Doc, JsonObject, as_object, as_str, count, identifier, number

BENCHMARK = "agents-last-exam"
LIMIT = 256 * 1024
SCHEMA_VERSION = 2
TASK_PATH_PARTS = 3
STATUSES = frozenset({"completed", "failed", "error", "timeout", "cancelled", "running"})
PHASES = frozenset({"setup", "agent", "eval", "evaluation", "verifier", "teardown"})
TERMINATIONS = STATUSES | {"max_steps", "max_tokens", "budget_exceeded", "interrupted"}
SOURCE_FIELDS = (
    "benchmark",
    "receipt_schema_version",
    "termination_reason",
    "companion_scope",
    "variant_index",
    "source_issues",
    "reward_source",
    "trajectory_source",
    "native_transcript_available",
    "submission_artifacts_available",
)


def label(value: object) -> str | None:
    text = as_str(value)
    try:
        return identifier(text) if text else None
    except ValueError:
        return None


def _task_path(path: str) -> bool:
    parts = path.split("/")
    return (
        len(parts) == TASK_PATH_PARTS
        and parts[0] == "tasks"
        and all(label(p) is not None and p not in {".", ".."} for p in parts[1:])
    )


def recognized(run: JsonObject) -> bool:
    """Versioned receipt shape, not prose, directory names or credentials/config."""
    task, agent = as_object(run.get("task")), as_object(run.get("agent"))
    path = as_str(task.get("path")) or ""
    return (
        type(run.get("schema_version")) is int
        and run["schema_version"] == SCHEMA_VERSION
        and bool(as_str(run.get("run_id")))
        and as_str(run.get("timestamp_utc")) is not None
        and label(task.get("slug")) is not None
        and _task_path(path)
        and count(task.get("variant_index")) is not None
        and label(agent.get("id")) is not None
        and isinstance(run.get("termination"), dict)
        and isinstance(run.get("timings"), dict)
    )


def _score(value: object) -> float | None:
    score = number(value, low=0)
    return score if score is not None and score <= 1 else None


def _reward(
    run: JsonObject, evaluation: JsonObject, issues: list[str], evaluation_present: bool
) -> tuple[float | None, str]:
    recorded = _score(run.get("score"))
    if not evaluation:
        issues.append("evaluation_unreadable" if evaluation_present else "evaluation_not_recorded")
        return (
            (None, "unknown")
            if evaluation_present
            else (recorded, "run" if recorded is not None else "unknown")
        )
    evaluated = _score(evaluation.get("score"))
    if evaluation.get("eval_status") != "success" or evaluated is None:
        issues.append("evaluation_unresolved")
        return None, "unknown"
    if recorded is not None and recorded != evaluated:
        issues.append("score_conflict")
        return None, "conflict"
    return evaluated, "evaluation"


def facts(run: JsonObject, evaluation: JsonObject, *, evaluation_present: bool = False) -> Doc:
    """Recognized receipt only. Conflicts are unknown, not silently resolved."""
    task, agent = as_object(run.get("task")), as_object(run.get("agent"))
    termination = as_object(run.get("termination"))
    issues: list[str] = []
    slug = label(task.get("slug"))
    path = as_str(task.get("path")) or ""
    if "/".join((slug or "").split("__", 1)) != path.removeprefix("tasks/"):
        slug = None
        issues.append("task_identity_conflict")
    reward, provenance = _reward(run, evaluation, issues, evaluation_present)
    status = as_str(run.get("status"))
    phase = as_str(termination.get("phase"))
    reason = as_str(termination.get("reason"))
    return {
        "benchmark": BENCHMARK,
        "receipt_schema_version": SCHEMA_VERSION,
        "termination_reason": reason if reason in TERMINATIONS else None,
        "companion_scope": "listed_files_only",
        "task": slug,
        "variant_index": count(task.get("variant_index")),
        "reward": reward,
        "reward_source": provenance,
        "status": status if status in STATUSES else None,
        "configured_agent": label(agent.get("id")),
        "configured_model": label(agent.get("model")),
        "duration_sec": number(as_object(run.get("timings")).get("duration_s"), low=0),
        "verifier_duration_sec": number(evaluation.get("eval_duration_s"), low=0),
        "failed_phase": phase if phase in PHASES else None,
        "source_issues": issues,
    }
