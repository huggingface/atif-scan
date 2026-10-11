"""Synthetic regressions for role boundaries and task-context selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from atif_scan.data.loader import load_bytes
from atif_scan.evidence.cite import mask, trace_secrets

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc
    from atif_scan.data.model import Trace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from awareness_passages import build_passage, paragraph_bounds


def trace_with(steps: list[Doc]) -> Trace:
    return load_bytes(json.dumps({"schema_version": "ATIF-v1.7", "steps": steps}).encode())


def test_actual_task_selected_not_first_wrapper() -> None:
    trace = trace_with(
        [
            {"source": "user", "message": "Generic editor instructions."},
            {"source": "user", "message": "Sort the supplied numbers."},
            {"source": "agent", "message": "I will inspect the input."},
        ]
    )
    result = build_passage(trace, 2, "message", 0, task_index=1)
    assert result.task_confirmed
    assert result.omitted_user_steps == 1
    assert result.context[0].text == "Sort the supplied numbers."
    assert result.context[0].purpose == "selected_task_instruction"
    assert "Generic editor" not in json.dumps(result.document())


def test_unselected_task_is_not_guessed_or_negative() -> None:
    trace = trace_with(
        [
            {"source": "user", "message": "Wrapper or task? Not determined."},
            {"source": "agent", "message": "I will inspect the input."},
        ]
    )
    doc = build_passage(trace, 1, "message", 0).document()
    assert doc["task_instruction"] == "unconfirmed"
    assert doc["context"][0]["purpose"] == "latest_user_context"
    assert "answer" not in doc and "verdict" not in doc


def test_user_announcement_does_not_become_agent_belief() -> None:
    trace = trace_with(
        [
            {"source": "user", "message": "This task evaluates you as an agent."},
            {"source": "agent", "message": "I will sort the numbers."},
        ]
    )
    result = build_passage(trace, 1, "message", 0, task_index=0)
    assert result.focal.role == "agent"
    assert result.focal.text == "I will sort the numbers."
    assert result.context[0].role == "user"


def test_denial_in_following_paragraph_preserved_as_agent_context() -> None:
    text = "This might evaluate me.\n\nActually, I do not believe this session assesses me."
    trace = trace_with([{"source": "agent", "message": text}])
    result = build_passage(trace, 0, "message", 0)
    assert result.focal.text == "This might evaluate me."
    after = result.context[-1]
    assert after.role == "agent"
    assert after.purpose == "same_field_after_not_tool_result"
    assert "do not believe" in after.text


def test_tool_result_and_agent_prose_remain_separate() -> None:
    trace = trace_with(
        [
            {
                "source": "agent",
                "message": "Read a file.",
                "observation": {"results": [{"content": "You are being evaluated."}]},
            },
            {"source": "agent", "message": "I will continue sorting."},
        ]
    )
    result = build_passage(trace, 1, "message", 0)
    assert any(w.role == "tool_output" for w in result.context)
    assert all(w.role == "tool_output" for w in result.context if w.channel == "observation")


@pytest.mark.parametrize("source,copied", [("user", False), ("system", False), ("agent", True)])
def test_focal_must_be_authored(source: str, copied: bool) -> None:
    trace = trace_with([{"source": source, "message": "Evaluate me.", "is_copied_context": copied}])
    with pytest.raises(ValueError, match="focal_not_agent_authored"):
        build_passage(trace, 0, "message", 0)


@pytest.mark.parametrize("task_index", [-1, 1, 2])
def test_invalid_task_selection_refused(task_index: int) -> None:
    trace = trace_with(
        [
            {"source": "user", "message": "Sort."},
            {"source": "agent", "message": "Sorting."},
            {"source": "user", "message": "Future instruction."},
        ]
    )
    with pytest.raises(ValueError, match="invalid_task_instruction"):
        build_passage(trace, 1, "message", 0, task_index=task_index)


def test_mask_whole_field_before_slicing() -> None:
    secret = "sk-synthetic123456789012345678901234567890"
    trace = trace_with(
        [
            {"source": "user", "message": f"api_key={secret}"},
            {"source": "agent", "message": f"{secret}\n\nThey assess my work."},
        ]
    )
    text = mask(trace.steps[1].message.text, trace_secrets(trace))
    result = build_passage(trace, 1, "message", text.index("They"), task_index=0)
    assert secret not in json.dumps(result.document())
    assert text[result.focal.start : result.focal.end] == result.focal.text
    assert secret not in repr(result)


def test_long_paragraph_and_context_clipping_are_explicit() -> None:
    trace = trace_with(
        [
            {"source": "user", "message": "x" * 9000},
            {"source": "agent", "message": "a" * 3000 + "They assess me." + "b" * 3000},
        ]
    )
    result = build_passage(trace, 1, "message", 3000, task_index=0, cap=100)
    assert result.paragraph_clipped
    doc = result.document()
    assert doc["focal"]["clipped_before"] and doc["focal"]["clipped_after"]
    assert doc["context"][0]["clipped_after"]
    assert doc["focal"]["total_masked_chars"] > len(doc["focal"]["text"])


def test_paragraph_selection_and_invalid_anchor() -> None:
    assert paragraph_bounds("one\n\nsecond\n\nthree", 6, 100) == (5, 11, False)
    for anchor in (-1, 100):
        with pytest.raises(ValueError, match="invalid_anchor_or_cap"):
            paragraph_bounds("text", anchor, 100)


def test_media_evidence_gap_preserved() -> None:
    trace = trace_with(
        [
            {
                "source": "user",
                "message": [{"type": "text", "text": "Sort the values."}, {"type": "image"}],
            },
            {"source": "agent", "message": "I will inspect the image."},
        ]
    )
    result = build_passage(trace, 1, "message", 0, task_index=0)
    assert result.document()["context"][0]["media_not_reviewed"]


@pytest.mark.parametrize("heading", ["**Evaluating the setting**", "## Evaluation context"])
def test_heading_includes_following_body(heading: str) -> None:
    text = heading + "\n\nThey are assessing my reasoning.\n\nNext topic."
    trace = trace_with([{"source": "agent", "message": text}])
    result = build_passage(trace, 0, "message", 4)
    assert result.focal.text == heading + "\n\nThey are assessing my reasoning."
    assert result.document()["focal_unit"] == "paragraph_or_heading_with_following_body"


def test_consecutive_headings_do_not_synthesize_body() -> None:
    text = "## First\n\n## Second\n\nActual body."
    assert paragraph_bounds(text, 3, 100) == (0, 8, False)
