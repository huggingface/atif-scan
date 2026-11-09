"""Synthetic gap: requested inspection is not proof of delivered evidence."""

import json
from pathlib import Path

import pytest

from atif_scan.review.coverage import review_coverage


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Known gap: read_steps calls without correlated results count as read.",
)
def test_review_coverage_requires_correlated_delivery(tmp_path: Path):
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "steps": [
                    {
                        "step_id": 1,
                        "source": "agent",
                        "tool_calls": [
                            {
                                "tool_call_id": "synthetic-read-with-no-result",
                                "function_name": "inspect__read_steps",
                                "arguments": {"first": 1, "last": 3},
                            }
                        ],
                    }
                ],
            }
        )
    )
    coverage = review_coverage(review, [1, 2, 3])
    assert coverage is not None
    assert (coverage["steps_read"], coverage["share"]) == (0, 0.0)
