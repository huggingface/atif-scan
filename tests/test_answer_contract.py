"""Synthetic answer contracts; no external models or traces."""

import json

import pytest

from atif_scan.review.answers import parse_answer
from atif_scan.review.catalogue import ANSWER_STEPS, BY_ID
from atif_scan.review.prompts import REASON_CHARS, schema

REPLY = {"answer": "used", "confidence": "high", "steps": [0, 2], "reason": "private sentinel"}
META = {"question": "lookup_used", "version": BY_ID["lookup_used"].version}


@pytest.mark.parametrize("field", REPLY)
def test_all_schema_fields_required(field):
    reply = {key: value for key, value in REPLY.items() if key != field}
    assert parse_answer(json.dumps(reply), META) is None


@pytest.mark.parametrize(
    "steps",
    [
        None,
        False,
        0,
        "",
        {},
        "2",
        [True],
        [False],
        [-1],
        [2.0],
        ["2"],
        [None],
        [2, -1],
        [2] * (ANSWER_STEPS + 1),
    ],
)
def test_invalid_steps_reject_entire_answer(steps):
    assert parse_answer(json.dumps({**REPLY, "steps": steps}), META) is None


@pytest.mark.parametrize("reason", [None, False, 1, [], {}, "x" * (REASON_CHARS + 1)])
def test_reason_must_be_bounded_string(reason):
    assert parse_answer(json.dumps({**REPLY, "reason": reason}), META) is None


@pytest.mark.parametrize("reason", ["", "é" * REASON_CHARS])
def test_valid_boundaries_and_private_reason(reason):
    reply = {**REPLY, "steps": list(range(ANSWER_STEPS)), "reason": reason}
    parsed = parse_answer(json.dumps(reply), META)
    assert parsed is not None
    assert parsed["steps"] == reply["steps"]
    assert "reason" not in parsed


@pytest.mark.parametrize("field", ["answer", "confidence"])
@pytest.mark.parametrize("value", [[], {}, True, None])
def test_enum_shapes_fail_closed(field, value):
    assert parse_answer(json.dumps({**REPLY, field: value}), META) is None


def test_schema_bounds_and_additional_fields():
    document = schema(BY_ID["lookup_used"])
    assert document["properties"]["steps"]["items"] == {"type": "integer", "minimum": 0}
    assert document["properties"]["reason"]["maxLength"] == REASON_CHARS
    assert parse_answer(json.dumps({**REPLY, "extra": "unallowed"}), META) is None
    # Valid duplicates retain the existing canonical report representation.
    parsed = parse_answer(json.dumps({**REPLY, "steps": [2, 0, 2]}), META)
    assert parsed is not None and parsed["steps"] == [0, 2]


@pytest.mark.parametrize(
    "qid", ["hack_hunt", "verification_hunt", "awareness_hunt", "concealment_hunt", "escape_hunt"]
)
def test_clear_mechanism_consistency_matches_schema(qid):
    question = BY_ID[qid]
    branches = schema(question)["anyOf"]
    for answer in question.answers:
        allowed = next(
            branch["properties"]["mechanism"]["enum"]
            for branch in branches
            if branch["properties"]["answer"]["enum"] == [answer]
        )
        for mechanism in question.mechanisms:
            reply = {**REPLY, "answer": answer, "mechanism": mechanism}
            parsed = parse_answer(json.dumps(reply), {"question": qid, "version": question.version})
            expected = (
                mechanism == "none"
                if answer in {"clean", "absent", "requests_only", "no_requests"}
                else mechanism != "none"
                if answer != "unclear"
                else True
            )
            assert (parsed is not None) == expected == (mechanism in allowed)
