"""Coverage uses executed, linked inspection dispatches, never authored Python text."""

import json

import pytest

from atif_scan.review.coverage import review_coverage


def dispatch(tool, arguments, *, status="completed", error=False, suffix="1"):
    return {
        "id": f"outer::ptc-{suffix}",
        "tool": f"inspect__{tool}",
        "arguments": arguments,
        "status": status,
        "is_error": error,
    }


def step(entries, tool="execute_python"):
    return {
        "tool_calls": [
            {
                "tool_call_id": "outer",
                "function_name": tool,
                "arguments": {"input": "await inspect__read_steps(first=99, last=99)"},
            }
        ],
        "observation": {
            "results": [
                {
                    "source_call_id": "outer",
                    "content": "inspection completed",
                    "extra": {
                        "ptc_calls_schema": "fast-agent.ptc-calls/v1",
                        "ptc_calls": entries,
                    },
                }
            ]
        },
    }


def coverage(tmp_path, value, *, required=()):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"steps": [value]}))
    return review_coverage(path, [1, 2, 3, 4, 99], required=required)


@pytest.mark.parametrize("tool", ["execute_python", "exec_sandbox"])
def test_runtime_ledger_counts_reads_and_outline_not_searches_or_authored_code(tmp_path, tool):
    value = step(
        [
            dispatch("trace_outline", {}, suffix="1"),
            dispatch("read_steps", {"first": 1, "last": 2}, suffix="2"),
            dispatch("read_step_segment", {"step_number": 3, "part": "call"}, suffix="3"),
            dispatch("search_trace", {"pattern": "synthetic"}, suffix="4"),
        ],
        tool=tool,
    )
    result = coverage(tmp_path, value)
    assert result == {
        "steps_read": 3,
        "share": 0.6,
        "basis": "steps",
        "outline": True,
        "media_steps": 0,
        "media_read": 0,
        "tool_calls": 5,
    }
    required = coverage(tmp_path, value, required=[2, 4])
    assert required is not None and required["share"] == 0.5
    assert required["basis"] == "required"


def test_completed_ledger_media_reads_preserve_upstream_media_coverage(tmp_path):
    value = step(
        [
            dispatch("read_media", {"step_number": 1}, suffix="1"),
            dispatch("read_media", {"step_number": 2}, error=True, suffix="2"),
        ]
    )
    coverage(tmp_path, value)
    result = review_coverage(tmp_path / "review.json", [1, 2], media_steps=[1, 2])
    assert result is not None
    assert result["media_steps"] == 2
    assert result["media_read"] == 1
    assert result["steps_read"] == 0


@pytest.mark.parametrize(
    ("status", "error"),
    [("raised", True), ("in_flight", None), ("cancelled", None), ("completed", True)],
)
def test_failed_or_unfinished_dispatch_does_not_count_as_read(tmp_path, status, error):
    value = step([dispatch("read_steps", {"first": 1}, status=status, error=error)])
    result = coverage(tmp_path, value)
    assert result is not None and result["steps_read"] == 0


@pytest.mark.parametrize(
    "corruption",
    ["schema", "parent", "duplicate_result", "duplicate_parent", "duplicate_child", "arguments"],
)
def test_invalid_or_ambiguous_ledger_never_grants_coverage(tmp_path, corruption):
    value = step([dispatch("read_steps", {"first": 1})])
    result = value["observation"]["results"][0]
    extra = result["extra"]
    if corruption == "schema":
        extra["ptc_calls_schema"] = "unknown"
    elif corruption == "parent":
        result["source_call_id"] = "other"
    elif corruption == "duplicate_result":
        value["observation"]["results"].append(result.copy())
    elif corruption == "duplicate_parent":
        value["tool_calls"].append(value["tool_calls"][0].copy())
    elif corruption == "duplicate_child":
        extra["ptc_calls"].append(extra["ptc_calls"][0].copy())
    else:
        extra["ptc_calls"][0].pop("arguments")
    measured = coverage(tmp_path, value)
    assert measured is not None and measured["steps_read"] == 0


def test_ledger_embedded_in_result_text_is_not_runtime_metadata(tmp_path):
    value = step([dispatch("read_steps", {"first": 1})])
    result = value["observation"]["results"][0]
    result["content"] = json.dumps(result.pop("extra"))
    measured = coverage(tmp_path, value)
    assert measured is not None and measured["steps_read"] == 0


@pytest.mark.parametrize("arguments", [{"first": 2}, {"first": 2, "last": None}])
def test_omitted_or_null_last_reads_one_step(tmp_path, arguments):
    direct = {"tool_calls": [{"function_name": "inspect__read_steps", "arguments": arguments}]}
    for value in (direct, step([dispatch("read_steps", arguments)])):
        result = coverage(tmp_path, value)
        assert result is not None and result["steps_read"] == 1


@pytest.mark.parametrize(
    "arguments",
    [{"first": 1, "last": 9}, {"first": 2, "last": 1}, {"first": 1, "last": "2"}],
)
def test_rejected_inspector_ranges_do_not_count(tmp_path, arguments):
    result = coverage(tmp_path, step([dispatch("read_steps", arguments)]))
    assert result is not None and result["steps_read"] == 0


@pytest.mark.parametrize("tool", ["other", "inspect__exec_sandbox", None, [], {}])
def test_unrecognized_or_malformed_outer_tool_does_not_grant_ledger_coverage(tmp_path, tool):
    value = step([dispatch("read_steps", {"first": 1})], tool=tool)
    result = coverage(tmp_path, value)
    assert result is not None and result["steps_read"] == 0
