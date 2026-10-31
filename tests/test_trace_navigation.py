"""Navigating one trace: outline tags, outline paging and bounded, accountable search.

Synthetic traces only. Regressions for evidence gaps a judge hit with the read-only MCP
tools: a capped search silently hid the late decisive match, a part's second match was
invisible behind a harmless first one, and an all-`bash` outline gave no hint where test
runs and edits were.
"""

from __future__ import annotations

import re
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest
from test_inspect_mcp import SERVER, RecordingServer

from atif_scan import parse_trace
from atif_scan.evidence.extract import grep, outline, read_segment, tag_summary, unsearchable
from atif_scan.evidence.navigation import all_tags, ranges


def bash(command: str, result: str = "ok") -> dict[str, object]:
    return {
        "source": "agent",
        "message": "working",
        "tool_calls": [
            {"tool_call_id": "c", "function_name": "bash", "arguments": {"command": command}}
        ],
        "observation": {"results": [{"source_call_id": "c", "content": result}]},
    }


def call(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "source": "agent",
        "message": "working",
        "tool_calls": [{"tool_call_id": "c", "function_name": name, "arguments": arguments}],
        "observation": {"results": [{"source_call_id": "c", "content": "ok"}]},
    }


def trace_of(steps: list[dict[str, object]]):
    numbered = [{"step_id": i + 1, **s} for i, s in enumerate(steps)]
    return parse_trace({"schema_version": "ATIF-v1.7", "steps": numbered})


def tags_by_command(commands: list[str]) -> list[list[str]]:
    trace = trace_of([{"source": "user", "message": "task"}, *map(bash, commands)])
    # Drop the user step and the last-agent marker: these cases are about the call.
    return [[t for t in tags if t != "last-agent"] for tags in all_tags(trace)[1:]]


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q tests/",
        "cd /app && python3 -m pytest -x",
        "python -m unittest discover",
        "npm test",
        "yarn run test:unit",
        "go test ./...",
        "cargo test --release",
        "./gradlew test",
        "make check",
        "bash /tests/run_tests.sh",
        "bash -lc 'cd /app && pytest'",
        "timeout 600 pytest -k slow",
        "python tests/test_parser.py",
    ],
)
def test_test_runs_are_tagged(command):
    assert "test-run" in tags_by_command([command])[0]


@pytest.mark.parametrize(
    "command",
    ["cat pytest.log", "grep -rn test src/", "echo pytest", "ls tests", "pip install pytest"],
)
def test_mentions_of_tests_are_not_test_runs(command):
    assert "test-run" not in tags_by_command([command])[0]


def test_edit_test_edit_and_verifier_path_tags():
    tags = tags_by_command(
        [
            "echo 4 > /app/out.txt",
            "cat > tests/test_extra.py <<'EOF'\ndef test_x(): pass\nEOF",
            "echo 1 > /logs/verifier/reward.txt",
            "ls /tests",
            "echo hi > /dev/null",
            "ls /app/tests",
        ]
    )
    assert tags[0] == ["edit"]
    assert tags[1] == ["edit", "test-edit"]
    assert tags[2] == ["edit", "test-edit", "verifier-path"]
    assert tags[3] == ["verifier-path"]
    assert tags[4] == [] and tags[5] == []


def test_write_tool_web_and_last_agent_tags():
    trace = trace_of(
        [
            call("write_file", {"path": "/app/conftest.py", "content": "x"}),
            call("web_fetch", {"url": "https://example.invalid/page"}),
            call("Read", {"file_path": "/app/main.py"}),
        ]
    )
    assert all_tags(trace) == [["edit", "test-edit"], ["web"], ["last-agent"]]


def test_outline_carries_tags_and_an_index_without_trace_text():
    trace = trace_of([bash("pytest -q", result="SECRET-LOOKING OUTPUT"), bash("echo 1 > a.txt")])
    lines = outline(trace)
    assert lines[0].endswith("#test-run") and lines[1].endswith("#edit #last-agent")
    summary = "\n".join(tag_summary(trace))
    assert "#test-run" in summary and "steps 1" in summary and "not findings" in summary
    assert "SECRET-LOOKING" not in "\n".join(lines) + summary
    assert outline(trace, (2, None)) == lines[1:]


def test_untagged_trace_says_so_and_is_not_a_negative():
    trace = trace_of([{"source": "user", "message": "task only"}])
    assert tag_summary(trace) == [
        "tags: none (tags are navigation hints from simple patterns over call arguments, "
        "not findings; a step without a tag may still hold the evidence)"
    ]


def test_ranges():
    assert ranges([1, 2, 3, 7, 9, 10]) == "1-3, 7, 9-10"
    assert ranges([1, 3, 5], limit=2) == "1, 3, … (+1 more steps)"


def test_grep_counts_every_match_and_locates_each_window():
    text = "harmless token here " + "filler " * 200 + "decisive token and answer 42"
    trace = trace_of([bash("cat notes.txt", result=text)])
    one = grep(trace, re.compile("token"), frozenset())
    assert len(one) == 1 and one[0]["matches"] == 2
    two = grep(trace, re.compile("token"), frozenset(), windows=2)
    assert [h["offset"] for h in two] == [m.start() for m in re.finditer("token", text)]
    seg = two[1]["segment"]
    assert seg == {"part": "result", "index": 0, "field": 0}
    page = read_segment(trace, 1, seg["part"], seg["index"], seg["field"], two[1]["offset"], 40)
    assert page["text"].startswith("token and answer 42")


def test_grep_span_and_media_gaps():
    trace = trace_of(
        [
            bash("echo token"),
            {
                "source": "agent",
                "message": "looking at the screenshot",
                "tool_calls": [
                    {"tool_call_id": "v", "function_name": "view_image", "arguments": {}}
                ],
                "observation": {
                    "results": [{"source_call_id": "v", "content": [{"type": "image"}]}]
                },
            },
            bash("echo token"),
        ]
    )
    assert [h["step"] for h in grep(trace, re.compile("token"), frozenset(), span=(2, 3))] == [3]
    assert unsearchable(trace, (None, None))["media"] == [2]
    assert unsearchable(trace, (3, None)) == {"media": [], "unreadable": []}


@pytest.fixture
def serve(monkeypatch):
    module = ModuleType("mcp.server.fastmcp")
    module.__dict__["FastMCP"] = RecordingServer
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", module)
    namespace = runpy.run_module(SERVER)

    def bound(trace):
        monkeypatch.setitem(namespace["serve"].__globals__, "load_trace", lambda path: trace)
        return namespace["serve"](Path("synthetic.json"))

    return bound


def noisy_trace(noise: int = 60, total: int = 200):
    """`token` in many early steps; the decisive one comes last."""
    steps = [bash("echo token noise") for _ in range(noise)]
    steps += [bash("echo filler") for _ in range(total - noise - 1)]
    steps.append(bash("cp /tests/expected.txt /app/out.txt  # token decisive"))
    return trace_of(steps)


def test_capped_search_names_the_steps_it_did_not_show(serve):
    # Regression: the first 20 hits were all noise; the decisive hit at the last step was
    # cut without notice, so the judge could read the cap as the whole answer.
    server = serve(noisy_trace())
    out = server.request("search_trace", pattern="token")
    assert "decisive" not in out
    assert "NOT shown" in out and "21-60, 200" in out
    assert "searched steps 1-200 (200 steps" in out
    narrowed = server.request("search_trace", pattern="token", first=101)
    assert "decisive" in narrowed and "NOT shown" not in narrowed
    assert "searched steps 101-200 (100 steps" in narrowed


def test_search_shows_counts_locators_and_range_errors(serve):
    text = "harmless token " + "filler " * 100 + "decisive token"
    server = serve(trace_of([bash("cat log", result=text)]))
    out = server.request("search_trace", pattern="token", windows_per_part=2)
    assert out.count("2 matches in this part") == 2 and "decisive" in out
    assert "read_step_segment(step_number=1, part=result, index=0, field=0)" in out
    assert server.request("search_trace", pattern="token", first=5).startswith("error:")
    assert server.request("search_trace", pattern="token", first=3, last=2).startswith("error:")
    none = server.request("search_trace", pattern="absent-word")
    assert none.startswith("no matches in steps 1-1 (1 steps")


def test_no_match_reports_media_it_could_not_search(serve):
    trace = trace_of(
        [
            {
                "source": "agent",
                "message": "screenshot",
                "tool_calls": [{"tool_call_id": "v", "function_name": "view", "arguments": {}}],
                "observation": {
                    "results": [{"source_call_id": "v", "content": [{"type": "image"}]}]
                },
            }
        ]
    )
    out = serve(trace).request("search_trace", pattern="total")
    assert out.startswith("no matches") and "media a text search can't read, at steps: 1" in out


def test_outline_pages_long_traces(serve):
    server = serve(trace_of([bash("pytest -q") for _ in range(450)]))
    out = server.request("trace_outline")
    assert "#test-run (a command that runs a test suite or test script): steps 1-450" in out
    assert "outline continues: steps 401-450 not shown; call trace_outline(first=401)" in out
    assert "\n  400 agent" in out and "\n  401 agent" not in out
    page = server.request("trace_outline", first=401)
    assert "\n  450 agent" in page and "continues" not in page
    assert server.request("trace_outline", first=999).startswith("error:")
