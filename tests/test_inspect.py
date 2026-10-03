"""--inspect: layout classification from names and sizes only (synthetic trees)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from atif_scan.jsonval import Doc

import pytest

from atif_scan.cli import main
from atif_scan.layout import document
from atif_scan.sources import list_input

# Deliberately not valid JSON: inspection must never read trace contents.
NOT_READ = "not json: inspection must not open this"


def write(root, files):
    for path in files:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(NOT_READ)


HARBOR_JOB = [
    "job.log",
    "config.json",
    "lock.json",
    "result.json",
    # A normal trial with a simulated user.
    "task-a__1/trial.log",
    "task-a__1/config.json",
    "task-a__1/result.json",
    "task-a__1/agent/trajectory.json",
    "task-a__1/agent/trajectory.summarization-1-summary.json",
    "task-a__1/user-agent/trajectory.json",
    "task-a__1/verifier/reward.txt",
    # A failed trial: no trajectory, an exception.
    "task-b__2/trial.log",
    "task-b__2/config.json",
    "task-b__2/result.json",
    "task-b__2/exception.txt",
    # A multi-step trial.
    "task-c__3/trial.log",
    "task-c__3/config.json",
    "task-c__3/result.json",
    "task-c__3/steps/build/agent/trajectory.json",
    "task-c__3/steps/build/verifier/reward.json",
]


def inspect(root, pattern="trajectory.json"):
    return document([list_input(str(root))], pattern)["inputs"][0]


def codes(item):
    return {a["code"]: a for a in item["anomalies"]}


def test_harbor_job_is_recognized_with_roles_and_anomalies(tmp_path):
    write(tmp_path, HARBOR_JOB)
    item = inspect(tmp_path)
    assert item["layout"] == "harbor_jobs"
    assert item["harbor"] == {
        "jobs": 1,
        "trials": 3,
        "trials_with_reward": 1,
        "multi_step_trials": 1,
    }
    assert item["would_scan"] == {
        "total": 3,
        "by_role": {"agent": 1, "simulated_user": 1, "step_agent": 1},
    }
    found = codes(item)
    assert found["simulated_user_trajectories_would_be_scanned"]["examples"] == [
        "task-a__1/user-agent"
    ]
    assert found["trials_without_agent_trajectory"]["examples"] == ["task-b__2"]
    assert found["trials_with_multiple_scanned_trajectories"]["examples"] == ["task-a__1"]
    assert found["trials_with_exception"]["examples"] == ["task-b__2"]
    assert found["unscanned_trajectory_like_files"]["examples"] == [
        "task-a__1/agent/trajectory.summarization-1-summary.json"
    ]


def test_single_harbor_trial_root(tmp_path):
    write(tmp_path, [p.split("/", 1)[1] for p in HARBOR_JOB if p.startswith("task-a__1/")])
    item = inspect(tmp_path)
    assert item["layout"] == "harbor_trials" and item["harbor"]["trials"] == 1
    assert item["would_scan"]["by_role"] == {"agent": 1, "simulated_user": 1}


def test_trajectory_folders_with_mixed_depth_and_companions(tmp_path):
    write(
        tmp_path,
        [
            "run/t1/trajectory.json",
            "run/t1/summary.json",
            "run/t2/trajectory.json",
            "run/replacements/r1/trials/t3/trajectory.json",
            "README.md",
        ],
    )
    item = inspect(tmp_path)
    assert item["layout"] == "trajectory_folders" and item["harbor"] is None
    assert item["folders"] == 3 and item["alongside"] == {"summary.json": 1}
    assert item["other_files"] == {"summary.json": 1, "README.md": 1}
    assert codes(item)["mixed_depths"]["examples"] == ["run/replacements/r1/trials/t3"]


def test_no_matches_is_reported_not_raised(tmp_path, capsys):
    write(tmp_path, ["a/summary.json"])
    item = inspect(tmp_path)
    assert item["layout"] == "no_trajectories" and "no_files_match_pattern" in codes(item)
    assert main(["--inspect", str(tmp_path), "--format", "json"]) == 0
    assert inspect(tmp_path, pattern="*.json")["would_scan"]["total"] == 1


def test_single_file(tmp_path):
    write(tmp_path, ["x.json"])
    item = inspect(tmp_path / "x.json")
    assert item["layout"] == "single_file" and item["would_scan"]["total"] == 1


def test_cli_inspect_text_and_json_never_echo_root(tmp_path, capsys):
    write(tmp_path, HARBOR_JOB)
    assert main(["--inspect", str(tmp_path), "--format", "text"]) == 0
    text = capsys.readouterr().out
    assert "layout: harbor_jobs" in text and "simulated_user" in text
    assert main(["--inspect", str(tmp_path), "--format", "json"]) == 0
    raw = capsys.readouterr().out
    assert json.loads(raw)["kind"] == "inspection"
    assert str(tmp_path) not in text + raw and NOT_READ not in text + raw


@pytest.mark.parametrize("argv", [["--inspect"], ["--inspect", "--manifest", "m.json"]])
def test_cli_inspect_needs_paths(argv, capsys):
    assert main(argv) == 2


def test_cli_inspect_missing_path(tmp_path, capsys):
    assert main(["--inspect", str(tmp_path / "nope")]) == 2
    assert "path_not_found" in capsys.readouterr().err


def test_inspect_hub_listing_uses_sizes_without_opening(monkeypatch, fake_hub):

    class ListingOnlyFS:
        files: ClassVar[dict[str, int]] = {
            "buckets/o/b/run/t1/trajectory.json": 10,
            "buckets/o/b/run/t1/summary.json": 2,
        }

        def info(self, path):
            return {"type": "directory"}

        def find(self, path, detail=False):
            return {p: {"type": "file", "size": n} for p, n in self.files.items()}

        def open(self, *args):  # pragma: no cover - must not be called
            raise AssertionError("inspection opened a file")

    fake_hub(ListingOnlyFS)
    item = document([list_input("hf://buckets/o/b/run")], "trajectory.json")["inputs"][0]
    assert item["remote"] and item["bytes"] == 12
    assert item["would_scan"]["total"] == 1 and item["alongside"] == {"summary.json": 1}


def segment_trace(text="synthetic result"):
    from atif_scan import parse_trace

    return parse_trace(
        {
            "schema_version": "ATIF-v1.7",
            "steps": [
                {
                    "step_id": 12,
                    "source": "agent",
                    "message": "sibling message must not be returned",
                    "reasoning_content": "sibling reasoning must not be returned",
                    "tool_calls": [
                        {
                            "tool_call_id": "a",
                            "function_name": "bash",
                            "arguments": {"command": "touch /tmp/never-execute-synthetic"},
                        }
                    ],
                    "observation": {"results": [{"source_call_id": "a", "content": text}]},
                }
            ],
        }
    )


def test_segment_paginates_past_prefix_without_sibling_fields():
    from atif_scan.extract import read_segment

    text = "synthetic " * 1600 + "final evidence"
    trace = segment_trace(text)
    offset = 0
    pages = []
    while True:
        page = read_segment(trace, 12, "result", offset=offset, limit=97)
        assert page["total_length"] == len(text)
        assert page["end_offset"] - page["offset"] <= 97
        assert page["truncated"] == (page["next_offset"] is not None)
        pages.append(page["text"])
        assert "sibling" not in json.dumps(page)
        assert "never-execute" not in json.dumps(page)
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert "".join(pages) == text


def test_segment_masks_whole_field_before_slicing_and_uses_masked_offsets():
    from atif_scan.cite import mask
    from atif_scan.extract import read_segment

    secret = "sk-" + "syntheticCredential123" * 3
    text = "x" * 35 + " " + secret + " tail"
    trace = segment_trace(text)
    expected = mask(text)
    pages = [
        read_segment(trace, 12, "result", offset=offset, limit=7)
        for offset in range(0, len(expected), 7)
    ]
    assert "".join(p["text"] for p in pages) == expected
    assert secret not in json.dumps(pages)
    assert all(p["total_length"] == len(expected) for p in pages)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step_number": 0},
        {"step_number": True},
        {"part": "results"},
        {"index": -1},
        {"index": 1},
        {"field": -1},
        {"field": 1},
        {"offset": -1},
        {"offset": 100},
        {"offset": True},
        {"limit": 0},
        {"limit": -1},
        {"limit": 6001},
        {"limit": True},
    ],
)
def test_segment_rejects_invalid_bounds(kwargs):
    from atif_scan.extract import read_segment

    arguments: Doc = {"step_number": 12, "part": "result", **kwargs}
    with pytest.raises(ValueError):
        read_segment(segment_trace(), **arguments)


def test_segment_empty_media_unreadable_and_end_bounds():
    from atif_scan.extract import read_segment

    empty = read_segment(segment_trace(""), 12, "result")
    assert empty["status"] == "empty"
    assert empty["total_length"] == 0 and empty["next_offset"] is None
    assert empty["text"] == "" and not empty["truncated"]
    media = read_segment(
        segment_trace([{"type": "image", "source": "https://invalid.synthetic/no-fetch"}]),
        12,
        "result",
    )
    assert media["status"] == "media" and media["media"]
    unreadable = read_segment(segment_trace(123), 12, "result")
    assert unreadable["status"] == "unreadable" and not unreadable["understood"]
    end = read_segment(segment_trace("abcd"), 12, "result", offset=4)
    assert end["text"] == "" and end["next_offset"] is None


def test_segment_call_and_message_selection():
    from atif_scan.extract import read_segment

    trace = segment_trace()
    call = read_segment(trace, 12, "call")
    assert call["channel"] == "command" and call["status"] == "text"
    assert "touch" in call["text"]  # only data, never executed
    message = read_segment(trace, 12, "message")
    assert message["text"] == "sibling message must not be returned"
    for index, field in ((1, 0), (0, 1)):
        with pytest.raises(ValueError):
            read_segment(trace, 12, "message", index=index, field=field)
    with pytest.raises(ValueError):
        read_segment(trace, 12, "call", field=99)


@pytest.mark.parametrize("method,call_index", [("position", 0), ("unique_remainder", 1)])
def test_inspection_pairing_provenance_consistent(method, call_index):
    from atif_scan import parse_trace
    from atif_scan.extract import read_segment, render, step_record

    results = [{"content": "first synthetic"}, {"content": "second synthetic"}]
    if method == "unique_remainder":
        # Observation 0 links uniquely to call 1, NOT the positional call 0.
        results[1]["source_call_id"] = "a"
    trace = parse_trace(
        {
            "steps": [
                {
                    "source": "agent",
                    "step_id": 12,
                    "tool_calls": [
                        {
                            "tool_call_id": cid,
                            "function_name": "bash",
                            "arguments": {"command": "printf synthetic"},
                        }
                        for cid in ("a", "b")
                    ],
                    "observation": {"results": results},
                }
            ]
        }
    )
    assert trace.steps[0].observations[0].pairing_method == method
    assert trace.steps[0].observations[0].source_call_index == call_index
    record = step_record(trace, 0, {"results"}, 200, frozenset())
    segment = read_segment(trace, 12, "result")
    for key in ("source_call_index", "pairing_method", "pairing_reconstructed"):
        assert record["results"][0][key] == segment[key]
    rendered = render(record)
    assert f"source_call_index={call_index}" in rendered
    if method == "position":
        assert "WARNING: positional pairing reconstructed" in rendered
    else:
        assert "WARNING: pairing reconstructed by unique remainder" in rendered
        assert "positional" not in rendered


def test_extraction_core_has_no_mandatory_optional_dependencies():
    import subprocess
    import sys

    code = """
import builtins
original = builtins.__import__
def offline(name, *args, **kwargs):
    if name.split('.')[0] in {'rich', 'huggingface_hub', 'mcp'}:
        raise AssertionError('optional dependency imported by core')
    return original(name, *args, **kwargs)
builtins.__import__ = offline
from atif_scan.extract import read_segment
from atif_scan import parse_trace
trace = parse_trace({'steps': [{'source': 'user', 'step_id': 1, 'message': 'synthetic'}]})
assert read_segment(trace, 1, 'message')['text'] == 'synthetic'
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_segment_masks_known_secret_from_sibling_field():
    from atif_scan import parse_trace
    from atif_scan.extract import read_segment

    trace = parse_trace(
        {
            "steps": [
                {
                    "source": "agent",
                    "step_id": 1,
                    "message": "password=syntheticSharedValue123",
                    "observation": {"results": [{"content": "bare syntheticSharedValue123 tail"}]},
                }
            ]
        }
    )
    page = read_segment(trace, 1, "result", limit=10)
    assert page["text"] == "bare *** t"
    assert "syntheticSharedValue" not in json.dumps(page)
