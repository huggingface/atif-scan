"""hunt --image-model: a judge reads stored image transcripts (synthetic only).

Evidence gap (TB4 Sol hunt, 19 of 61 rewarded trials with inline images): the judge's
trace tools only said "media at step N", so a `clean` answer silently assumed the images
held nothing relevant. `hunt --image-model MODEL` transcribes the traces' images first
(stored like `atif-scan --image-model`), and the read-only server's `read_media` returns
those transcripts. It never asks a model itself, and only a complete transcript stands in
for an image: partial, missing and unmatched images stay unread.
"""

from __future__ import annotations

import json
import os
import runpy
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from test_image_model import image_message, png, reply, sha, uri
from test_inspect_mcp import RecordingServer

from atif_scan.cli import hunt, images
from atif_scan.cli.fast_agent import INSTRUCTION, PTC_BATCHING_NOTE
from atif_scan.evidence.transcripts import STORE_VERSION, TranscriptStore, model_key, stored_path
from atif_scan.review.catalogue import BY_ID
from atif_scan.review.prompts import schema

SERVER = "atif_scan.review.inspect_server"
MODEL = "vision-model"
SECRET = "sk-syntheticCredential123456789"


def trajectory(tmp_path: Path) -> Path:
    steps = [
        {"step_id": 1, "source": "user", "message": "Build the part from the drawing."},
        {"step_id": 2, "source": "agent", "message": "Looking at the drawing."},
        {"step_id": 3, "source": "user", "message": image_message(png(1), png(2))},
        {"step_id": 4, "source": "agent", "message": "Done."},
        {
            "step_id": 5,
            "source": "user",
            "message": [{"type": "image", "source": {"path": "/app/render.png"}}],
        },
        {"step_id": 6, "source": "user", "message": f"see {uri(png(3))}"},
    ]
    path = tmp_path / "trial__a" / "agent" / "trajectory.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": "ATIF-v1.7", "steps": steps}))
    return path


def store(tmp_path: Path, image: bytes, answer: dict[str, object]) -> Path:
    folder = tmp_path / "images"
    folder.mkdir(exist_ok=True)
    stored_path(folder, sha(image), MODEL).write_text(json.dumps({"model": MODEL, **answer}))
    return folder


@pytest.fixture
def server(monkeypatch):
    module = ModuleType("mcp.server.fastmcp")
    module.__dict__["FastMCP"] = RecordingServer
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", module)
    return runpy.run_module(SERVER)


def test_read_media_returns_stored_transcripts_masked_and_framed(tmp_path, server):
    folder = store(tmp_path, png(1), json.loads(reply(f"PART A-12 key {SECRET}")))
    store(tmp_path, png(2), json.loads(reply("", "no_text")))
    bound = server["serve"](trajectory(tmp_path), TranscriptStore(folder, MODEL))
    got = bound.request("read_media", step_number=3)
    first, second = got["media"]
    assert first["status"] == "all" and first["complete"] is True
    assert first["text"].startswith("<trace-excerpt>\n") and "PART A-12" in first["text"]
    assert SECRET not in json.dumps(got)  # masked like the trace
    assert second["status"] == "no_text" and second["complete"] is True
    assert "not a description" in second["note"]
    assert "text" not in second


def test_unread_images_are_never_empty(tmp_path, server):
    folder = store(tmp_path, png(1), json.loads(reply("PART", "partial")))
    bound = server["serve"](trajectory(tmp_path), TranscriptStore(folder, MODEL))
    partial, missing = bound.request("read_media", step_number=3)["media"]
    assert (partial["status"], partial["complete"]) == ("partial", False)
    assert "may show more" in partial["note"]
    assert missing["status"] == "not_transcribed" and "unread" in missing["note"]
    # A file reference has no payload to match: unread, not "no image".
    [ref] = bound.request("read_media", step_number=5)["media"]
    assert ref["status"] == "unidentified"
    assert bound.request("read_media", step_number=4)["media"] == []
    assert "error" in bound.request("read_media", step_number=99)


def test_without_transcripts_every_image_reads_as_unread(tmp_path, server):
    bound = server["serve"](trajectory(tmp_path))
    statuses = [m["status"] for m in bound.request("read_media", step_number=3)["media"]]
    assert statuses == ["not_transcribed", "not_transcribed"]
    # No hint to a tool that has nothing to give.
    assert "read_media" not in bound.request("trace_outline")
    assert "read_media" not in bound.request("search_trace", pattern="drawing")


def test_the_outline_and_searches_point_at_read_media(tmp_path, server):
    bound = server["serve"](trajectory(tmp_path), TranscriptStore(tmp_path / "none", MODEL))
    assert "media at steps 3, 5-6: read_media" in bound.request("trace_outline")
    found = bound.request("search_trace", pattern="drawing")
    assert "media a text search can't read, at steps: 3, 5-6; read_media" in found


def test_a_digest_is_never_a_path(tmp_path):
    assert TranscriptStore(tmp_path, MODEL).get("../../etc/passwd") is None
    assert TranscriptStore(tmp_path, MODEL).get("A" * 64) is None


def test_the_server_command_binds_the_image_store(tmp_path):
    checker = images.ImageChecker(MODEL, store=tmp_path / "images")
    words = shlex.split(hunt.server_command("/t/trajectory.json", checker))
    at = words.index("--image-model")
    assert words[at : at + 4] == ["--image-model", MODEL, "--image-store", str(tmp_path / "images")]
    assert "--image-model" not in shlex.split(hunt.server_command("/t/trajectory.json"))


def test_the_server_reads_its_arguments(server, tmp_path):
    args = server["_arguments"](["t.json", "--image-model", MODEL, "--image-store", str(tmp_path)])
    assert (args.image_model, args.image_store) == (MODEL, tmp_path)
    assert server["_arguments"](["t.json"]).image_model is None


def bundle(tmp_path: Path, trace: Path, questions: int = 2) -> Path:
    root = tmp_path / "bundle"
    (root / "schemas").mkdir(parents=True)
    (root / "index.jsonl").write_text("")
    question = BY_ID["hack_hunt"]
    (root / "schemas" / "hack_hunt.json").write_text(json.dumps(schema(question)))
    for n in range(questions):
        folder = root / f"input-{n}"
        folder.mkdir()
        meta = {
            "question": "hack_hunt",
            "version": question.version,
            "input_id": f"input-{n}",
            "trace_path": str(trace),
        }
        (folder / "hack_hunt.json").write_text(json.dumps(meta))
        (folder / "hack_hunt.md").write_text("Synthetic prompt")
    return root


ANSWER = {"answer": "clean", "confidence": "high", "mechanism": "none", "steps": [], "reason": "x"}


def test_hunt_transcribes_each_image_once_then_binds_the_judge(tmp_path, monkeypatch):
    monkeypatch.setenv("ATIF_SCAN_HOME", str(tmp_path / "home"))
    root = bundle(tmp_path, trajectory(tmp_path))
    sent, asked = [], []

    def image_model(executable, command, timeout=None):
        sent.append(command)
        return subprocess.CompletedProcess(command, 0, reply("PART A-12"), "")

    def judge(executable, command):
        asked.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(ANSWER), "")

    monkeypatch.setattr(images, "run_go", image_model)
    monkeypatch.setattr(hunt, "run_go", judge)
    monkeypatch.setattr(hunt.shutil, "which", lambda _: "synthetic-runner")
    argv = ["--model", "judge", "--questions", str(root), "--inspect-tool"]
    assert hunt.main([*argv, "--image-model", MODEL]) == 0
    assert len(sent) == 3  # three distinct inline images, one trace behind two questions
    assert all(MODEL in c for c in sent)
    for command in asked:
        server = command[command.index("--stdio") + 1]
        assert f"--image-model {MODEL}" in server
    stored = list((tmp_path / "home" / "images").iterdir())
    assert len(stored) == 3

    # Answered again (--force): stored transcripts are reused, nothing is re-sent.
    assert hunt.main([*argv, "--image-model", MODEL, "--force"]) == 0
    assert len(sent) == 3


def test_image_model_needs_the_inspection_tool(tmp_path, monkeypatch, capsys):
    root = bundle(tmp_path, trajectory(tmp_path))
    monkeypatch.setattr(hunt.shutil, "which", lambda _: "synthetic-runner")
    with pytest.raises(SystemExit):
        hunt.main(["--model", "judge", "--questions", str(root), "--image-model", MODEL])
    assert "--inspect-tool" in capsys.readouterr().err


def test_no_images_are_sent_without_the_flag(tmp_path, monkeypatch):
    root = bundle(tmp_path, trajectory(tmp_path))
    sent = []
    monkeypatch.setattr(images, "run_go", lambda *a, **k: sent.append(a))
    monkeypatch.setattr(
        hunt,
        "run_go",
        lambda e, c: subprocess.CompletedProcess(c, 0, json.dumps(ANSWER), ""),
    )
    monkeypatch.setattr(hunt.shutil, "which", lambda _: "synthetic-runner")
    assert hunt.main(["--model", "judge", "--questions", str(root), "--inspect-tool"]) == 0
    assert sent == []


def test_a_clean_answer_that_left_images_unread_is_counted_apart(tmp_path, capsys):
    from atif_scan.cli import main

    trace = trajectory(tmp_path)
    qdir = tmp_path / "q"
    asked = ["--question-scope", "all", "--question", "hack_hunt", "--blind", "--format", "json"]
    main([str(trace), "--questions", str(qdir), *asked])
    capsys.readouterr()
    [meta] = [m for m in qdir.glob("*/hack_hunt.json") if m.parent.name != "schemas"]
    meta.with_suffix(".answer.json").write_text(json.dumps(ANSWER))
    calls = [
        {"function_name": "uv__read_steps", "arguments": {"first": 1, "last": 6}},
        {"function_name": "uv__read_media", "arguments": {"step_number": 3}},
    ]
    meta.with_suffix(".review.atif.json").write_text(json.dumps({"steps": [{"tool_calls": calls}]}))
    argv = [str(trace), "--answers", str(qdir), "--format", "json"]
    main(argv)
    (row,) = json.loads(capsys.readouterr().out)["inputs"][0]["answers"]
    assert (row["coverage"]["media_steps"], row["coverage"]["media_read"]) == (3, 1)
    assert row["coverage"]["share"] == 1.0  # every step read, yet two steps' images not
    main([*argv, "--brief"])
    assert json.loads(capsys.readouterr().out)["answers_images_unread"] == {"hack_hunt": 1}
    from atif_scan.output.brief_view.run import _answers

    brief = {"answers": {"hack_hunt": {"clean": 1}}, "answers_images_unread": {"hack_hunt": 1}}
    [line] = _answers(brief)
    assert "1 clean/absent without reading all of the trace's images" in line


def judge_writing(system: str):
    """A judge run that records `system` as its system prompt, then answers clean."""

    def run(executable, command):
        out = command[command.index("--trajectory-output") + 1]
        steps = [{"step_id": 1, "source": "system", "message": system}]
        Path(out).write_text(json.dumps({"steps": steps}))
        os.utime(out, (100, 100))  # deterministic freshness for the immediate mock write
        return subprocess.CompletedProcess(command, 0, json.dumps(ANSWER), "")

    return run


@pytest.mark.parametrize(
    ("system", "answered"),
    [
        (INSTRUCTION, True),
        (INSTRUCTION + "\n\n" + PTC_BATCHING_NOTE, True),
        (INSTRUCTION + "\n" + PTC_BATCHING_NOTE + "\n# project instructions", False),
        (INSTRUCTION + "\n# project instructions\n" + PTC_BATCHING_NOTE, False),
        (INSTRUCTION + "\n" + PTC_BATCHING_NOTE.replace("read-only", "arbitrary"), False),
        (PTC_BATCHING_NOTE, False),
        # fast-agent's default: the working directory's AGENTS.md and environment.
        ("You are a helpful AI Agent.\n\n# atif-scan development\nRead README.md", False),
        (INSTRUCTION + "\n# project instructions", False),
    ],
)
def test_an_answer_under_a_foreign_system_prompt_is_discarded(
    tmp_path, monkeypatch, system, answered
):
    root = bundle(tmp_path, trajectory(tmp_path), questions=1)
    monkeypatch.setattr(hunt.time, "time", lambda: 100)
    monkeypatch.setattr(hunt, "run_go", judge_writing(system))
    monkeypatch.setattr(hunt.shutil, "which", lambda _: "synthetic-runner")
    status = hunt.main(["--model", "judge", "--questions", str(root)])
    [meta] = root.glob("input-*/hack_hunt.json")
    assert meta.with_suffix(".answer.json").exists() is answered
    assert status == (0 if answered else 1)
    if not answered:
        assert "system prompt was not atif-scan's" in (root / "ask-errors.log").read_text()


def test_transcripts_asked_under_the_old_system_prompt_are_not_read(tmp_path):
    # Stored before atif-scan set its own system prompt: named without STORE_VERSION.
    old = tmp_path / f"{sha(png(1))}.{model_key(MODEL)}.json"
    old.write_text(reply("PART A-12"))
    assert TranscriptStore(tmp_path, MODEL).get(sha(png(1))) is None
    assert stored_path(tmp_path, sha(png(1)), MODEL).name.endswith(f".{STORE_VERSION}.json")
