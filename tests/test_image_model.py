"""--image-model: images that block a check, transcribed by an image model (synthetic).

Regression (fast-agent DeepSeek TB2.1 run): ten trials' unprompted-recall checks were
unknown only because an image the agent attached (its own rendered frame) came before
it wrote "tbench". A complete transcript is read as text the agent was shown: a legible
image without the word makes the recall a match, never "clean".
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import struct
import sys
import zlib
from typing import TYPE_CHECKING

import pytest

from atif_scan import Engine, Status, builtin_detectors, parse_trace
from atif_scan.checks import Context, ImageReading
from atif_scan.cli.images import no_session_history, parse_answer
from atif_scan.data.content import content
from atif_scan.data.jsonval import Doc, as_object, as_str
from atif_scan.data.media import find_media

if TYPE_CHECKING:
    from pathlib import Path

RECALL = "recall.benchmark_unprompted"


def png(seed: int = 0) -> bytes:
    """A valid 1x1 PNG (stdlib only); `seed` changes its colour, so its hash."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixel = zlib.compress(bytes([0, seed % 256, 0, 0]))
    chunks = chunk(b"IHDR", header) + chunk(b"IDAT", pixel) + chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + chunks


def uri(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_message(*images: bytes) -> list[dict[str, object]]:
    """fast-agent's shape: the data URI in `source.path`."""
    return [
        {"type": "image", "source": {"media_type": "image/png", "path": uri(i)}} for i in images
    ]


def trace(*images: bytes, later: str = "I recall this might be from tbench tasks") -> Doc:
    steps: list[dict[str, object]] = [
        {"step_id": 1, "source": "user", "message": "Render the first frame."},
        {"step_id": 2, "source": "agent", "message": "Attaching the frame."},
    ]
    if images:
        steps.append({"step_id": 3, "source": "user", "message": image_message(*images)})
    steps.append({"step_id": 4, "source": "agent", "message": "", "reasoning_content": later})
    return {"schema_version": "ATIF-v1.7", "steps": steps}


def recall(raw: Doc, images: dict[str, ImageReading] | None = None) -> Status:
    context = Context(images=images or {})
    found = Engine(builtin_detectors()).evaluate(parse_trace(raw), context)
    return next(a.result.status for a in found if a.spec.id == RECALL)


# --- identities ---------------------------------------------------------------------------


def test_inline_images_are_identified_by_payload_hash_only():
    frame = png()
    block = content(image_message(frame))
    assert block.media and block.media_ids == (sha(frame),)
    assert uri(frame) not in block.text  # the payload never becomes text
    anthropic = {"type": "image", "source": {"media_type": "image/png", "data": uri(frame)[22:]}}
    assert content([anthropic]).media_ids == (sha(frame),)
    assert [m.digest for m in find_media(trace(frame))] == [sha(frame)]


@pytest.mark.parametrize(
    "value",
    [
        "[Image 1]",  # a placeholder: nothing to identify
        [{"type": "image", "source": {"media_type": "image/png", "path": "images/1.png"}}],
        ["[Image 1]", *image_message(png())],  # one part unidentified
        [{"type": "image", "source": {"media_type": "image/png", "data": "!!not base64!!"}}],
    ],
)
def test_media_without_a_readable_payload_stays_unidentified(value):
    found = content(value)
    assert found.media and found.media_ids == ()


# --- the recall check ----------------------------------------------------------------------


def read(text: str, legible: str = "complete") -> ImageReading:
    return ImageReading(text, legible)  # ty: ignore[invalid-argument-type] - tests pass codes


def test_an_unread_image_leaves_a_later_word_unknown():
    assert recall(trace(png())) == Status.UNKNOWN
    assert recall(trace()) == Status.MATCH  # no image: named from memory


@pytest.mark.parametrize(
    ("reading", "expected"),
    [
        (read("DOOM  id Software (c) 1993"), Status.MATCH),  # it didn't show it: recalled
        (read(""), Status.MATCH),  # legible, no text at all
        (read("Terminal-Bench 2.0 leaderboard"), Status.NO_MATCH),  # spelling variant
        (read("DOOM", "partial"), Status.UNKNOWN),  # the rest might have shown it
        (read("", "none"), Status.UNKNOWN),
    ],
)
def test_a_transcript_is_read_as_what_the_agent_was_shown(reading, expected):
    frame = png()
    assert recall(trace(frame), {sha(frame): reading}) == expected


def test_every_image_before_the_word_must_be_transcribed():
    first, second = png(1), png(2)
    raw = trace(first, second)
    assert recall(raw, {sha(first): read("DOOM")}) == Status.UNKNOWN
    assert recall(raw, {sha(first): read("DOOM"), sha(second): read("DOOM")}) == Status.MATCH
    shown = {sha(first): read("DOOM"), sha(second): read("tbench run 3")}
    assert recall(raw, shown) == Status.NO_MATCH


def test_readings_are_bounded():
    with pytest.raises(ValueError, match="invalid_image_reading"):
        read("x" * 20_001)
    with pytest.raises(ValueError, match="invalid_image_reading"):
        read("x", "mostly")


# --- the reply ------------------------------------------------------------------------------


def reply(text: str = "DOOM", text_read: str = "all", **extra: object) -> str:
    fields = {"text": text, "text_read": text_read, "instructions": False, "sensitive": "none"}
    return json.dumps({**fields, **extra})


def test_replies_must_match_the_schema():
    answer = parse_answer("tool status line\n" + reply())
    assert answer is not None and (answer.text, answer.legible) == ("DOOM", "complete")
    assert parse_answer(reply(sensitive="secret")) is None
    assert parse_answer(reply(text_read="mostly")) is None
    # The first schema's answers (`legible`) are re-asked, never guessed at.
    old = {"text": "", "legible": "none", "instructions": False, "sensitive": "none"}
    assert parse_answer(json.dumps(old)) is None
    assert parse_answer(reply(instructions="yes")) is None
    assert parse_answer("Error: model does not accept images") is None


@pytest.mark.parametrize(
    ("text", "text_read", "legible"),
    [
        ("", "no_text", "complete"),  # a chess board or a cell image: nothing to read
        ("", "unreadable", "none"),  # text it can't make out
        ("e4 e5", "no_text", "complete"),  # said no text, transcribed some: judge it
        ("Revenue", "partial", "partial"),
    ],
)
def test_an_image_without_text_is_a_complete_reading(text, text_read, legible):
    # Regression (fast-agent DeepSeek run, gpt-6-luna transcripts): chess boards and
    # microscopy came back `legible: none`, read as unreadable, so they never cleared.
    answer = parse_answer(reply(text, text_read))
    assert answer is not None and answer.legible == legible


def test_an_over_long_transcript_is_cut_and_not_complete():
    answer = parse_answer(reply("x" * 30_000))
    assert answer is not None and answer.legible == "partial" and len(answer.text) == 20_000


# --- the CLI ----------------------------------------------------------------------------------

FAKE = """#!{python}
import json, os, sys
args = sys.argv[1:]
image = args[args.index("--attach") + 1]
prompt = open(args[args.index("--prompt-file") + 1]).read()
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps({{"args": args, "image": open(image, "rb").read().hex(),
                           "prompt": prompt,
                           "session_history": os.environ.get("SESSION_HISTORY")}}) + "\\n")
if os.environ.get("FAKE_FAIL"):
    print("Error: model does not accept images")
    sys.exit(1)
print(os.environ["FAKE_ANSWER"])
"""


@pytest.fixture
def fake_fast_agent(tmp_path: Path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "fast-agent"
    script.write_text(FAKE.format(python=sys.executable))
    script.chmod(0o755)
    log = tmp_path / "fake.log"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LOG", str(log))

    def calls() -> list[dict[str, object]]:
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    return calls


def scan(tmp_path: Path, capsys, raw: Doc, *extra: str) -> Doc:
    from atif_scan.cli import main

    trial = tmp_path / "trial__a" / "agent"
    trial.mkdir(parents=True, exist_ok=True)
    (trial / "trajectory.json").write_text(json.dumps(raw))
    main([str(trial / "trajectory.json"), "--format", "json", "--no-cache", *extra])
    return json.loads(capsys.readouterr().out)["inputs"][0]


def status(item: Doc, check: str = RECALL) -> str | None:
    found = (as_object(a) for a in item["assessments"])
    return next(as_str(a["status"]) for a in found if a["id"] == check)


def test_image_model_transcribes_the_blocking_image(tmp_path, capsys, monkeypatch, fake_fast_agent):
    frame = png()
    monkeypatch.setenv("FAKE_ANSWER", reply("DOOM  id Software"))
    item = scan(tmp_path, capsys, trace(frame, frame), "--image-model", "vision-model")
    assert status(item) == "match"
    assert item["image_checks"] == {
        "images": 1,
        "over_cap": 0,
        "unanswered": 0,
        "all": 1,
        "no_text": 0,
        "partial": 0,
        "unreadable": 0,
        "instructions": 0,
        "sensitive": 0,
    }
    [call] = fake_fast_agent()  # the same image twice is asked about once
    args = call["args"]
    assert args[:3] == ["go", "--model", "vision-model"]
    assert {"--no-shell", "--no-subagents", "--json-schema"} <= set(args)
    assert call["session_history"] == "false"  # the image isn't saved in fast-agent's home
    assert bytes.fromhex(call["image"]) == frame  # the decoded image, not the data URI
    assert "tbench" not in call["prompt"]  # nothing from the trace but the image
    # Reports carry counts, never the transcript or the image.
    text = json.dumps(item)
    assert "DOOM" not in text and sha(frame) not in text


def test_answers_are_stored_privately_and_reused(tmp_path, capsys, monkeypatch, fake_fast_agent):
    from atif_scan.data.paths import home

    frame = png()
    monkeypatch.setenv("FAKE_ANSWER", reply("tbench", instructions=True, sensitive="credential"))
    item = scan(tmp_path, capsys, trace(frame), "--image-model", "vision-model")
    assert status(item) == "no_match"
    assert (item["image_checks"]["instructions"], item["image_checks"]["sensitive"]) == (1, 1)
    [stored] = (home() / "images").iterdir()
    assert stored.name.startswith(sha(frame))
    assert stat.S_IMODE(stored.stat().st_mode) == 0o600
    assert stat.S_IMODE(stored.parent.stat().st_mode) == 0o700
    monkeypatch.setenv("FAKE_FAIL", "1")  # a rescan must not ask again
    assert status(scan(tmp_path, capsys, trace(frame), "--image-model", "vision-model")) == (
        "no_match"
    )
    assert len(fake_fast_agent()) == 1


def test_a_partial_transcript_stays_unknown(tmp_path, capsys, monkeypatch, fake_fast_agent):
    monkeypatch.setenv("FAKE_ANSWER", reply("DOOM", "partial"))
    item = scan(tmp_path, capsys, trace(png()), "--image-model", "vision-model")
    assert status(item) == "unknown"
    assert (item["image_checks"]["all"], item["image_checks"]["partial"]) == (0, 1)


def test_an_image_without_text_clears(tmp_path, capsys, monkeypatch, fake_fast_agent):
    monkeypatch.setenv("FAKE_ANSWER", reply("", "no_text"))
    item = scan(tmp_path, capsys, trace(png()), "--image-model", "vision-model")
    assert status(item) == "match"
    assert item["image_checks"]["no_text"] == 1


def test_only_images_before_the_word_are_sent(tmp_path, capsys, monkeypatch, fake_fast_agent):
    # Regression (financial-document-processor, video-processing: ~30 images each): every
    # image in the trial was sent, up to 20, so the ones that mattered could be cut off.
    early, late = png(1), png(2)
    raw = trace(early)
    raw["steps"].append({"step_id": 5, "source": "user", "message": image_message(late)})
    monkeypatch.setenv("FAKE_ANSWER", reply("DOOM"))
    item = scan(tmp_path, capsys, raw, "--image-model", "vision-model")
    assert status(item) == "match"
    assert [bytes.fromhex(c["image"]) for c in fake_fast_agent()] == [early]
    assert item["image_checks"]["images"] == 1


def test_images_over_the_cap_stay_unread(tmp_path, capsys, monkeypatch, fake_fast_agent):
    from atif_scan.cli import images

    monkeypatch.setattr(images, "MAX_IMAGES", 2)
    monkeypatch.setenv("FAKE_ANSWER", reply("DOOM"))
    item = scan(tmp_path, capsys, trace(png(1), png(2), png(3)), "--image-model", "vision-model")
    assert status(item) == "unknown"
    assert (item["image_checks"]["images"], item["image_checks"]["over_cap"]) == (3, 1)
    assert len(fake_fast_agent()) == 2


def test_a_failed_image_check_is_still_unknown(tmp_path, capsys, monkeypatch, fake_fast_agent):
    monkeypatch.setenv("FAKE_FAIL", "1")
    item = scan(tmp_path, capsys, trace(png()), "--image-model", "vision-model")
    assert status(item) == "unknown"
    assert item["image_checks"]["images"] == 1
    assert item["image_checks"]["unanswered"] == 1


def test_no_model_call_without_a_blocking_image(tmp_path, capsys, monkeypatch, fake_fast_agent):
    monkeypatch.setenv("FAKE_ANSWER", reply())
    quiet = trace(png(), later="Rendering works now.")  # nothing written after it to confirm
    item = scan(tmp_path, capsys, quiet, "--image-model", "vision-model")
    assert status(item) == "no_match"
    assert "image_checks" not in item
    assert fake_fast_agent() == []


def test_without_the_flag_nothing_is_sent(tmp_path, capsys, fake_fast_agent):
    item = scan(tmp_path, capsys, trace(png()))
    assert status(item) == "unknown"
    assert fake_fast_agent() == []


def test_fast_agent_runs_without_session_history(monkeypatch):
    # Both model calls (--image-model and hunt) would otherwise copy prompts and images
    # into fast-agent's home sessions/ folder.
    monkeypatch.setenv("SESSION_HISTORY", "true")
    assert no_session_history()["SESSION_HISTORY"] == "false"


def test_the_brief_totals_image_checks():
    from atif_scan.output.brief import image_check_totals

    one = {"images": 3, "all": 1, "no_text": 1, "partial": 1, "sensitive": 0}
    two = {"images": 2, "all": 2, "no_text": 0, "partial": 0, "sensitive": 1}
    items = [{"image_checks": one}, {"image_checks": two}, {}]
    totals = image_check_totals(items)
    assert totals == {
        "trials": 2,
        "images": 5,
        "all": 3,
        "no_text": 1,
        "partial": 1,
        "sensitive": 1,
    }
    assert image_check_totals([{}]) is None
