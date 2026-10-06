"""Synthetic masked focus regressions; no trace commands or URLs are executed."""

from __future__ import annotations

import json
from threading import Thread
from typing import TYPE_CHECKING

import pytest
from test_browser_server import request
from test_browser_session import fixture

from atif_scan.browser.focus import MARK, masked_focus
from atif_scan.browser.server import create_server
from atif_scan.browser.session import Session
from atif_scan.evidence.cite import mask

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("text", "needle", "known", "status"),
    [
        ("api_key=synthetic123\ncandidate", "candidate", frozenset(), "exact"),
        ("before synthetic123 after", "synthetic123", frozenset({"synthetic123"}), "exact"),
        ("api_key=synthetic123\nend", "synthetic", frozenset(), "masked_region"),
        ("before synthetic123 after", "thetic", frozenset({"synthetic123"}), "masked_region"),
        ("abcSECRETxyz", "SECRET", frozenset({"abcSECRET", "SECRETxyz"}), "masked_region"),
        (
            "candidate api_key=synthetic123\nend",
            "candidate api_key=synthetic",
            frozenset(),
            "masked_region",
        ),
        ("<script>candidate</script>", "candidate", frozenset(), "exact"),
    ],
)
def test_mapping(text: str, needle: str, known: frozenset[str], status: str):
    start = text.index(needle)
    focus = masked_focus(text, (start, start + len(needle)), known)
    assert focus["focus_status"] == status
    highlighted = mask(text, known)[focus["highlight_start"] : focus["highlight_end"]]
    assert highlighted
    assert "synthetic123" not in highlighted
    assert set(focus) == {"focus_status", "highlight_start", "highlight_end"}


@pytest.mark.parametrize("span", [None, [], [0, 0], [-1, 2], [0, 100], [True, 2], ["0", 2]])
def test_invalid_span(span: object):
    assert masked_focus("candidate", span) == {
        "focus_status": "field",
        "highlight_start": None,
        "highlight_end": None,
    }


def test_private_key_and_collision():
    key = "-----BEGIN PRIVATE KEY-----\nsynthetic123456\n-----END PRIVATE KEY-----"
    text = f"before {key} after"
    start = text.index("synthetic")
    focus = masked_focus(text, (start, start + 4))
    assert focus["focus_status"] == "masked_region"
    assert mask(text)[focus["highlight_start"] : focus["highlight_end"]] == "[private key]"
    assert masked_focus(MARK + "candidate", (len(MARK), len(MARK) + 9))["focus_status"] == "field"


def test_marker_context_change_is_rejected():
    # A marker creates a token boundary absent in the original: reject its mask.
    text = "xsk-123456789abcdefghi"
    assert masked_focus(text, (1, len(text)))["focus_status"] == "field"


def test_deep_unicode_first_trigger_and_literal_html(tmp_path: Path):
    text = "api_key=synthetic123\n" + "🙂é" * 6000 + "<b>candidate</b> candidate"
    record, item = fixture(tmp_path, message=text)
    evidence = item["assessments"][0]["evidence"]
    evidence.extend(
        [
            {**evidence[0], "span": [text.rindex("candidate"), len(text)]},
            {"step_id": 7, "channel": "command", "call": 0, "field": 0, "span": [5, 14]},
        ]
    )
    session = Session([record], [item], tmp_path / "feedback")
    result = session.focus("0", "0", 0)
    expected = mask(text).index("candidate")
    assert result["step"] == 7
    assert result["highlight_start"] == expected
    assert result["highlight_end"] == expected + len("candidate")
    assert result["offset"] == expected - 160
    assert result["limit"] == 3000
    assert "<b>candidate</b>" in result["text"]
    assert result["text"] == mask(text)[result["offset"] : result["end_offset"]]
    assert "span" not in json.dumps(session.overview())
    # Duplicate locators in one field must not shift selection of another field.
    locations = session.trial("0")["findings"][0]["locations"]
    assert len(locations) == 2
    assert session.focus("0", "0", 1)["part"] == "call"


def test_missing_span_and_binding(tmp_path: Path):
    record, item = fixture(tmp_path)
    item["assessments"][0]["evidence"][0].pop("span", None)
    session = Session([record], [item], tmp_path / "feedback")
    result = session.focus("0", "0", 0)
    assert result["focus_status"] == "field"
    assert result["highlight_start"] is result["highlight_end"] is None
    assert result["offset"] == 0
    for location in (-1, True, 100):
        with pytest.raises(ValueError, match="invalid_location"):
            session.focus("0", "0", location)
    with pytest.raises(ValueError, match="unknown_finding"):
        session.focus("0", "missing", 0)
    assert record[0].local is not None
    record[0].local.write_text("{}")
    with pytest.raises(ValueError, match="trace_source_changed"):
        session.focus("0", "0", 0)


def test_call_metadata_focus_is_field(tmp_path: Path):
    record, item = fixture(tmp_path)
    item["assessments"][0]["evidence"] = [
        {"step_id": 7, "channel": "metadata", "call": 0, "span": [0, 3]}
    ]
    session = Session([record], [item], tmp_path / "feedback")
    result = session.focus("0", "0", 0)
    assert result["part"] == "call_info"
    assert result["metadata_only"]
    assert result["focus_status"] == "field"


def test_focus_endpoint(tmp_path: Path):
    record, item = fixture(tmp_path)
    session = Session([record], [item], tmp_path / "feedback")
    body = {"trial": "0", "finding": "0", "location": 0}
    with create_server(session) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            status, _, data = request(server, "/api/focus", json.dumps(body).encode())
            assert status == 200
            assert json.loads(data)["focus_status"] == "exact"
            for changes in ({"location": True}, {"location": -1}, {"finding": 0}, {"extra": 1}):
                assert request(server, "/api/focus", json.dumps(body | changes).encode())[0] == 400
            assert request(server, "/api/focus", b"{}")[0] == 400
            assert request(server, "/api/focus", method="GET")[0] == 405
            assert (
                request(
                    server,
                    "/api/focus",
                    json.dumps(body).encode(),
                    headers={"Authorization": None},
                )[0]
                == 403
            )
            assert record[0].local is not None
            record[0].local.write_text("{}")
            status, _, data = request(server, "/api/focus", json.dumps(body).encode())
            assert status == 400
            assert json.loads(data) == {"error": "trace_source_changed"}
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_unavailable_trace(tmp_path: Path):
    record, item = fixture(tmp_path)
    item["input_status"] = "unavailable"
    session = Session([record], [item], tmp_path / "feedback")
    with pytest.raises(ValueError, match="trace_unavailable"):
        session.focus("0", "0", 0)


def test_highlight_end_is_global_not_page_clipped(tmp_path: Path):
    text = "candidate " + "🙂" * 4000
    record, item = fixture(tmp_path, message=text)
    item["assessments"][0]["evidence"][0]["span"] = [0, len(text)]
    session = Session([record], [item], tmp_path / "feedback")
    result = session.focus("0", "0", 0)
    assert result["highlight_end"] == len(text)
    assert result["end_offset"] == 3000
