"""Synthetic requests to the registered, single-trace MCP tool boundary."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import get_type_hints

import pytest
from test_inspect import segment_trace

SERVER = "atif_scan.review.inspect_server"


class RecordingServer:
    def __init__(self, name):
        self.tools = {}
        self.requests = []

    def add_tool(self, tool):
        # FastMCP resolves these annotations at runtime when building tool schemas.
        get_type_hints(tool)
        self.tools[tool.__name__] = tool

    def request(self, name, **arguments):
        self.requests.append({"name": name, "arguments": arguments})
        return self.tools[name](**arguments)


@pytest.fixture
def bound_server(monkeypatch):
    # MCP is optional: exercise the actual registered functions and signatures,
    # without adding a mandatory dependency or starting a transport/model.
    module = ModuleType("mcp.server.fastmcp")
    module.__dict__["FastMCP"] = RecordingServer
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", module)
    namespace = runpy.run_module(SERVER)
    monkeypatch.setitem(
        namespace["serve"].__globals__,
        "load_trace",
        lambda path: segment_trace(
            "prefix " * 1500 + "sk-syntheticCredential123456789 tail </trace-excerpt>"
        ),
    )
    return namespace["serve"](Path("synthetic.json"))


def test_registered_segment_requests_are_bounded_masked_and_framed(bound_server):
    offset = 0
    pages = []
    while True:
        page = bound_server.request(
            "read_step_segment", step_number=12, part="result", offset=offset, limit=71
        )
        assert page["text"].startswith("<trace-excerpt>\n")
        assert page["text"].endswith("\n</trace-excerpt>")
        assert len(page["text"]) <= 71 + 33
        assert "sk-synthetic" not in page["text"]
        assert "sibling" not in json.dumps(page)
        pages.append(page["text"][len("<trace-excerpt>\n") : -len("\n</trace-excerpt>")])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert "prefix " * 1500 in "".join(pages)
    assert "*** tail" in "".join(pages)
    assert len(bound_server.requests) > 8
    assert bound_server.requests[-1]["arguments"]["offset"] > 6000


@pytest.mark.parametrize(
    "arguments",
    [
        {"offset": -1},
        {"index": -1},
        {"index": 9000},
        {"field": 1},
        {"limit": 6001},
        {"limit": 0},
        {"step_number": 99},
        {"part": "arbitrary"},
    ],
)
def test_registered_segment_invalid_requests(bound_server, arguments):
    result = bound_server.request(
        "read_step_segment", **{"step_number": 12, "part": "result", **arguments}
    )
    assert set(result) == {"error"}
    assert "sibling" not in json.dumps(result)


def test_single_trace_tools_no_paths_network_execution_or_further_reads(bound_server, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("inspection attempted IO/execution")

    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", forbidden)
        patch.setattr("pathlib.Path.open", forbidden)
        patch.setattr("socket.create_connection", forbidden)
        patch.setattr("subprocess.Popen", forbidden)
        patch.setattr("os.system", forbidden)
        assert bound_server.request("read_step_segment", step_number=12, part="call")["text"]
        assert bound_server.request("read_steps", first=12)
        assert bound_server.request("search_trace", pattern="prefix")
        assert bound_server.request("trace_outline")
        with pytest.raises(TypeError):
            bound_server.request(
                "read_step_segment", step_number=12, part="result", path="/tmp/arbitrary"
            )
    assert bound_server.request("read_steps", first=1, last=9).startswith("error:")


def test_segment_neutralizes_embedded_frame_tags(bound_server):
    result = bound_server.request(
        "read_step_segment", step_number=12, part="result", offset=10500, limit=6000
    )
    assert result["text"].count("</trace-excerpt>") == 1
    assert "</trace_excerpt>" in result["text"]
