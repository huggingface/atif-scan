"""Real loopback HTTP requests over synthetic, explicitly selected evidence."""

from __future__ import annotations

import json
import socket
from http.client import HTTPConnection
from importlib.resources import files
from threading import Thread
from typing import TYPE_CHECKING

import pytest
from test_browser_session import fixture

from atif_scan.browser.server import CSP, MAX_BODY, MAX_URL, create_server
from atif_scan.browser.session import Session

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from atif_scan.browser.server import DeskServer


@pytest.fixture
def desk(tmp_path: Path) -> Iterator[DeskServer]:
    record, item = fixture(
        tmp_path, message="candidate Authorization: Bearer synthetic-secret-123456789"
    )
    session = Session([record], [item], tmp_path / "feedback")
    with create_server(session) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def request(
    desk: DeskServer,
    path: str = "/api/overview",
    body: bytes | None = None,
    *,
    method: str = "POST",
    headers: Mapping[str, str | None] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    if body is None and method == "POST":
        body = b"{}"
    values = {
        "Host": desk.host,
        "Origin": desk.origin,
        "Authorization": f"Bearer {desk.token}",
        "Content-Type": "application/json",
    }
    for key, value in (headers or {}).items():
        if value is None:
            values.pop(key, None)
        else:
            values[key] = value
    connection = HTTPConnection("127.0.0.1", desk.server_port, timeout=5)
    try:
        connection.request(method, path, body, values)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def post(desk: DeskServer, path: str, body: object) -> tuple[int, dict[str, str], bytes]:
    return request(desk, path, json.dumps(body).encode())


def test_link_binding_and_headers(desk: DeskServer):
    assert desk.server_address == ("127.0.0.1", desk.server_port)
    assert desk.url == f"{desk.origin}/#token={desk.token}"
    assert len(desk.token) >= 43
    assert desk.socket.gettimeout() is not None
    status, headers, body = request(desk)
    assert status == 200
    assert json.loads(body)["trials"][0]["task"] == "synthetic-task"
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["Content-Security-Policy"] == CSP
    assert "Set-Cookie" not in headers
    assert not any(key.lower().startswith("access-control") for key in headers)
    assert desk.token.encode() not in body


@pytest.mark.parametrize("path", ["/", "/index.html", "/browser.js", "/browser.css"])
def test_public_fixed_assets(desk: DeskServer, path: str):
    status, headers, body = request(
        desk, path, method="GET", headers={"Authorization": None, "Origin": None}
    )
    assert status == 200
    assert headers["Cache-Control"] == "no-store"
    name = "index.html" if path == "/" else path.removeprefix("/")
    assert body == files("atif_scan.browser").joinpath("static", name).read_bytes()
    assert int(headers["Content-Length"]) == len(body)
    assert desk.token.encode() not in body


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": None},
        {"Authorization": "Bearer wrong"},
        {"Authorization": "bearer wrong"},
        {"Origin": None},
        {"Origin": "null"},
        {"Origin": "http://localhost"},
        {"Origin": "https://evil.invalid"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Host": "localhost"},
        {"Host": "127.0.0.1"},
        {"Host": "evil.invalid"},
    ],
)
@pytest.mark.parametrize("payload", [b"", b"{}", b"x" * MAX_BODY])
def test_auth_and_origin(desk: DeskServer, headers: dict[str, str | None], payload: bytes):
    status, response_headers, body = request(desk, body=payload, headers=headers)
    assert status == 403
    assert json.loads(body) == {"error": "request_rejected"}
    assert response_headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("name", ["Host", "Origin", "Authorization"])
def test_duplicate_headers(desk: DeskServer, name: str):
    values = {"Host": desk.host, "Origin": desk.origin, "Authorization": f"Bearer {desk.token}"}
    wire = (
        "POST /api/overview HTTP/1.1\r\n"
        + "".join(f"{key}: {value}\r\n" for key, value in values.items())
        + f"{name}: {values[name]}\r\n"
        + "Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{}"
    ).encode()
    data = raw_request(desk, wire)
    assert b" 403 " in data
    assert desk.token.encode() not in data


def raw_request(desk: DeskServer, wire: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", desk.server_port), timeout=5) as connection:
        connection.sendall(wire)
        connection.shutdown(socket.SHUT_WR)
        chunks = []
        while chunk := connection.recv(65536):
            chunks.append(chunk)
        return b"".join(chunks)


@pytest.mark.parametrize(
    "path",
    [
        "/../session.py",
        "/%2e%2e/session.py",
        "/static/",
        "/browser.js?token=private",
        "//index.html",
        "/api/unknown",
        "http://evil.invalid/index.html",
    ],
)
def test_no_traversal_or_url_following(desk: DeskServer, path: str):
    status, _, body = request(desk, path, method="GET")
    assert status == 404
    assert json.loads(body) == {"error": "request_rejected"}


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "PUT", "DELETE", "TRACE"])
def test_api_requires_post(desk: DeskServer, method: str):
    status, headers, body = request(desk, method=method)
    assert status >= 400
    assert headers["Cache-Control"] == "no-store"
    assert not body if method == "HEAD" else b"request_rejected" in body


@pytest.mark.parametrize("body", [b"", b"[1]", b"null", b"{", b"\xff", b'{"x":1}', b'{"x":NaN}'])
def test_malformed_body(desk: DeskServer, body: bytes):
    status, _, response = request(desk, body=body)
    assert status == 400
    assert json.loads(response) == {"error": "invalid_request"}


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/trial", {"trial": True}),
        ("/api/trial", {"trial": 0}),
        ("/api/trial", {"trial": "x" * 129}),
        ("/api/trial", {"trial": "/etc/passwd"}),
        ("/api/trial", {"trial": "https://evil.invalid"}),
        ("/api/segment", {"trial": "0", "step": True, "part": "message"}),
        ("/api/segment", {"trial": "0", "step": 7, "part": "message", "field": False}),
        ("/api/segment", {"trial": "0", "step": 7, "part": "message", "offset": -1}),
        ("/api/segment", {"trial": "0", "step": 7.0, "part": "message"}),
        ("/api/segment", {"trial": "0", "step": 7, "part": "file"}),
        ("/api/search", {"trial": "0", "query": ""}),
        ("/api/search", {"trial": "0", "query": "x" * 151}),
        ("/api/search", {"trial": "0", "query": []}),
        ("/api/feedback", {"trial": "0", "finding": "0", "verdict": "oops", "note": ""}),
        ("/api/feedback", {"trial": "0", "finding": "0", "verdict": "unclear", "note": 1}),
        ("/api/feedback", {"trial": "0", "finding": "0", "verdict": "unclear", "note": "x" * 2001}),
        ("/api/overview", {"path": "/etc/passwd"}),
        ("/api/trial", {}),
    ],
)
def test_field_validation(desk: DeskServer, path: str, body: object):
    status, _, response = post(desk, path, body)
    assert status == 400
    assert json.loads(response) == {"error": "invalid_request"}


def test_limits_and_framing(desk: DeskServer):
    assert request(desk, body=b" " * (MAX_BODY + 1))[0] == 400
    assert request(desk, "/" + "x" * MAX_URL, method="GET")[0] == 414
    for headers in [
        {"Content-Length": "-1"},
        {"Content-Length": "wat"},
        {"Content-Length": "10"},  # Incomplete body times out, not an indefinite hang.
        {"Transfer-Encoding": "chunked"},
        {"Content-Type": "text/plain"},
    ]:
        assert request(desk, headers=headers)[0] in (400, 503)


def test_endpoints_redaction_and_feedback(desk: DeskServer):
    status, _, body = post(desk, "/api/trial", {"trial": "0"})
    assert status == 200
    trial = json.loads(body)
    finding = trial["findings"][0]["id"]
    assert trial["steps"][0]["step"] == 7
    status, _, body = post(desk, "/api/segment", {"trial": "0", "step": 7, "part": "message"})
    assert status == 200
    assert b"candidate" in body
    assert b"synthetic-secret-123456789" not in body
    status, _, body = post(desk, "/api/search", {"trial": "0", "query": "candidate"})
    assert status == 200
    assert json.loads(body)["matches"]
    status, _, body = post(
        desk,
        "/api/feedback",
        {"trial": "0", "finding": finding, "verdict": "unclear", "note": "Synthetic note"},
    )
    assert status == 200
    assert json.loads(body) == {"verdict": "unclear", "note": "Synthetic note"}
    updated = json.loads(request(desk)[2])
    assert updated["trials"][0]["findings"][0]["feedback"]["verdict"] == "unclear"


def test_changed_source_fixed_error(
    desk: DeskServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    (tmp_path / "synthetic.json").write_text("sensitive-path-and-text")
    status, _, body = post(desk, "/api/trial", {"trial": "0"})
    assert status == 400
    assert json.loads(body) == {"error": "trace_source_changed"}
    assert str(tmp_path).encode() not in body
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("port", [-1, True, 65536])
def test_invalid_port(tmp_path: Path, port: int):
    with pytest.raises(ValueError, match="invalid_port"):
        create_server(Session([], [], tmp_path / "feedback"), port)


def test_tokens_unique_and_explicit_port(tmp_path: Path):
    session = Session([], [], tmp_path / "feedback")
    with create_server(session) as first, create_server(session) as second:
        port = first.server_port
        assert first.token != second.token
    with create_server(session, port=port) as rebound:
        assert rebound.server_port == port


@pytest.mark.parametrize(
    "error,status,code",
    [
        (ValueError("trace_unavailable"), 400, "trace_unavailable"),
        (ValueError("/private/path credential=synthetic"), 400, "invalid_request"),
        (PermissionError("/private/path credential=synthetic"), 503, "session_unavailable"),
    ],
)
def test_session_errors_are_allowlisted(
    desk: DeskServer,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status: int,
    code: str,
    capsys: pytest.CaptureFixture[str],
):
    def fail() -> None:
        raise error

    monkeypatch.setattr(desk.session, "overview", fail)
    actual, _, body = request(desk)
    assert actual == status
    assert json.loads(body) == {"error": code}
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    "extra",
    [
        "Content-Length: 2\r\nContent-Length: 2\r\n",
        "Transfer-Encoding: chunked\r\nContent-Length: 2\r\n",
        "",
    ],
)
def test_reject_ambiguous_or_missing_length(desk: DeskServer, extra: str):
    wire = (
        f"POST /api/overview HTTP/1.1\r\nHost: {desk.host}\r\n"
        f"Origin: {desk.origin}\r\nAuthorization: Bearer {desk.token}\r\n"
        f"Content-Type: application/json\r\n{extra}\r\n{{}}"
    ).encode()
    assert b" 400 " in raw_request(desk, wire)


def test_missing_host_and_static_origin_checks(desk: DeskServer):
    assert b" 403 " in raw_request(desk, b"GET / HTTP/1.1\r\n\r\n")
    assert request(desk, "/", method="GET", headers={"Origin": "null"})[0] == 403
    assert request(desk, "/", method="GET", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert request(desk, "/api/overview?x=1")[0] == 404
    assert request(desk, "//api/overview")[0] == 404
    assert request(desk, "/index.html")[0] == 404


@pytest.mark.parametrize("payload", [b"{}", b"x" * MAX_BODY])
def test_get_rejects_body_without_truncating_error(desk: DeskServer, payload: bytes):
    status, headers, body = request(desk, "/browser.js", body=payload, method="GET")
    assert status == 400
    assert json.loads(body) == {"error": "request_rejected"}
    assert int(headers["Content-Length"]) == len(body)


def test_get_explicit_empty_body(desk: DeskServer):
    status, _, body = request(
        desk, "/browser.js", body=b"", method="GET", headers={"Content-Length": "0"}
    )
    assert status == 200
    assert body == files("atif_scan.browser").joinpath("static", "browser.js").read_bytes()


@pytest.mark.parametrize(
    "framing",
    [
        "Content-Length: 0\r\nContent-Length: 0\r\n",
        "Content-Length: -1\r\n",
        "Transfer-Encoding: chunked\r\n",
    ],
)
def test_get_rejects_invalid_framing(desk: DeskServer, framing: str):
    wire = (f"GET /browser.js HTTP/1.1\r\nHost: {desk.host}\r\n{framing}\r\n").encode()
    data = raw_request(desk, wire)
    assert b" 400 " in data
    assert data.endswith(b'{"error": "request_rejected"}')


def test_call_metadata_anchor_is_allowlisted_and_index_checked(desk: DeskServer):
    status, _, body = post(
        desk,
        "/api/segment",
        {"trial": "0", "step": 7, "part": "call_info", "index": 0},
    )
    assert status == 200
    page = json.loads(body)
    assert page["metadata_only"] is True
    assert "Recorded tool call 0." in page["text"]
    assert "candidate" not in page["text"]
    status, _, _ = post(
        desk,
        "/api/segment",
        {"trial": "0", "step": 7, "part": "call_info", "index": 99},
    )
    assert status == 400
