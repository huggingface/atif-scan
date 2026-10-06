"""Loopback-only, bearer-authenticated transport for the private evidence desk."""

from __future__ import annotations

import json
import secrets
import socket
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from typing import TYPE_CHECKING

from ..data.jsonval import as_str, count, is_object
from .feedback import MAX_NOTE, VERDICTS
from .session import MAX_QUERY

if TYPE_CHECKING:
    from ..data.jsonval import Doc, JsonObject
    from .session import Session

MAX_BODY = 16 * 1024
MAX_URL = 2 * 1024
MAX_ID = 128
MAX_PORT = 65535
SOCKET_TIMEOUT = 3
CLOSE_TIMEOUT = 0.1
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "form-action 'none'; base-uri 'none'; frame-ancestors 'none'"
)
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/browser.js": ("browser.js", "text/javascript; charset=utf-8"),
    "/browser.css": ("browser.css", "text/css; charset=utf-8"),
}
FIELDS = {
    "/api/overview": (set(), set()),
    "/api/trial": ({"trial"}, set()),
    "/api/segment": ({"trial", "step", "part"}, {"index", "field", "offset"}),
    "/api/focus": ({"trial", "finding", "location"}, set()),
    "/api/search": ({"trial", "query"}, set()),
    "/api/feedback": ({"trial", "finding", "verdict", "note"}, set()),
}


def _string(body: JsonObject, key: str, limit: int, *, empty: bool = False) -> str:
    value = as_str(body.get(key))
    if value is None or len(value) > limit or (not empty and not value):
        raise ValueError("invalid_request")
    return value


def _integer(body: JsonObject, key: str, default: int | None = None) -> int:
    value = count(body.get(key, default))
    if value is None:
        raise ValueError("invalid_request")
    return value


def _segment(session: Session, body: JsonObject) -> Doc:
    part = _string(body, "part", MAX_ID)
    if part not in {"message", "reasoning", "call", "call_info", "result"}:
        raise ValueError("invalid_request")
    return session.segment(
        _string(body, "trial", MAX_ID),
        _integer(body, "step"),
        part,
        _integer(body, "index", 0),
        _integer(body, "field", 0),
        _integer(body, "offset", 0),
    )


def _feedback(session: Session, body: JsonObject) -> Doc:
    verdict = _string(body, "verdict", MAX_ID)
    if verdict not in VERDICTS:
        raise ValueError("invalid_request")
    return session.save_feedback(
        _string(body, "trial", MAX_ID),
        _string(body, "finding", MAX_ID),
        verdict,
        _string(body, "note", MAX_NOTE, empty=True),
    )


def _dispatch(session: Session, path: str, body: JsonObject) -> Doc:
    required, optional = FIELDS[path]
    if not required <= body.keys() or body.keys() - required - optional:
        raise ValueError("invalid_request")
    if path == "/api/overview":
        return session.overview()
    if path == "/api/segment":
        return _segment(session, body)
    if path == "/api/feedback":
        return _feedback(session, body)
    return _inspection(session, path, body)


def _inspection(session: Session, path: str, body: JsonObject) -> Doc:
    if path == "/api/focus":
        return session.focus(
            _string(body, "trial", MAX_ID),
            _string(body, "finding", MAX_ID),
            _integer(body, "location"),
        )
    trial = _string(body, "trial", MAX_ID)
    return (
        session.trial(trial)
        if path == "/api/trial"
        else session.search(trial, _string(body, "query", MAX_QUERY))
    )


class DeskServer(HTTPServer):
    """One serialized session, with a random token unique to this bound server."""

    def __init__(self, session: Session, port: int = 0) -> None:
        if count(port) is None or port > MAX_PORT:
            raise ValueError("invalid_port")
        self.session = session
        self.token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), Handler)
        self.host = f"127.0.0.1:{self.server_port}"
        self.origin = f"http://{self.host}"
        self.url = f"{self.origin}/#token={self.token}"
        self.socket.settimeout(SOCKET_TIMEOUT)

    # Python 3.11 stdlib has no override decorator; keep this transport dependency-free.
    def get_request(self) -> tuple[socket.socket, object]:  # ty: ignore[missing-override-decorator]
        connection, address = super().get_request()
        connection.settimeout(SOCKET_TIMEOUT)
        return connection, address

    # Python 3.11 stdlib has no override decorator; keep this transport dependency-free.
    def handle_error(self, request: object, client_address: object) -> None:  # ty: ignore[missing-override-decorator]
        # Last-resort stdlib boundary: never print tracebacks containing private data.
        # Expected input/storage errors are translated in the handler, not here.
        pass


class Handler(BaseHTTPRequestHandler):
    """Fixed routes only; neither client paths nor trace URLs become resources."""

    server: DeskServer

    # Python 3.11 stdlib has no override decorator; keep this transport dependency-free.
    def log_message(self, format: str, *args: object) -> None:  # ty: ignore[missing-override-decorator]
        """No request, token, path or exception logging."""

    # Python 3.11 stdlib has no override decorator; keep this transport dependency-free.
    def finish(self) -> None:  # ty: ignore[missing-override-decorator]
        try:
            super().finish()
        finally:
            self._close_write_and_drain()

    def _close_write_and_drain(self) -> None:
        # Deliver the complete response before discarding unread request bytes. Closing
        # with unread TCP data can reset the connection and truncate even static assets.
        # Neither an unlimited body nor a slow sender may monopolize this serial server.
        try:
            self.connection.shutdown(socket.SHUT_WR)
            deadline = time.monotonic() + CLOSE_TIMEOUT
            remaining = MAX_BODY + 1
            while remaining > 0:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                self.connection.settimeout(timeout)
                data = self.connection.recv(min(remaining, 8192))
                if not data:
                    break
                remaining -= len(data)
        except OSError:
            pass  # A disconnected peer or expired drain needs no further response.

    def _reply(self, status: int, content: bytes, content_type: str) -> None:
        self.close_connection = True
        self.send_response(status)
        for key, value in {
            "Content-Type": content_type,
            "Content-Length": str(len(content)),
            "Connection": "close",
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": CSP,
        }.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(content)

    def _json(self, status: int, document: Doc) -> None:
        self._reply(status, json.dumps(document).encode("utf-8"), "application/json")

    # Python 3.11 stdlib has no override decorator; keep this transport dependency-free.
    def send_error(  # ty: ignore[missing-override-decorator]
        self, code: int, message: str | None = None, explain: str | None = None
    ) -> None:
        # BaseHTTPRequestHandler's default errors echo the untrusted request.
        self._json(code, {"error": "request_rejected"})

    # Python 3.11 stdlib has no override decorator; keep this transport dependency-free.
    def parse_request(self) -> bool:  # ty: ignore[missing-override-decorator]
        if not super().parse_request():
            return False
        if len(self.path.encode("utf-8")) > MAX_URL:
            self.send_error(414)
            return False
        if not self._trusted_headers():
            self.send_error(403)
            return False
        return True

    def _trusted_headers(self) -> bool:
        if any(
            len(self.headers.get_all(name, [])) > 1
            for name in ("Host", "Origin", "Authorization", "Sec-Fetch-Site")
        ):
            return False
        origin = self.headers.get("Origin")
        return (
            self.headers.get("Host") == self.server.host
            and (origin is None or origin == self.server.origin)
            and self.headers.get("Sec-Fetch-Site", "").lower() != "cross-site"
        )

    def do_GET(self) -> None:
        # GET has no body semantics here. Reject ambiguous framing too; never serve
        # a large asset while silently leaving a declared request body unread.
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get_all("Transfer-Encoding") or lengths not in ([], ["0"]):
            self.send_error(400)
            return
        asset = ASSETS.get(self.path)
        if asset is None or self.requestline.split()[1] != self.path:
            self.send_error(405 if self.path in FIELDS else 404)
            return
        name, content_type = asset
        try:
            content = files("atif_scan.browser").joinpath("static", name).read_bytes()
        except OSError:
            self._json(500, {"error": "asset_unavailable"})
            return
        self._reply(200, content, content_type)

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "").encode("utf-8")
        expected = f"Bearer {self.server.token}".encode("ascii")
        return self.headers.get("Origin") == self.server.origin and secrets.compare_digest(
            supplied, expected
        )

    def _body(self) -> JsonObject:
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
            raise ValueError("invalid_request")
        length = int(lengths[0])
        if not 0 < length <= MAX_BODY or self.headers.get_all("Transfer-Encoding"):
            raise ValueError("invalid_request")
        if self.headers.get_all("Content-Type") != ["application/json"]:
            raise ValueError("invalid_request")
        data = self.rfile.read(length)
        if len(data) != length:
            raise ValueError("invalid_request")
        value: object = json.loads(data.decode("utf-8"))
        if not is_object(value):
            raise ValueError("invalid_request")
        return value

    def do_POST(self) -> None:
        if not self._authorized():
            self.send_error(403)
            return
        if self.path not in FIELDS or self.requestline.split()[1] != self.path:
            self.send_error(404)
            return
        try:
            result = _dispatch(self.server.session, self.path, self._body())
        except (ValueError, UnicodeError, RecursionError) as exc:
            code = (
                str(exc)
                if str(exc) in {"trace_source_changed", "trace_unavailable"}
                else ("invalid_request")
            )
            self._json(400, {"error": code})
        except OSError:
            self._json(503, {"error": "session_unavailable"})
        else:
            self._json(200, result)


def create_server(session: Session, port: int = 0) -> DeskServer:
    """Bind IPv4 loopback; callers own serving and closing (or a context manager)."""
    return DeskServer(session, port)
