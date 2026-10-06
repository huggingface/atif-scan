"""Synthetic CLI coverage: no network, model calls, real traces or live server."""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from atif_scan.cli import main
from atif_scan.cli.args import _check_combinations, build_parser

if TYPE_CHECKING:
    from pathlib import Path

browse_cli = importlib.import_module("atif_scan.cli.browse")


@pytest.fixture
def trace(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "steps": [{"step_id": 1, "source": "agent", "message": "PRIVATE SYNTHETIC TEXT"}],
            }
        )
    )
    return path


@pytest.fixture
def transport(monkeypatch):
    server = MagicMock()
    server.url = "http://127.0.0.1:12345/#token=synthetic"
    server.__enter__.return_value = server
    server.serve_forever.side_effect = KeyboardInterrupt
    factory = MagicMock(return_value=server)
    monkeypatch.setattr(browse_cli.server, "create_server", factory)
    return factory, server


@pytest.mark.parametrize(
    "flags",
    [
        ["--inspect"],
        ["--no-sync"],
        ["--cite"],
        ["--cite-check", "synthetic.*"],
        ["--questions", "private"],
        ["--answers", "private"],
        ["--judge-prompts", "private"],
        ["--judge", "private"],
        ["--question", "hack_hunt"],
        ["--judge-scope", "all"],
    ],
)
def test_conflicts(flags, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["synthetic.json", "--browse", *flags])
    assert exc.value.code == 2
    assert "--browse cannot be combined" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [["--browse-port", "0"], ["--feedback-dir", "private"]])
def test_associated_options_require_browse(flags, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["synthetic.json", *flags])
    assert exc.value.code == 2
    assert "require --browse" in capsys.readouterr().err


@pytest.mark.parametrize("port", ["-1", "65536", "not-a-port"])
def test_bad_port(port):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--browse", "--browse-port", port])


@pytest.mark.parametrize("port", ["0", "65535"])
def test_valid_port(port):
    parser = build_parser()
    args = parser.parse_args(["--browse", "--browse-port", port])
    _check_combinations(parser, args)
    assert args.browse_port == int(port)


def test_browse_private_output_and_default_feedback(
    trace, tmp_path, monkeypatch, transport, capsys
):
    monkeypatch.setenv("ATIF_SCAN_HOME", str(tmp_path / "home"))
    assert main([str(trace), "--browse", "--no-cache", "--packs", "none"]) == 0
    factory, server = transport
    desk = factory.call_args.args[0]
    assert desk.overview()["trials"][0]["task"] is None
    assert factory.call_args.kwargs == {"port": 0}
    server.serve_forever.assert_called_once()
    server.__exit__.assert_called_once()
    assert (tmp_path / "home" / "feedback").is_dir()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert server.url in captured.err
    assert "1 input(s) scanned" in captured.err
    assert "private" in captured.err
    assert str(trace) not in captured.err
    assert "PRIVATE SYNTHETIC TEXT" not in captured.err


def test_explicit_task_port_feedback(trace, tmp_path, transport):
    feedback = tmp_path / "custom-feedback"
    assert (
        main(
            [
                str(trace),
                "--browse",
                "--no-cache",
                "--packs",
                "none",
                "--task",
                "synthetic-task",
                "--browse-port",
                "12345",
                "--feedback-dir",
                str(feedback),
            ]
        )
        == 0
    )
    factory, _ = transport
    assert factory.call_args.kwargs == {"port": 12345}
    assert factory.call_args.args[0].overview()["trials"][0]["task"] == "synthetic-task"
    assert feedback.is_dir()


def test_streamed_manifest_rejected(tmp_path, transport, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"inputs": [{"id": "synthetic", "path": "hf://datasets/o/r/a"}]})
    )
    assert main(["--manifest", str(manifest), "--browse", "--packs", "none"]) == 2
    transport[0].assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "browser sources unavailable (details withheld)" in captured.err
    assert "hf://" not in captured.err


def test_digest_precedes_scan_and_report_passed(trace, tmp_path, monkeypatch, transport):
    events = []
    original_scan = browse_cli._scan_all

    def digests(records):
        assert records[0][0].local == trace
        events.append("digest")
        return ["synthetic-digest"]

    def scan(*args):
        assert events == ["digest"]
        events.append("scan")
        return original_scan(*args)

    constructor = MagicMock()
    monkeypatch.setattr(browse_cli.session, "source_digests", digests)
    monkeypatch.setattr(browse_cli, "_scan_all", scan)
    monkeypatch.setattr(browse_cli.session, "Session", constructor)
    assert (
        main(
            [
                str(trace),
                "--browse",
                "--no-cache",
                "--packs",
                "none",
                "--feedback-dir",
                str(tmp_path / "feedback"),
            ]
        )
        == 0
    )
    assert events == ["digest", "scan"]
    kwargs = constructor.call_args.kwargs
    assert kwargs["expected_digests"] == ["synthetic-digest"]
    assert kwargs["report"]["inputs"] == constructor.call_args.args[1]


@pytest.mark.parametrize("stage", ["digest", "session", "server"])
@pytest.mark.parametrize("error", [OSError("PRIVATE PATH"), ValueError("PRIVATE PATH")])
def test_startup_errors_withheld(trace, tmp_path, monkeypatch, transport, capsys, stage, error):
    targets = {
        "digest": (browse_cli.session, "source_digests"),
        "session": (browse_cli.session, "Session"),
        "server": (browse_cli.server, "create_server"),
    }
    target, name = targets[stage]
    monkeypatch.setattr(target, name, MagicMock(side_effect=error))
    assert (
        main(
            [
                str(trace),
                "--browse",
                "--no-cache",
                "--packs",
                "none",
                "--feedback-dir",
                str(tmp_path / "feedback"),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "details withheld" in captured.err
    assert "PRIVATE PATH" not in captured.err


@pytest.mark.parametrize(
    ("invalid", "failed", "sync_failed", "expected"),
    [(False, False, 0, 0), (False, True, 0, 1), (True, True, 0, 2), (False, True, 3, 2)],
)
def test_exit_status_after_interrupt(
    trace, tmp_path, monkeypatch, transport, capsys, invalid, failed, sync_failed, expected
):
    original_prepare = browse_cli._prepare
    original_scan = browse_cli._scan_all

    def prepare(args):
        result = original_prepare(args)
        args.sync_failures = [sync_failed]
        return result

    def scan(scanner, records, threshold):
        assert threshold == browse_cli.Severity.HIGH
        items, _, _ = original_scan(scanner, records, threshold)
        return items, invalid, failed

    monkeypatch.setattr(browse_cli, "_prepare", prepare)
    monkeypatch.setattr(browse_cli, "_scan_all", scan)
    assert (
        main(
            [
                str(trace),
                "--browse",
                "--no-cache",
                "--packs",
                "none",
                "--fail-on",
                "high",
                "--feedback-dir",
                str(tmp_path / "feedback"),
            ]
        )
        == expected
    )
    transport[1].__exit__.assert_called_once()
    if sync_failed:
        assert "3 sync file(s) unavailable" in capsys.readouterr().err


def test_invalid_preparation(transport, capsys):
    assert main(["--browse"]) == 2
    transport[0].assert_not_called()
    assert capsys.readouterr().out == ""


def test_remote_url_reuses_sync(trace, tmp_path, monkeypatch, transport, capsys):
    inputs_cli = importlib.import_module("atif_scan.cli.inputs")
    sync = MagicMock(return_value=(trace, {"downloaded": 1, "up_to_date": 0, "failed": 0}))
    monkeypatch.setattr(inputs_cli, "sync_remote", sync)
    url = "https://huggingface.co/buckets/synthetic/traces/tree/run"
    assert (
        main(
            [
                url,
                "--browse",
                "--no-cache",
                "--packs",
                "none",
                "--sync-dir",
                str(tmp_path / "sync"),
                "--feedback-dir",
                str(tmp_path / "feedback"),
            ]
        )
        == 0
    )
    sync.assert_called_once()
    assert sync.call_args.args[0] == url
    transport[1].serve_forever.assert_called_once()
    assert url not in capsys.readouterr().err


def test_scan_time_edit_never_served(trace, tmp_path, monkeypatch, transport, capsys):
    original_scan = browse_cli._scan_all

    def scan(*args):
        result = original_scan(*args)
        trace.write_text('{"steps": []}')
        return result

    monkeypatch.setattr(browse_cli, "_scan_all", scan)
    assert (
        main(
            [
                str(trace),
                "--browse",
                "--no-cache",
                "--packs",
                "none",
                "--feedback-dir",
                str(tmp_path / "feedback"),
            ]
        )
        == 2
    )
    transport[0].assert_not_called()
    assert "browser startup failed (details withheld)" in capsys.readouterr().err


def test_symlink_never_scanned(trace, tmp_path, monkeypatch, transport, capsys):
    link = tmp_path / "link.json"
    link.symlink_to(trace)
    scan = MagicMock()
    monkeypatch.setattr(browse_cli, "_scan_all", scan)
    assert (
        main(
            [
                str(link),
                "--browse",
                "--no-cache",
                "--packs",
                "none",
                "--feedback-dir",
                str(tmp_path / "feedback"),
            ]
        )
        == 2
    )
    scan.assert_not_called()
    transport[0].assert_not_called()
    assert str(link) not in capsys.readouterr().err
