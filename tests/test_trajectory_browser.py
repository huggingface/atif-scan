"""Synthetic-only prototype boundaries and dependency-free JavaScript navigation tests."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ASSETS = Path(__file__).resolve().parents[1] / "examples" / "trajectory-browser"


def test_browser_evidence_navigation():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional; only the static browser prototype uses JavaScript.")
    result = subprocess.run(
        [node, str(ASSETS / "evidence.test.cjs")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "PASS" in result.stdout


def test_browser_has_no_network_or_html_interpretation():
    html = (ASSETS / "index.html").read_text()
    script = (ASSETS / "browser.js").read_text()
    assert "SYNTHETIC PROTOTYPE" in html
    assert "connect-src 'none'" in html
    assert "img-src 'none'" in html
    assert "script-src 'self'" in html
    assert "unsafe-inline" not in html
    assert 'content="no-referrer"' in html
    assert "http://" not in html and "https://" not in html
    assert "innerHTML" not in script and "insertAdjacentHTML" not in script
    assert "fetch(" not in script and "eval(" not in script
    assert 'type="file"' not in html


def test_browser_distinguishes_gaps_from_clearance():
    script = (ASSETS / "browser.js").read_text()
    fixture = (ASSETS / "fixture.js").read_text()
    assert "Stale answer — not current evidence clearance." in script
    assert "receipt remains unknown" in script
    assert "inferred: true" in fixture
    assert 'status: "not_reviewed"' in fixture
    assert 'status: "absent"' in fixture and 'status: "media"' in fixture
    assert "production inspection must mask whole fields" in (ASSETS / "index.html").read_text()
