"""The static marketing page stays script-free and its check count stays accurate."""

from __future__ import annotations

import re
from pathlib import Path

from atif_scan.access import access_rules
from atif_scan.detectors.builtin import builtin_detectors

PAGE = Path(__file__).resolve().parents[1] / "docs" / "marketing" / "index.html"
ALLOWED_HOSTS = {"github.com", "fonts.googleapis.com", "fonts.gstatic.com"}


def test_marketing_check_count_matches_catalogue():
    html = PAGE.read_text()
    count = len(builtin_detectors()) + len(access_rules())
    assert f"{count} built-in checks" in html


def test_marketing_page_has_no_scripts_and_known_hosts():
    html = PAGE.read_text()
    assert "<script" not in html
    assert "script-src 'none'" in html and "connect-src 'none'" in html
    hosts = set(re.findall(r"https://([A-Za-z0-9.-]+)", html))
    assert hosts <= ALLOWED_HOSTS
