"""Private frontend regressions: optional Node, synthetic in-memory transport only."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_private_browser_frontend():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional; scanner and browser server require only Python.")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("browser_static.test.cjs"))],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "fail 0" in result.stdout
