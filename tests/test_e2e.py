"""End to end: real runs, scanned through the CLI, against their gold snapshots.

Opt-in and machine-local: real run identifiers never live in the repository. Point
ATIF_SCAN_E2E at a private JSON file listing the cases, each a gold snapshot name plus the
source and arguments it was taken with:

    {"cases": [
      {"name": "local-run", "source": "/path/to/run", "args": ["--task-from", "trial-dir"]},
      {"name": "hub-job", "source": "harbor://jobs/<uuid>", "args": []},
      {"name": "hf-run", "source": "hf://buckets/<org>/<bucket>/<run>", "args": []}
    ]}

Take a snapshot once (`uv run python tools/gold.py snapshot NAME SOURCE ARGS…`), then:

    ATIF_SCAN_E2E=~/.cache/atif-scan/e2e.json uv run pytest tests/test_e2e.py -v

A failure prints the per-check and DQ differences (tools/gold.py diff). An intended change
is accepted by re-taking the snapshot. Covers resolving (local, Harbor Hub, Hugging Face),
syncing, scanning and the run brief, which unit tests only reach through fakes.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc

REPO = Path(__file__).resolve().parents[1]
CONFIG = os.environ.get("ATIF_SCAN_E2E")
pytestmark = pytest.mark.skipif(not CONFIG, reason="set ATIF_SCAN_E2E to a cases file")


def _cases() -> list[Doc]:
    if not CONFIG:
        return []
    cases: list[Doc] = json.loads(Path(CONFIG).expanduser().read_text())["cases"]
    return cases


def _gold():
    os.environ.setdefault("ATIF_SCAN_GOLD_DIR", str(REPO / "reports" / "gold"))
    spec = importlib.util.spec_from_file_location("gold", REPO / "tools" / "gold.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_run_matches_its_gold_snapshot(case, capsys):
    gold = _gold()
    if not (gold.GOLD / f"{case['name']}.json").exists():
        pytest.skip(f"no snapshot: tools/gold.py snapshot {case['name']} …")
    changed = gold.main(["diff", case["name"], case["source"], *case.get("args", [])])
    out = capsys.readouterr().out
    assert changed == 0, out
