"""Where atif-scan keeps local data: one root with documented subfolders.

    $ATIF_SCAN_HOME   (default $XDG_CACHE_HOME/atif-scan, else ~/.cache/atif-scan)
      hf/        mirrors of hf:// inputs, laid out as hf/<path>/         (sync)
      harbor/    mirrors of harbor:// jobs, harbor/<job id>/              (sync)
      results/   the per-trace result cache                              (--cache)
      labels/    the label store and its run splits                      (atif-scan labels)
      feedback/  finding-level browser annotations and private notes     (--browse)
      gold/      gold snapshots of scan results                          (tools/gold.py, e2e)
      bundles/   question and hunt bundles                               (--questions, hunt)

Everything under it names real runs or holds trace text: it is private (folders 0700,
files 0600 where atif-scan writes them), never committed. Mirrors/results are expendable;
labels, feedback, gold and bundles may not be recoverable after deletion. Two older
variables still win for their part: ATIF_SCAN_SYNC_DIR (hf/, harbor/, results/) and
ATIF_SCAN_GOLD_DIR (gold/).
"""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    """$ATIF_SCAN_HOME, else $XDG_CACHE_HOME/atif-scan, else ~/.cache/atif-scan."""
    if os.environ.get("ATIF_SCAN_HOME"):
        return Path(os.environ["ATIF_SCAN_HOME"]).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "atif-scan"


def sync_root() -> Path:
    """Parent of the hf/ and harbor/ mirrors and of results/ ($ATIF_SCAN_SYNC_DIR wins)."""
    if os.environ.get("ATIF_SCAN_SYNC_DIR"):
        return Path(os.environ["ATIF_SCAN_SYNC_DIR"]).expanduser()
    return home()


def labels_dir() -> Path:
    return home() / "labels"


def gold_dir() -> Path:
    """Gold snapshots ($ATIF_SCAN_GOLD_DIR wins)."""
    if os.environ.get("ATIF_SCAN_GOLD_DIR"):
        return Path(os.environ["ATIF_SCAN_GOLD_DIR"]).expanduser()
    return home() / "gold"


def bundles_dir() -> Path:
    return home() / "bundles"
