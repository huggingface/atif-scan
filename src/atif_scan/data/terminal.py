"""Terminal-screen results: what a recorded terminal capture says about its own gaps.

Terminus 2 (Harbor) types a step's commands into one tmux pane and records the pane's
new output once for the whole batch ("multi-command batches share one terminal output",
harbor `terminus_2.py`; single commands are linked since harbor#2087). Its tmux history
is effectively unbounded, so a capture misses output only when it says so:

- `[... output limited to 10000 bytes; N interior bytes omitted ...]`: the middle of the
  output was cut before it was recorded.
- `Current Terminal Screen:` instead of `New Terminal Output:`: only the visible pane was
  recorded. Terminus falls back to this when it can't find the previous capture in the
  buffer (`clear`, a full-screen program) and also when nothing new was printed; the two
  read the same, so the screen may be missing output that scrolled past.
"""

from __future__ import annotations

import re
from typing import Literal

OUTPUT_OMITTED = re.compile(
    r"^\[\.\.\. output limited to \d+ bytes; \d+ interior bytes omitted \.\.\.\]$", re.M
)
SCREEN_ONLY = re.compile(r"^Current Terminal Screen:$", re.M)
NEW_OUTPUT = re.compile(r"^New Terminal Output:$", re.M)

TerminalGap = Literal["result_truncated", "terminal_screen_only"]


def terminal_gap(text: str) -> TerminalGap | None:
    """Why a terminal capture may be missing output, when it records why."""
    if OUTPUT_OMITTED.search(text):
        return "result_truncated"
    if SCREEN_ONLY.search(text) and not NEW_OUTPUT.search(text):
        return "terminal_screen_only"
    return None
