"""Running `fast-agent go` for atif-scan's model calls (`--image-model`, `hunt`).

Every call is `--isolated` (fast-agent 0.10.43+): config, secrets and model aliases come
from the fast-agent home, but nothing is written there (no session history, file logs or
telemetry) and no skills, cards, plugins, hooks or shell/filesystem/subagent tools load.
Prompts here carry trace text and images, so neither may end up in the home's sessions/
or reach a plugin.

An older fast-agent rejects the flag: the call is retried once without it, with
`--no-shell --no-subagents` (always passed) and session history off through the
environment, and a warning that the home's plugins and hooks may still load.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

ISOLATED = "--isolated"
# Click/Typer: "No such option: --isolated" or, for a command group, "No such command".
REJECTED = re.compile(r"No such (?:option|command)\W+--isolated\b")
_UNSUPPORTED: set[tuple[str, ...]] = set()
_LOCK = threading.Lock()


def no_session_history() -> dict[str, str]:
    """This environment, with fast-agent's session history off: the fallback for a
    fast-agent without --isolated (which would otherwise save the prompt and any attached
    image in its home's sessions/ folder)."""
    return {**os.environ, "SESSION_HISTORY": "false"}


def run_go(
    base: Sequence[str], go_args: Sequence[str], timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """`<base> go --isolated <go_args>`, or without --isolated on a fast-agent that
    doesn't have it. Raises like subprocess.run (OSError, TimeoutExpired)."""
    key = tuple(base)
    if key not in _UNSUPPORTED:
        run = _run([*base, "go", ISOLATED, *go_args], timeout)
        if run.returncode == 0 or not REJECTED.search(_plain(run.stderr + run.stdout)):
            return run
        _fallback(key)
    return _run([*base, "go", *go_args], timeout)


def _run(command: list[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed arguments, no shell
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=no_session_history(),
    )


def _plain(text: str) -> str:
    """Without terminal colour codes (Typer colours its error box)."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _fallback(key: tuple[str, ...]) -> None:
    with _LOCK:
        if key in _UNSUPPORTED:
            return
        _UNSUPPORTED.add(key)
    print(
        "atif-scan: this fast-agent has no --isolated (0.10.43+); running with --no-shell "
        "--no-subagents and session history off, but its home's skills, plugins and hooks "
        "may load. Upgrade fast-agent to isolate these calls.",
        file=sys.stderr,
    )
