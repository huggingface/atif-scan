"""atif-scan's fast-agent calls (`--image-model`, `hunt`) run `--isolated`, and fall back
with a warning on a fast-agent that predates it (0.10.42 and earlier). Every call passes
atif-scan's own system prompt: `--isolated` leaves fast-agent's default one, which reads
the working directory's AGENTS.md (regression: judges got this repo's AGENTS.md)."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from atif_scan.cli import fast_agent
from atif_scan.cli.fast_agent import INSTRUCTION, run_go

if TYPE_CHECKING:
    from pathlib import Path

# 0.10.42's actual rejection (a Typer command group), inside its coloured error box.
OLD = """#!{python}
import json, os, sys
args = sys.argv[1:]
if "--instruction" in args:
    at = args.index("--instruction") + 1
    args[at] = "instruction:" + open(args[at]).read()
with open(os.environ["STUB_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
if "--isolated" in sys.argv:
    sys.stderr.write("Usage: fast-agent go [OPTIONS] COMMAND [ARGS]...\\n"
                     "\\x1b[31m| No such command '--isolated'. |\\x1b[0m\\n")
    sys.exit(2)
print(os.environ.get("SESSION_HISTORY"))
"""
# A current fast-agent failing for another reason (exit 2: a bad option value).
FAILS = """#!{python}
import json, os, sys
args = sys.argv[1:]
if "--instruction" in args:
    at = args.index("--instruction") + 1
    args[at] = "instruction:" + open(args[at]).read()
with open(os.environ["STUB_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
sys.stderr.write("Error: Invalid value for '--timeout'\\n")
sys.exit(2)
"""


@pytest.fixture(autouse=True)
def fresh_support(monkeypatch):
    monkeypatch.setattr(fast_agent, "_UNSUPPORTED", set())


def stub(tmp_path: Path, monkeypatch, body: str) -> tuple[list[str], Path]:
    script = tmp_path / "fast-agent"
    script.write_text(body.format(python=sys.executable))
    script.chmod(0o755)
    log = tmp_path / "calls.log"
    monkeypatch.setenv("STUB_LOG", str(log))
    return [str(script)], log


OWN = ["--instruction", "instruction:" + INSTRUCTION]


def calls(log: Path) -> list[list[str]]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_an_older_fast_agent_falls_back_once_with_a_warning(tmp_path, monkeypatch, capsys):
    base, log = stub(tmp_path, monkeypatch, OLD)
    run = run_go(base, ["--model", "m", "--no-shell"])
    assert run.returncode == 0
    assert run.stdout.strip() == "false"  # session history still off without --isolated
    assert calls(log) == [
        ["go", "--isolated", *OWN, "--model", "m", "--no-shell"],
        ["go", *OWN, "--model", "m", "--no-shell"],
    ]
    assert "no --isolated" in capsys.readouterr().err
    # Remembered: later calls skip the rejected attempt and don't warn again.
    run_go(base, ["--model", "m"])
    assert calls(log)[2:] == [["go", *OWN, "--model", "m"]]
    assert capsys.readouterr().err == ""


def test_other_failures_are_not_retried_without_isolation(tmp_path, monkeypatch, capsys):
    base, log = stub(tmp_path, monkeypatch, FAILS)
    run = run_go(base, ["--model", "m"])
    assert run.returncode == 2
    assert calls(log) == [["go", "--isolated", *OWN, "--model", "m"]]
    assert capsys.readouterr().err == ""


def test_every_call_has_atif_scans_own_system_prompt(tmp_path, monkeypatch):
    # fast-agent's default system prompt has {{file_silent:AGENTS.md}} and {{env}}: run
    # from a project checkout, the judge got that project's development instructions.
    (tmp_path / "AGENTS.md").write_text("# project instructions for coding agents")
    monkeypatch.chdir(tmp_path)
    base, log = stub(tmp_path, monkeypatch, FAILS)
    run_go(base, ["--model", "m"])
    [args] = calls(log)
    assert args[2:4] == OWN
    assert "{{" not in INSTRUCTION  # nothing for fast-agent to fill from the workspace
    assert "project instructions" not in json.dumps(args)
    assert not list(tmp_path.glob("atif-instruction-*"))
