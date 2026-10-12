"""Synthetic executable-discovery regressions: looking up curl does not run it."""

import pytest

from atif_scan import Engine, Status, builtin_detectors, parse_trace


def result(command, tool="terminal"):
    trace = parse_trace(
        {
            "steps": [
                {
                    "source": "agent",
                    "tool_calls": [
                        {
                            "tool_call_id": "c",
                            "function_name": tool,
                            "arguments": {"command": command},
                        }
                    ],
                }
            ]
        }
    )
    return next(
        a.result
        for a in Engine(builtin_detectors()).evaluate(trace)
        if a.spec.id == "network.http_or_git"
    )


@pytest.mark.parametrize(
    "command",
    [
        "which curl jq python3",
        "command -v curl && python3 --version",
        "command -V wget && echo done",
        "type -a curl wget",
        "whereis curl wget",
        "env LANG=C which curl jq",
        "sudo which curl jq",
    ],
)
def test_discovery_is_not_a_network_command(command):
    assert result(command).status is Status.NO_MATCH


@pytest.mark.parametrize(
    "command",
    [
        "which curl jq && curl https://example.org/data",
        "command -v curl; command curl https://example.org/data",
        "type curl; wget https://example.org/data",
        'which "$(curl https://example.org/name)"',
        "git fetch origin",
        "python -c 'import requests; requests.get(\"https://example.org/data\")'",
    ],
)
def test_network_operations_remain_visible(command):
    assert result(command).status is Status.MATCH


def test_citation_points_to_invocation_not_prior_lookup():
    command = "which curl jq && curl https://example.org/data"
    found = result(command)
    span = found.evidence[0].span
    assert span and span[0] == command.rindex("curl")


def test_unknown_harness_keeps_the_generic_evidence_fallback():
    assert result("curl https://example.org/data", "new_executor").status is Status.MATCH


def test_unparsed_shell_keeps_the_pattern_fallback():
    assert result('which curl jq; echo "unterminated').status is Status.MATCH
