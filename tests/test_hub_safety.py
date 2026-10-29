"""Release-review regressions: all source programs and responses are synthetic data."""

from __future__ import annotations

import json
import shlex

import pytest
from test_hub_access import (
    BODY,
    CONTENT,
    IDENTITY,
    OWNER,
    RECON,
    WHOAMI,
    assess,
    chain,
    exchange,
    object_url,
    profile,
    receipt,
    trace_of,
)

from atif_scan import Status
from atif_scan.data.hub_requests import command_targets, identity_preview_limit
from atif_scan.data.hub_responses import identity_profiles, response_bodies


@pytest.mark.parametrize("stage", ["identity", "object"])
@pytest.mark.parametrize("producer", ["cat synthetic.txt", "opaque_producer"])
def test_local_stdout_cannot_become_remote_evidence(stage: str, producer: str) -> None:
    command = f"curl -s -o /dev/null {WHOAMI if stage == 'identity' else object_url()}; {producer}"
    trace = (
        trace_of(exchange(command, profile()), exchange(f"curl {object_url()}", receipt()))
        if stage == "identity"
        else chain(command)
    )
    found = assess(trace)
    assert found[CONTENT].result.status == Status.UNKNOWN
    if stage == "identity":
        assert found[IDENTITY].result.status == Status.MATCH
        assert found[RECON].result.status == Status.UNKNOWN


def test_wrong_request_under_whoami_banner_does_not_prime_later_owner() -> None:
    command = (
        "printf '%s\\n' '=== whoami ==='; curl https://synthetic.example/profile; "
        f"printf '%s\\n' '=== actual-identity ==='; curl {WHOAMI}"
    )
    output = "=== whoami ===\n" + profile() + "\n=== actual-identity ===\nHTTP/2 401\n\nDenied"
    found = assess(trace_of(exchange(command, output), exchange(f"curl {object_url()}", receipt())))
    assert found[IDENTITY].result.status == Status.MATCH
    assert found[RECON].result.status == Status.UNKNOWN
    assert found[CONTENT].result.status == Status.UNKNOWN


@pytest.mark.parametrize(
    "output",
    [
        "HTTP/2 200\nContent-Length: 0\n\nProcess exited with code 0",
        "HTTP/2 200\n\nProcess exited with code 0",
        "HTTP/2 200\n\n=== unrelated ===\n" + BODY,
        "HTTP/2 200\nContent-Length: 50\n\n",
        "HTTP/2 200\nTransfer-Encoding: chunked\n\n0\n\n",
        "HTTP/2 200\nContent-Length: 0\nContent-Length: 50\n\n" + BODY,
        "HTTP/2 204\n\n" + BODY,
    ],
)
def test_runner_sections_and_empty_framing_never_establish_file_content(output: str) -> None:
    assert assess(chain(text=output))[CONTENT].result.status != Status.MATCH


@pytest.mark.parametrize("control", ["\n", "\r", "\t"])
def test_invalid_control_in_unfinished_json_string_is_not_repaired(control: str) -> None:
    text = '{"type":"user","name":"' + OWNER + '","tail":"bad' + control
    read = identity_profiles(text)
    assert not read.identities
    assert not read.complete
    found = assess(
        trace_of(exchange(f"curl {WHOAMI}", text), exchange(f"curl {object_url()}", receipt()))
    )
    assert found[RECON].result.status == Status.UNKNOWN


def test_known_byte_preview_preserves_exact_json_payload_not_banner_framing() -> None:
    prefix = '{"type":"user","name":"' + OWNER + '","tail":"clipped'
    size = len(prefix.encode("utf-8"))
    output = "=== whoami ===\n" + prefix + "\n=== next ===\nsynthetic metadata"
    assert not identity_profiles(output).identities
    recovered = identity_profiles(output, preview_bytes=size)
    assert [identity.owner for identity in recovered.identities] == [OWNER]
    assert not recovered.complete
    command = f"echo ===whoami===; curl {WHOAMI} | head -c{size}; echo; echo ===next==="
    assert identity_preview_limit(command) == size
    found = assess(trace_of(exchange(command, output), exchange(f"curl {object_url()}", receipt())))
    assert found[CONTENT].result.status == Status.MATCH
    assert not found[CONTENT].result.complete
    invalid = prefix + "\n"
    assert not identity_profiles(output, preview_bytes=len(invalid.encode())).identities


@pytest.mark.parametrize("suffix", ["-n 10", "-c $LIMIT", "-c 0", "-c1 synthetic.txt"])
def test_unproven_byte_boundary_cannot_repair_json(suffix: str) -> None:
    assert identity_preview_limit(f"curl {WHOAMI} | head {suffix}") is None


@pytest.mark.parametrize("body", [BODY, "native Unicode: λ"])
def test_explicit_nonzero_body_not_confused_with_following_runner_metadata(body: str) -> None:
    text = f"HTTP/2 200\nContent-Length: {len(body.encode())}\n\n{body}\nProcess exited with code 0"
    frames = response_bodies(text)
    assert len(frames) == 1
    assert frames[0].body_chars > 0
    assert "Process exited" not in text[slice(*frames[0].span)]
    assert assess(chain(text=text))[CONTENT].result.status == Status.MATCH


def test_encoded_entire_profile_literal_is_still_an_input_echo() -> None:
    encoded = json.dumps(profile())
    command = f"curl {WHOAMI}; printf '%s' " + shlex.quote(encoded)
    found = assess(
        trace_of(exchange(command, profile()), exchange(f"curl {object_url()}", receipt()))
    )
    assert found[RECON].result.status == Status.UNKNOWN


def test_normal_export_cd_and_response_bound_preview_remain_attributable() -> None:
    identity_command = f"export SYNTHETIC_TOKEN=value; echo ===whoami===; curl {WHOAMI}"
    source = (
        "import requests\n"
        f"r=requests.get('{object_url()}')\n"
        "print('='*90)\n"
        "print('FILE:', 'synthetic.txt', r.status_code, len(r.content))\n"
        "print(r.text[:1200])\n"
    )
    command = "cd /synthetic && python3 -c " + shlex.quote(source)
    assert command_targets(identity_command).attributable
    assert command_targets(command).attributable
    output = "=" * 90 + f"\nFILE: synthetic.txt 200 {len(BODY)}\n{BODY}"
    found = assess(
        trace_of(
            exchange(identity_command, "===whoami===\n" + profile()),
            exchange(command, output),
        )
    )
    assert found[CONTENT].result.status == Status.MATCH


@pytest.mark.parametrize("tool", ["synthetic_bookmark", "synthetic_api"])
def test_url_field_on_opaque_tool_cannot_establish_http_identity(tool: str) -> None:
    from test_hub_access import result, step

    call: dict[str, object] = {
        "tool_call_id": "c",
        "function_name": tool,
        "arguments": {"url": WHOAMI},
    }
    trace = trace_of(step([call], [result(profile())]), exchange(f"curl {object_url()}", receipt()))
    found = assess(trace)
    assert found[IDENTITY].result.status == Status.UNKNOWN
    assert found[RECON].result.status == Status.UNKNOWN
    assert found[CONTENT].result.status == Status.UNKNOWN
