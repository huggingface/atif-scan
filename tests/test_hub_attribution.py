"""Synthetic stdout provenance and literal binding regressions; no fixture is executed."""

from __future__ import annotations

import shlex

import pytest

from atif_scan.data.hub_requests import HubTarget, RequestRead, command_targets, url_targets

WHOAMI = "https://hf.co/api/whoami-v2"
OTHER = "https://synthetic.example/search"


def python_read(source: str) -> RequestRead:
    return command_targets("python3 -c " + shlex.quote(source))


@pytest.mark.parametrize(
    "producer",
    [
        "cat synthetic.json",
        "cat <<'EOF'\n{}\nEOF",
        "read synthetic",
        "opaque_producer",
        'echo \'{"name":"synthetic"}\'',
        "printf '%s\\n' '{}'",
        "head -c1500",
        "node synthetic.js",
    ],
)
def test_foreign_stdout_preserves_candidate_not_attribution(producer: str) -> None:
    read = command_targets(f"curl {WHOAMI}; {producer}")
    assert read.targets == (HubTarget("identity"),)
    assert not read.attributable


@pytest.mark.parametrize(
    "options",
    [
        "-o synthetic.json",
        "-osynthetic.json",
        "-sSosynthetic.json",
        "--output synthetic.json",
        "--output=synthetic.json",
        "-o -",
        "-O",
        "--remote-name",
        "-w '%{http_code}'",
        "--write-out=%{http_code}",
    ],
)
def test_curl_output_suppression_never_authorizes_response(options: str) -> None:
    assert command_targets(f"curl {options} {WHOAMI}") == RequestRead(
        (HubTarget("identity"),), attributable=False
    )


@pytest.mark.parametrize(
    "command",
    [
        f"curl {WHOAMI} > synthetic.json",
        f"echo $(curl {WHOAMI})",
        f"curl {WHOAMI}; wget -qOsynthetic.json {OTHER}",
        f"curl {WHOAMI}; wget --output-document=synthetic.json {OTHER}",
    ],
)
def test_capture_redirect_and_wget_files_are_not_stdout(command: str) -> None:
    read = command_targets(command)
    assert read.targets == (HubTarget("identity"),)
    assert not read.attributable


@pytest.mark.parametrize(
    "command",
    [
        f"curl {WHOAMI}",
        f"wget -qO- {WHOAMI}",
        f"wget --output-document=- {WHOAMI}",
        f"echo; curl {WHOAMI} | head -c1500; echo",
        f"printf '\\n'; curl {WHOAMI} | head -c 1500; printf '\\n'",
    ],
)
def test_simple_identity_and_bounded_filters_remain_attributable(command: str) -> None:
    assert command_targets(command) == RequestRead((HubTarget("identity"),))


@pytest.mark.parametrize(
    "banner",
    [
        "echo '=== whoami ==='",
        "printf '%s\\n' '=== whoami ==='",
        "printf '\\n=== https://hf.co/api/whoami-v2 ===\\n'",
        "echo ===whoami===",
    ],
)
@pytest.mark.parametrize("getter", ["curl", "wget -qO-"])
def test_identity_banner_is_immediate_and_bounded(banner: str, getter: str) -> None:
    text = (
        f"TOKEN=synthetic; {banner}; {getter} {WHOAMI} | head -c1500; "
        f"echo; echo ===search===; curl {OTHER}"
    )
    assert command_targets(text) == RequestRead(
        (HubTarget("identity", labelled=True),), other_requests=1
    )
    assert command_targets(f"{banner}; {getter} {WHOAMI} | head -c1500; echo") == RequestRead(
        (HubTarget("identity", labelled=True),)
    )


@pytest.mark.parametrize(
    "command",
    [
        f"echo ===whoami===; curl {OTHER}; echo ===search===; curl {WHOAMI}",
        f"echo ===whoami===; curl {OTHER}; curl {WHOAMI}",
        f"echo ===whoami===; curl {WHOAMI}; curl {OTHER}",
        f"echo ===whoami===; curl {WHOAMI} {OTHER}",
        f"echo ===whoami===; curl {WHOAMI}; echo; curl {OTHER}",
        f"echo ===whoami===; curl {WHOAMI}; opaque",
        f"echo ===whoami===; true; curl {WHOAMI}",
        f"echo $BANNER; curl {WHOAMI}",
        f"printf '%s' '===whoami==='; curl {WHOAMI}",
        f"echo ===whoami===; bash -c 'curl {WHOAMI}'",
        f"bash -c 'echo ===whoami==='; curl {WHOAMI}",
        f"curl {WHOAMI}; echo ===whoami===",
    ],
)
def test_marker_presence_is_not_identity_request_labelling(command: str) -> None:
    read = command_targets(command)
    assert read.targets == (HubTarget("identity"),)


def test_only_identity_targets_can_be_labelled() -> None:
    read = command_targets(
        "echo ===whoami===; curl https://hf.co/api/buckets/sample-owner; echo ===done==="
    )
    assert read.targets == (HubTarget("inventory", "sample-owner"),)


@pytest.mark.parametrize("segment", [".", "..", "%2e", "%2E%2e", ".%2e", "%2e."])
@pytest.mark.parametrize(
    "path",
    [
        "/api/buckets/sample-owner/{segment}/tree",
        "/buckets/sample-owner/example/resolve/main/{segment}/synthetic.json",
        "/datasets/sample-owner/example/resolve/{segment}/synthetic.json",
        "/api/{segment}/whoami",
        "/sample-owner/example/resolve/main/{segment}",
    ],
)
def test_dot_segments_never_infer_an_owner(segment: str, path: str) -> None:
    read = url_targets("https://hf.co" + path.format(segment=segment))
    assert read == RequestRead(complete=False)
    assert read.attributable  # Native input provenance is independent of route validity.


@pytest.mark.parametrize("interpreter", ["node", "nodejs", "deno", "bun", "ruby", "perl", "php"])
def test_recognized_unsupported_interpreters_are_unknown(interpreter: str) -> None:
    read = command_targets(f"{interpreter} synthetic-script")
    assert not read.complete
    assert not read.targets


@pytest.mark.parametrize("interpreter", ["bash", "sh", "zsh", "dash", "ksh", "ash"])
def test_named_shell_scripts_are_never_opened(interpreter: str) -> None:
    read = command_targets(f"{interpreter} synthetic-script.sh")
    assert not read.complete
    assert not read.targets


@pytest.mark.parametrize(
    "binding",
    [
        "match value:\n case requests: pass",
        "match value:\n case [*requests]: pass",
        "match value:\n case {'key': _, **requests}: pass",
        "try:\n pass\nexcept Exception as requests:\n pass",
        "del requests",
        "del requests.get",
        "requests.get = synthetic",
        "requests['get'] = synthetic",
        "requests.adapters['get'] = synthetic",
        "del requests.adapters['get']",
    ],
)
def test_python_network_capture_delete_and_mutation_are_invalidations(binding: str) -> None:
    read = python_read(f"import requests\n{binding}\nrequests.get('{WHOAMI}')")
    assert not read.complete
    assert not read.targets


@pytest.mark.parametrize(
    "binding",
    [
        "match value:\n case send: pass",
        "match value:\n case [*send]: pass",
        "match value:\n case {**send}: pass",
        "try:\n pass\nexcept Exception as send:\n pass",
        "del send",
        "send.attribute = synthetic",
        "send[0] = synthetic",
    ],
)
def test_imported_network_alias_invalidations(binding: str) -> None:
    read = python_read(f"from requests import get as send\n{binding}\nsend('{WHOAMI}')")
    assert not read.complete
    assert not read.targets


@pytest.mark.parametrize("assignment", ["r =", "r: object ="])
@pytest.mark.parametrize("separator", ["\n", "; "])
def test_python_file_receipt_is_coupled_to_one_http_result(assignment: str, separator: str) -> None:
    source = separator.join(
        [
            "import requests",
            "f='synthetic.json'",
            f"{assignment} requests.get('{WHOAMI}')",
            "print('===whoami===')",
            "print('FILE:', f, r.status_code, len(r.content))",
            "print(r.text[:1500])",
            "print()",
            "print('===done===')",
        ]
    )
    read = python_read(source)
    assert read.complete
    assert read.attributable
    assert read.targets == (HubTarget("identity"),)  # No shell-banner guessing in Python.


@pytest.mark.parametrize(
    "extra",
    [
        "print('{}')",
        "print(body)",
        "print(make_body())",
        "print(open('synthetic.json').read())",
        "body = open('synthetic.json').read()",
        "body = Path('synthetic.json').read_text()",
        "json.dump({}, sys.stdout)",
        "r = synthetic",
        "r.text = '{}'",
        "del r",
        "del r.text",
        "r['text'] = '{}'",
        "match value:\n case r: pass",
        "match value:\n case [*r]: pass",
        "match value:\n case {**r}: pass",
        "try:\n pass\nexcept Exception as r:\n pass",
        "other = r\nother.text = '{}'",
        f"r = requests.get('{OTHER}')",
    ],
)
def test_python_response_stdout_ambiguity_retains_request(extra: str) -> None:
    read = python_read(
        f"import requests\nr = requests.get('{WHOAMI}')\n{extra}\n"
        "print('FILE:', f, r.status_code, len(r.content))\nprint(r.text[:1500])"
    )
    assert read.targets[0] == HubTarget("identity")
    assert not read.attributable


@pytest.mark.parametrize(
    "printing",
    [
        "print(r.text)",
        "print(r.text[:n])",
        "print(r.text[:0])",
        "print(r.text[:1500], other)",
        "print('FILE:', f, other.status_code, len(r.content))",
        "print('FILE:', f, r.status_code, 26)",
        "print(r.text[:1500], file=target)",
    ],
)
def test_python_only_allowlisted_result_prints_are_attributable(printing: str) -> None:
    read = python_read(f"r = requests.get('{WHOAMI}')\n{printing}")
    assert read.targets == (HubTarget("identity"),)
    assert not read.attributable


def test_python_print_before_assignment_does_not_couple_result() -> None:
    read = python_read(f"print(r.text[:1500]); r = requests.get('{WHOAMI}')")
    assert read.targets == (HubTarget("identity"),)
    assert not read.attributable


def test_completeness_and_stdout_attribution_are_independent() -> None:
    read = python_read("requests.get(url)")
    assert read == RequestRead(complete=False)
    read = python_read(f"r = requests.get('{WHOAMI}'); print('{{}}')")
    assert read == RequestRead((HubTarget("identity"),), attributable=False)


@pytest.mark.parametrize(
    "command",
    [
        f"echo ===whoami=== >&2; curl {WHOAMI}",
        f"echo ===whoami=== > synthetic.txt; curl {WHOAMI}",
        f"echo ===whoami=== & curl {WHOAMI}",
        f"echo '  ===whoami==='; curl {WHOAMI}",
        f"echo ===whoami===; curl {WHOAMI}; echo '  ===search==='; curl {OTHER}",
        f"bash -c 'echo ===whoami===; curl {WHOAMI}'; curl {OTHER}",
    ],
)
def test_banners_must_reach_stdout_and_close_in_the_same_source(command: str) -> None:
    assert command_targets(command).targets == (HubTarget("identity"),)


@pytest.mark.parametrize(
    "command",
    [
        f"bash -c 'echo ===whoami===; curl {WHOAMI}'",
        f"bash -c 'echo ===whoami===; curl {WHOAMI}; echo ===done==='; curl {OTHER}",
    ],
)
def test_nested_source_can_bound_its_own_banner(command: str) -> None:
    read = command_targets(command)
    assert read.targets == (HubTarget("identity", labelled=True),)
    assert read.attributable


@pytest.mark.parametrize("segment", ["%2e%2e%2felsewhere", "folder%2f..", "%2f.%2f"])
def test_decoded_slashes_cannot_hide_dot_segments(segment: str) -> None:
    read = url_targets(f"https://hf.co/buckets/sample-owner/example/resolve/main/{segment}")
    assert read == RequestRead(complete=False)


def test_default_wget_download_is_not_stdout() -> None:
    assert command_targets(f"wget {WHOAMI}") == RequestRead(
        (HubTarget("identity"),), attributable=False
    )


def test_syntactically_incomplete_stdout_is_not_attributable() -> None:
    read = command_targets(f"curl {WHOAMI}; printf 'unterminated")
    assert read.targets == (HubTarget("identity"),)
    assert not read.complete
    assert not read.attributable


def test_response_class_mutation_is_not_safe_coupling() -> None:
    read = python_read(
        f"r = requests.get('{WHOAMI}')\nrequests.Response.text = '{{}}'\nprint(r.text[:1500])"
    )
    assert read.targets == (HubTarget("identity"),)
    assert not read.complete
    assert not read.attributable
