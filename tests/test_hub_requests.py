"""Synthetic, offline regressions for the small Hub request data reader."""

from __future__ import annotations

import shlex
from dataclasses import FrozenInstanceError, fields
from typing import Literal

import pytest

from atif_scan.data.hub_requests import HubTarget, RequestRead, command_targets, url_targets


def python_command(source: str) -> str:
    return "python3 -c " + shlex.quote(source)


def owned(kind: Literal["inventory", "object"]) -> RequestRead:
    return RequestRead((HubTarget(kind, "sample-owner"),))


@pytest.mark.parametrize("host", ["huggingface.co", "hf.co", "HUGGINGFACE.CO", "hf.co:443"])
@pytest.mark.parametrize("path", ["/api/whoami", "/api/whoami-v2", "/api/whoami/"])
def test_identity_exact_hosts(host: str, path: str) -> None:
    assert url_targets(f"https://{host}{path}") == RequestRead((HubTarget("identity"),))


@pytest.mark.parametrize(
    "path",
    [
        "/api/buckets/sample-owner",
        "/api/buckets/sample-owner/tree",
        "/api/buckets/sample-owner/example/tree/main",
        "/api/datasets?author=sample-owner",
        "/api/models?search=synthetic&author=sample-owner&limit=3",
        "/api/spaces?author=sample%2Downer",
        "/api/datasets/sample-owner/example/tree/main",
        "/api/models/sample-owner/example/tree/main/folder",
        "/api/spaces/sample-owner/example/tree/main",
        "/api/%62uckets/sample-owner/tree",
    ],
)
def test_owner_inventory(path: str) -> None:
    assert url_targets("https://huggingface.co" + path) == owned("inventory")


@pytest.mark.parametrize(
    "path",
    [
        "/buckets/sample-owner/example/resolve/main/synthetic.json",
        "/datasets/sample-owner/example/resolve/main/synthetic.json",
        "/spaces/sample-owner/example/resolve/main/app.py",
        "/sample-owner/example/resolve/main/config.json",
        "/sample-owner/example.repo/resolve/main/config.json",
        "/buckets/sample-owner/example/resolve/${FILE}",
    ],
)
def test_objects(path: str) -> None:
    assert url_targets("https://hf.co" + path) == owned("object")


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co.evil.example/api/whoami",
        "https://www.huggingface.co/api/whoami",
        "https://huggingface.co@other.example/api/whoami",
        "https://hf.co.other.example/api/whoami",
        "https://other.example/?next=https://huggingface.co/api/whoami",
        "https://[::1]/api/whoami",
        "https://huggingface.co./api/whoami",
    ],
)
def test_non_hub_host_counted_without_positive(url: str) -> None:
    assert url_targets(url) == RequestRead(other_requests=1)


@pytest.mark.parametrize(
    "path",
    [
        "/api/datasets?search=synthetic",
        "/api/models",
        "/api/spaces?author_hint=sample-owner",
        "/api/whoami-not-real",
        "/api/whoami/extra",
        "//api/whoami",
        "/buckets/sample-owner/example/tree/main",
        "/datasets/sample-owner/example/blob/main/config.json",
        "/api/models/sample-owner/example/commits/main",
    ],
)
def test_unrelated_hub_routes_not_owner_inventory(path: str) -> None:
    assert url_targets("https://huggingface.co" + path) == RequestRead(other_requests=1)


@pytest.mark.parametrize(
    "author",
    ["", "$OWNER", "{owner}", "%24OWNER", "a%2Fb", "a.b", "a--b", "-a", "a-", "a" * 97],
)
def test_unknown_or_invalid_author_remains_unknown(author: str) -> None:
    result = url_targets("https://huggingface.co/api/models?author=" + author)
    assert result == RequestRead((HubTarget("inventory"),), complete=False)


def test_duplicate_authors_are_not_arbitrarily_selected() -> None:
    assert url_targets("https://hf.co/api/models?author=one&author=two") == RequestRead(
        (HubTarget("inventory"),), complete=False
    )


@pytest.mark.parametrize("owner", ["$OWNER", "{owner}", "sample%2Fowner", "sample.owner", ""])
def test_dynamic_namespace_not_a_binding(owner: str) -> None:
    result = url_targets(f"https://hf.co/api/buckets/{owner}/tree")
    assert result == RequestRead((HubTarget("inventory"),), complete=False)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a URL",
        "https://$HOST/api/whoami",
        "https://{host}/api/whoami",
        "https://hf.co:bad/api/whoami",
        "https://[broken/api/whoami",
        "https://hf.co/api/buckets/%ZZ",
        "https://hf.co/api/models?author=%FF",
        "https://hf.co/api/whoami\n",
        "file:///api/whoami",
        "https://hf.co/api/models?" + "x=y&" * 65,
        "https://hf.co/" + "x" * 16_384,
    ],
)
def test_invalid_or_bounded_url_unknown(url: str) -> None:
    result = url_targets(url)
    assert not result.complete
    assert not result.targets
    assert result.other_requests == 0


@pytest.mark.parametrize(
    "command",
    [
        "curl -fsSL https://hf.co/api/buckets/sample-owner",
        "wget -qO- https://hf.co/api/buckets/sample-owner",
        "curl --url=https://hf.co/api/buckets/sample-owner",
        "curl --url https://hf.co/api/buckets/sample-owner",
        "sudo -u runner env SYNTHETIC=1 /usr/bin/curl -- https://hf.co/api/buckets/sample-owner",
        "bash -lc 'curl https://hf.co/api/buckets/sample-owner'",
    ],
)
def test_invoked_downloader_targets_not_option_values(command: str) -> None:
    assert command_targets(command) == owned("inventory")


@pytest.mark.parametrize(
    "command",
    [
        "echo 'curl https://hf.co/api/whoami'",
        "printf '%s' 'https://hf.co/api/whoami'",
        "grep 'https://hf.co/api/whoami' synthetic.txt",
        "# curl https://hf.co/api/whoami\necho done",
        "cat > synthetic.py <<'PY'\nrequests.get('https://hf.co/api/whoami')\nPY",
        "tee synthetic.sh <<'SH'\ncurl https://hf.co/api/whoami\nSH",
        "curl -H https://hf.co/api/whoami -o https://hf.co/api/whoami "
        "-e https://hf.co/api/whoami --write-out https://hf.co/api/whoami",
    ],
)
def test_data_and_comments_are_not_requests(command: str) -> None:
    assert command_targets(command) == RequestRead(attributable=False)


@pytest.mark.parametrize(
    "command",
    [
        "curl $URL",
        "wget ${URL}",
        "curl --unknown-option https://hf.co/api/whoami",
        "curl -K synthetic.cfg",
        "wget -i synthetic.txt",
        "hf api whoami",
        "huggingface-cli whoami",
        "python3 synthetic.py",
        "python3 -m synthetic",
        "python3 -",
        "python3 -c",
        "curl 'unterminated",
        "$FETCH https://hf.co/api/whoami",
        "curl -G --data-urlencode author=$OWNER https://hf.co/api/models",
    ],
)
def test_unknown_shell_targets_not_clean(command: str) -> None:
    assert not command_targets(command).complete


@pytest.mark.parametrize(
    "source",
    [
        "import requests; requests.get('https://hf.co/api/whoami')",
        "import httpx; httpx.post(url='https://hf.co/api/whoami')",
        "import urllib.request; urllib.request.urlopen('https://hf.co/api/whoami')",
        "from urllib.request import urlopen; urlopen(fullurl='https://hf.co/api/whoami')",
        "import requests as req; req.get('https://hf.co/api/whoami')",
        "from httpx import post as send; send('https://hf.co/api/whoami')",
        "if synthetic:\n requests.get('https://hf.co/api/whoami')",
    ],
)
def test_inline_python_lexical_requests(source: str) -> None:
    assert command_targets(python_command(source)) == RequestRead((HubTarget("identity"),))


def test_actual_stdin_heredoc_and_nested_shell() -> None:
    body = "import requests\nrequests.get('https://hf.co/api/buckets/sample-owner')\n"
    assert command_targets("python3 - <<'PY'\n" + body + "PY") == owned("inventory")
    text = "bash <<'SH'\npython3 - <<'PY'\n" + body + "PY\nSH"
    assert command_targets(text) == owned("inventory")
    assert command_targets("python3 <<-'PY'\n\t" + body.replace("\n", "\n\t") + "PY") == owned(
        "inventory"
    )


def test_heredoc_must_be_the_selected_python_input() -> None:
    body = "requests.get('https://hf.co/api/whoami')\n"
    text = "python3 -c 'print(1)' <<'PY'\n" + body + "PY"
    assert command_targets(text) == RequestRead(attributable=False)
    text = "python3 - <<'PY' < synthetic.py\n" + body + "PY"
    assert command_targets(text) == RequestRead(complete=False)
    text = "python3 - 2<<'PY'\n" + body + "PY"
    assert command_targets(text) == RequestRead(complete=False)


@pytest.mark.parametrize(
    "source",
    [
        "requests.get(f'https://hf.co/buckets/sample-owner/example/resolve/{filename}')",
        "httpx.post(f'https://hf.co/datasets/sample-owner/example/resolve/main/{filename}')",
        "requests.get(f'https://hf.co/sample-owner/example/resolve/main/{filename}')",
    ],
)
def test_python_fstring_known_owner_and_route(source: str) -> None:
    assert command_targets(python_command(source)) == owned("object")


@pytest.mark.parametrize(
    "source",
    [
        "owner='sample-owner'; requests.get(f'https://hf.co/api/buckets/{owner}')",
        "requests.get(f'https://{host}/api/whoami')",
        "requests.get(f'https://hf.co/buckets/{owner}/example/resolve/main/file')",
        "requests.get(url)",
        "requests.get(BASE + '/api/whoami')",
        "requests.get('https://hf.co/' + route)",
        "requests.get(make_url())",
        "requests.get(*urls)",
        "requests.get(**kwargs)",
        "requests.get('https://hf.co/api/models', params={'author': owner})",
        "import huggingface_hub; huggingface_hub.HfApi().whoami()",
        "from huggingface_hub import HfApi; HfApi().list_buckets()",
        "requests.Session().get('https://hf.co/api/whoami')",
        "getattr(requests, method)('https://hf.co/api/whoami')",
        "exec(synthetic)",
    ],
)
def test_python_dynamic_targets_or_unsupported_network_unknown(source: str) -> None:
    assert not command_targets(python_command(source)).complete


@pytest.mark.parametrize(
    "source",
    [
        "def unused():\n requests.get('https://hf.co/api/whoami')",
        "async def unused():\n httpx.post('https://hf.co/api/whoami')",
        "unused = lambda: requests.get('https://hf.co/api/whoami')",
        "class Unused:\n response=requests.get('https://hf.co/api/whoami')",
        "deferred=(requests.get('https://hf.co/api/whoami') for x in values)",
        "def unused():\n from huggingface_hub import snapshot_download as d\n d('synthetic')",
    ],
)
def test_deferred_bodies_are_not_positives_but_not_clean(source: str) -> None:
    assert command_targets(python_command(source)) == RequestRead(complete=False)


def test_python_strings_comments_and_unused_imports_are_data() -> None:
    source = (
        "import huggingface_hub\n"
        "snippet=\"requests.get('https://hf.co/api/whoami')\"\n"
        "# httpx.get('https://hf.co/api/whoami')\n"
        "def unused():\n return 'https://hf.co/api/whoami'\n"
        "print(snippet)"
    )
    assert command_targets(python_command(source)) == RequestRead(attributable=False)


def test_request_order_duplicates_and_conservative_other_count() -> None:
    text = (
        "curl https://hf.co/api/whoami https://other.example/synthetic; "
        + python_command(
            "requests.get('https://hf.co/api/whoami'); httpx.post('https://other.example')"
        )
        + "; curl $URL"
    )
    assert command_targets(text) == RequestRead(
        (HubTarget("identity"), HubTarget("identity")), complete=False, other_requests=2
    )


def test_limits_and_parsing_errors_remain_unknown() -> None:
    assert not command_targets("echo " + "x" * 131_072).complete
    assert not command_targets(python_command("requests.get(")).complete
    assert not command_targets(python_command("(" * 1000)).complete
    assert not command_targets(python_command("a=1\n" * 6000)).complete


def test_frozen_minimal_interface_and_repr_do_not_expose_owner_or_url() -> None:
    result = url_targets(
        "https://synthetic-user:synthetic-secret@hf.co/buckets/sample-owner/example/resolve/"
        "main/synthetic.json?token=synthetic-token"
    )
    assert result == owned("object")
    assert [item.name for item in fields(HubTarget)] == ["kind", "owner", "labelled"]
    assert [item.name for item in fields(RequestRead)] == [
        "targets",
        "complete",
        "other_requests",
        "attributable",
    ]
    assert repr(result) == (
        "RequestRead(targets=(HubTarget(kind='object', labelled=False),), "
        "complete=True, other_requests=0, attributable=True)"
    )
    with pytest.raises(FrozenInstanceError):
        result.__setattr__("complete", False)
    with pytest.raises(FrozenInstanceError):
        result.targets[0].__setattr__("owner", "replacement")


@pytest.mark.parametrize(
    "source",
    [
        "requests=synthetic; requests.get('https://hf.co/api/whoami')",
        "import requests as r; r=synthetic; r.get('https://hf.co/api/whoami')",
        "from requests import get; get=synthetic; get('https://hf.co/api/whoami')",
        "from requests import get\ndef get(url): pass\nget('https://hf.co/api/whoami')",
        "class requests: pass\nrequests.get('https://hf.co/api/whoami')",
        "factory().requests.get('https://hf.co/api/whoami')",
        "factory().urllib.request.urlopen('https://hf.co/api/whoami')",
        "request_fn=requests.get; request_fn('https://hf.co/api/whoami')",
        "import synthetic_sdk; synthetic_sdk.download('synthetic')",
        "requests[method]('https://hf.co/api/whoami')",
    ],
)
def test_opaque_or_rebound_callees_never_guessed_as_literal_requests(source: str) -> None:
    assert command_targets(python_command(source)) == RequestRead(
        complete=False, attributable=False
    )


def test_definition_headers_are_active_but_not_the_body() -> None:
    source = "def unused(response=requests.get('https://hf.co/api/whoami')):\n return 1"
    assert command_targets(python_command(source)) == RequestRead((HubTarget("identity"),))
    source = "(item for item in requests.get('https://hf.co/api/whoami'))"
    assert command_targets(python_command(source)) == RequestRead((HubTarget("identity"),))


def test_only_final_stdin_heredoc_is_python_source() -> None:
    text = (
        "python3 - <<'FIRST' <<'LAST'\n"
        "requests.get('https://hf.co/api/whoami')\nFIRST\n"
        "requests.get('https://hf.co/api/buckets/sample-owner')\nLAST"
    )
    assert command_targets(text) == owned("inventory")
    text = "python3 - 02<<'PY'\nrequests.get('https://hf.co/api/whoami')\nPY"
    assert command_targets(text) == RequestRead(complete=False)
    assert command_targets(
        "python3 -- -c " + shlex.quote("requests.get('https://hf.co/api/whoami')")
    ) == (RequestRead(complete=False))


def test_ast_reader_does_not_run_assertions_or_load_source_imports() -> None:
    source = (
        "import huggingface_hub\n"
        "import synthetic_nonexistent_module\n"
        "assert False, 'synthetic source must not execute'\n"
        "requests.get('https://hf.co/api/whoami')"
    )
    assert command_targets(python_command(source)) == RequestRead((HubTarget("identity"),))


def test_every_input_prefix_is_safe_to_read() -> None:
    text = (
        "python3 - <<'PY'\n"
        "requests.get(f'https://hf.co/buckets/sample-owner/example/resolve/{filename}')\nPY\n"
        "curl --url https://hf.co/api/whoami"
    )
    for end in range(len(text) + 1):
        assert isinstance(command_targets(text[:end]), RequestRead)
        assert isinstance(url_targets(text[:end]), RequestRead)
