"""Synthetic response syntax only: no endpoints, files or source programs are run."""

from __future__ import annotations

import json

import pytest

from atif_scan.data.hub_responses import IdentityRead, identity_profiles, response_bodies


@pytest.mark.parametrize("owner", ["synthetic-zebra-731", "Q_8", "a", "n" * 96])
@pytest.mark.parametrize("order", [False, True])
def test_root_user_identity_has_validated_private_owner(owner: str, order: bool) -> None:
    profile = {"type": "user", "name": owner} if order else {"name": owner, "type": "user"}
    text = json.dumps(profile)
    read = identity_profiles(text)
    assert read.complete
    assert len(read.identities) == 1
    identity = read.identities[0]
    assert identity.owner == owner
    assert identity.complete
    assert not identity.scoped
    assert text[slice(*identity.span)] == text
    assert "owner=" not in repr(identity)
    assert "owner=" not in repr(read)


@pytest.mark.parametrize(
    "text",
    [
        '{"type":"user","name":"synthetic-prefix"',
        '{"name":"synthetic-prefix","type":"user","tail":',
        '{"type":"user","name":"synthetic-prefix","orgs":[{"name":"clipped',
    ],
)
def test_fully_decoded_identity_prefix_is_usable_but_partial(text: str) -> None:
    read = identity_profiles(text)
    assert not read.complete
    assert len(read.identities) == 1
    assert read.identities[0].owner == "synthetic-prefix"
    assert not read.identities[0].complete
    start, end = read.identities[0].span
    assert 0 <= start < end <= len(text)


@pytest.mark.parametrize(
    "text",
    [
        '{"type":"user","name":"synthetic-clipped',
        '{"type":"us',
        '{"type":"user",',
        '{"name":"synthetic-owner", "type":',
        '{"type":"user" "name":"synthetic-owner"}',
        '{"type":"user","name":"synthetic-owner" INVALID}',
    ],
)
def test_invalid_or_unfinished_identity_fields_never_bind(text: str) -> None:
    read = identity_profiles(text)
    assert not read.complete
    assert not read.identities


@pytest.mark.parametrize(
    "text",
    [
        '{"type":"org","name":"synthetic-org"}',
        '{"name":"synthetic-model","description":"type user name synthetic-user"}',
        '{"type":"model","name":"synthetic-model","user":{"type":"user","name":"x"}}',
        '{"wrapper":{"type":"user","name":"synthetic-nested"}}',
        '{"orgs":[{"type":"user","name":"synthetic-org-member"}]}',
        '[{"type":"user","name":"synthetic-array"}]',
        'Model prose: {"type":"user","name":"synthetic-prose"}',
        '{"type":"user","name":null}',
        '{"type":"user","name":42}',
        '{"type":"user","name":{"name":"synthetic-object"}}',
    ],
)
def test_only_root_user_profiles_establish_identity(text: str) -> None:
    assert not identity_profiles(text).identities


def test_multiline_nested_identity_is_not_rediscovered_as_root() -> None:
    text = '{"wrapper":\n {"type":"user","name":"synthetic-nested"}\n}'
    assert identity_profiles(text) == IdentityRead()
    malformed = '{"wrapper":\n {"type":"user","name":"synthetic-nested"}\n'
    read = identity_profiles(malformed)
    assert not read.identities
    assert not read.complete


@pytest.mark.parametrize("owner", ["", "a--b", "-bad", "bad-", "a.b", "a/b", "$OWNER", "n" * 97])
def test_invalid_namespaces_cannot_escape_response_reader(owner: str) -> None:
    read = identity_profiles(json.dumps({"type": "user", "name": owner}))
    assert not read.identities


@pytest.mark.parametrize(
    "tail",
    ['"name":"synthetic-other"}', '"name":"synthetic-owner"}', '"name":null}'],
)
def test_duplicate_top_level_name_is_unknown_even_if_values_agree(tail: str) -> None:
    read = identity_profiles('{"type":"user","name":"synthetic-owner",' + tail)
    assert not read.identities
    assert not read.complete


def test_nested_duplicate_name_does_not_invalidate_unique_root_name() -> None:
    text = (
        '{"type":"user","name":"synthetic-owner",'
        '"org":{"name":"synthetic-org","name":"synthetic-other-org"}}'
    )
    read = identity_profiles(text)
    assert read.complete
    assert [identity.owner for identity in read.identities] == ["synthetic-owner"]


def test_escaped_root_keys_and_values_are_decoded_not_regex_matched() -> None:
    text = r'{"ty\u0070e":"user","na\u006de":"synthetic\u002downer"}'
    read = identity_profiles(text)
    assert read.complete
    assert [identity.owner for identity in read.identities] == ["synthetic-owner"]


def test_whoami_sections_scope_profiles_without_losing_locators() -> None:
    text = (
        '--- unrelated ---\n{"type":"user","name":"synthetic-unrelated"}\n'
        '=== https://hf.co/api/whoami-v2 ===\n{"type":"user","name":"synthetic-whoami"}\n'
        '--- models ---\n{"type":"user","name":"synthetic-model"}\n'
    )
    read = identity_profiles(text)
    assert read.complete
    assert [identity.scoped for identity in read.identities] == [False, True, False]
    for identity in read.identities:
        assert json.loads(text[slice(*identity.span)])["name"] == identity.owner


def test_size_object_count_field_count_and_depth_bounds_fail_closed() -> None:
    profile = '{"type":"user","name":"synthetic-bounded"}'
    oversized = profile + " " * 262_144
    assert identity_profiles(oversized) == IdentityRead(complete=False)
    assert not response_bodies("HTTP/2 200\n\n" + "x" * 262_144)
    many = identity_profiles("\n".join([profile] * 33))
    assert len(many.identities) <= 32
    assert not many.complete
    fields = ",".join(f'"field{i}":0' for i in range(129))
    assert not identity_profiles("{" + fields + ',"type":"user","name":"x"}').complete
    deep = '{"nested":' + "[" * 2000 + "0" + "]" * 2000 + "}"
    assert not identity_profiles(deep).identities


@pytest.mark.parametrize("version", ["1.0", "1.1", "2", "3"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_http_body_requires_explicit_header_boundary(version: str, newline: str) -> None:
    body = "synthetic response body"
    text = newline.join(
        [f"HTTP/{version} 206 Partial Content", "Content-Type: text/plain", "", body]
    )
    frames = response_bodies(text)
    assert len(frames) == 1
    assert frames[0].status == 206
    assert frames[0].body_chars == len(body)
    assert text[slice(*frames[0].span)] == body


@pytest.mark.parametrize(
    "text",
    [
        "synthetic body",
        "200",
        "status=200",
        "exit code: 0",
        "HTTP/2 200",
        "HTTP/2 200\nContent-Length: 5\n",
        "HTTP/2 200\n\n   \n",
        "FILE: /tmp/synthetic.json 200 9",
        "FILE: /tmp/synthetic.json 200 0\npreview",
    ],
)
def test_status_headers_and_empty_file_preview_are_not_receipts(text: str) -> None:
    assert not any(
        200 <= frame.status < 300 and frame.body_chars for frame in response_bodies(text)
    )


def test_tunnel_acknowledgement_is_not_origin_receipt() -> None:
    text = "HTTP/1.1 200 Connection established\n\nsynthetic proxy note"
    frames = response_bodies(text)
    assert len(frames) == 1
    assert frames[0].status == 200
    assert frames[0].body_chars == 0
    origin = response_bodies(text + "\nHTTP/2 200\n\nsynthetic origin body")
    assert [frame.body_chars > 0 for frame in origin] == [False, True]


@pytest.mark.parametrize("status", [200, 206, 302, 403])
def test_file_status_size_preview_syntax_is_not_itself_success(status: int) -> None:
    text = f"FILE: /tmp/synthetic-file.json {status} 7\nsynthetic-preview"
    frames = response_bodies(text)
    assert len(frames) == 1
    assert frames[0].status == status
    assert frames[0].body_chars == 7
    assert text[slice(*frames[0].span)].strip() == "synthetic-preview"


def test_file_preview_cannot_borrow_next_section_as_body() -> None:
    text = "FILE: /tmp/synthetic-empty 200 23\n--- next ---\nsynthetic unrelated body"
    assert response_bodies(text)[0].body_chars == 0


def test_multiple_http_frames_keep_independent_status_body_and_spans() -> None:
    text = "HTTP/2 302\nLocation: synthetic\n\n\nHTTP/2 403\n\nDenied\nHTTP/2 200\n\nReceived"
    frames = response_bodies(text)
    assert [frame.status for frame in frames] == [302, 403, 200]
    assert [text[slice(*frame.span)].strip() for frame in frames] == ["", "Denied", "Received"]


def test_every_prefix_is_safe_and_locators_remain_in_bounds() -> None:
    text = '=== whoami ===\n{"type":"user","name":"synthetic-prefix","orgs":[]}\nHTTP/2 200\n\nbody'
    for end in range(len(text) + 1):
        prefix = text[:end]
        spans = [identity.span for identity in identity_profiles(prefix).identities]
        spans.extend(frame.span for frame in response_bodies(prefix))
        assert all(0 <= start <= stop <= len(prefix) for start, stop in spans)


def test_response_reader_repr_does_not_disclose_dynamic_namespace() -> None:
    owner = "synthetic-private-namespace-812"
    read = identity_profiles(json.dumps({"type": "user", "name": owner}))
    assert read.identities
    assert owner not in repr(read)
    assert owner not in repr(read.identities[0])


@pytest.mark.parametrize("tail", ['"name":', '"name":"clipped'])
def test_duplicate_name_key_with_clipped_value_is_not_unique_identity(tail: str) -> None:
    read = identity_profiles('{"type":"user","name":"synthetic-first",' + tail)
    assert not read.complete
    assert not read.identities
