"""Synthetic recorded metadata only; no processes are started or inspected."""

import json
from itertools import permutations

import pytest

from atif_scan.data.process_results import (
    MAX_JSON_DEPTH,
    MAX_OS_PID,
    MAX_PROCESS_ID_CHARS,
    MAX_TEXT_CHARS,
    ProcessResult,
    parse_process_result,
)

UNKNOWN = ProcessResult()
METADATA = {
    "effective_lifecycle": "persistent",
    "process_id": "process-demo",
    "os_pid": 1234,
}
EXPECTED = ProcessResult("persistent", "process-demo", 1234)
DURABLE_BANNERS = [
    "Managed background process is still running.",
    "Managed background process completed before launch acknowledgement.",
    "Managed background process failed before launch acknowledgement.",
    "Managed background process was stopped during launch.",
    "Managed background process supervisor is unavailable.",
]


@pytest.mark.parametrize("banner", DURABLE_BANNERS)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("spool", [False, True])
def test_installed_durable_launch_format(banner: str, newline: str, spool: bool):
    lines = [banner, *(f"{k}: {v}" for k, v in METADATA.items())]
    if spool:
        lines.append("output_spool_path: /synthetic/output.log")
    lines.extend(["Synthetic launch explanation.", "os_pid: 5678"])
    # All five statuses preserve the configuration claim, not proof of running.
    assert parse_process_result(newline.join(lines)) == EXPECTED


@pytest.mark.parametrize("banner", DURABLE_BANNERS)
def test_durable_banner_does_not_infer_metadata(banner: str):
    assert parse_process_result(banner) == UNKNOWN
    assert parse_process_result(banner + "\n") == UNKNOWN
    assert parse_process_result(banner + "\nos_pid: 1234") == ProcessResult(os_pid=1234)


@pytest.mark.parametrize("banner", DURABLE_BANNERS)
@pytest.mark.parametrize(
    "prefix", ["stdout:\n", "stderr:\n", "ordinary output\n", "\n", " ", "\t", "```text\n"]
)
def test_durable_banner_spoof_in_output_is_not_searched(banner: str, prefix: str):
    launch = banner + "\n" + "\n".join(f"{k}: {v}" for k, v in METADATA.items())
    assert parse_process_result(prefix + launch) == UNKNOWN
    assert parse_process_result("effective_lifecycle: foreground\n" + prefix + launch) == (
        ProcessResult("foreground")
    )
    assert parse_process_result(json.dumps({"stdout": launch})) == UNKNOWN
    assert parse_process_result(json.dumps({**METADATA, "stdout": launch})) == EXPECTED


@pytest.mark.parametrize("banner", DURABLE_BANNERS)
@pytest.mark.parametrize(
    "boundary", ["", "stdout:", "explanation", "output_spool_path: /synthetic"]
)
def test_durable_metadata_must_be_contiguous(banner: str, boundary: str):
    fields = "\n".join(f"{k}: {v}" for k, v in METADATA.items())
    assert parse_process_result(f"{banner}\n{boundary}\n{fields}") == UNKNOWN
    assert parse_process_result(
        f"{banner}\neffective_lifecycle: foreground\n{boundary}\n{fields}"
    ) == (ProcessResult("foreground"))


@pytest.mark.parametrize("banner", DURABLE_BANNERS)
def test_durable_banner_must_match_exactly(banner: str):
    fields = "\n".join(f"{k}: {v}" for k, v in METADATA.items())
    for altered in (banner.lower(), banner[:-1], banner + " ", "Status: " + banner):
        assert parse_process_result(altered + "\n" + fields) == UNKNOWN
    assert parse_process_result(banner + "\n" + banner + "\n" + fields) == UNKNOWN


@pytest.mark.parametrize("banner", DURABLE_BANNERS)
@pytest.mark.parametrize("pid", ["None", "0", "invalid", str(MAX_OS_PID + 1)])
def test_durable_invalid_pid_retains_strict_rejection(banner: str, pid: str):
    text = f"{banner}\neffective_lifecycle: persistent\nprocess_id: process-demo\nos_pid: {pid}"
    assert parse_process_result(text) == UNKNOWN


@pytest.mark.parametrize("banner", DURABLE_BANNERS)
@pytest.mark.parametrize("key", list(METADATA))
def test_durable_duplicate_fields_still_rejected(banner: str, key: str):
    fields = "\n".join(f"{k}: {v}" for k, v in METADATA.items())
    assert parse_process_result(f"{banner}\n{fields}\n{key}: {METADATA[key]}") == UNKNOWN


@pytest.mark.parametrize("lifecycle", ["persistent", "foreground", "background"])
@pytest.mark.parametrize("style", ["json", "text"])
def test_supported_lifecycles(lifecycle: str, style: str):
    data = {**METADATA, "effective_lifecycle": lifecycle}
    text = json.dumps(data) if style == "json" else "\n".join(f"{k}: {v}" for k, v in data.items())
    assert parse_process_result(text) == ProcessResult(lifecycle, "process-demo", 1234)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_text_prefix_any_field_order(newline: str):
    for fields in permutations(METADATA.items()):
        text = newline.join(f"{k}: {v}" for k, v in fields)
        assert parse_process_result(text + newline) == EXPECTED


@pytest.mark.parametrize("lifecycle", ["session", "unknown", "PERSISTENT", "", " persistent"])
def test_unknown_lifecycle_is_not_inferred(lifecycle: str):
    data = {**METADATA, "effective_lifecycle": lifecycle}
    assert parse_process_result(json.dumps(data)) == ProcessResult(None, "process-demo", 1234)
    text = "\n".join(f"{k}: {v}" for k, v in data.items())
    assert parse_process_result(text) == ProcessResult(None, "process-demo", 1234)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"effective_lifecycle": "foreground"}', ProcessResult("foreground")),
        ("effective_lifecycle: foreground", ProcessResult("foreground")),
        ('{"process_id": "process-demo"}', ProcessResult(process_id="process-demo")),
        ("process_id: process-demo", ProcessResult(process_id="process-demo")),
        ('{"os_pid": 1234}', ProcessResult(os_pid=1234)),
        ("os_pid: 1234", ProcessResult(os_pid=1234)),
    ],
)
def test_partial_metadata(text: str, expected: ProcessResult):
    assert parse_process_result(text) == expected


def test_json_whole_object_and_top_level_only():
    data = {
        **METADATA,
        "stdout": "effective_lifecycle: foreground\nos_pid: 5678",
        "stderr": {"effective_lifecycle": "background", "process_id": "process-other"},
        "messages": [{"effective_lifecycle": "foreground"}],
    }
    assert parse_process_result(" \n" + json.dumps(data) + "\t\r\n") == EXPECTED


@pytest.mark.parametrize("key", ["stdout", "stderr", "messages", "result", "content"])
def test_no_nested_json_search(key: str):
    assert parse_process_result(json.dumps({key: METADATA})) == UNKNOWN
    assert parse_process_result(json.dumps({key: json.dumps(METADATA)})) == UNKNOWN


@pytest.mark.parametrize(
    "prefix",
    ["output\n", "stdout:\n", "stderr:\n", "\n", " ", "\t", "```text\n", "> "],
)
def test_no_buried_or_indented_text_metadata(prefix: str):
    text = "\n".join(f"{k}: {v}" for k, v in METADATA.items())
    assert parse_process_result(prefix + text) == UNKNOWN


@pytest.mark.parametrize("boundary", ["stdout:", "stderr:", "ordinary output", "", "status: ok"])
def test_text_never_resumes_after_body_boundary(boundary: str):
    text = f"effective_lifecycle: foreground\n{boundary}\nprocess_id: process-demo\nos_pid: 1234"
    assert parse_process_result(text) == ProcessResult("foreground")
    # Conflicting metadata-looking body lines are not part of the metadata prefix.
    assert parse_process_result(text + "\neffective_lifecycle: persistent") == ProcessResult(
        "foreground"
    )


@pytest.mark.parametrize("key", list(METADATA))
@pytest.mark.parametrize("conflict", [False, True])
def test_duplicate_fields_reject_whole_result(key: str, conflict: bool):
    value = {"effective_lifecycle": "foreground", "process_id": "other", "os_pid": 5678}[key]
    value = value if conflict else METADATA[key]
    encoded = json.dumps(METADATA)[:-1] + f", {json.dumps(key)}: {json.dumps(value)}}}"
    assert parse_process_result(encoded) == UNKNOWN
    prefix = "\n".join(f"{k}: {v}" for k, v in METADATA.items())
    assert parse_process_result(prefix + f"\n{key}: {value}\nstdout:\nbody") == UNKNOWN


@pytest.mark.parametrize(
    "text",
    [
        '{"effective_lifecycle":"persistent","effective_\\u006cifecycle":"persistent"}',
        '{"effective_lifecycle":"persistent","extra":1,"extra":1}',
        '{"effective_lifecycle":"persistent","extra":{"x":1,"x":2}}',
    ],
)
def test_json_duplicate_keys_even_escaped_or_nested(text: str):
    assert parse_process_result(text) == UNKNOWN


@pytest.mark.parametrize("key", list(METADATA))
@pytest.mark.parametrize("value", [None, True, False, [], {}, 1.5])
def test_wrong_json_field_types_invalidate_all(key: str, value: object):
    assert parse_process_result(json.dumps({**METADATA, key: value})) == UNKNOWN


@pytest.mark.parametrize("key", ["effective_lifecycle", "process_id"])
def test_integer_is_not_a_string_field(key: str):
    assert parse_process_result(json.dumps({**METADATA, key: 1234})) == UNKNOWN


@pytest.mark.parametrize("value", ["1234", 0, -1, MAX_OS_PID + 1, 1234.0])
def test_invalid_json_pid(value: object):
    assert parse_process_result(json.dumps({**METADATA, "os_pid": value})) == UNKNOWN


@pytest.mark.parametrize(
    "value", ["", "0", "-1", "+1", "01", "1.0", "1e3", "true", "null", "１２３", "1234 ", "9" * 100]
)
def test_invalid_text_pid(value: str):
    assert parse_process_result(f"effective_lifecycle: persistent\nos_pid: {value}") == UNKNOWN


@pytest.mark.parametrize(
    "value", ["", "process demo", "../demo", "demo/path", "demo:1", "é", "demo\x00", "a" * 129]
)
def test_invalid_identifiers(value: str):
    assert parse_process_result(json.dumps({**METADATA, "process_id": value})) == UNKNOWN
    assert parse_process_result(f"effective_lifecycle: persistent\nprocess_id: {value}") == UNKNOWN


def test_identifier_and_pid_boundaries():
    for pid in (1, MAX_OS_PID):
        identifier = "p" * MAX_PROCESS_ID_CHARS
        expected = ProcessResult("persistent", identifier, pid)
        data = {**METADATA, "process_id": identifier, "os_pid": pid}
        assert parse_process_result(json.dumps(data)) == expected
        assert parse_process_result("\n".join(f"{k}: {v}" for k, v in data.items())) == expected
    assert parse_process_result("process_id: Demo_123.x-y") == ProcessResult(
        process_id="Demo_123.x-y"
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "{}",
        "[]",
        "null",
        '[{"effective_lifecycle":"persistent"}]',
        '"effective_lifecycle: persistent"',
        '{"effective_lifecycle":"persistent",}',
        '{"effective_lifecycle":"persistent"',
        '{"effective_lifecycle":"persistent"}\nextra',
        '{"effective_lifecycle":"persistent"} {}',
        '```json\n{"effective_lifecycle":"persistent"}\n```',
        '{"effective_lifecycle":"persistent","extra":NaN}',
        '{"effective_lifecycle":"persistent","extra":Infinity}',
        "effective_lifecycle:persistent",
        "effective_lifecycle: persistent\nos_pid:1234",
        "effective_lifecycle: persistent\nprocess_id",
        "effective_lifecycle: persistent\nos_pid: 1234\nos_pid",
    ],
)
def test_malformed_or_unsupported_input(text: str):
    assert parse_process_result(text) == UNKNOWN


def test_total_size_limit_rejects_instead_of_truncating():
    prefix = "effective_lifecycle: persistent\nstdout:\n"
    at_limit = prefix + "x" * (MAX_TEXT_CHARS - len(prefix))
    assert parse_process_result(at_limit) == ProcessResult("persistent")
    assert parse_process_result(at_limit + "x") == UNKNOWN
    base = json.dumps({**METADATA, "stdout": ""})
    at_limit_json = json.dumps({**METADATA, "stdout": "x" * (MAX_TEXT_CHARS - len(base))})
    assert len(at_limit_json) == MAX_TEXT_CHARS
    assert parse_process_result(at_limit_json) == EXPECTED
    assert parse_process_result(at_limit_json + " ") == UNKNOWN


def test_json_depth_limit():
    prefix = '{"effective_lifecycle":"persistent","extra":'
    for depth, expected in [
        (MAX_JSON_DEPTH - 1, ProcessResult("persistent")),
        (MAX_JSON_DEPTH, UNKNOWN),
    ]:
        assert parse_process_result(prefix + "[" * depth + "0" + "]" * depth + "}") == expected
    assert parse_process_result(prefix + "[" * 2000 + "0" + "]" * 2000 + "}") == UNKNOWN


def test_json_brackets_and_escaped_quotes_in_strings_are_not_depth():
    stdout = '[{\\"' * 100
    assert parse_process_result(json.dumps({**METADATA, "stdout": stdout})) == EXPECTED


def test_huge_json_integer_is_unknown_without_exception():
    assert parse_process_result('{"os_pid":' + "9" * 5000 + "}") == UNKNOWN


def test_private_identifiers_are_not_in_repr():
    assert "process-demo" not in repr(EXPECTED)
    assert "1234" not in repr(EXPECTED)
