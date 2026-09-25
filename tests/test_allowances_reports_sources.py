from __future__ import annotations

import io
import json
import sys

import pytest

from atif_scan import (
    Allowance,
    CheckSpec,
    Context,
    Engine,
    Ref,
    RegexDetector,
    Rule,
    Severity,
    Status,
    TraceError,
    document,
    parse_trace,
    report,
    to_text,
)
from atif_scan import sources as sources_module
from atif_scan.cli import main
from atif_scan.policy import load_rules
from atif_scan.report import render_rich
from atif_scan.sources import SourceError, resolve

SECRET = "SECRET-sk-live-0000"


def trajectory(message="", command=None):
    step = {"source": "agent", "message": message}
    if command is not None:
        step["tool_calls"] = [
            {"tool_call_id": "c1", "function_name": "bash", "arguments": {"command": command}}
        ]
    return {"schema_version": "ATIF-v1.7", "steps": [step]}


def detector(key, pattern, severity=Severity.MEDIUM, tasks=frozenset()):
    return RegexDetector(CheckSpec(key, severity, tasks=tasks), pattern)


def by_id(assessments):
    return {a.spec.id: a for a in assessments}


# --- Allowances -------------------------------------------------------------------------


def test_unconditional_allowance_excuses_match_and_score():
    checks = [
        detector("net", "download"),
        Allowance(CheckSpec("expected.net"), frozenset({"net"})),
    ]
    assessments = Engine(checks).evaluate(parse_trace(trajectory("download it")))
    net = by_id(assessments)["net"]
    assert net.result.status == Status.MATCH  # the fact is kept
    assert net.expected_by == ("expected.net",) and not net.counts
    output = report(assessments)
    assert output["score"] == 0 and output["severity"] is None
    assert output["expected_matches"] == 1
    rows = {a["id"]: a for a in output["assessments"]}
    assert rows["net"]["expected_by"] == ["expected.net"] and rows["net"]["score"] is None
    assert rows["expected.net"]["kind"] == "allowance"
    assert rows["expected.net"]["covers"] == ["net"]
    assert rows["expected.net"]["severity"] is None


def test_allowance_not_applied_on_unknown_or_false_condition():
    scoped = frozenset({"t"})
    checks = [
        detector("net", "download"),
        detector("pip_only", "pip", tasks=scoped),
        Allowance(CheckSpec("expected.net"), frozenset({"net"}), Ref("pip_only")),
    ]
    engine = Engine(checks)
    trace = parse_trace(trajectory("download it"))
    # No task: condition unknown -> not applied, and the report is incomplete.
    unknown = engine.evaluate(trace)
    assert by_id(unknown)["expected.net"].result.status == Status.UNKNOWN
    assert by_id(unknown)["net"].counts
    assert report(unknown)["incomplete"] and report(unknown)["score"] == 50
    # Condition false -> not applied.
    false = engine.evaluate(trace, Context("t"))
    assert by_id(false)["expected.net"].result.status == Status.NO_MATCH
    assert by_id(false)["net"].counts
    # Condition true -> applied, citing the condition's evidence.
    true = engine.evaluate(parse_trace(trajectory("download pip")), Context("t"))
    assert by_id(true)["expected.net"].result.status == Status.MATCH
    assert by_id(true)["expected.net"].result.evidence
    assert not by_id(true)["net"].counts


def test_allowance_task_scope_and_non_matching_covered_check():
    checks = [
        detector("net", "download"),
        Allowance(CheckSpec("expected.net", tasks=frozenset({"a"})), frozenset({"net"})),
    ]
    other = Engine(checks).evaluate(parse_trace(trajectory("download")), Context("b"))
    assert by_id(other)["expected.net"].result.status == Status.NOT_APPLICABLE
    assert by_id(other)["net"].counts
    clean = Engine(checks).evaluate(parse_trace(trajectory("nothing")), Context("a"))
    assert by_id(clean)["net"].expected_by == ()  # only matches get excused


def test_rules_see_raw_results_unless_rule_is_covered():
    checks = [
        detector("net", "download", Severity.INFO),
        Rule(CheckSpec("escalate", Severity.HIGH), Ref("net")),
        Allowance(CheckSpec("expected.net"), frozenset({"net"})),
    ]
    trace = parse_trace(trajectory("download"))
    assert report(Engine(checks).evaluate(trace))["score"] == 75
    checks[2] = Allowance(CheckSpec("expected.net"), frozenset({"net", "escalate"}))
    assert report(Engine(checks).evaluate(trace))["score"] == 0


def test_allowance_graph_validation():
    base = [detector("net", "download")]
    with pytest.raises(ValueError, match="missing_dependency"):
        Engine([*base, Allowance(CheckSpec("x"), frozenset({"missing"}))])
    with pytest.raises(ValueError, match="missing_dependency"):
        Engine([*base, Allowance(CheckSpec("x"), frozenset({"net"}), Ref("missing"))])
    with pytest.raises(ValueError, match="missing_dependency"):  # rules can't use allowances
        Engine(
            [
                *base,
                Allowance(CheckSpec("x"), frozenset({"net"})),
                Rule(CheckSpec("r"), Ref("x")),
            ]
        )
    with pytest.raises(ValueError, match="duplicate_check_id"):
        Engine([*base, Allowance(CheckSpec("net"), frozenset({"net"}))])
    with pytest.raises(ValueError):
        Allowance(CheckSpec("x"), frozenset())


def test_policy_allow_section():
    checks = load_rules(
        {
            "allow": [
                {"id": "a1", "covers": ["net"]},
                {"id": "a2", "covers": ["net"], "when": {"not": "net"}, "tasks": ["t"]},
            ]
        }
    )
    assert [type(c) for c in checks] == [Allowance, Allowance]
    assert checks[1].spec.tasks == {"t"} and checks[1].dependencies == ("net",)
    for bad in [
        {"allow": [{"id": "a", "covers": []}]},
        {"allow": [{"id": "a"}]},
        {"allow": [{"id": "a", "covers": ["net"], "severity": "high"}]},
        {"allow": {"id": "a"}},
        {},
    ]:
        with pytest.raises(ValueError):
            load_rules(bad)


def test_cli_fail_on_ignores_expected_matches(tmp_path, capsys):
    trace = tmp_path / "t.json"
    trace.write_text(json.dumps(trajectory(command="ls /tests")))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"allow": [{"id": "ok", "covers": ["access.test_path"]}]}))
    assert main([str(trace), "--fail-on", "medium", "--format", "json"]) == 1
    capsys.readouterr()
    args = [str(trace), "--fail-on", "medium", "--rules", str(policy), "--format", "json"]
    assert main(args) == 0
    item = json.loads(capsys.readouterr().out)["inputs"][0]
    assert item["expected_matches"] == 1 and item["score"] == 0


# --- Text reports -----------------------------------------------------------------------


def scanned_doc():
    checks = [
        detector("net", "download", Severity.HIGH),
        detector("other", "zzz"),
        Allowance(CheckSpec("expected.x"), frozenset({"other"})),
    ]
    item = report(Engine(checks).evaluate(parse_trace(trajectory(f"download {SECRET}"))))
    item.update(
        input_id="trial-1",
        input_status="available",
        partial=False,
        agent_steps=1,
        tool_calls=0,
        unrecognized_tool_calls=0,
    )
    return document([item], "0.0.0")


def test_plain_text_report_lists_findings_without_trace_text():
    text = to_text(scanned_doc())
    assert "trial-1" in text and "score 75 (high)" in text
    assert "net" in text and "step 0 message" in text
    assert SECRET not in text


def test_rich_report_without_trace_text():
    buffer = io.StringIO()
    render_rich(scanned_doc(), file=buffer)
    out = buffer.getvalue()
    assert "trial-1" in out and "net" in out and SECRET not in out


def test_cli_text_falls_back_to_plain_without_rich(tmp_path, capsys, monkeypatch):
    trace = tmp_path / "t.json"
    trace.write_text(json.dumps(trajectory(f"{SECRET}", command="ls /tests")))
    monkeypatch.setitem(sys.modules, "rich.console", None)
    assert main([str(trace), "--format", "text"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("atif-scan ") and "access.test_path" in out
    assert SECRET not in out and str(tmp_path) not in out


# --- Sources ----------------------------------------------------------------------------


def test_directory_expansion_is_sorted_relative_and_pattern_scoped(tmp_path, capsys):
    for name in ["b-trial/agent", "a-trial/agent", "has space/agent"]:
        (tmp_path / name).mkdir(parents=True)
        (tmp_path / name / "trajectory.json").write_text(json.dumps(trajectory("x")))
    (tmp_path / "a-trial" / "agent" / "other.json").write_text("{}")
    labels = [s.label for s in resolve([str(tmp_path)])]
    assert labels == [
        "a-trial/agent",
        "b-trial/agent",
        "input-0003",
    ]
    assert len(resolve([str(tmp_path)], pattern="*.json")) == 4
    assert main([str(tmp_path), "--format", "json"]) == 0
    out = capsys.readouterr().out
    assert str(tmp_path) not in out
    assert json.loads(out)["coverage"]["inputs"] == 3


def test_empty_directory_is_an_error_not_a_clean_result(tmp_path, capsys):
    with pytest.raises(SourceError, match="no_files_match_pattern"):
        resolve([str(tmp_path)])
    assert main([str(tmp_path)]) == 2
    assert "no_files_match_pattern" in capsys.readouterr().err


class FakeHubFS:
    """Mimics the HfFileSystem subset used: paths come back without the hf:// prefix."""

    def __init__(self, files, fail_open=False):
        self.files = files
        self.fail_open = fail_open

    def info(self, path):
        if path in self.files:
            return {"type": "file", "size": len(self.files[path])}
        if any(p.startswith(path + "/") for p in self.files):
            return {"type": "directory", "size": 0}
        raise FileNotFoundError(f"https://huggingface.co/{path}?token={SECRET}")

    def find(self, path, detail=False):
        return {
            p: {"type": "file", "size": len(data)}
            for p, data in self.files.items()
            if p.startswith(path + "/")
        }

    def open(self, path, mode):
        if self.fail_open or path not in self.files:
            raise OSError(f"401 for https://huggingface.co/{path}?token={SECRET}")
        return io.BytesIO(self.files[path])


def hub_files():
    data = json.dumps(trajectory(command="ls /tests")).encode()
    return {
        "buckets/org/runs/job1/t2/agent/trajectory.json": data,
        "buckets/org/runs/job1/t1/agent/trajectory.json": data,
        "buckets/org/runs/job1/t1/agent/log.txt": b"not a trace",
    }


def test_hf_bucket_prefix_expansion_and_loading():
    fs = FakeHubFS(hub_files())
    found = resolve(["hf://buckets/org/runs/job1/"], fs=fs)
    assert [s.label for s in found] == ["t1/agent", "t2/agent"]
    assert found[0].load().agent_steps == 1
    (single,) = resolve(["hf://buckets/org/runs/job1/t1/agent/trajectory.json"], fs=fs)
    assert single.label == "input-0001" and single.load().tool_calls == 1


def test_hf_read_errors_are_withheld_and_size_capped(monkeypatch):
    fs = FakeHubFS(hub_files(), fail_open=True)
    (source, _) = resolve(["hf://buckets/org/runs/job1"], fs=fs)
    with pytest.raises(TraceError) as error:
        source.load()
    assert SECRET not in str(error.value) and error.value.__cause__ is None
    monkeypatch.setattr(sources_module, "MAX_BYTES", 10)
    (source, _) = resolve(["hf://buckets/org/runs/job1"], fs=FakeHubFS(hub_files()))
    with pytest.raises(TraceError, match="trace_too_large"):
        source.load()


def test_cli_hf_paths_web_urls_and_manifest(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sources_module, "hf_filesystem", lambda: FakeHubFS(hub_files()))
    web = "https://huggingface.co/buckets/org/runs/tree/job1"
    for location in ["hf://buckets/org/runs/job1", web]:
        assert main([location, "--fail-on", "medium", "--format", "json"]) == 1
        out = capsys.readouterr().out
        assert "buckets/org" not in out
        assert [x["input_id"] for x in json.loads(out)["inputs"]] == ["t1/agent", "t2/agent"]
    manifest = tmp_path / "inputs.json"
    remote = "hf://buckets/org/runs/job1/t1/agent/trajectory.json"
    manifest.write_text(json.dumps({"inputs": [{"id": "one", "path": remote}]}))
    assert main(["--manifest", str(manifest), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["inputs"][0]["input_id"] == "one"


def test_cli_hf_missing_path_is_an_error_without_echoing_remote_message(monkeypatch, capsys):
    monkeypatch.setattr(sources_module, "hf_filesystem", lambda: FakeHubFS(hub_files()))
    assert main(["hf://buckets/org/runs/nope"]) == 2
    err = capsys.readouterr().err
    assert "hf_path_not_found_or_no_access" in err and SECRET not in err


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://huggingface.co/buckets/o/b/tree/run/x",
            "hf://buckets/o/b/run/x",
        ),
        ("https://huggingface.co/buckets/o/b", "hf://buckets/o/b"),
        (
            "https://huggingface.co/buckets/o/b/resolve/run/trajectory.json",
            "hf://buckets/o/b/run/trajectory.json",
        ),
        (
            "https://huggingface.co/datasets/o/d/tree/main/traces/run%201",
            "hf://datasets/o/d@main/traces/run 1",
        ),
        ("https://hf.co/datasets/o/d/blob/v2/t.json", "hf://datasets/o/d@v2/t.json"),
        ("https://huggingface.co/o/model/tree/main/logs", "hf://o/model@main/logs"),
        ("hf://buckets/o/b/x", "hf://buckets/o/b/x"),
        ("relative/dir", "relative/dir"),
    ],
)
def test_hub_web_urls_normalize_to_hf_paths(url, expected):
    assert sources_module.normalize(url) == expected


@pytest.mark.parametrize(
    "url",
    ["https://example.org/trace.json", "s3://bucket/x", "https://huggingface.co/buckets/o"],
)
def test_other_urls_are_rejected(url):
    with pytest.raises(SourceError):
        sources_module.normalize(url)


def test_core_import_does_not_load_optional_io_libraries():
    import subprocess

    code = (
        "import sys, atif_scan, atif_scan.cli; "
        "assert not {'huggingface_hub', 'rich'} & set(sys.modules), sorted(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
