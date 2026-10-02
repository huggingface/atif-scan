"""Inferred associations are explicit warnings, never silent provenance repair."""

import copy
import json
import re

import pytest

from atif_scan import Context, Engine, Status, builtin_detectors, parse_trace
from atif_scan.cli import main
from atif_scan.extract import grep, outline, render, step_record
from atif_scan.questions import BY_ID, Answers, Writer, build, trace_digest

UNRESOLVED = "integrity.observation_pairing_unresolved"
WARNING = "integrity.observation_pairing_reconstructed"


def call(cid, name="bash", arguments=None):
    return {
        "tool_call_id": cid,
        "function_name": name,
        "arguments": arguments if arguments is not None else {"command": "printf synthetic"},
    }


def raw(results=None, calls=None):
    return {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {"source": "user", "step_id": 1, "message": "A synthetic task."},
            {
                "source": "agent",
                "step_id": 2,
                "message": "Working.",
                "tool_calls": calls if calls is not None else [call("a"), call("b")],
                "observation": {
                    "results": results
                    if results is not None
                    else [
                        {"content": "first synthetic result"},
                        {"content": "second synthetic result"},
                    ]
                },
            },
        ],
    }


def assessments(t):
    return {a.spec.id: a for a in Engine(builtin_detectors()).evaluate(t, Context())}


def test_matching_counts_reconstruct_and_warn_without_mutating_input():
    r = raw()
    before = copy.deepcopy(r)
    t = parse_trace(r)
    s = t.steps[1]
    assert r == before
    assert [o.source_call_id for o in s.observations] == ["a", "b"]
    assert all(o.pairing_reconstructed for o in s.observations)
    assert [o.source_call_index for o in s.observations] == [0, 1]
    assert all(o.pairing_method == "position" for o in s.observations)
    assert [j for j, _ in s.results_for(s.calls[0])] == [0]
    assert [j for j, _ in s.results_for(s.calls[1])] == [1]
    found = assessments(t)
    assert found[WARNING].result.status == Status.MATCH
    assert found[WARNING].spec.severity.name == "LOW"
    assert [e.observation for e in found[WARNING].result.evidence] == [0, 1]
    assert found["observation.credentials_exposed"].result.complete


@pytest.mark.parametrize(
    "results",
    [
        [{"content": "one"}],
        [{"content": "one"}, {"content": "two"}, {"content": "three"}],
        [{"source_call_id": "a", "content": "one"}, {"content": "two"}, {"content": "three"}],
    ],
)
def test_count_mismatch_leaves_results_unlinked_and_reported(results):
    t = parse_trace(raw(results))
    obs = t.steps[1].observations
    assert all(o.pairing_unresolved for o in obs if o.source_call_id is None)
    assert not any(o.pairing_reconstructed for o in obs)
    assert assessments(t)[UNRESOLVED].result.status == Status.MATCH


@pytest.mark.parametrize("ids", [("a", None), (None, "b"), ("", "b")])
def test_mixed_compatible_links_only_mark_inferred_observations(ids):
    t = parse_trace(raw([{"source_call_id": cid, "content": "synthetic"} for cid in ids]))
    assert [o.source_call_id for o in t.steps[1].observations] == ["a", "b"]
    assert [o.pairing_reconstructed for o in t.steps[1].observations] == [not cid for cid in ids]


@pytest.mark.parametrize("ids", [("foreign", None)])
def test_contradictory_explicit_links_are_not_reordered(ids):
    t = parse_trace(raw([{"source_call_id": cid, "content": "synthetic"} for cid in ids]))
    obs = t.steps[1].observations
    assert [o.source_call_id for o in obs] == list(ids)  # nothing reordered or re-linked
    assert [o.pairing_unresolved for o in obs] == [cid is None for cid in ids]


def test_explicit_links_override_order_and_support_multiple_output_chunks():
    t = parse_trace(
        raw([{"source_call_id": cid, "content": "synthetic"} for cid in ("b", "a", "a")])
    )
    s = t.steps[1]
    assert [i for i, _ in s.results_for(s.calls[0])] == [1, 2]
    assert [i for i, _ in s.results_for(s.calls[1])] == [0]
    assert not any(o.pairing_reconstructed for o in s.observations)
    assert assessments(t)[WARNING].result.status == Status.NO_MATCH


def test_no_output_remains_missing_evidence_not_a_reconstruction_error():
    t = parse_trace(raw([]))
    assert not t.steps[1].observations
    assert not assessments(t)["observation.credentials_exposed"].result.complete
    assert assessments(t)[WARNING].result.status == Status.NO_MATCH


def test_single_call_multichunk_fallback_unchanged():
    t = parse_trace(raw(calls=[call("a")]))
    s = t.steps[1]
    assert len(s.results_for(s.calls[0])) == 2
    assert not any(o.pairing_reconstructed for o in s.observations)


def test_empty_call_id_cannot_be_reconstructed():
    # TB4 Codex: hosted web-search calls without an ID. The trace is still scanned.
    t = parse_trace(raw(calls=[call(""), call("b")]))
    assert all(o.pairing_unresolved and o.source_call_id is None for o in t.steps[1].observations)
    assert assessments(t)[UNRESOLVED].result.status == Status.MATCH


def test_copied_duplicate_ids_cannot_be_reconstructed():
    r = raw(calls=[call("a"), call("a")])
    r["steps"][1]["is_copied_context"] = True
    t = parse_trace(r)
    assert all(o.pairing_unresolved for o in t.steps[1].observations)


def test_code_mode_synthetic_calls_do_not_change_recorded_count():
    calls = [
        call("a", "exec", {"input": 'await tools.shell({command: "printf synthetic"})'}),
        call("b"),
    ]
    t = parse_trace(raw(calls=calls))
    s = t.steps[1]
    assert len(s.calls) == 3
    assert [o.source_call_id for o in s.observations] == ["a", "b"]
    assert [i for i, _ in s.results_for(s.calls[2])] == [0]


def test_inspector_and_search_expose_warning():
    t = parse_trace(raw())
    assert "PAIRING-RECONSTRUCTED" in outline(t)[1]
    record = step_record(t, 1, {"results"}, 200, frozenset())
    assert all(o["pairing_reconstructed"] for o in record["results"])
    assert "WARNING: positional pairing reconstructed" in render(record)
    assert all(
        "PAIRING-RECONSTRUCTED" in h["part"]
        for h in grep(t, re.compile("synthetic result"), frozenset())
    )


def test_question_prompt_carries_real_reconstruction_warning(tmp_path):
    t = parse_trace(
        raw(calls=[call("a", arguments={"command": "curl https://example.org"}), call("b")])
    )
    a = tuple(assessments(t).values())
    built = build(BY_ID["network_outcome"], t, a, Context(), "synthetic")
    assert built is not None
    prompt, meta = built
    assert WARNING in prompt and "reconstructed" in prompt
    assert meta["question"] == "network_outcome"
    writer = Writer(tmp_path, ["network_outcome"])
    writer.add(t, a, Context(), "synthetic")
    writer.close()
    (tmp_path / "synthetic/network_outcome.answer.json").write_text(
        json.dumps(
            {
                "answer": "unclear",
                "confidence": "low",
                "steps": [2],
                "reason": "Synthetic evidence.",
            }
        )
    )
    answers = Answers.load(tmp_path)
    assert answers.annotate("synthetic", t)[0]["status"] == "answered"
    explicit = raw(calls=[call("a", arguments={"command": "curl https://example.org"}), call("b")])
    for result, cid in zip(explicit["steps"][1]["observation"]["results"], ("a", "b"), strict=True):
        result["source_call_id"] = cid
    assert trace_digest(t) != trace_digest(parse_trace(explicit))
    assert answers.annotate("synthetic", parse_trace(explicit))[0]["status"] == "stale"


def test_cli_mismatch_is_scanned_and_no_raw_text_emitted(tmp_path, capsys):
    p = tmp_path / "synthetic.json"
    p.write_text(json.dumps(raw([{"content": "DO_NOT_ECHO_SYNTHETIC"}])))
    main([str(p), "--format", "json"])
    output = capsys.readouterr()
    assert "DO_NOT_ECHO_SYNTHETIC" not in output.out + output.err
    doc = json.loads(output.out)
    assert doc["coverage"]["available"] == 1
    item = doc["inputs"][0]
    assert item["incomplete"]  # checks that need the pairing can't clear this step
    assert next(a for a in item["assessments"] if a["id"] == UNRESOLVED)["status"] == "match"


def test_citations_keep_reconstruction_provenance():
    from atif_scan.cite import cite
    from atif_scan.model import Channel, Locator
    from atif_scan.report import citation_lines

    t = parse_trace(raw())
    for at in (
        Locator(1, Channel.COMMAND, call=0, field=0),
        Locator(1, Channel.OBSERVATION, observation=0),
        Locator(1, Channel.METADATA, observation=0),
    ):
        citation = cite(t, at)
        assert citation["pairing_reconstructed"]
        assert any(label == "warning" for label, _ in citation_lines(citation))
    explicit = raw(
        [
            {"source_call_id": "a", "content": "first synthetic result"},
            {"source_call_id": "b", "content": "second synthetic result"},
        ]
    )
    citation = cite(parse_trace(explicit), Locator(1, Channel.COMMAND, call=0, field=0))
    assert "pairing_reconstructed" not in citation
    assert not any(label == "warning" for label, _ in citation_lines(citation))


def test_default_brief_prints_reconstruction_as_warning(tmp_path, capsys):
    p = tmp_path / "synthetic.json"
    p.write_text(json.dumps(raw()))
    assert main([str(p), "--brief", "--format", "text"]) == 0
    text = capsys.readouterr().out
    evidence = text[text.index("EVIDENCE") :]
    assert "⚠ 1 trial" in evidence
    assert "pairing links inferred" in evidence or "tool result links inferred" in evidence
    # Real plurals: one trial "isn't" in the total.
    assert "the 1 trial without usage isn't in the total" in text


@pytest.mark.parametrize("ids, target", [(("b", None), 0), ((None, "a"), 1)])
def test_valid_explicit_remainder_intentionally_overrides_contradictory_position(ids, target):
    # Formerly unresolved: one explicit one-to-one link proves the sole remainder,
    # but does not prove result order. Retain both the original IDs and the warning.
    s = parse_trace(raw([{"source_call_id": cid, "content": "synthetic"} for cid in ids])).steps[1]
    missing = ids.index(None)
    assert [o.source_call_id for o in s.observations] == list(ids)
    assert s.observations[missing].source_call_index == target
    assert s.observations[missing].pairing_method == "unique_remainder"
    assert s.observations[missing].pairing_reconstructed
    assert [j for j, _ in s.results_for(s.calls[target])] == [missing]


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("web_search_call", {"action_type": "search", "query": "synthetic"}),
        ("bash", {"command": "printf synthetic"}),
    ],
)
def test_missing_provider_id_unique_remainder_preserves_identifiers_and_input(name, arguments):
    r = raw(
        [{"content": "synthetic remainder"}, {"source_call_id": "b", "content": "explicit"}],
        [call("", name, arguments), call("b")],
    )
    before = copy.deepcopy(r)
    t = parse_trace(r)
    s = t.steps[1]
    assert r == before
    assert [c.id for c in s.calls] == ["", "b"]
    assert [o.source_call_id for o in s.observations] == [None, "b"]
    assert [o.source_call_index for o in s.observations] == [0, None]
    assert [j for j, _ in s.results_for(s.calls[0])] == [0]
    assert [j for j, _ in s.results_for(s.calls[1])] == [1]
    assert s.observations[0].pairing_method == "unique_remainder"
    assert assessments(t)[WARNING].result.status == Status.MATCH


def test_multipart_explicit_links_leave_one_unique_recorded_remainder():
    t = parse_trace(
        raw(
            [
                {"source_call_id": "b", "content": "first chunk"},
                {"content": "remainder"},
                {"source_call_id": "b", "content": "second chunk"},
            ],
            [call(""), call("b")],
        )
    )
    s = t.steps[1]
    assert [j for j, _ in s.results_for(s.calls[0])] == [1]
    assert [j for j, _ in s.results_for(s.calls[1])] == [0, 2]
    assert [o.pairing_method for o in s.observations] == [None, "unique_remainder", None]


@pytest.mark.parametrize(
    "calls, results",
    [
        ([call(""), call("")], [{"content": "one"}]),
        ([call(""), call("b")], [{"content": "one"}, {"content": "two"}]),
        ([call(""), call("b")], [{"source_call_id": "foreign"}, {"content": "one"}]),
        ([call("a"), call("a"), call("")], [{"source_call_id": "a"}, {"content": "one"}]),
    ],
)
def test_ambiguous_or_invalid_explicit_links_never_create_index_links(calls, results):
    r = raw(results, calls)
    r["steps"][1]["is_copied_context"] = True  # duplicate IDs are allowed in copied steps
    s = parse_trace(r).steps[1]
    assert not any(o.source_call_index is not None for o in s.observations)
    assert all(o.pairing_unresolved for o in s.observations if o.source_call_id is None)


def test_derived_call_uses_indexed_recorded_parent_with_nonempty_id():
    s = parse_trace(
        raw(
            [{"source_call_id": "b", "content": "explicit"}, {"content": "parent result"}],
            [
                call("a", "exec", {"input": 'await tools.shell({command: "printf synthetic"})'}),
                call("b"),
            ],
        )
    ).steps[1]
    assert len(s.calls) == 3
    assert s.observations[1].source_call_id is None
    assert s.observations[1].source_call_index == 0
    assert [j for j, _ in s.results_for(s.calls[2])] == [1]


def test_empty_parent_id_does_not_collide_with_unrelated_empty_id_calls():
    from atif_scan.model import Content, Observation, Step, ToolCall

    calls = (
        ToolCall(0, "inert", (), "", "exec", None),
        ToolCall(1, "shell", (), "", "bash", None),
        ToolCall(2, "shell", (), "derived", "bash", None, result_id=""),
    )
    s = Step(
        0,
        "agent",
        False,
        Content(),
        Content(),
        calls,
        (Observation(None, Content("synthetic"), source_call_index=1),),
    )
    assert not s.results_for(calls[0])
    assert [j for j, _ in s.results_for(calls[1])] == [0]
    assert not s.results_for(calls[2])
