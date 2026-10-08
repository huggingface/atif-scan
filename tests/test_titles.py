"""Check titles: every bundled check has one, and the report carries them in a catalog."""

from __future__ import annotations

import json

import pytest

from atif_scan import CheckSpec, Severity, builtin_detectors
from atif_scan.access import access_rules
from atif_scan.cli import main
from atif_scan.detectors.priming import builtin_allowances
from atif_scan.output.document import document
from atif_scan.policy import load_rules

MAX_STYLE = 48


def bundled_specs(tmp_path, monkeypatch) -> dict[str, CheckSpec]:
    monkeypatch.setenv("ATIF_SCAN_REFERENCE", str(tmp_path))
    from atif_scan.packs import reference, tb4, tb21

    specs: dict[str, CheckSpec] = {}
    for check in [
        *builtin_detectors(),
        *access_rules(),
        *builtin_allowances(),
        *tb21.checks(),
        *tb4.checks(),
        *reference.checks(),
    ]:
        # A spec may be shared (a detector reused inside another); it must be the same.
        assert specs.setdefault(check.spec.id, check.spec) == check.spec
    return specs


def test_every_bundled_check_has_a_short_unique_title(tmp_path, monkeypatch):
    specs = bundled_specs(tmp_path, monkeypatch)
    assert len(specs) > 90
    for spec in specs.values():
        assert spec.title, spec.id
        assert len(spec.title) <= MAX_STYLE, spec.id
        assert not spec.title.endswith("."), spec.id
        assert spec.title[0].isupper(), spec.id
        assert spec.id not in spec.title, spec.id
    titles = [spec.title for spec in specs.values()]
    assert len(set(titles)) == len(titles)


def test_check_spec_title_is_optional_and_positional_specs_still_work():
    spec = CheckSpec("demo.check", Severity.LOW, "2")
    assert spec.title == ""
    assert CheckSpec("demo.check", title="Demo thing seen").title == "Demo thing seen"


@pytest.mark.parametrize(
    "title",
    [
        "x" * 61,
        "Two\nlines",
        "Tab\there",
        "See https://example.invalid",
        "file://thing",
        b"bytes",
        None,
    ],
)
def test_check_spec_title_is_validated(title):
    with pytest.raises(ValueError, match="invalid_check_title"):
        CheckSpec("demo.check", title=title)


def test_rule_files_accept_a_validated_title():
    rule, allow = load_rules(
        {
            "rules": [
                {"id": "r.x", "severity": "low", "when": "awareness.benchmark", "title": "X seen"}
            ],
            "allow": [{"id": "a.x", "covers": ["r.x"], "title": "X is expected"}],
        }
    )
    assert (rule.spec.title, allow.spec.title) == ("X seen", "X is expected")
    for title in (3, "http://x"):
        with pytest.raises(ValueError, match="invalid_check_title"):
            load_rules({"rules": [{"id": "r", "severity": "low", "when": "a", "title": title}]})


def test_document_catalog_uses_assessment_severity_names():
    catalog = [
        ("rule", CheckSpec("b.rule", Severity.HIGH, title="B happened")),
        ("detector", CheckSpec("a.detector", Severity.LOW)),
        ("allowance", CheckSpec("c.allow", title="C expected")),
        ("context", CheckSpec("d.context", title="D known")),
    ]
    doc = document([], "0.0.0", catalog)
    assert list(doc["checks"]) == ["a.detector", "b.rule", "c.allow", "d.context"]
    assert doc["checks"] == {
        "a.detector": {"severity": "low", "title": None},
        "b.rule": {"severity": "high", "title": "B happened"},
        "c.allow": {"severity": None, "title": "C expected"},
        "d.context": {"severity": None, "title": "D known"},
    }
    assert document([], "0.0.0")["checks"] == {}


def test_scan_report_lists_every_check_it_ran_even_from_cache(tmp_path, capsys):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"schema_version": "ATIF-v1.7", "steps": [{"source": "agent"}]}))
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps({"rules": [{"id": "demo.r", "severity": "medium", "when": "access.test_path"}]})
    )
    args = [str(path), "--format", "json", "--cache", str(tmp_path / "c"), "--rules", str(rules)]
    docs = []
    for _ in range(2):  # the second scan reads the cached trace results
        main(args)
        docs.append(json.loads(capsys.readouterr().out))
    assert docs[0]["checks"] == docs[1]["checks"]
    doc = docs[1]
    assessments = {a["id"]: a for a in doc["inputs"][0]["assessments"]}
    assert list(doc["checks"]) == sorted(assessments)
    for check_id, entry in doc["checks"].items():
        assert entry["severity"] == assessments[check_id]["severity"]
    assert doc["checks"]["tamper.reward_write"] == {
        "severity": "high",
        "title": "Reward file written",
    }
    assert doc["checks"]["context.rewarded"]["severity"] is None
    assert doc["checks"]["demo.r"] == {"severity": "medium", "title": None}
