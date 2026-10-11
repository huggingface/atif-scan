"""Synthetic-only tests for the offline awareness ledger."""

from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc

TOOL = Path(__file__).resolve().parents[1] / "tools" / "awareness_report.py"
sys.path.insert(0, str(TOOL.parent))
spec = importlib.util.spec_from_file_location("awareness_report_tool", TOOL)
assert spec and spec.loader
report = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = report
spec.loader.exec_module(report)

BOARD = "11111111-1111-4111-8111-111111111111"
ROW = "22222222-2222-4222-8222-222222222222"
TRIAL = "33333333-3333-4333-8333-333333333333"
OTHER = "44444444-4444-4444-8444-444444444444"
PAGE = (
    "https://hub.harborframework.com/datasets/synthetic-org/demo-pack/7"
    "?tab=leaderboard&leaderboard=test-board"
)


def assessment(check: str, status: str = "no_match", complete: bool = True) -> Doc:
    return {
        "id": check,
        "status": status,
        "complete": complete,
        "expected_by": [],
        "evidence": [],
    }


def trial(identity: str = TRIAL) -> Doc:
    return {
        "hub_trial_id": identity,
        "input_id": "/private/never-export",
        "task": "synthetic-task",
        "input_status": "available",
        "partial": False,
        "compacted": False,
        "reward": 0,
        "coverage_gaps": {"behavioural": [], "telemetry": ["integrity.synthetic"]},
        "recording_gaps": {"trials": 0, "pairing_unresolved": 0, "web_results_not_recorded": 0},
        "assessments": [assessment(c) for c in report.CHECKS],
    }


def entry(items: list[Doc], declared: int = 1):
    return report.Entry(
        ROW,
        "Synthetic entry",
        declared,
        report.trials_from({"inputs": items}),
        "test-version",
        "hash",
        "loaded",
        len(items),
    )


def snapshot() -> Doc:
    return {
        "leaderboard": {"id": BOARD, "package": "synthetic-org/demo-pack", "name": "test-board"},
        "rows": [
            {
                "id": ROW,
                "n_trials": 2,
                "metadata": {
                    "agent_display": "Synthetic Agent",
                    "model_display": "Synthetic Model",
                },
            }
        ],
    }


def test_explained_overlap_and_all_locators() -> None:
    item = trial()
    for a in item["assessments"][:2]:
        a["status"] = "match"
        a["expected_by"] = ["synthetic.allowance"]
        a["evidence"] = [
            {"step": 2, "step_id": 8, "channel": "reasoning", "field": 0, "span": [1, 3]},
            {"step": 4, "channel": "message", "text": "SECRET-TRACE"},
        ]
    e = entry([item])
    assert report.metric(e, (0, 1)) == {
        "hits": 1,
        "negative": 0,
        "unknown": 0,
        "explained": 1,
        "declared": 1,
    }
    page = report.render((e,), BOARD, "snapshot-hash")
    assert "synthetic.allowance" in page
    assert "step 2 · reasoning · step_id 8 · field 0" in page
    assert "step 4 · message" in page
    assert "step is the zero-based trajectory position; step_id is the recorded" in page
    assert page.count("2 evidence locators") == 2
    assert "SECRET-TRACE" not in page
    assert f"trial-{ROW}-{TRIAL}-awareness.benchmark" in page


@pytest.mark.parametrize("mode", ["missing", "unknown", "incomplete", "malformed", "duplicate"])
def test_unknown_never_negative(mode: str) -> None:
    item = trial()
    a = item["assessments"][0]
    if mode == "missing":
        item["assessments"].pop(0)
    elif mode == "duplicate":
        item["assessments"].append(a.copy())
    elif mode == "incomplete":
        a["complete"] = False
    else:
        a["status"] = mode
    assert report.metric(entry([item]), (0, 1))["unknown"] == 1


def test_partial_positive_and_separate_recall() -> None:
    item = trial()
    item["partial"] = True
    item["compacted"] = True
    item["assessments"][0].update(status="match", complete=False)
    item["assessments"][2].update(status="match")
    e = entry([item])
    assert report.metric(e, (0, 1))["hits"] == 1
    assert report.metric(e, (1,))["unknown"] == 1
    assert report.metric(e, (2,))["hits"] == 1
    page = report.render((e,), BOARD, "hash")
    assert "partial: yes" in page and "compacted: yes" in page
    assert "Complete: False" in page
    assert "does not establish training exposure" in page


def test_telemetry_not_behavioural_and_targeted_gaps() -> None:
    item = trial()
    item["incomplete"] = True
    assert report.metric(entry([item]), (0, 1))["negative"] == 1
    item["coverage_gaps"]["behavioural"] = [report.CHECKS[0]]
    assert report.metric(entry([item]), (0, 1))["unknown"] == 1
    assert report.metric(entry([item]), (1,))["negative"] == 1
    item["recording_gaps"]["trials"] = 1
    assert "recording gaps: present" in report.render((entry([item]),), BOARD, "hash")


def test_duplicate_and_invalid_identity_are_unknown() -> None:
    item = trial()
    item["assessments"][0]["status"] = "match"
    e = entry([item, item, trial("/private/trace")], declared=3)
    assert len(e.trials) == 2
    assert report.metric(e, (0, 1))["hits"] == 0
    assert report.metric(e, (0, 1))["unknown"] == 3
    page = report.render((e,), BOARD, "hash")
    assert "/private" not in page
    assert "opaque-trial-3" in page
    assert "unknown (n/a)" in page


@pytest.mark.parametrize("mutation", ["duplicate", "bad-id", "bad-count", "bad-board", "empty"])
def test_malformed_snapshot_rejected(mutation: str) -> None:
    doc = snapshot()
    if mutation == "duplicate":
        doc["rows"].append(doc["rows"][0])
    elif mutation == "bad-id":
        doc["rows"][0]["id"] = "../../private"
    elif mutation == "bad-count":
        doc["rows"][0]["n_trials"] = True
    elif mutation == "bad-board":
        doc["leaderboard"]["id"] = "<script>"
    else:
        doc = {}
    with pytest.raises(ValueError):
        report.snapshot_rows(doc)


def test_missing_malformed_and_extra_reports(tmp_path: Path) -> None:
    rows = report.snapshot_rows(snapshot())[1]
    e = report.load_entries(tmp_path, rows)[0]
    assert e.report_status.startswith("unscanned")
    assert report.metric(e, (0, 1))["unknown"] == 2
    (tmp_path / f"{ROW}.report.json").write_text('{"inputs": "bad"}')
    assert not report.load_entries(tmp_path, rows)[0].trials
    (tmp_path / f"{OTHER}.report.json").write_text('{"inputs": []}')
    assert len(report.load_entries(tmp_path, rows)) == 1


def test_export_allowlist_permissions_and_refuse_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    doc = snapshot()
    doc["rows"][0]["metadata"]["agent_display"] = '<img src=x onerror="boom">'
    doc["rows"][0]["metadata"]["private"] = "DO-NOT-EXPORT"
    (source / "leaderboard.json").write_text(json.dumps(doc))
    item = trial()
    item.update(input_error="RAW-ERROR", raw_trace="RAW-TRACE", task="/secret/task")
    a = item["assessments"][0]
    a.update(status="match", expected_by=["/secret/allowance"])
    a["evidence"] = [
        {"step": 1, "channel": "message", "field": "/secret/field", "text": "RAW-EVIDENCE"},
        {"step": 2, "channel": "<script>alert(1)</script>"},
    ]
    (source / f"{ROW}.report.json").write_text(
        json.dumps(
            {
                "scanner_version": "0.test",
                "inputs": [item],
                "raw_metadata": "DO-NOT-EXPORT",
            }
        )
    )
    output = tmp_path / "new"
    report.export(source, output, PAGE)
    page = (output / "index.html").read_text()
    for forbidden in (
        "RAW-",
        "/secret",
        "/private",
        "DO-NOT-EXPORT",
        "<img",
        "<script",
        "onerror",
    ):
        assert forbidden not in page
    assert "Build timestamp (UTC):" in page
    assert "locator withheld (malformed)" in page
    assert "automated-unreviewed" in page
    assert "default-src 'none'; style-src 'self'" in page
    assert "style=" not in page
    assert (
        f"https://hub.harborframework.com/datasets/synthetic-org/demo-pack/7/leaderboards/test-board/rows/{ROW}"
        in page
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in output.iterdir())
    with pytest.raises(FileExistsError):
        report.export(source, output)


def test_report_symlink_not_followed(tmp_path: Path) -> None:
    private = tmp_path / "private.json"
    private.write_text('{"inputs": []}')
    (tmp_path / f"{ROW}.report.json").symlink_to(private)
    rows = report.snapshot_rows(snapshot())[1]
    assert report.load_entries(tmp_path, rows)[0].report_status.startswith("unscanned")


def test_all_outcomes_tasks_and_recall_not_awareness() -> None:
    first, second = trial(), trial(OTHER)
    first["assessments"][2]["status"] = "match"
    second.update(task="another-synthetic-task", reward=1)
    e = entry([first, second], 2)
    assert report.metric(e, (0, 1))["hits"] == 0
    page = report.render((e,), BOARD, "hash")
    assert "2 distinct reported task labels" in page
    assert "reward 0 (included)" in page and "reward 1 (included)" in page


def test_excess_membership_disables_rates() -> None:
    page = report.render((entry([trial(), trial(OTHER)]),), BOARD, "hash")
    assert "Declared minus recorded: -1" in page
    assert "unknown (n/a)" in page


def test_unknown_coverage_shape_and_repository_output_refused() -> None:
    item = trial()
    item["coverage_gaps"]["behavioural"] = [False]
    assert report.metric(entry([item]), (0, 1))["unknown"] == 1
    with pytest.raises(ValueError, match="outside the repository"):
        report.export(Path("/unused-synthetic"), TOOL.parent / "never-create-export")


def test_malformed_allowances_do_not_explain_a_positive() -> None:
    item = trial()
    item["assessments"][0].update(status="match", expected_by=[None, {}, "/private/rule"])
    counts = report.metric(entry([item]), (0, 1))
    assert counts["hits"] == 1
    assert counts["explained"] == 0


def test_duplicate_alone_cannot_support_comparable_rate() -> None:
    item = trial()
    item["assessments"][0]["status"] = "match"
    e = entry([item, item], declared=2)
    assert len(e.trials) == 1
    assert not e.trials[0].trusted
    assert report.metric(e, (0, 1))["unknown"] == 2
    assert "unknown (n/a)" in report.metric_html(e, "Awareness", (0, 1))
    assert "2 trial records" in report.overview_html((e,))


def test_full_overview_in_snapshot_order_with_unscanned_dashes(tmp_path: Path) -> None:
    from uuid import UUID

    rows = [
        {"id": str(UUID(int=i + 1)), "n_trials": 2, "metadata": {"agent_display": f"Agent {i}"}}
        for i in range(35)
    ]
    for row in rows[:3]:
        (tmp_path / f"{row['id']}.report.json").write_text(json.dumps({"inputs": [trial()]}))
    entries = report.load_entries(tmp_path, rows)
    page = report.render(entries, BOARD, "synthetic-hash")
    overview = page.split('<section id="overview">', 1)[1].split("</section>", 1)[0]
    assert overview.count('<th scope="row">') == 35
    assert "3 / 35 reports loaded · 3 trial records" in overview
    assert overview.count("— / 2") == 32 * 3
    assert overview.count("<td>2 / 2</td>") == 32
    assert overview.count("<td>—</td>") == 32
    assert overview.index("Agent 0") < overview.index("Agent 34")
    assert f'href="#entry-{rows[-1]["id"]}"' in overview
    assert 'href="#snapshot-hash"' in overview
    assert "Declared denominator" not in overview
    assert "overflow-x:auto" in report.CSS
    assert 'tabindex="0"' in overview
    assert page.index("<table>") < page.index("<article")
    assert "— / 2</strong>" in page


def test_trial_priority_is_stable_and_preserves_anchors() -> None:
    from uuid import UUID

    items = [trial(str(UUID(int=i + 1))) for i in range(6)]
    for i in (1, 3):
        items[i]["assessments"][0]["status"] = "unknown"
    for i in (2, 5):
        items[i]["assessments"][0]["status"] = "match"
    e = entry(items, declared=6)
    page = report.entry_html(e, BOARD)
    order = [page.index(f'id="trial-{ROW}-{items[i]["hub_trial_id"]}"') for i in (2, 5, 1, 3, 0, 4)]
    assert order == sorted(order)
    assert page.count("AWARENESS HIT ·") == 2
    assert "Awareness: 2 hits · 0 explained (included) · 2 unknown · 2 no expression" in page
    assert f'id="counts-{ROW}"' in page
    assert '<summary title="Expand' in page


def test_expected_is_not_an_assessment_status() -> None:
    item = trial()
    item["assessments"][0].update(status="expected", expected_by=["synthetic.allowance"])
    counts = report.metric(entry([item]), (0, 1))
    assert counts["unknown"] == 1
    assert counts["hits"] == counts["explained"] == 0


def test_explained_positive_from_actual_report_serializer() -> None:
    from atif_scan.checks import CheckSpec, Detection, Status
    from atif_scan.engine import Assessment
    from atif_scan.output.document import report as scanner_report

    explained = Assessment(
        CheckSpec(report.CHECKS[0]),
        Detection(Status.MATCH, complete=False),
        expected_by=("synthetic.allowance",),
    )
    serialized = scanner_report((explained,))
    assert serialized["assessments"][0]["status"] == "match"
    item = trial()
    item["assessments"][0] = serialized["assessments"][0]
    counts = report.metric(entry([item]), (0, 1))
    assert counts["hits"] == counts["explained"] == 1
    assert counts["unknown"] == 0


def test_structured_display_labels_ignore_urls(tmp_path: Path) -> None:
    doc = snapshot()
    meta = doc["rows"][0]["metadata"]
    meta["agent_display"] = {"label": "Synthetic Harness", "url": "https://secret.invalid"}
    meta["model_display"] = {"label": "Synthetic Model 2", "url": "javascript:alert(1)"}
    entries = report.load_entries(tmp_path, doc["rows"])
    assert entries[0].title == "Synthetic Harness · Synthetic Model 2 · not recorded"
    page = report.render(entries, BOARD, "hash")
    assert "secret.invalid" not in page
    assert "javascript:" not in page
    assert report.display_label({"label": "<script>", "url": "ignored"}) == "not recorded"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("recorded", "recorded"),
        ("withheld", "withheld"),
        ("none", "none"),
        ("unknown", "unknown"),
        ("RAW-REASONING <script>secret</script>", "unknown"),
        ("Recorded", "unknown"),
        (None, "unknown"),
        ({"text": "RAW-REASONING"}, "unknown"),
        (["recorded"], "unknown"),
        (True, "unknown"),
    ],
)
def test_reasoning_allowlist_and_independent_negative(value: object, expected: str) -> None:
    item = trial()
    item["reasoning"] = value
    e = entry([item])
    assert e.trials[0].reasoning == expected
    assert report.metric(e, (0, 1))["negative"] == 1
    page = report.render((e,), BOARD, "hash")
    assert f"Reasoning: {expected} —" in page
    assert "RAW-REASONING" not in page
    assert "<script>" not in page


def test_reasoning_distribution_missing_and_default() -> None:
    items = []
    for i, state in enumerate(("recorded", "recorded", "withheld", "none", "unknown", None)):
        item = trial(f"00000000-0000-4000-8000-{i:012d}")
        if state is not None:
            item["reasoning"] = state
        items.append(item)
    e = entry(items, declared=8)
    assert report.reasoning_counts(e) == {
        "recorded": 2,
        "withheld": 1,
        "none": 1,
        "unknown": 2,
    }
    page = report.render((e,), BOARD, "hash")
    assert "recorded 2 / 6 trials" in page
    assert "recorded: 2 · withheld: 1 · none: 1 · unknown: 2" in page
    assert "Missing trials: 2 (visibility unknown; outside recorded distribution)" in page
    assert report.Trial(TRIAL, True, "synthetic", (), (), "unknown").reasoning == "unknown"
    assert page.index("Models expose differing reasoning") < page.index('id="overview"')
    assert "not comparative internal awareness" in page
    assert "none — no reasoning text/tokens recorded" in page
    assert "may be a summary, not complete chain" in page
    empty = entry([], declared=2)
    assert "recorded 0 / 0 trials" in report.overview_row(empty)
    assert "Missing trials: 2" in report.reasoning_html(empty)


def test_build_timestamp_and_discovery_scope() -> None:
    import re
    from datetime import UTC, datetime

    before = datetime.now(UTC).replace(microsecond=0)
    page = report.render((), BOARD, "synthetic-discovery-digest")
    after = datetime.now(UTC)
    match = re.search(r'Build timestamp \(UTC\): <time datetime="([^"]+)">([^<]+)</time>', page)
    assert match and match[1] == match[2] and match[1].endswith("Z")
    assert before <= datetime.fromisoformat(match[1]) <= after
    assert "discovery snapshot, not a frozen per-row trial manifest" in page
    assert "Scans selected exact row membership at fetch time" in page
    assert "snapshot does not pin membership" in page


@pytest.mark.parametrize(
    ("package", "name"),
    [
        (None, None),
        ("synthetic-org/demo-pack", None),
        (None, "test-board"),
        ("../demo-pack", "test-board"),
        ("synthetic-org/demo-pack/extra", "test-board"),
        ("synthetic-org/demo-pack", "../test-board"),
        ("https://evil.invalid/pkg", "test-board"),
        ("synthetic-org/demo-pack", 'test-board"><a href="https://evil.invalid'),
        ("synthetic-org/demo-pack", "test-board?redirect=https://evil.invalid"),
        ("synthetic-org/demo-pack", "%2e%2e"),
        ({"label": "synthetic-org/demo-pack"}, "test-board"),
        ("synthetic-org/demo-pack", ["test-board"]),
        ("synthetic-org/demo-pack", "test-board\\n"),
    ],
)
def test_invalid_source_metadata_leaves_row_unlinked(package: object, name: object) -> None:
    doc = snapshot()
    doc["leaderboard"].update(package=package, name=name)
    prefix = report.source_row_prefix(doc, PAGE)
    assert prefix is None
    page = report.render((entry([]),), BOARD, "hash", prefix)
    assert f"Snapshot row {ROW}" in page
    assert f"Snapshot leaderboard UUID: <code>{BOARD}</code>" in page
    assert "https://" not in page


def test_canonical_source_link_and_missing_metadata() -> None:
    doc = snapshot()
    prefix = report.source_row_prefix(doc, PAGE)
    page = report.render((entry([]),), BOARD, "hash", prefix)
    url = (
        "https://hub.harborframework.com/datasets/synthetic-org/demo-pack/7/"
        f"leaderboards/test-board/rows/{ROW}"
    )
    assert f'href="{url}"' in page
    assert f"/leaderboards/{BOARD}" not in page
    doc["leaderboard"] = {"id": BOARD}
    assert report.source_row_prefix(doc) is None
    assert "https://" not in report.render((entry([]),), BOARD, "hash")
    assert "<a " not in report.source_row_html("../invalid", prefix)
    assert "<a " not in report.source_row_html(ROW, "https://evil.invalid/")


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.invalid/datasets/synthetic-org/demo-pack/7",
        PAGE + "&selected=anything",
        PAGE + "#row",
        PAGE.replace("https:", "http:"),
        PAGE.replace("hub.harborframework.com", "hub.harborframework.com.evil.invalid"),
        PAGE.replace("hub.harborframework.com", "user@hub.harborframework.com"),
        PAGE.replace("/7?", "/../7?"),
        PAGE.replace("/7?", "/%37?"),
        PAGE.replace("/7?", "/0?"),
        PAGE + "\\n",
        PAGE.replace("tab=leaderboard", "tab=tasks"),
    ],
)
def test_invalid_leaderboard_url_refused(url: str, tmp_path: Path) -> None:
    assert report.source_page_url(url) is None
    assert report.source_row_prefix(snapshot(), url) is None
    with pytest.raises(ValueError):
        report.export(tmp_path, tmp_path / "out", url)
    assert not (tmp_path / "out").exists()


def test_no_version_inferred_from_snapshot() -> None:
    doc = snapshot()
    doc["leaderboard"]["dataset_version_ids"] = [OTHER]
    assert report.source_row_prefix(doc) is None
    assert "/leaderboards/" not in report.render((entry([]),), BOARD, "hash")


@pytest.mark.parametrize(
    "field,value",
    [
        ("package", "other-org/other-package"),
        ("name", "other-board"),
        ("package", None),
    ],
)
def test_source_page_fallback_is_not_row_selection(field: str, value: object) -> None:
    doc = snapshot()
    doc["leaderboard"][field] = value
    prefix = report.source_row_prefix(doc, PAGE)
    assert prefix is None
    page = report.render((entry([]),), BOARD, "hash", prefix, leaderboard_url=PAGE)
    assert 'href="' + PAGE.replace("&", "&amp;") + '"' in page
    assert "Source leaderboard page" in page
    assert "not a direct row selection" in page
    assert f"Snapshot row {ROW}" in page
    assert "/rows/" not in page


def test_versioned_route_does_not_hardcode_dataset_version() -> None:
    for version in ("1", "42"):
        url = PAGE.replace("/7?", f"/{version}?")
        prefix = report.source_row_prefix(snapshot(), url)
        assert prefix == (
            "https://hub.harborframework.com/datasets/synthetic-org/demo-pack/"
            f"{version}/leaderboards/test-board/rows/"
        )
    assert "<a " not in report.source_row_html(
        ROW,
        "https://hub.harborframework.com/datasets/synthetic-org/demo-pack/"
        "leaderboards/test-board/rows/",
    )


def test_explained_subset_requires_only_one_matching_check() -> None:
    item = trial()
    item["assessments"][0].update(status="match", expected_by=["synthetic.allowance"])
    item["assessments"][1].update(status="match")
    e = entry([item])
    assert report.metric(e, (0, 1))["explained"] == 1
    assert e.trials[0].checks[0].explained
    assert not e.trials[0].checks[1].explained
    assert "not necessarily all expressions explained" in report.render((e,), BOARD, "hash")


@pytest.mark.parametrize("status", ["unavailable", "invalid", None, True, "AVAILABLE"])
def test_input_availability_is_explicit_and_not_a_negative(status: object) -> None:
    missing = trial(OTHER)
    missing["input_status"] = status
    missing["input_error"] = "SECRET-RAW-ERROR"
    e = entry([trial(), missing], declared=3)
    assert [t.available for t in e.trials] == [True, False]
    assert report.metric(e, (0, 1))["unknown"] == 2
    page = report.render((e,), BOARD, "hash")
    assert "1 inputs available / 3 declared" in page
    assert "1 unavailable or invalid input records" in page
    assert "Input unavailable, invalid or unverified; availability not established." in page
    assert "SECRET-RAW-ERROR" not in page
    assert "1 / 1 reports loaded · 2 trial records" in page
    assert "completed entries" not in page
    assert "scanned inputs" not in page


def test_ambiguous_inputs_do_not_prove_availability() -> None:
    e = entry([trial(), trial(), trial("invalid")], declared=3)
    assert not any(t.available for t in e.trials)
    assert e.input_count == 3
    assert len(e.trials) == 2
    page = report.entry_html(e, None)
    assert "3 unavailable or invalid input records" in page
    assert report.metric(e, (0, 1))["unknown"] == 3


def test_incomplete_positive_is_not_lost_to_availability() -> None:
    item = trial()
    item["input_status"] = "invalid"
    item["partial"] = True
    item["assessments"][0] = assessment(report.CHECKS[0], "match", False)
    e = entry([item], declared=2)
    assert not e.trials[0].available
    assert report.metric(e, (0, 1))["hits"] == 1
    assert report.metric(e, (0, 1))["unknown"] == 1


def test_zero_loaded_and_empty_loaded_are_distinct(tmp_path: Path) -> None:
    _, rows = report.snapshot_rows(snapshot())
    unloaded = report.load_entries(tmp_path, rows)
    page = report.overview_html(unloaded)
    assert "0 / 1 reports loaded · 0 trial records" in page
    assert "0 inputs available / 2 declared" in page
    assert "— / 2" in page
    (tmp_path / f"{ROW}.report.json").write_text('{"inputs": []}')
    loaded = report.load_entries(tmp_path, rows)
    page = report.overview_html(loaded)
    assert "1 / 1 reports loaded · 0 trial records" in page
    assert "— / 2" not in page
    assert report.metric(loaded[0], (0, 1))["unknown"] == 2


def test_optional_benchmark_title_is_allowlisted(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    doc = snapshot()
    doc["leaderboard"]["title"] = "Synthetic Benchmark 4"
    (source / "leaderboard.json").write_text(json.dumps(doc))
    target = tmp_path / "export"
    report.export(source, target)
    page = (target / "index.html").read_text()
    assert '<p class="intro">Synthetic Benchmark 4</p>' in page
    assert "Synthetic Benchmark 4" not in report.render((), BOARD, "hash")
    assert "SECRET" not in report.render((), BOARD, "hash", benchmark_title="<b>SECRET</b>")
