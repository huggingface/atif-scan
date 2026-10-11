"""Synthetic only: opted-in masking, revalidation, and source confinement."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from atif_scan.checks import Context
from atif_scan.data.loader import load_bytes
from atif_scan.detectors.builtin import builtin_detectors
from atif_scan.engine import Engine
from atif_scan.output.document import report

if TYPE_CHECKING:
    from atif_scan.data.jsonval import Doc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def modules():
    import awareness_excerpts
    import awareness_report

    return awareness_excerpts, awareness_report


excerpts, overview = modules()

ROW = "22222222-2222-4222-8222-222222222222"
JOB = "11111111-1111-4111-8111-111111111111"
TRIAL = "33333333-3333-4333-8333-333333333333"


def engine() -> Engine:
    return Engine([d for d in builtin_detectors() if d.spec.id in overview.CHECKS])


def fixture() -> tuple[bytes, Doc, str]:
    secret = "sk-synthetic123456789012345678901234567890"
    data = json.dumps(
        {
            "schema_version": "ATIF-v1.7",
            "steps": [
                {"step_id": 1, "source": "user", "message": f"api_key={secret}"},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": (
                        f"This is a benchmark task. Terminal-Bench. {secret} "
                        "<script>alert(1)</script>"
                    ),
                },
            ],
        }
    ).encode()
    trace = load_bytes(data)
    doc = report(engine().evaluate(trace, Context(task="synthetic-task")), trace.step_numbers)
    item: Doc = {**doc, "hub_trial_id": TRIAL, "task": "synthetic-task", "partial": False}
    item["assessments"][0]["expected_by"] = ["synthetic.explained"]
    return data, item, secret


def test_masked_explained_evidence_and_provenance() -> None:
    data, item, secret = fixture()
    result = excerpts.extract(item, data, engine())
    assert len(result["source_sha256"]) == 64
    found = result["checks"]["awareness.benchmark"]
    assert found["status"] == "available"
    assert found["excerpts"][0]["step_id"] == 2
    assert secret not in json.dumps(result)
    assert "benchmark" in json.dumps(result)
    assert "expected_by" not in json.dumps(result)  # stays in the bound report


def test_changed_positions_and_version_refused() -> None:
    data, item, _ = fixture()
    saved = next(a for a in item["assessments"] if a["id"] == "awareness.benchmark")
    saved["version"] = "old-version"
    assert excerpts.extract(item, data, engine())["checks"]["awareness.benchmark"]["status"] == (
        "changed_check_or_locations"
    )
    _, item, _ = fixture()
    changed = data.replace(b"This is", b"A prefix. This is")
    assert excerpts.extract(item, changed, engine())["checks"]["awareness.benchmark"]["status"] == (
        "changed_check_or_locations"
    )


def test_missing_and_symlink_sources(tmp_path: Path) -> None:
    data, item, _ = fixture()
    p = tmp_path / "trace.json"
    p.write_bytes(data)
    link = tmp_path / "link.json"
    link.symlink_to(p)
    for path in (None, link, tmp_path / "missing"):
        found = excerpts.trial_excerpts(item, path, tmp_path, engine())
        assert not found["checks"]["awareness.benchmark"]["excerpts"]
    result = excerpts.trial_excerpts(item, p, tmp_path, engine())
    assert result["checks"]["awareness.benchmark"]["status"] == "available"


def test_listing_confinement_and_duplicate_ids(tmp_path: Path) -> None:
    folder = tmp_path / JOB
    folder.mkdir()
    listing = {"job": JOB, "rows": [{"id": TRIAL, "name": "../../secret"}]}
    (folder / "hub-listing.json").write_text(json.dumps(listing))
    document: Doc = {"runs": [{"leaderboard": {"jobs": [JOB, "../../other"]}}]}
    index = excerpts.source_index(tmp_path, document)
    assert index[TRIAL] == folder / TRIAL / "trajectory.json"
    listing["rows"].append({"id": TRIAL, "name": "another"})
    (folder / "hub-listing.json").write_text(json.dumps(listing))
    assert not excerpts.source_index(tmp_path, document)


def test_private_artifact_bound_to_report(tmp_path: Path) -> None:
    _, item, _ = fixture()
    document: Doc = {"inputs": [item], "runs": []}
    result = excerpts.build_row(document, "hash", ROW, tmp_path, engine())
    assert result["report_sha256"] == "hash"
    assert result["trials"][TRIAL]["checks"]["awareness.benchmark"]["status"].startswith(
        "source_unavailable"
    )
    output = tmp_path / "artifact.json"
    excerpts.write_private(output, result)
    assert output.stat().st_mode & 0o777 == 0o600


def test_view_requires_bound_report_and_escapes_text(tmp_path: Path) -> None:
    import awareness_excerpt_view as view

    data, item, secret = fixture()
    trial = excerpts.extract(item, data, engine())
    artifact = {
        "format": excerpts.FORMAT,
        "row_id": ROW,
        "report_sha256": "bound",
        "trials": {TRIAL: trial},
    }
    path = tmp_path / "sidecar.json"
    path.write_text(json.dumps(artifact))
    assert view.load(path, ROW, "other-report") == {}
    bank = view.load(path, ROW, "bound")
    shown = view.evidence_html(bank[TRIAL]["awareness.benchmark"])
    assert "<mark>" in shown and "benchmark" in shown
    assert "<script>" not in shown
    assert "&lt;script&gt;" in shown
    assert secret not in shown
    assert "current source" in shown.lower()


def test_view_rejects_bad_status_and_malformed_locators() -> None:
    import awareness_excerpt_view as view

    assert view.parse_evidence({"status": "<script>raw-secret"}, "").status == "invalid_sidecar"
    assert (
        view.parse_evidence(
            {"status": "available", "excerpts": [{"step_id": True}], "total_locations": 1},
            "a" * 64,
        ).status
        == "invalid_sidecar"
    )


def test_excerpt_queue_opt_in_and_reconciliation(tmp_path: Path) -> None:
    data, item, secret = fixture()
    item.update(input_status="available", compacted=False, coverage_gaps={"behavioural": []})
    e = overview.Entry(
        ROW,
        "Synthetic entry",
        1,
        overview.trials_from({"inputs": [item]}),
        "test",
        "bound",
        "loaded",
        1,
    )
    artifact = {
        "format": excerpts.FORMAT,
        "row_id": ROW,
        "report_sha256": "bound",
        "trials": {TRIAL: excerpts.extract(item, data, engine())},
    }
    (tmp_path / f"{ROW}.excerpts.json").write_text(json.dumps(artifact))
    assert "This is a " not in overview.render((e,), JOB, "hash")
    attached = overview.attach_excerpts((e,), tmp_path)
    page = overview.render(attached, JOB, "hash")
    assert "Read matched excerpts" in page
    assert "This is a " in page
    assert secret not in page
    assert overview.metric(e, (0, 1)) == overview.metric(attached[0], (0, 1))
    assert f'id="excerpt-body-{ROW}"' in page
    assert f'href="#excerpt-body-{ROW}"' in page


def test_observations_keep_explained_unexplained_overlap() -> None:
    import awareness_observations

    _, item, _ = fixture()
    e = overview.Entry(
        ROW,
        "Synthetic entry",
        1,
        overview.trials_from({"inputs": [item]}),
        "test",
        "bound",
        "loaded",
        1,
    )
    assert awareness_observations.counts(e) == (1, 1, 1, 1)
    page = awareness_observations.render((e,))
    assert "columns overlap" in page
    assert "not controlled model-only comparisons" in page


def test_source_parent_traversal_and_symlink_refused(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    other = tmp_path / "other"
    other.write_text("synthetic")
    assert not excerpts.confined(root, root / ".." / "other")
    assert not excerpts.confined(root, other)
    link = tmp_path / "linked-parent"
    link.symlink_to(root, target_is_directory=True)
    assert not excerpts.confined(link, link / "file")
