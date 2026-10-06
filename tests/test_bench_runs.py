"""Synthetic bench-run metadata only; never import the external checkout."""

import hashlib
import json
from pathlib import Path

import pytest

from atif_scan.cli import main
from atif_scan.sources.bench import catalogue, load_run


def prepared(root: Path) -> Path:
    run = root / "runs" / "demo"
    run.mkdir(parents=True)
    configs = {}
    for part in ("a", "b"):
        data = json.dumps(
            {"job_name": f"demo-{part}", "jobs_dir": "/untrusted/ignored", "agents": []}
        ).encode()
        (run / f"{part}.json").write_bytes(data)
        configs[f"{part}.json"] = hashlib.sha256(data).hexdigest()
        (root / "jobs" / f"demo-{part}").mkdir(parents=True)
    (run / "receipt.json").write_text(
        json.dumps({"schema_version": 1, "identity": {"run_id": "demo"}, "configs": configs})
    )
    (root / "records").mkdir()
    (root / "records" / "cohorts.json").write_text(
        json.dumps(
            {"schema_version": 1, "cohorts": {"demo": {"status": "active", "reason": "SECRET"}}}
        )
    )
    return run


def test_parts_and_allowlisted_listing(tmp_path: Path, capsys):
    prepared(tmp_path)
    assert load_run(tmp_path, "demo").scan_paths() == [
        str(tmp_path / "jobs" / "demo-a"),
        str(tmp_path / "jobs" / "demo-b"),
    ]
    assert main(["runs", "--bench-root", str(tmp_path), "--format", "json"]) == 0
    output = capsys.readouterr().out
    assert "SECRET" not in output
    assert str(tmp_path) not in output
    assert json.loads(output)["runs"] == [
        {"run": "demo", "status": "active", "parts": 2, "local_parts": 2}
    ]


def test_missing_part_not_silently_skipped(tmp_path: Path):
    prepared(tmp_path)
    (tmp_path / "jobs" / "demo-b").rmdir()
    assert catalogue(tmp_path)[0]["local_parts"] == 1
    with pytest.raises(ValueError, match="missing_job_parts"):
        load_run(tmp_path, "demo").scan_paths()


def test_drift_is_unknown_not_zero(tmp_path: Path):
    run = prepared(tmp_path)
    (run / "a.json").write_text("{}")
    with pytest.raises(ValueError, match="digest_mismatch"):
        load_run(tmp_path, "demo")
    row = catalogue(tmp_path)[0]
    assert row["parts"] is None
    assert "issue" in row


@pytest.mark.parametrize("name", ["../demo", "/demo", "https://example.org", "."])
def test_invalid_selection(tmp_path: Path, name: str):
    with pytest.raises(ValueError, match="invalid_name"):
        load_run(tmp_path, name)


def test_external_symlink_rejected(tmp_path: Path):
    root = tmp_path / "root"
    run = prepared(root)
    outside = tmp_path / "outside.json"
    (run / "a.json").rename(outside)
    (run / "a.json").symlink_to(outside)
    with pytest.raises(ValueError, match="outside_root"):
        load_run(root, "demo")


@pytest.mark.parametrize("extra", [["x.json"], ["--manifest", "x.json"], ["--submission", "x"]])
def test_selection_conflicts(extra: list[str]):
    with pytest.raises(SystemExit) as error:
        main(["--run", "demo", *extra])
    assert error.value.code == 2


def test_scan_assembles_parts_with_repeated_trial_names(tmp_path: Path, capsys):
    prepared(tmp_path)
    synthetic = Path("examples/synthetic.json").read_bytes()
    for part in ("a", "b"):
        trial = tmp_path / "jobs" / f"demo-{part}" / "same-trial" / "agent"
        trial.mkdir(parents=True)
        (trial / "trajectory.json").write_bytes(synthetic)
    result = main(
        ["--bench-root", str(tmp_path), "--run", "demo", "--format", "json", "--no-cache"]
    )
    assert result in (0, 1)
    output = capsys.readouterr().out
    assert "part-1/" in output
    assert "part-2/" in output
    assert str(tmp_path) not in output


def test_oversized_metadata_rejected(tmp_path: Path):
    run = prepared(tmp_path)
    (run / "receipt.json").write_bytes(b" " * (2 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="metadata_too_large"):
        load_run(tmp_path, "demo")


def test_malformed_receipt_stays_visible(tmp_path: Path):
    run = prepared(tmp_path)
    (run / "receipt.json").write_text('{"configs": []}')
    assert catalogue(tmp_path)[0]["parts"] is None


def diagnostic(root: Path, *, kind: str = "codex-diagnostic", run_id: str = "codex-demo") -> Path:
    """A synthetic codex-diagnostic receipt: top-level run_id, no identity object."""
    run = root / "runs" / run_id
    run.mkdir(parents=True)
    configs = {}
    for part in ("hf-basic", "hf-upgrade"):
        data = json.dumps(
            {
                "job_name": f"{run_id}-{part}",
                "jobs_dir": "/untrusted/ignored",
                "agents": [],
            }
        ).encode()
        (run / f"{part}.json").write_bytes(data)
        configs[f"{part}.json"] = hashlib.sha256(data).hexdigest()
        (root / "jobs" / f"{run_id}-{part}").mkdir(parents=True)
    (run / "plan.json").write_text('{"job_name": "not-selected"}')
    (run / "receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": kind,
                "run_id": run_id,
                "arm": "oauth",
                "plan_path": str(root / "outside-plan.json"),
                "configs": configs,
            }
        )
    )
    (root / "records").mkdir(exist_ok=True)
    (root / "records" / "cohorts.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cohorts": {run_id: {"status": "diagnostic", "reason": "SECRET"}},
            }
        )
    )
    return run


def test_diagnostic_receipt_selects_pinned_parts_not_plan(tmp_path: Path):
    diagnostic(tmp_path)
    (tmp_path / "outside-plan.json").write_text('{"job_name": "leaked"}')
    assert load_run(tmp_path, "codex-demo").scan_paths() == [
        str(tmp_path / "jobs" / "codex-demo-hf-basic"),
        str(tmp_path / "jobs" / "codex-demo-hf-upgrade"),
    ]
    row = catalogue(tmp_path)[0]
    assert row["parts"] == 2
    assert row["local_parts"] == 2
    assert "issue" not in row
    assert "SECRET" not in json.dumps(row)


def test_diagnostic_scan_picks_up_trajectories(tmp_path: Path, capsys):
    diagnostic(tmp_path)
    synthetic = Path("examples/synthetic.json").read_bytes()
    for part in ("hf-basic", "hf-upgrade"):
        trial = tmp_path / "jobs" / f"codex-demo-{part}" / "same-trial" / "agent"
        trial.mkdir(parents=True)
        (trial / "trajectory.json").write_bytes(synthetic)
    result = main(
        [
            "--bench-root",
            str(tmp_path),
            "--run",
            "codex-demo",
            "--format",
            "json",
            "--no-cache",
        ]
    )
    assert result in (0, 1)
    output = capsys.readouterr().out
    assert "part-1/" in output
    assert "part-2/" in output
    assert "same-trial" in output
    assert "not-selected" not in output
    assert str(tmp_path) not in output


def test_diagnostic_run_id_must_match_selection(tmp_path: Path):
    run = diagnostic(tmp_path)
    receipt = json.loads((run / "receipt.json").read_text())
    receipt["run_id"] = "other-demo"
    (run / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="identity_mismatch"):
        load_run(tmp_path, "codex-demo")


@pytest.mark.parametrize("kind", ["replacement", "codex-diagnostic-replacement", "other"])
def test_other_receipt_kinds_stay_unknown(tmp_path: Path, kind: str):
    diagnostic(tmp_path, kind=kind)
    with pytest.raises(ValueError, match="unsupported_receipt"):
        load_run(tmp_path, "codex-demo")
    assert catalogue(tmp_path)[0]["parts"] is None
