"""Offline bench-run receipts. Never import bench-run or follow config paths."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..data.jsonval import Doc, JsonObject, as_object, as_str, load_object

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

LIMIT = 2 * 1024 * 1024
# Codex diagnostic cohorts bind a top-level run_id. plan_path is not a scan input.
DIAGNOSTIC_KIND = "codex-diagnostic"


def _name(value: object) -> str:
    text = as_str(value)
    if text is None or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", text) is None:
        raise ValueError("bench_invalid_name")
    return text


def _path(root: Path, *parts: str) -> Path:
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("bench_path_outside_root")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("bench_path_outside_root")
    return path


def _read(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise ValueError("bench_metadata_too_large")
    return data


def _object(path: Path) -> JsonObject:
    value = load_object(_read(path), LIMIT)
    if not value:
        raise ValueError("bench_invalid_metadata")
    return value


def _bound_run_id(receipt: JsonObject) -> object:
    """Standard cohorts use identity.run_id; diagnostics use a top-level run_id.

    Any other kind stays unsupported. This does not read plan_path or jobs_dir.
    """
    kind = receipt.get("kind")
    if kind == DIAGNOSTIC_KIND:
        if "identity" in receipt:
            raise ValueError("bench_unsupported_receipt")
        return receipt.get("run_id")
    if kind is not None:
        raise ValueError("bench_unsupported_receipt")
    return as_object(receipt.get("identity")).get("run_id")


@dataclass(frozen=True)
class BenchRun:
    name: str
    jobs: tuple[Path, ...]
    # Partition (config file stem, e.g. "hf-basic") -> its job folder, with the digests
    # replacement receipts must name to count as this run's (see sources.selection).
    partitions: Mapping[str, Path] = field(default_factory=dict)
    receipt_sha256: str = ""
    config_sha256: Mapping[str, str] = field(default_factory=dict)

    def scan_paths(self) -> list[str]:
        if any(not job.is_dir() for job in self.jobs):
            raise ValueError("bench_missing_job_parts")
        return [str(job) for job in self.jobs]


def _job_parts(
    root: Path, name: str, configs: JsonObject
) -> tuple[dict[str, Path], dict[str, str]]:
    """Partition -> job folder and config digest, from receipt-pinned configs."""
    partitions: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for filename, digest in sorted(configs.items()):
        data = _read(_path(root, "runs", name, _name(filename)))
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("bench_config_digest_mismatch")
        config = load_object(data, LIMIT)
        partition = filename.removesuffix(".json")
        partitions[partition] = _path(root, "jobs", _name(config.get("job_name")))
        digests[partition] = str(digest)
    return partitions, digests


def load_run(root: Path, name: str) -> BenchRun:
    """Use receipt-pinned configs; jobs are always under the selected root/jobs."""
    name = _name(name)
    receipt_bytes = _read(_path(root, "runs", name, "receipt.json"))
    receipt = load_object(receipt_bytes, LIMIT)
    if not receipt:
        raise ValueError("bench_invalid_metadata")
    if type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1:
        raise ValueError("bench_unsupported_receipt")
    if _bound_run_id(receipt) != name:
        raise ValueError("bench_run_identity_mismatch")
    configs = as_object(receipt.get("configs"))
    if not configs:
        raise ValueError("bench_no_job_parts")
    partitions, digests = _job_parts(root, name, configs)
    jobs = list(partitions.values())
    if len(set(jobs)) != len(jobs):
        raise ValueError("bench_duplicate_job_parts")
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    return BenchRun(name, tuple(jobs), partitions, receipt_sha, digests)


def catalogue(root: Path) -> list[Doc]:
    """Allowlisted labels/counts only; malformed receipts remain visible as unknown."""
    catalog = _object(_path(root, "records", "cohorts.json"))
    if catalog.get("schema_version") != 1 or not as_object(catalog.get("cohorts")):
        raise ValueError("bench_invalid_catalogue")
    rows: list[Doc] = []
    for raw_name, entry in sorted(as_object(catalog.get("cohorts")).items()):
        name = _name(raw_name)
        status = as_object(entry).get("status")
        if status not in (
            "active",
            "candidate",
            "published",
            "reference",
            "diagnostic",
            "written-off",
            "stopped",
            "superseded",
        ):
            status = "unknown"
        row: Doc = {"run": name, "status": status, "parts": None, "local_parts": None}
        try:
            run = load_run(root, name)
            row.update(parts=len(run.jobs), local_parts=sum(p.is_dir() for p in run.jobs))
        except (OSError, ValueError):
            row["issue"] = "unavailable_or_invalid_receipt"
        rows.append(row)
    return rows
