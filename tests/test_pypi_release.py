"""Synthetic release metadata and archives; no credentials or network publishing."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "pypi_release.py"
spec = importlib.util.spec_from_file_location("pypi_release", TOOL)
assert spec and spec.loader
release = importlib.util.module_from_spec(spec)
sys.modules["pypi_release"] = release
spec.loader.exec_module(release)

VERSION = "1.2.3"
SHA = "a" * 40


@pytest.mark.parametrize("tag", ["v0.8.15", "v1.2.3", "v0.0.0"])
def test_stable_tag(tag: str) -> None:
    assert release.checked_tag(tag) == tag


@pytest.mark.parametrize(
    "tag",
    ["main", "refs/tags/v1.2.3", "v01.2.3", "v1.2.3rc1", "v1.2.3\n", "v1.2.3;echo bad"],
)
def test_untrusted_tag_cannot_be_a_ref_or_shell_fragment(tag: str) -> None:
    with pytest.raises(release.ReleaseError):
        release.checked_tag(tag)


@pytest.mark.parametrize(
    "document",
    [None, [], {}, {"project": "bad"}, {"project": {"name": "another", "version": VERSION}}],
)
def test_project_identity_is_required(document: object) -> None:
    with pytest.raises(release.ReleaseError):
        release.project_version("v" + VERSION, document)


def test_project_version_must_match_tag() -> None:
    document = {"project": {"name": "atif-scan", "version": VERSION}}
    assert release.project_version("v" + VERSION, document) == VERSION
    with pytest.raises(release.ReleaseError, match="version differ"):
        release.project_version("v1.2.4", document)


def test_resolve_requires_main_ancestry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def git(_source: Path, *args: str) -> str:
        calls.append(args)
        if args[0] == "rev-parse":
            return SHA + "\n"
        if args[0] == "show":
            return f'[project]\nname = "atif-scan"\nversion = "{VERSION}"\n'
        assert args == ("merge-base", "--is-ancestor", SHA, "refs/remotes/origin/main")
        return ""

    monkeypatch.setattr(release, "git", git)
    assert release.resolve(tmp_path, "v" + VERSION) == release.Release("v" + VERSION, SHA, VERSION)
    assert calls[1][0] == "merge-base"

    def off_main(_source: Path, *args: str) -> str:
        if args[0] == "merge-base":
            raise subprocess.CalledProcessError(1, "git")
        return SHA

    monkeypatch.setattr(release, "git", off_main)
    with pytest.raises(subprocess.CalledProcessError):
        release.resolve(tmp_path, "v" + VERSION)


@pytest.mark.parametrize("name", ["/absolute", "../escape", "root/../escape", "root\\escape"])
def test_archive_paths_cannot_escape(name: str) -> None:
    with pytest.raises(release.ReleaseError):
        release.member_path(name)


def wheel(path: Path, *, extra: str | None = None, version: str = VERSION) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"atif_scan-{VERSION}.dist-info/METADATA",
            f"Name: atif-scan\nVersion: {version}\n",
        )
        archive.writestr("atif_scan/__init__.py", "# Synthetic fixture\n")
        if extra is not None:
            archive.writestr(extra, "SYNTHETIC_ONLY")


def sdist(path: Path, *, extra: str | None = None, link: bool = False) -> None:
    files = {"pyproject.toml": b"# Synthetic fixture\n", "PKG-INFO": b"Name: atif-scan\n"}
    if extra is not None:
        files[extra] = b"SYNTHETIC_ONLY"
    with tarfile.open(path, "w:gz") as archive:
        for name, body in files.items():
            member = tarfile.TarInfo(f"atif_scan-{VERSION}/{name}")
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))
        if link:
            member = tarfile.TarInfo(f"atif_scan-{VERSION}/link")
            member.type = tarfile.SYMTYPE
            member.linkname = "/synthetic-outside"
            archive.addfile(member)


def distributions(tmp_path: Path) -> tuple[Path, Path]:
    whl = tmp_path / f"atif_scan-{VERSION}-py3-none-any.whl"
    tar = tmp_path / f"atif_scan-{VERSION}.tar.gz"
    wheel(whl)
    sdist(tar)
    return whl, tar


def test_distribution_allowlist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    distributions(tmp_path)
    monkeypatch.setattr(release, "git", lambda *_: "pyproject.toml\0")
    release.verify(tmp_path, tmp_path, VERSION)
    (tmp_path / "private-note.txt").write_text("SYNTHETIC_ONLY")
    with pytest.raises(release.ReleaseError, match="exactly one"):
        release.verify(tmp_path, tmp_path, VERSION)


def test_untracked_source_note_is_not_publishable(tmp_path: Path) -> None:
    _, tar = distributions(tmp_path)
    sdist(tar, extra="private-note.txt")
    with pytest.raises(release.ReleaseError, match="untracked"):
        release.check_sdist(tar, VERSION, {"pyproject.toml"})


def test_source_symlink_is_not_publishable(tmp_path: Path) -> None:
    _, tar = distributions(tmp_path)
    sdist(tar, link=True)
    with pytest.raises(release.ReleaseError, match="non-regular"):
        release.check_sdist(tar, VERSION, {"pyproject.toml"})


@pytest.mark.parametrize("extra", ["../escape", "/escape", "private-note.txt"])
def test_wheel_members_are_bounded(tmp_path: Path, extra: str) -> None:
    whl, _ = distributions(tmp_path)
    wheel(whl, extra=extra)
    with pytest.raises(release.ReleaseError):
        release.check_wheel(whl, VERSION)


def test_wheel_version_cannot_differ(tmp_path: Path) -> None:
    whl, _ = distributions(tmp_path)
    wheel(whl, version="9.9.9")
    with pytest.raises(release.ReleaseError, match="metadata differs"):
        release.check_wheel(whl, VERSION)


def test_runtime_floors(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {"certifi": "2026.10.18", "filelock": "4.2.1"}
    monkeypatch.setattr(release, "version", versions.__getitem__)
    release.runtime_dependencies()
    versions["certifi"] = "2026.7.22"
    with pytest.raises(release.ReleaseError, match="security floor"):
        release.runtime_dependencies()


def test_absent_optional_dependency_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_package: str) -> str:
        raise release.PackageNotFoundError

    monkeypatch.setattr(release, "version", missing)
    release.runtime_dependencies()


def test_prerelease_cannot_pass_as_a_patched_stable_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release, "version", lambda _: "4.2.1rc1")
    with pytest.raises(release.ReleaseError, match="stable"):
        release.runtime_dependencies()


def test_publisher_job_has_no_source_execution_or_stored_token() -> None:
    workflow = (TOOL.parents[1] / ".github/workflows/publish.yml").read_text()
    publisher = workflow.split("\n  publish:\n", 1)[1]
    assert workflow.count("id-token: write") == 1
    assert "id-token: write" in publisher
    assert "actions/checkout@" not in publisher
    assert "\n        run:" not in publisher
    assert "password:" not in publisher
    assert "skip-existing: false" in publisher


def test_publication_depends_on_same_workflow_release_checks() -> None:
    workflow = (TOOL.parents[1] / ".github/workflows/publish.yml").read_text()
    assert "needs: [resolve, check]" in workflow
    assert "needs: [resolve, build]" in workflow
    for command in ["pytest -q", "ruff check .", "ruff format --check .", "ty check"]:
        assert f"uv run {command}" in workflow
    assert "github.ref == 'refs/heads/main'" in workflow
    assert "persist-credentials: false" in workflow


def test_resolver_rejects_output_injection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(release, "git", lambda *_: SHA + "\nversion=9.9.9")
    with pytest.raises(release.ReleaseError, match="invalid release commit"):
        release.resolve(tmp_path, "v" + VERSION)
