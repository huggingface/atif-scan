"""Validate release tags and distribution boundaries; never execute package contents."""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import tarfile
import tomllib
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath


class ReleaseError(ValueError):
    """A safe, fixed release-validation diagnostic."""


@dataclass(frozen=True)
class Release:
    tag: str
    commit: str
    version: str


def checked_tag(tag: str) -> str:
    if not re.fullmatch(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", tag):
        raise ReleaseError("expected a stable vMAJOR.MINOR.PATCH tag")
    return tag


def project_version(tag: str, document: object) -> str:
    checked_tag(tag)
    project = document.get("project") if isinstance(document, dict) else None
    if not isinstance(project, dict) or project.get("name") != "atif-scan":
        raise ReleaseError("unexpected project metadata")
    if project.get("version") != tag[1:]:
        raise ReleaseError("tag and project version differ")
    return tag[1:]


def git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=source, text=True, capture_output=True, check=True
    ).stdout


def resolve(source: Path, tag: str) -> Release:
    tag = checked_tag(tag)
    commit = git(source, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ReleaseError("invalid release commit")
    git(source, "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main")
    document = tomllib.loads(git(source, "show", f"{commit}:pyproject.toml"))
    return Release(tag, commit, project_version(tag, document))


def member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ReleaseError("unsafe archive member path")
    return path


def check_wheel(path: Path, release_version: str) -> None:
    metadata_dir = f"atif_scan-{release_version}.dist-info"
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            parts = member_path(member.filename).parts
            if not parts or parts[0] not in {"atif_scan", metadata_dir}:
                raise ReleaseError("unexpected wheel member")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ReleaseError("wheel contains a symbolic link")
        metadata = Parser().parsestr(archive.read(f"{metadata_dir}/METADATA").decode())
        if metadata["Name"] != "atif-scan" or metadata["Version"] != release_version:
            raise ReleaseError("wheel metadata differs from release")


def check_sdist(path: Path, release_version: str, tracked: set[str]) -> None:
    root = f"atif_scan-{release_version}"
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            parts = member_path(member.name).parts
            if not parts or parts[0] != root:
                raise ReleaseError("unexpected source archive root")
            if member.isdir():
                continue
            if not member.isfile():
                raise ReleaseError("source archive contains a non-regular file")
            relative = "/".join(parts[1:])
            if relative not in tracked | {"PKG-INFO"}:
                raise ReleaseError("source archive contains an untracked file")


def verify(source: Path, dist: Path, release_version: str) -> None:
    checked_tag("v" + release_version)
    names = {
        f"atif_scan-{release_version}-py3-none-any.whl",
        f"atif_scan-{release_version}.tar.gz",
    }
    files = list(dist.iterdir())
    if {path.name for path in files} != names or any(
        path.is_symlink() or not path.is_file() for path in files
    ):
        raise ReleaseError("expected exactly one release wheel and one source archive")
    tracked = set(git(source, "ls-files", "-z").split("\0")) - {""}
    check_wheel(dist / f"atif_scan-{release_version}-py3-none-any.whl", release_version)
    check_sdist(dist / f"atif_scan-{release_version}.tar.gz", release_version, tracked)


def runtime_dependencies() -> None:
    floors = {"certifi": (2026, 10, 18), "filelock": (4, 2, 1)}
    for package, minimum in floors.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            continue
        if not re.fullmatch(r"\d+\.\d+\.\d+", installed):
            raise ReleaseError("dependency version is not a stable three-part version")
        if tuple(int(part) for part in installed.split(".")) < minimum:
            raise ReleaseError("installed dependency is below the security floor")
        print(f"{package} {installed}: security floor satisfied")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tag = commands.add_parser("resolve")
    tag.add_argument("--tag", required=True)
    tag.add_argument("--source", type=Path, default=Path("."))
    artifact = commands.add_parser("verify")
    artifact.add_argument("--source", type=Path, required=True)
    artifact.add_argument("--dist", type=Path, required=True)
    artifact.add_argument("--version", required=True)
    commands.add_parser("runtime")
    args = parser.parse_args()
    try:
        run(args)
    except ReleaseError as exc:
        parser.exit(1, f"release validation failed: {exc}\n")
    except (OSError, subprocess.CalledProcessError, tarfile.TarError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"release validation failed ({type(exc).__name__})\n")


def run(args: argparse.Namespace) -> None:
    if args.command == "resolve":
        release = resolve(args.source, args.tag)
        output = os.environ.get("GITHUB_OUTPUT")
        if output:
            with Path(output).open("a") as stream:
                stream.write(
                    f"tag={release.tag}\nsha={release.commit}\nversion={release.version}\n"
                )
        print(f"Validated {release.tag} at {release.commit}")
    elif args.command == "verify":
        verify(args.source, args.dist, args.version)
        print("Validated distribution metadata and archive boundaries")
    else:
        runtime_dependencies()


if __name__ == "__main__":
    main()
