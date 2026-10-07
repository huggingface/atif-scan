"""What a trial submitted: the files its git patch changes (standard library only).

Benchmarks that grade a committed diff (DeepSWE's `artifacts/model.patch`, captured by
`pre_artifacts.sh` as `git diff <base> HEAD`) record the submission beside the
trajectory. This reads its `diff --git` headers into paths and change kinds, and keeps
each file's added lines in memory for fixed predicates. Paths, lines and patch text are
never exported: reports carry check IDs, trace step locators and counts only.

A patch that can't be read is not an empty submission: `parse_patch` marks what it
couldn't parse, and callers pass `None` (unknown) when there is no patch file at all.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

MAX_BYTES = 8 * 1024 * 1024  # DeepSWE patches observed reach ~120 KB; larger is refused
MAX_FILES = 5000

HEADER = re.compile(r"^diff --git (?P<a>\"?a/.*?\"?) (?P<b>\"?b/.*?\"?)$")
CHANGES = ("added", "modified", "deleted", "renamed")


@dataclass(frozen=True)
class PatchFile:
    path: str  # repository-relative, the post-change path (the old one for a deletion)
    change: str  # one of CHANGES
    # Added lines (without the leading `+`): memory only, hidden from repr, never exported.
    added: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if self.change not in CHANGES:
            raise ValueError("invalid_patch_change")


@dataclass(frozen=True)
class Submission:
    files: tuple[PatchFile, ...]
    digest: str  # sha256 of the patch bytes (cache identity, not exported as text)
    understood: bool = True  # False: some of the patch couldn't be parsed

    @property
    def empty(self) -> bool:
        return not self.files and self.understood


def _unquote(value: str) -> str:
    """A git header path without its `a/`/`b/` prefix. C-quoted paths (special
    characters) keep their escapes: they are compared, never opened."""
    value = value.strip()
    if value.startswith('"') and value.endswith('"') and len(value) > 1:
        value = value[1:-1]
    return value[2:] if value[:2] in ("a/", "b/") else value


@dataclass
class _File:
    old: str
    new: str
    change: str = "modified"
    added: list[str] = field(default_factory=list)

    def done(self) -> PatchFile:
        path = self.old if self.change == "deleted" else self.new
        return PatchFile(path, self.change, tuple(self.added))


def _header_line(current: _File, line: str) -> None:
    if line.startswith("new file mode"):
        current.change = "added"
    elif line.startswith("deleted file mode"):
        current.change = "deleted"
    elif line.startswith("rename to "):
        current.change, current.new = "renamed", line[len("rename to ") :].strip()
    elif line.startswith("+++ ") and line[4:].strip() != "/dev/null":
        current.new = _unquote(line[4:])


@dataclass
class _Parser:
    files: list[PatchFile] = field(default_factory=list)
    current: _File | None = None
    understood: bool = True
    in_hunk: bool = False

    def feed(self, line: str) -> None:
        if found := HEADER.match(line):
            self.close()
            self.current, self.in_hunk = _File(_unquote(found["a"]), _unquote(found["b"])), False
        elif self.current is None:
            self.understood = self.understood and not line.strip()  # text before any header
        elif line.startswith("@@"):
            self.in_hunk = True
        elif self.in_hunk:
            if line.startswith("+"):
                self.current.added.append(line[1:])
        else:
            _header_line(self.current, line)

    def close(self) -> None:
        if self.current is not None:
            self.files.append(self.current.done())
            self.current = None


def parse_patch(data: bytes) -> Submission:
    """The files a `git diff` patch changes. Lines before the first header, or a file
    count over MAX_FILES, mark the submission as not fully understood."""
    parser = _Parser()
    for line in data.decode("utf-8", errors="replace").splitlines():
        parser.feed(line)
    parser.close()
    files, understood = parser.files, parser.understood
    if len(files) > MAX_FILES:
        files, understood = files[:MAX_FILES], False
    return Submission(tuple(files), hashlib.sha256(data).hexdigest(), understood)
