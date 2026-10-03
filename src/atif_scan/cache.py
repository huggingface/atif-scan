"""Per-trace result cache for repeat scans of large runs (`--cache DIR`).

Stores the allowlisted, trace-derived part of a per-trace report item (never citations,
trace text or run facts from result.json / the Hub, which are merged fresh on every
scan), keyed by the trajectory file's fingerprint (path, size, mtime) plus a hash of the
scanner's own source code, the loaded check set (IDs, versions, task scopes, rule and
allowance expressions, plugin module sources) and the trace's context (task, partial,
reward). Any change to those misses the cache, so a parsing, detector or rule fix never
reuses results computed before it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .jsonval import Doc, is_object
from .output.document import SCHEMA_VERSION
from .rules import Allowance, Rule

if TYPE_CHECKING:
    from .checks import Context
    from .engine import Engine

PACKAGE = Path(__file__).parent


def code_fingerprint() -> str:
    """Hash of every module in the package: version strings aren't bumped per change."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.rglob("*.py")):
        digest.update(path.relative_to(PACKAGE).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def module_fingerprint(obj: object) -> str | None:
    """Hash of the source file defining `obj`'s class, for code outside the package (a
    plugin); None for the package's own modules (covered by `code_fingerprint`)."""
    module = sys.modules.get(type(obj).__module__)
    file = getattr(module, "__file__", None)
    if not file:
        return None
    path = Path(file).resolve()
    if path.is_relative_to(PACKAGE.resolve()):
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unreadable"  # never matches a readable file's hash


def logic(check: object) -> list[object]:
    """What a check decides with beyond its spec: a rule's `when`, an allowance's covers
    and `when` (frozen dataclasses: their repr is canonical), and plugin source."""
    if isinstance(check, Rule):
        expression: list[object] = [repr(check.expression)]
    elif isinstance(check, Allowance):
        expression = [sorted(check.covers), repr(check.when)]
    else:
        expression = []
    return [*expression, module_fingerprint(check)]


def checks_signature(engine: Engine) -> list[list[object]]:
    checks = [*engine.checks.values(), *engine.allowances]
    return sorted(
        [c.spec.id, c.spec.version, sorted(c.spec.tasks), int(c.spec.severity), logic(c)]
        for c in checks
    )


class ResultCache:
    def __init__(
        self, directory: Path, scanner_version: str, signature: list[list[object]]
    ) -> None:
        self.directory = directory
        # The report schema too: an item cached under an older layout must not be reused.
        self.base = json.dumps(
            [scanner_version, code_fingerprint(), SCHEMA_VERSION, signature], sort_keys=True
        )

    def key(self, fingerprint: str, context: Context) -> str:
        material = json.dumps(
            [self.base, fingerprint, context.task, context.partial, context.reward],
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Doc | None:
        try:
            value = json.loads(self._path(key).read_text())
        except (OSError, ValueError):
            return None
        # Written by `put` from allowlisted report fields; any other shape is a miss.
        return value if is_object(value) else None

    def put(self, key: str, entry: Doc) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Unique per process, so concurrent scans never interleave one temp file.
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(entry))
            tmp.replace(path)
        except OSError:
            pass  # a cache is an optimisation, never a failure
