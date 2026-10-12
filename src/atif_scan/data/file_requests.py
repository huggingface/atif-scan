"""Literal shell file-access requests, not evidence of execution or returned content.

Only named readers, directory probes and copy sources are interpreted. Other command
text stays undecidable for a caller's target predicate. No filesystem access or shell
execution; generated heredoc file contents are not requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from . import shell

READERS = frozenset({"cat", "head", "tail", "wc", "md5sum", "sha256sum", "diff", "cmp"})
# Options consuming a value rather than a file operand. cat -n is a flag, whereas
# head -n consumes a count; sharing these options would silently lose cat's target.
READ_VALUES = {
    "head": frozenset({"-n", "-c", "--lines", "--bytes"}),
    "tail": frozenset({"-n", "-c", "--lines", "--bytes", "--pid", "--sleep-interval"}),
    "diff": frozenset({"-I", "-x", "-X", "-L", "--ignore-matching-lines", "--exclude", "--label"}),
    "cmp": frozenset({"-i", "-n", "--ignore-initial", "--bytes"}),
    "ls": frozenset({"-I", "--ignore", "--hide", "--time-style", "--format", "--sort"}),
}
NON_READERS = frozenset({"echo", "printf", "true", "false", "mkdir", "touch", "rm", "tee"})


@dataclass(frozen=True)
class PathRequest:
    path: str
    span: shell.Span


@dataclass(frozen=True)
class PathRead:
    requests: tuple[PathRequest, ...] = ()
    undecidable: tuple[str, ...] = ()
    complete: bool = True


def _operands(command: shell.Command) -> list[tuple[str, shell.Span]] | None:
    argv = command.argv()
    if command.name in READERS or command.name == "ls":
        return shell.operands(argv, READ_VALUES.get(command.name, frozenset()))
    if command.name == "cp":
        destinations = {span for _, span in shell.writes(command)}
        return [p for p in shell.operands(argv, shell.COPY_VALUES) if p[1] not in destinations]
    if command.name in {"cd", "find"}:
        return shell.operands(argv)[:1] if command.name == "cd" else _find_roots(argv)
    return [] if command.name in NON_READERS else None


def _find_roots(argv: list[tuple[str, shell.Span]]) -> list[tuple[str, shell.Span]]:
    # Search roots precede expressions: a -name pattern is not a requested file.
    roots = []
    for word, span in argv[1:]:
        if not roots and word in {"-H", "-L", "-P", "--"}:
            continue
        if word.startswith("-") or word in {"(", "!"}:
            break
        roots.append((word, span))
    return roots


def _literal(path: str) -> bool:
    return not any(c in path for c in ("$", "`", "*", "?", "{"))


def _read_command(command: shell.Command, text: str) -> PathRead:
    operands = _operands(command)
    unknown = [text[slice(*command.span)]] if operands is None else []
    targets = [*(operands or []), *((p, s) for op, p, s in command.redirects if op == "<")]
    requests = tuple(PathRequest(path, span) for path, span in targets if _literal(path))
    unknown.extend(path for path, _ in targets if not _literal(path))
    return PathRead(requests, tuple(unknown), all(_literal(path) for path, _ in targets))


@lru_cache(maxsize=512)
def command_paths(text: str) -> PathRead:
    script = shell.parse(text)
    reads = [_read_command(command, text) for command in script.commands]
    bodies = tuple(
        text[start:end]
        for (start, end), command in zip(script.bodies, script.body_commands, strict=True)
        if (start, end) not in script.file_bodies and command.name not in READERS | NON_READERS
    )
    return PathRead(
        tuple(request for read in reads for request in read.requests),
        (*bodies, *(text for read in reads for text in read.undecidable)),
        script.complete and all(read.complete for read in reads),
    )
