"""Static reading of tool calls inside a Python tool program (fast-agent programmatic tool
calling).

fast-agent can expose its tools only through one `execute_python` tool whose raw `input`
is a Python program: `r = await shell(command="git show HEAD:a.py"); print(r["stdout"])`.
This legacy static projection reads tool dispatches, not arbitrary direct Python I/O
(which some runtimes permit). Valid runtime dispatch ledgers are handled by `ptc`.
This module parses the program with `ast` (never compiled or run) and reads each call
of a name the program doesn't bind (`shell`, `read_text_file`, a helper kept from an
earlier program, …): its keyword arguments when they are *literals* (strings, numbers,
lists, dicts), a name bound
once at the top level to one, `+`-joined or f-string text of those, or `"sep".join` of a
literal list. A call in a `for` loop or comprehension over a fixed list is read once per
element (a superset of what ran). Anything else is `UNREAD`. A program that doesn't parse,
a call with `**kwargs`, and any other use of a tool name (`f = shell`) or of `eval`-like
builtins add a `("?", UNREAD)` call, so callers treat them as unknown, not absent.

Variables and functions persist between fast-agent programs, so a name bound only by an
earlier program is runtime state: unread.
"""

from __future__ import annotations

import ast
import builtins
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from math import prod
from typing import TYPE_CHECKING

from .jslit import UNREAD

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

MAX_NESTING = 50  # literal nesting depth read before giving up (unreadable)
MAX_VALUE_WORK = 4096
MAX_VALUE_CHARS = 262_144
MAX_INTEGER_BITS = 4096
MAX_EXPANSION = 64  # loop element combinations read per call before leaving it unread
UNKNOWN_CALL = ("?", UNREAD)
# Builtins that run or reach code by name: what they call can't be read statically.
DYNAMIC = frozenset(
    {"eval", "exec", "compile", "getattr", "globals", "locals", "vars", "__import__"}
)
BUILTINS = frozenset(dir(builtins)) - DYNAMIC


class _ReadLimit(Exception):
    """A finite evaluation-work or materialized-text budget was exhausted."""


def parse(program: str) -> ast.Module | None:
    """The program's syntax tree, or None when it doesn't parse (nothing is compiled)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # invalid escapes etc. would print to stderr
            return ast.parse(program)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def _binding_names(node: ast.AST) -> Iterator[str]:
    """Names a node binds (assignment and loop targets, parameters, imports, defs…)."""
    if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
        yield node.id
    elif isinstance(node, (ast.Global, ast.Nonlocal)):
        yield from node.names
    elif isinstance(node, ast.alias):
        yield (node.asname or node.name).split(".")[0]
    else:
        name = _declared_name(node)
        if name:
            yield name


def _declared_name(node: ast.AST) -> str | None:
    """The name a definition, parameter, `except … as` or `match` capture binds."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
        return node.name
    return node.rest if isinstance(node, ast.MatchMapping) else None


def bindings(tree: ast.AST) -> Counter[str]:
    """How often each name is bound anywhere in the program (any scope)."""
    return Counter(name for node in ast.walk(tree) for name in _binding_names(node))


@dataclass
class _Program:
    tree: ast.Module
    # The tool functions the harness declared to programs, when the trace records them.
    declared: frozenset[str] | None
    # Recognized tool names, for traces that don't declare theirs.
    known: Callable[[str], bool]
    value_work: int = MAX_VALUE_WORK
    value_chars: int = MAX_VALUE_CHARS
    bound: Counter[str] = field(init=False)
    # Names bound once, by a plain top-level `NAME = value`: (where it ends, the value).
    constants: dict[str, tuple[tuple[int, int], ast.expr]] = field(init=False)

    def __post_init__(self) -> None:
        self.bound = bindings(self.tree)
        self.constants = {}
        for stmt in self.tree.body:
            target, value = _simple_assignment(stmt)
            if target is not None and value is not None and self.bound[target] == 1:
                self.constants[target] = (_end(stmt), value)

    def unbound(self, name: str) -> bool:
        return not self.bound[name] and name not in BUILTINS

    def calls_tool(self, name: str) -> bool:
        """Calling `name` may call a tool: a declared tool, or, when none are declared,
        any name the program doesn't bind (a helper kept from an earlier program could
        wrap one)."""
        return self.unbound(name) and (self.declared is None or name in self.declared)

    def shadowed_tool(self, name: str) -> bool:
        """A tool-shaped name with unresolved lexical binding is not proven absent."""
        names = self.declared
        known = name in names if names is not None else self.known(name)
        return bool(self.bound[name]) and known

    def names_tool(self, name: str) -> bool:
        tools = self.declared
        return self.unbound(name) and (name in tools if tools is not None else self.known(name))


def _simple_assignment(stmt: ast.stmt) -> tuple[str | None, ast.expr | None]:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        return (target.id, stmt.value) if isinstance(target, ast.Name) else (None, None)
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id, stmt.value
    return None, None


def _end(node: ast.stmt | ast.expr) -> tuple[int, int]:
    return (node.end_lineno or node.lineno, node.end_col_offset or 0)


def _start(node: ast.expr) -> tuple[int, int]:
    return (node.lineno, node.col_offset)


# A loop variable's elements, by name, for the loops a call is inside.
Loops = dict[str, list[object]]


class _Values:
    """Literal values of expressions, under one assignment of loop variables."""

    def __init__(self, program: _Program, loop: dict[str, object]) -> None:
        self.program = program
        self.loop = loop

    def read(self, node: ast.expr, depth: int = 0) -> object:
        self.program.value_work -= 1
        if self.program.value_work < 0:
            raise _ReadLimit
        if depth > MAX_NESTING:
            return UNREAD
        if isinstance(node, ast.Constant):
            value = _constant(node.value)
            if isinstance(value, str):
                self.program.value_chars -= len(value)
                if self.program.value_chars < 0:
                    raise _ReadLimit
            return value
        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Name)):
            return self.container(node, depth)
        return self.text(node, depth)

    def container(
        self, node: ast.List | ast.Tuple | ast.Set | ast.Dict | ast.Name, depth: int
    ) -> object:
        """A list, tuple, set or dict display, or a name (bound to one, or to text)."""
        if isinstance(node, ast.Name):
            return self.name(node, depth)
        if isinstance(node, ast.Dict):
            return self.mapping(node, depth)
        if any(isinstance(e, ast.Starred) for e in node.elts):
            return UNREAD
        return [self.read(e, depth + 1) for e in node.elts]

    def mapping(self, node: ast.Dict, depth: int) -> object:
        out: dict[str, object] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            literal = self.read(key, depth + 1) if key is not None else UNREAD
            if not isinstance(literal, str):
                return UNREAD  # `**spread` or a computed key: unknown extra keys
            out[literal] = self.read(value, depth + 1)
        return out

    def name(self, node: ast.Name, depth: int) -> object:
        if node.id in self.loop:
            return self.loop[node.id]
        bound = self.program.constants.get(node.id)
        if bound is None or bound[0] > _start(node):
            return UNREAD  # rebound, bound elsewhere, earlier program state, or not yet
        return self.read(bound[1], depth + 1)

    def text(self, node: ast.expr, depth: int) -> object:
        """`a + b`, an f-string or `"sep".join([...])` of literal strings and integers."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return _join(self.read(node.left, depth + 1), self.read(node.right, depth + 1))
        if isinstance(node, ast.JoinedStr):
            return self.fstring(node, depth)
        join = _join_call(node)
        return self.joined(join, depth) if join is not None else UNREAD

    def fstring(self, node: ast.JoinedStr, depth: int) -> object:
        parts: list[str] = []
        for part in node.values:
            value = self.hole(part, depth)
            if not _joinable(value):
                return UNREAD
            parts.append(str(value))
        return _text_join("", parts)

    def hole(self, part: ast.expr, depth: int) -> object:
        """An f-string's literal text, or a `{value}` with no conversion or format spec
        (its text is `str(value)`)."""
        if isinstance(part, ast.Constant):
            return self.read(part, depth + 1)
        if not isinstance(part, ast.FormattedValue) or part.format_spec is not None:
            return UNREAD
        return self.read(part.value, depth + 1) if part.conversion in (-1, ord("s")) else UNREAD

    def joined(self, join: tuple[ast.expr, ast.expr], depth: int) -> object:
        sep = self.read(join[0], depth + 1)
        items = self.read(join[1], depth + 1)
        if not isinstance(sep, str) or not isinstance(items, list):
            return UNREAD
        return _text_join(sep, items)


def _constant(value: object) -> object:
    """str, int, float, bool or None as written; bytes, complex and `...` are unread."""
    if isinstance(value, int) and value.bit_length() > MAX_INTEGER_BITS:
        return UNREAD
    return UNREAD if isinstance(value, (bytes, complex)) or value is Ellipsis else value


def _text_join(separator: str, values: Sequence[object]) -> object:
    if any(not isinstance(value, str) for value in values):
        return UNREAD
    texts = [value for value in values if isinstance(value, str)]
    size = sum(map(len, texts)) + max(0, len(texts) - 1) * len(separator)
    return separator.join(texts) if size <= MAX_VALUE_CHARS else UNREAD


def _join_call(node: ast.expr) -> tuple[ast.expr, ast.expr] | None:
    """(separator, items) of `SEP.join(ITEMS)`."""
    if not isinstance(node, ast.Call) or len(node.args) != 1 or node.keywords:
        return None
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "join":
        return func.value, node.args[0]
    return None


def _joinable(value: object) -> bool:
    """A string or integer: what `+` and f-strings turn into text exactly as written."""
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


def _join(left: object, right: object) -> object:
    if isinstance(left, str) and isinstance(right, str):
        return left + right if len(left) + len(right) <= MAX_VALUE_CHARS else UNREAD
    if _joinable(left) and _joinable(right) and isinstance(left, int) and isinstance(right, int):
        return left + right
    return UNREAD  # Python doesn't add a string and a number


_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)


def _comprehension_body(
    node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
) -> list[ast.expr]:
    conditions = [c for g in node.generators for c in g.ifs]
    if isinstance(node, ast.DictComp):
        return [node.key, node.value, *conditions]
    return [node.elt, *conditions]


class _Reader:
    """Walks the program's syntax tree, collecting the calls it reads."""

    def __init__(self, program: _Program) -> None:
        self.program = program
        self.loops: Loops = {}
        self.calls: list[tuple[str, object]] = []

    def visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.Call):
            self.call(node)
        elif isinstance(node, ast.Name):
            self.reference(node)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            self.looped([(node.target, node.iter)], node.body, self.program.bound)
            self.visit_all(node.orelse)
        elif isinstance(node, _COMPREHENSIONS):
            pairs = [(g.target, g.iter) for g in node.generators]
            # Comprehension targets are local: unrelated bindings elsewhere cannot
            # reassign them. Nested bindings still conservatively prevent expansion.
            self.looped(pairs, _comprehension_body(node), bindings(node))
        else:
            self.visit_all(ast.iter_child_nodes(node))

    def visit_all(self, nodes: Iterable[ast.AST]) -> None:
        for node in nodes:
            self.visit(node)

    def call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in DYNAMIC and not self.program.bound[func.id]:
            self.calls.append(UNKNOWN_CALL)
        elif isinstance(func, ast.Name) and self.program.calls_tool(func.id):
            self.calls.extend((func.id, args) for args in self.arguments(node))
        elif isinstance(func, ast.Name) and self.program.shadowed_tool(func.id):
            self.calls.append(UNKNOWN_CALL)
        else:
            self.visit(func)
        self.visit_all((*node.args, *node.keywords))

    def reference(self, node: ast.Name) -> None:
        # A tool function used other than by calling it (`f = shell`, `run(shell)`).
        if isinstance(node.ctx, ast.Load) and (
            self.program.names_tool(node.id) or self.program.shadowed_tool(node.id)
        ):
            self.calls.append(UNKNOWN_CALL)

    def looped(
        self,
        pairs: list[tuple[ast.expr, ast.expr]],
        body: Sequence[ast.AST],
        bound: Counter[str],
    ) -> None:
        """Visit a loop's body once, with each loop variable over a fixed list bound to
        its elements (variables over anything else stay unread)."""
        added: Loops = {}
        for target, source in pairs:
            self.visit(source)
            elements = self.elements(target, source, added, bound)
            if elements is not None and isinstance(target, ast.Name):
                added[target.id] = elements
            else:
                self.visit(target)
        saved = self.loops
        self.loops = {**saved, **added}
        self.visit_all(body)
        self.loops = saved

    def elements(
        self, target: ast.expr, source: ast.expr, added: Loops, bound: Counter[str]
    ) -> list[object] | None:
        """A loop variable's values: a never-rebound name in this scope over a literal
        list of strings and integers."""
        if not isinstance(target, ast.Name) or bound[target.id] != 1:
            return None
        values = _Values(self.program, {}).read(source)
        if not isinstance(values, list) or target.id in added:
            return None
        elements: list[object] = list(values)
        return elements if elements and all(map(_joinable, elements)) else None

    def arguments(self, node: ast.Call) -> list[object]:
        """A tool call's arguments, once per combination of the loop variables it reads."""
        if any(k.arg is None for k in node.keywords):
            return [UNREAD]  # `**kwargs`: unknown keys and values
        used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} & self.loops.keys()
        names = sorted(used)
        combos = prod(len(self.loops[n]) for n in names)
        if combos > MAX_EXPANSION:
            return [self.read(node, {})]
        return [
            self.read(node, dict(zip(names, values, strict=True)))
            for values in product(*(self.loops[n] for n in names))
        ]

    def read(self, node: ast.Call, loop: dict[str, object]) -> object:
        values = _Values(self.program, loop)
        args: dict[str, object] = {k.arg: values.read(k.value) for k in node.keywords if k.arg}
        if node.args:
            # Tool functions take keywords only; positionals reach an unknown helper.
            args["args"] = values.read(ast.List(elts=node.args, ctx=ast.Load()))
        return args


def tool_calls(
    program: str, declared: frozenset[str] | None, known: Callable[[str], bool]
) -> list[tuple[str, object]]:
    """(name, arguments) of each tool call, in source order. `declared` are the tool
    functions the harness gave programs (None: not recorded, so every call of a name the
    program doesn't define counts); `known` recognizes tool names when none are declared.
    Any other use of a tool's name is unknown."""
    tree = parse(program)
    if tree is None:
        return [UNKNOWN_CALL]
    reader = _Reader(_Program(tree, declared, known))
    try:
        reader.visit(tree)
    except (_ReadLimit, RecursionError, MemoryError):
        return [UNKNOWN_CALL]
    return reader.calls


# Patch text is file content, read through the call it feeds.
_PATCH = re.compile(r"^\*\*\* (?:Begin Patch|End Patch|(?:Add|Update|Delete) File:|Move to:)", re.M)
# As in `jslit`: a lone word (a dict key, `"stdout"`, a method name) is code, not a command
# or path. Kept are literals with a space, `/`, `.`, `:` or `-`.
_WORDY = re.compile(r"\S[\s/.:-]|[/.:-]\S")


def program_strings(program: str, read: set[str]) -> list[str]:
    """The program's string literals and f-string text (holes are a space), in source
    order, except text already read as a call argument (`read`), patch text and lone
    words: the constants a computed argument is built from. Nothing is executed."""
    tree = parse(program)
    if tree is None:
        return []
    out: list[str] = []
    for text in _literals(tree):
        if text not in read and _WORDY.search(text.strip()) and not _PATCH.search(text):
            out.append(text)
    return out


def _literals(tree: ast.Module) -> list[str]:
    inside: set[int] = set()  # constants that are part of an f-string
    found: list[tuple[tuple[int, int], str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            parts = [_fstring_part(v) for v in node.values]
            inside.update(id(v) for v in ast.walk(node) if isinstance(v, ast.Constant))
            found.append(((node.lineno, node.col_offset), "".join(parts)))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in inside:
                found.append(((node.lineno, node.col_offset), node.value))
    return [text for _, text in sorted(found, key=lambda f: f[0])]


def _fstring_part(node: ast.expr) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return " "


def strings(value: object) -> list[str]:
    """The strings in a read argument value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return []
