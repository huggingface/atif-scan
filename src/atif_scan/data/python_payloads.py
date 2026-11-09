"""Proven local Python string payloads, read as syntax only; never execute source.

Follow literal strings through assignments and ordinary local text transformations.
Unknown consumers are uncertainty, not proof that a literal is inert or requested.
Offsets are character offsets into the supplied source, including non-ASCII text.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass, field

from .pysource import parse_module

Span = tuple[int, int]
LOCAL_SINKS = frozenset({"print", "open", "io.open"})
CONSTRUCTORS = frozenset({"Path", "pathlib.Path"})
LOCAL_METHODS = frozenset(
    {"write", "writelines", "write_text", "write_bytes", "read_text", "read_bytes"}
)
TEXT_METHODS = frozenset(
    {
        "replace",
        "format",
        "join",
        "split",
        "splitlines",
        "strip",
        "lstrip",
        "rstrip",
        "encode",
        "decode",
        "lower",
        "upper",
        "startswith",
        "endswith",
    }
)
LOCAL_FUNCTIONS = frozenset(
    {"str", "repr", "len", "json.dumps", "json.loads", "re.sub", "collections.OrderedDict"}
)
HTTP_METHODS = frozenset(
    {"get", "post", "put", "patch", "delete", "head", "options", "request", "stream"}
)
PROCESS_CALLS = frozenset(
    {
        "os.system",
        "os.popen",
        "subprocess.run",
        "subprocess.call",
        "subprocess.Popen",
        "subprocess.check_call",
        "subprocess.check_output",
    }
)


@dataclass(frozen=True)
class PythonPayloads:
    inert: tuple[Span, ...] = ()
    uncertain: tuple[Span, ...] = ()
    complete: bool = True
    active: bool = False


def _spelling(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _spelling(node.value) + "." + node.attr
    return ""


def _aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            aliases.update((a.asname or a.name.split(".")[0], a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            aliases.update(
                (a.asname or a.name, (node.module or "") + "." + a.name) for a in node.names
            )
    _binding_aliases(tree, aliases)
    return aliases


def _binding_aliases(tree: ast.Module, aliases: dict[str, str]) -> None:
    bindings: dict[str, list[ast.AST]] = {}
    for target, value in _bindings(tree):
        if isinstance(target, ast.Name):
            bindings.setdefault(target.id, []).append(value)
    definitions = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for name in definitions:
        aliases[name] = "__unknown__." + name
    original = dict(aliases)
    for name, values in bindings.items():
        aliases[name] = _binding_name(values, original)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            root = _spelling(node).split(".", maxsplit=1)[0]
            aliases[root] = "__unknown__." + root


def _binding_name(values: list[ast.AST], aliases: dict[str, str]) -> str:
    if len(values) != 1:
        return "__unknown__"
    value = values[0]
    if isinstance(value, (ast.Name, ast.Attribute)):
        return _resolved(_spelling(value), aliases)
    if isinstance(value, ast.Call):
        name = _resolved(_spelling(value.func), aliases)
        if name in {"requests.Session", "httpx.Client", "httpx.AsyncClient"}:
            return name
    return "__unknown__"


def _resolved(name: str, aliases: dict[str, str]) -> str:
    root, dot, rest = name.partition(".")
    return aliases.get(root, root) + dot + rest


def _request_call(name: str) -> bool:
    root = name.split(".", maxsplit=1)[0]
    method = name.rsplit(".", 1)[-1]
    return (
        (root in {"requests", "httpx", "aiohttp"} and method in HTTP_METHODS)
        or name in {"urllib.request.urlopen", "urllib.request.urlretrieve"}
        or name in PROCESS_CALLS
    )


@dataclass
class _Flow:
    parents: dict[ast.AST, ast.AST]
    loads: dict[str, list[ast.Name]]
    aliases: dict[str, str]
    files: set[str]
    paths: set[str]
    texts: set[str]
    unsafe_effects: bool
    # 0 = inert, 1 = unknown consumer, 2 = lexical request consumer.
    seen: set[ast.AST] = field(default_factory=set)

    def local_method(self, call: ast.Call) -> int:
        if self.unsafe_effects or not isinstance(call.func, ast.Attribute):
            return 1
        receiver = call.func.value
        spelling = _spelling(receiver)
        constructor = isinstance(receiver, ast.Call) and _resolved(
            _spelling(receiver.func), self.aliases
        ) in CONSTRUCTORS | {"open", "io.open"}
        return 0 if spelling in self.files | self.paths or constructor else 1

    def call(self, call: ast.Call) -> int | None:
        name = _resolved(_spelling(call.func), self.aliases)
        method = name.rsplit(".", 1)[-1]
        if _request_call(name):
            return 2
        return self.local_call(call, name, method)

    def local_call(self, call: ast.Call, name: str, method: str) -> int | None:
        if name in LOCAL_SINKS:
            return 1 if self.unsafe_effects else 0
        if method in LOCAL_METHODS:
            return self.local_method(call)
        if name in LOCAL_FUNCTIONS | CONSTRUCTORS:
            return 1 if self.unsafe_effects else None
        return self.text_method(call) if method in TEXT_METHODS else 1

    def text_method(self, call: ast.Call) -> int | None:
        if isinstance(call.func, ast.Attribute) and _text_expression(
            call.func.value, self.aliases, self.files, self.paths, self.texts
        ):
            return None
        return 1

    def assignment(self, node: ast.Assign | ast.AnnAssign) -> int:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not all(isinstance(target, ast.Name) for target in targets):
            return 1
        return max(
            (
                self.follow(load)
                for target in targets
                if isinstance(target, ast.Name)
                for load in self.loads.get(target.id, [])
            ),
            default=0,
        )

    def parent_effect(self, parent: ast.AST | None) -> int:
        if isinstance(parent, ast.Call):
            effect = self.call(parent)
            return self.follow(parent) if effect is None else effect
        if isinstance(parent, (ast.Assign, ast.AnnAssign)):
            return self.assignment(parent)
        if isinstance(parent, ast.Expr):
            return 0
        return self.comparison(parent)

    def comparison(self, node: ast.AST | None) -> int:
        if isinstance(node, ast.Compare) and all(
            _text_expression(value, self.aliases, self.files, self.paths, self.texts)
            for value in [node.left, *node.comparators]
        ):
            return 0
        return 1

    def follow(self, node: ast.AST) -> int:
        if _deferred(node, self.parents):
            return 1
        if node in self.seen:
            return 0  # Cycles add no sink; all other uses are still inspected.
        self.seen.add(node)
        parent = self.parents.get(node)
        if isinstance(
            parent,
            (
                ast.Attribute,
                ast.keyword,
                ast.List,
                ast.Tuple,
                ast.Set,
                ast.Dict,
                ast.BinOp,
                ast.JoinedStr,
                ast.FormattedValue,
            ),
        ):
            return self.follow(parent)
        return self.parent_effect(parent)


def _offset(source: str, line: int, column: int) -> int:
    lines = source.splitlines(keepends=True)
    prefix = "".join(lines[: line - 1])
    # AST columns are UTF-8 bytes, unlike tokenizer columns.
    return len(prefix) + len(lines[line - 1].encode()[:column].decode())


def _span(source: str, node: ast.Constant) -> Span:
    return (
        _offset(source, node.lineno, node.col_offset),
        _offset(source, node.end_lineno or node.lineno, node.end_col_offset or 0),
    )


def _deferred(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return True
    return False


def _comments(source: str) -> list[Span]:
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    return [
        (starts[t.start[0] - 1] + t.start[1], starts[t.end[0] - 1] + t.end[1])
        for t in tokenize.generate_tokens(io.StringIO(source).readline)
        if t.type == tokenize.COMMENT
    ]


def _bindings(tree: ast.Module) -> list[tuple[ast.AST, ast.AST]]:
    bindings: list[tuple[ast.AST, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            bindings.extend((target, node.value) for target in node.targets)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            bindings.append((node.optional_vars, node.context_expr))
    return bindings


def _mutated_names(tree: ast.Module) -> set[str]:
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(_spelling(node).split(".", maxsplit=1)[0])
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    return names


def _dynamic_bindings(tree: ast.Module, aliases: dict[str, str]) -> bool:
    """Reflective mutation/evaluation invalidates local-call spelling proofs."""
    return any(
        (
            isinstance(node, ast.Call)
            and (
                _spelling(node.func)
                in {"exec", "eval", "setattr", "delattr", "globals", "locals", "vars", "__import__"}
                or _resolved(_spelling(node.func), aliases) in PROCESS_CALLS
            )
        )
        for node in ast.walk(tree)
    )


def _local_receivers(tree: ast.Module, aliases: dict[str, str]) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    paths: set[str] = set()
    receivers = {"open": files, "io.open": files, "Path": paths, "pathlib.Path": paths}
    bindings = _bindings(tree)
    names = [
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    ]
    blocked = _mutated_names(tree)
    for target, value in bindings:
        if (
            isinstance(target, ast.Name)
            and names.count(target.id) == 1
            and target.id not in blocked
            and isinstance(value, ast.Call)
        ):
            name = _resolved(_spelling(value.func), aliases)
            if name in receivers:
                receivers[name].add(target.id)
    return files, paths


def _text_expression(
    node: ast.AST, aliases: dict[str, str], files: set[str], paths: set[str], texts: set[str]
) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.Name):
        return node.id in texts
    if isinstance(node, ast.Call):
        return _text_call(node, aliases, files, paths, texts)
    return isinstance(node, ast.JoinedStr)


def _text_call(
    call: ast.Call, aliases: dict[str, str], files: set[str], paths: set[str], texts: set[str]
) -> bool:
    name = _resolved(_spelling(call.func), aliases)
    if name in {"str", "repr", "json.dumps", "re.sub"}:
        return True
    if not isinstance(call.func, ast.Attribute):
        return False
    receiver = call.func.value
    method = call.func.attr
    if method in {"read_text", "read"}:
        local = _spelling(receiver) in files | paths
        constructor = isinstance(receiver, ast.Call) and _resolved(
            _spelling(receiver.func), aliases
        ) in CONSTRUCTORS | {"open", "io.open"}
        return local or constructor
    return method in TEXT_METHODS and _text_expression(receiver, aliases, files, paths, texts)


def _text_names(
    tree: ast.Module, aliases: dict[str, str], files: set[str], paths: set[str]
) -> set[str]:
    bindings = [
        (target.id, value) for target, value in _bindings(tree) if isinstance(target, ast.Name)
    ]
    texts: set[str] = {name for name, _ in bindings}
    while True:
        remaining: set[str] = {
            name
            for name in texts
            if all(
                _text_expression(value, aliases, files, paths, texts)
                for bound, value in bindings
                if bound == name
            )
        }
        if remaining == texts:
            return texts
        texts = remaining


def _classify(source: str, tree: ast.Module) -> PythonPayloads:
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    loads: dict[str, list[ast.Name]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loads.setdefault(node.id, []).append(node)
    aliases = _aliases(tree)
    files, paths = _local_receivers(tree, aliases)
    texts = _text_names(tree, aliases, files, paths)
    unsafe = _dynamic_bindings(tree, aliases)
    spans: dict[int, list[Span]] = {0: _comments(source), 1: [], 2: []}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        flow = _Flow(parents, loads, aliases, files, paths, texts, unsafe)
        effect = 1 if _deferred(node, parents) else flow.follow(node)
        spans[effect].append(_span(source, node))
    return PythonPayloads(tuple(spans[0]), tuple(spans[1]), active=bool(spans[2]))


def python_payloads(source: str) -> PythonPayloads:
    """Classify literal payloads; computed/unknown execution is never called inert."""
    try:
        tree = parse_module(source)
        return _classify(source, tree)
    except (SyntaxError, ValueError, RecursionError, tokenize.TokenError):
        return PythonPayloads(uncertain=((0, len(source)),), complete=False)
