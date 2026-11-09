"""Syntax-only source-read requests in bounded, already-extracted inline Python.

Facts are candidates, not proof of execution, file existence or import origin.
``complete`` describes this recognizer's coverage, not arbitrary Python behavior.
Function bodies are inert; only bounded literal string loops are reconstructed.
Other control flow is unknown, never explored.
No paths are opened, imports resolved or submitted expressions executed.
"""

from __future__ import annotations

import ast
import posixpath
from dataclasses import dataclass, field

from .pysource import parse_module

MAX_CODE = 256 * 1024
MAX_NODES = 20_000
MAX_DEPTH = 80
MAX_VALUE = 8192
MAX_LOOP_ITERATIONS = 32
_OPEN_MODE_INDEX = 1
_INSERT_ARITY = 2
_READ_MODES = frozenset({"r", "rt", "rb", "r+", "rt+", "r+t", "rb+", "r+b"})
_WRITE_MODES = frozenset(
    {
        "w",
        "wt",
        "wb",
        "w+",
        "wt+",
        "w+t",
        "wb+",
        "w+b",
        "a",
        "at",
        "ab",
        "a+",
        "at+",
        "a+t",
        "ab+",
        "a+b",
        "x",
        "xt",
        "xb",
        "x+",
        "xt+",
        "x+t",
        "xb+",
        "x+b",
    }
)


@dataclass(frozen=True)
class ModuleSource:
    module: str
    search_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class PythonSourceAccess:
    paths: tuple[str, ...] = ()
    modules: tuple[ModuleSource, ...] = ()
    complete: bool = True


@dataclass(frozen=True)
class _Imported:
    name: str
    paths: tuple[str, ...] = ()
    # Dotted, unaliased imports bind a root, but each member has its own timestamp.
    members: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class _Path:
    text: str


@dataclass(frozen=True)
class _PathMethod:
    path: str
    name: str


@dataclass(frozen=True)
class _Deferred:
    """A local callable whose body is intentionally not interpreted."""


_Value = str | _Imported | _Path | _PathMethod | _Deferred | None


def _loop_strings(node: ast.expr) -> tuple[str, ...] | None:
    if not isinstance(node, (ast.List, ast.Tuple)) or len(node.elts) > MAX_LOOP_ITERATIONS:
        return None
    values = [
        item.value
        for item in node.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    ]
    return tuple(values) if len(values) == len(node.elts) else None


def _bounded_tree(tree: ast.Module) -> bool:
    pending: list[tuple[ast.AST, int]] = [(tree, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            return False
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return True


def _text(value: _Value) -> str | None:
    if isinstance(value, _Path):
        return value.text
    return value if isinstance(value, str) else None


def _joined(parts: list[str]) -> str | None:
    if not parts or sum(map(len, parts)) > MAX_VALUE:
        return None
    return posixpath.join(*parts)


def _argument(args: list[_Value], keywords: dict[str, _Value], index: int, name: str) -> _Value:
    return args[index] if len(args) > index else keywords.get(name)


def _attribute(base: _Value, name: str) -> _Value:
    if isinstance(base, _Path):
        return _PathMethod(base.text, name)
    if isinstance(base, _Imported) and name not in {"__file__", "__version__"}:
        canonical = base.name + "." + name
        paths = dict(base.members).get(canonical, base.paths)
        return _Imported(canonical, paths, base.members)
    return None


@dataclass
class _Parser:
    values: dict[str, _Value] = field(default_factory=lambda: {"open": _Imported("builtins.open")})
    search_paths: tuple[str, ...] = ()
    paths: list[str] = field(default_factory=list)
    modules: list[ModuleSource] = field(default_factory=list)
    complete: bool = True
    source_names: set[str] = field(default_factory=lambda: {"open"})
    iterations_left: int = MAX_LOOP_ITERATIONS
    loop_depth: int = 0
    unknown_epoch: int = 0

    def unknown(self) -> None:
        self.unknown_epoch += 1
        self.complete = False
        self.values.clear()
        self.search_paths = ()

    def evaluate(self, node: ast.expr) -> _Value:
        if isinstance(node, ast.Lambda):
            self.defaults(node.args)
            return _Deferred()
        if isinstance(
            node,
            (
                ast.BoolOp,
                ast.IfExp,
                ast.NamedExpr,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        ):
            self.unknown()
            return None
        return self._evaluate_unconditional(node)

    def _evaluate_unconditional(self, node: ast.expr) -> _Value:
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, ast.BinOp):
            return self.binary(node)
        if isinstance(node, ast.Attribute):
            return _attribute(self.evaluate(node.value), node.attr)
        return self._evaluate_leaf(node)

    def _evaluate_leaf(self, node: ast.expr) -> _Value:
        if isinstance(node, ast.Name):
            return self.values.get(node.id)
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) else None
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.evaluate(child)
        return None

    def binary(self, node: ast.BinOp) -> _Value:
        left, right = self.evaluate(node.left), self.evaluate(node.right)
        if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str):
            return left + right if len(left) + len(right) <= MAX_VALUE else None
        if isinstance(node.op, ast.Div) and isinstance(left, _Path):
            tail = _text(right)
            joined = _joined([left.text, tail]) if tail is not None else None
            return _Path(joined) if joined is not None else None
        return None

    def call(self, node: ast.Call) -> _Value:
        function = self.evaluate(node.func)
        args = [self.evaluate(arg) for arg in node.args]
        keywords = {kw.arg: self.evaluate(kw.value) for kw in node.keywords}
        if None in keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
            self.complete = False
            return None
        named = {key: value for key, value in keywords.items() if key is not None}
        if isinstance(function, _PathMethod):
            return self.path_method(function, args)
        if isinstance(function, _Imported):
            return self.imported_call(function, node, args, named)
        self.unresolved_call(function, node.func)
        return None

    def unresolved_call(self, function: _Value, target: ast.expr) -> None:
        dynamic_execution = isinstance(target, ast.Name) and target.id in {
            "exec",
            "eval",
            "compile",
            "__import__",
        }
        if isinstance(function, _Deferred) or dynamic_execution:
            self.unknown()
        elif self._source_shaped(target):
            self.complete = False

    def _source_shaped(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.source_names
        return isinstance(node, ast.Attribute) and node.attr in {
            "getsource",
            "getsourcelines",
            "read_text",
            "read_bytes",
        }

    def imported_call(
        self,
        function: _Imported,
        node: ast.Call,
        args: list[_Value],
        keywords: dict[str, _Value],
    ) -> _Value:
        name = function.name
        if name == "builtins.open":
            self.open_call(args, keywords)
        elif name in {"inspect.getsource", "inspect.getsourcelines"}:
            self.inspect_call(_argument(args, keywords, 0, "object"))
        elif name in {"pathlib.Path", "pathlib.PosixPath"}:
            parts = [_text(arg) for arg in args]
            joined = _joined([part for part in parts if part is not None])
            return _Path(joined) if joined is not None and None not in parts else None
        elif name == "os.path.join":
            parts = [arg for arg in args if isinstance(arg, str)]
            return _joined(parts) if len(parts) == len(args) else None
        elif name.startswith("sys.path."):
            self.search_path_call(name, node, args)
        else:
            self.unresolved_call(function, node.func)
        return None

    def path_method(self, function: _PathMethod, args: list[_Value]) -> _Value:
        if function.name in {"read_text", "read_bytes"}:
            self.paths.append(function.path)
        elif function.name == "joinpath":
            parts = [_text(arg) for arg in args]
            joined = _joined([function.path, *[p for p in parts if p is not None]])
            return _Path(joined) if joined is not None and None not in parts else None
        return None

    def open_call(self, args: list[_Value], keywords: dict[str, _Value]) -> None:
        mode = _argument(args, keywords, 1, "mode")
        if len(args) <= _OPEN_MODE_INDEX and "mode" not in keywords:
            mode = "r"
        if isinstance(mode, str) and mode in _WRITE_MODES:
            return
        path = _text(_argument(args, keywords, 0, "file"))
        if path is None or not isinstance(mode, str) or mode not in _READ_MODES:
            self.complete = False
        else:
            self.paths.append(path)

    def inspect_call(self, target: _Value) -> None:
        if isinstance(target, _Imported):
            self.modules.append(ModuleSource(target.name, target.paths))
        else:
            self.complete = False

    def search_path_call(self, name: str, node: ast.Call, args: list[_Value]) -> None:
        if name == "sys.path.append":
            path = args[0] if len(args) == 1 else None
            if isinstance(path, str):
                self.search_paths += (path,)
                return
        elif len(node.args) == _INSERT_ARITY and isinstance(node.args[0], ast.Constant):
            path = args[1]
            if (
                type(node.args[0].value) is int
                and node.args[0].value == 0
                and isinstance(path, str)
            ):
                self.search_paths = (path, *self.search_paths)
                return
        self.complete = False
        self.search_paths = ()

    def invalidate(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.values.pop(target.id, None)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self.invalidate(element)
        else:
            # Attribute/subscript writes can change module objects and sys.path.
            self.unknown()

    def assign(self, node: ast.Assign | ast.AnnAssign) -> None:
        value = self.evaluate(node.value) if node.value is not None else None
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            self.invalidate(target)
            if isinstance(target, ast.Name) and isinstance(value, (str, _Path, _Deferred)):
                self.values[target.id] = value

    def import_statement(self, node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.ImportFrom) and (
            node.level or any(alias.name == "*" for alias in node.names)
        ):
            self.unknown()
            return
        for alias in node.names:
            self.import_alias(node, alias)

    def import_alias(self, node: ast.Import | ast.ImportFrom, alias: ast.alias) -> None:
        if isinstance(node, ast.ImportFrom):
            local = alias.asname or alias.name
            canonical = f"{node.module}.{alias.name}"
            self.values[local] = _Imported(canonical, self.search_paths)
            if canonical in {"builtins.open", "inspect.getsource", "inspect.getsourcelines"}:
                self.source_names.add(local)
            return
        if alias.asname:
            self.values[alias.asname] = _Imported(alias.name, self.search_paths)
            return
        root = alias.name.split(".")[0]
        previous = self.values.get(root)
        binding = (
            previous
            if isinstance(previous, _Imported) and previous.name == root
            else (_Imported(root, self.search_paths))
        )
        members = (*binding.members, (alias.name, self.search_paths))
        self.values[root] = _Imported(root, binding.paths, members)

    def definition(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Defaults and decorators execute now; the function body does not.
        for expr in node.decorator_list:
            self.evaluate(expr)
        self.defaults(node.args)
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if node.returns is not None or any(arg.annotation is not None for arg in arguments):
            # Annotation execution depends on Python version/future imports.
            self.unknown()
        self.values[node.name] = _Deferred()
        if node.decorator_list:
            self.complete = False

    def defaults(self, args: ast.arguments) -> None:
        for expr in [*args.defaults, *args.kw_defaults]:
            if expr is not None:
                self.evaluate(expr)

    def with_statement(self, node: ast.With) -> None:
        for item in node.items:
            self.evaluate(item.context_expr)
            if item.optional_vars is not None:
                self.invalidate(item.optional_vars)
        self.statements(node.body)

    def statement(self, node: ast.stmt) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self.import_statement(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            self.assign(node)
        elif isinstance(node, ast.Expr):
            self.evaluate(node.value)
        elif isinstance(node, ast.With):
            self.with_statement(node)
        else:
            self.other_statement(node)

    def other_statement(self, node: ast.stmt) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.definition(node)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                self.invalidate(target)
        elif isinstance(node, ast.For):
            self.literal_loop(node)
        elif not isinstance(node, ast.Pass):
            self.unknown()

    def literal_loop(self, node: ast.For) -> None:
        items = _loop_strings(node.iter)
        if not isinstance(node.target, ast.Name) or items is None:
            self.unknown()
            return
        self.loop_depth += 1
        try:
            self.loop_items(node, node.target.id, items)
        finally:
            self.loop_depth -= 1

    def loop_items(self, node: ast.For, target: str, items: tuple[str, ...]) -> None:
        for value in items:
            if not self.iterations_left:
                self.unknown()
                return
            self.iterations_left -= 1
            self.values[target] = value  # The loop target shadows any previous binding.
            if not self.statements(node.body):
                return
        self.statements(node.orelse)

    def statements(self, nodes: list[ast.stmt]) -> bool:
        for node in nodes:
            before = self.unknown_epoch
            self.statement(node)
            if self.loop_depth and self.unknown_epoch != before:
                # Never infer a subsequent iteration/read after unknown exits or flow.
                return False
        return True


def python_source_access(code: str) -> PythonSourceAccess:
    """Return bounded syntax facts; unknown coverage is explicitly incomplete."""
    if len(code) > MAX_CODE:
        return PythonSourceAccess(complete=False)
    try:
        tree = parse_module(code)
        if not _bounded_tree(tree):
            return PythonSourceAccess(complete=False)
        parser = _Parser()
        parser.statements(tree.body)
    except (SyntaxError, ValueError, RecursionError):
        return PythonSourceAccess(complete=False)
    return PythonSourceAccess(
        tuple(dict.fromkeys(parser.paths)),
        tuple(dict.fromkeys(parser.modules)),
        parser.complete,
    )
