"""Small, syntax-only filesystem call facts for directly supplied Python modules.

Operations are lexical requests, not proof of execution or success: branches,
exceptions, breaks and reachability are not evaluated. ``repeated`` means nested
in a literal ``while True``/``while 1`` body, not a guaranteed second execution.
Only the documented call spellings and string literals are recognized; imports,
aliases, bindings, path expressions and user functions are not resolved. In
particular, names are assumed to denote their usual APIs, not proven to do so.

``complete`` describes extraction within this limited vocabulary, NOT coverage
of arbitrary Python effects. Computed arguments and traversal/parsing failures
leave it false. Definition bodies and deferred expressions are not executed or
searched for operations. Paths are private evidence, never report-ready fields.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .pysource import parse_module


@dataclass(frozen=True)
class PythonOperation:
    path: str
    writes: bool
    repeated: bool


@dataclass(frozen=True)
class PythonActivity:
    operations: tuple[PythonOperation, ...]
    complete: bool
    # An unresolved filesystem operation specifically inside a literal forever loop.
    repeated_unread: bool = False


_PATH_METHODS = {
    "read_text": False,
    "read_bytes": False,
    "write_text": True,
    "write_bytes": True,
    "unlink": True,
}
_OS_CALLS = {
    "os.remove": True,
    "os.unlink": True,
    "os.rmdir": True,
    "os.listdir": False,
    "os.scandir": False,
}
_COPY_CALLS = frozenset({"shutil.copy", "shutil.copyfile", "shutil.copy2", "shutil.move"})


def _name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return ""


def _string(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _argument(call: ast.Call, index: int, name: str) -> ast.expr | None:
    # Unpacking can change argument positions or duplicate a keyword. Do not guess.
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
        kw.arg is None for kw in call.keywords
    ):
        return None
    values = [kw.value for kw in call.keywords if kw.arg == name]
    if len(call.args) > index:
        values.append(call.args[index])
    return values[0] if len(values) == 1 else None


def _literal_forever(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and (
        node.value is True or (type(node.value) is int and node.value == 1)
    )


def _definition_expressions(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda,
) -> list[ast.expr]:
    """Headers can run even though bodies are deliberately outside our scope."""
    if isinstance(node, ast.ClassDef):
        return [*node.decorator_list, *node.bases, *(kw.value for kw in node.keywords)]
    defaults = [*node.args.defaults, *(v for v in node.args.kw_defaults if v is not None)]
    return defaults if isinstance(node, ast.Lambda) else [*node.decorator_list, *defaults]


@dataclass
class _Collector:
    operations: list[PythonOperation] = field(default_factory=list)
    complete: bool = True
    repeated_unread: bool = False

    def unread(self, repeated: bool) -> None:
        self.complete = False
        self.repeated_unread |= repeated

    def record(self, path: ast.expr | None, writes: bool, repeated: bool) -> None:
        literal = _string(path)
        if literal is None:
            self.unread(repeated)
        else:
            self.operations.append(PythonOperation(literal, writes, repeated))

    def open_call(self, call: ast.Call, repeated: bool) -> None:
        mode = _argument(call, 1, "mode")
        # The default applies only if no positional/keyword mode or unpacking exists.
        default = len(call.args) <= 1 and all(kw.arg not in {None, "mode"} for kw in call.keywords)
        literal = "r" if mode is None and default else _string(mode)
        if literal is None:
            self.unread(repeated)
            return  # Unknown mode cannot safely be labelled read-only or writable.
        self.record(_argument(call, 0, "file"), any(c in literal for c in "wax+"), repeated)

    def call(self, call: ast.Call, repeated: bool) -> None:
        name = _name(call.func)
        if name in {"open", "io.open"}:
            self.open_call(call, repeated)
        elif name in _OS_CALLS:
            self.record(_argument(call, 0, "path"), _OS_CALLS[name], repeated)
        elif name in _COPY_CALLS:
            self.record(_argument(call, 0, "src"), name == "shutil.move", repeated)
            self.record(_argument(call, 1, "dst"), True, repeated)
        elif isinstance(call.func, ast.Attribute) and call.func.attr in _PATH_METHODS:
            constructor = call.func.value
            if isinstance(constructor, ast.Call) and _name(constructor.func) in {
                "Path",
                "pathlib.Path",
            }:
                # Joining multiple components is a computed path, not one literal.
                path = constructor.args[0] if len(constructor.args) == 1 else None
                if constructor.keywords:
                    path = None
                self.record(path, _PATH_METHODS[call.func.attr], repeated)

    def while_loop(self, node: ast.While, repeated: bool) -> None:
        self.visit(node.test, repeated)
        for statement in node.body:
            self.visit(statement, repeated or _literal_forever(node.test))
        for statement in node.orelse:
            self.visit(statement, repeated)

    def visit(self, node: ast.AST, repeated: bool = False) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            for expression in _definition_expressions(node):
                self.visit(expression, repeated)
        elif isinstance(node, ast.GeneratorExp):
            # Only the outer iterator is evaluated when a generator is created.
            self.visit(node.generators[0].iter, repeated)
            self.unread(repeated)  # Consumption, if any, is outside this analysis.
        elif isinstance(node, ast.While):
            self.while_loop(node, repeated)
        else:
            if isinstance(node, ast.Call):
                self.call(node, repeated)
            for child in ast.iter_child_nodes(node):
                self.visit(child, repeated)


def python_activity(source: str) -> PythonActivity:
    """Extract scoped lexical operations without executing source or opening paths.

    Results retain duplicates in syntax-tree order (copy source before destination).
    Known operations survive a traversal limit, but coverage then remains unknown.
    """
    collector = _Collector()
    try:
        collector.visit(parse_module(source))
    except (SyntaxError, ValueError, RecursionError):
        collector.complete = False
    return PythonActivity(
        tuple(collector.operations), collector.complete, collector.repeated_unread
    )
