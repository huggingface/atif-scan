"""Bounded inline Python read requests; parse syntax only, never run submitted code.

Recognize builtin open and explicitly imported h5py.File at module/with scope.
Only literal paths or preceding single-name literal assignments are resolved.
Control flow, functions, aliases assigned through expressions and external scripts
are not reconstructed. These are requests, not proof the calls completed.
"""

from __future__ import annotations

import ast
import re

from .pysource import parse_module

PYTHON = re.compile(r"python(?:[23](?:\.\d+)?)?$")
READ_MODES = frozenset({"r", "rb", "rt", "r+", "rb+", "r+b", "rt+", "r+t"})
MAX_CODE = 256 * 1024


def _string(node: ast.expr | None, literals: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return literals.get(node.id) if isinstance(node, ast.Name) else None


def _argument(call: ast.Call, index: int, name: str) -> ast.expr | None:
    if len(call.args) > index:
        return call.args[index]
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def _call_path(call: ast.Call, literals: dict[str, str], readers: set[str]) -> str | None:
    if any(kw.arg is None for kw in call.keywords):
        return None
    name = ast.unparse(call.func)
    if name not in readers:
        return None
    path_key = "file" if name == "open" else "name"
    mode_node = _argument(call, 1, "mode")
    mode = _string(mode_node, literals) if mode_node is not None else "r"
    if mode not in READ_MODES:
        return None
    return _string(_argument(call, 0, path_key), literals)


def _calls(expr: ast.expr) -> list[ast.Call]:
    # Never infer invocation from function/lambda bodies or deferred comprehensions.
    if isinstance(expr, (ast.Lambda, ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
        return []
    found: list[ast.Call] = [expr] if isinstance(expr, ast.Call) else []
    for child in ast.iter_child_nodes(expr):
        if isinstance(child, ast.expr):
            found.extend(_calls(child))
    return found


def _invalidate(target: ast.AST, literals: dict[str, str], readers: set[str]) -> None:
    for node in ast.walk(target):
        if isinstance(node, ast.Name):
            literals.pop(node.id, None)
            readers.difference_update(r for r in tuple(readers) if r.split(".")[0] == node.id)


def _import(stmt: ast.Import | ast.ImportFrom, readers: set[str]) -> None:
    for alias in stmt.names:
        if alias.name == "*":
            readers.clear()
            continue
        local = alias.asname or alias.name.split(".")[0]
        readers.difference_update(r for r in tuple(readers) if r.split(".")[0] == local)
        if isinstance(stmt, ast.Import) and alias.name == "h5py":
            readers.add(local + ".File")
        elif isinstance(stmt, ast.ImportFrom) and stmt.module == "h5py" and alias.name == "File":
            readers.add(local)


def _expressions(stmt: ast.stmt) -> list[ast.expr]:
    if isinstance(stmt, (ast.Expr, ast.Assign, ast.AnnAssign)) and stmt.value is not None:
        return [stmt.value]
    if isinstance(stmt, ast.With) and len(stmt.items) == 1:
        return [stmt.items[0].context_expr]
    return []


def _assign(stmt: ast.Assign, literals: dict[str, str], readers: set[str]) -> None:
    value = _string(stmt.value, literals)
    for target in stmt.targets:
        _invalidate(target, literals, readers)
    if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and value is not None:
        literals[stmt.targets[0].id] = value


def _statement_paths(stmt: ast.stmt, literals: dict[str, str], readers: set[str]) -> list[str]:
    paths = []
    for expr in _expressions(stmt):
        # Do not interpret conditional execution, even with constant conditions.
        # Skip the whole expression: a branch/argument may mutate trusted bindings
        # before another call in it. Assignment expressions have the same risk.
        if any(isinstance(n, (ast.BoolOp, ast.IfExp, ast.NamedExpr)) for n in ast.walk(expr)):
            literals.clear()
            readers.clear()
            continue
        for call in _calls(expr):
            path = _call_path(call, literals, readers)
            if path is not None:
                paths.append(path)
    return paths


def _update(stmt: ast.stmt, literals: dict[str, str], readers: set[str]) -> list[str]:
    if isinstance(stmt, ast.Assign):
        _assign(stmt, literals, readers)
    elif isinstance(stmt, ast.With):
        if len(stmt.items) != 1:
            literals.clear()
            readers.clear()
            return []  # context-manager bindings can change subsequent read targets
        for item in stmt.items:
            if item.optional_vars is not None:
                _invalidate(item.optional_vars, literals, readers)
        return _statements(stmt.body, literals, readers)
    elif not isinstance(stmt, ast.Expr):
        # Unsupported control flow/rebindings must not leave stale constants/readers.
        literals.clear()
        readers.clear()
    return []


def _statements(
    statements: list[ast.stmt], literals: dict[str, str], readers: set[str]
) -> list[str]:
    paths = []
    for stmt in statements:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            _import(stmt, readers)
            literals.clear()
            continue
        paths.extend(_statement_paths(stmt, literals, readers))
        paths.extend(_update(stmt, literals, readers))
    return paths


def python_paths(code: str) -> tuple[str, ...]:
    """Return bounded recognized read targets; absence never establishes coverage."""
    if len(code) > MAX_CODE:
        return ()
    try:
        tree = parse_module(code)
        return tuple(_statements(tree.body, {}, {"open"}))
    except (SyntaxError, ValueError, RecursionError):
        return ()
