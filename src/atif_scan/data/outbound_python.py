"""Narrow inline Python HTTP syntax; never evaluate expressions or read files.

Only module-level calls on explicitly imported, still-bound canonical modules are
supported. No aliases, persistent bindings, constructors alone, control flow or
response-derived credentials are inferred.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Locator
from .outbound_types import (
    MAX_CHARS,
    MAX_DEPTH,
    MAX_ITEMS,
    Carrier,
    Requests,
    Role,
    body,
    pairs,
    request,
    scalar,
)
from .pysource import parse_module

_URL_OPEN_ARGS = 2
_MODULES = frozenset({"requests", "httpx", "urllib.request"})
_HTTP_OPERATIONS = frozenset(
    {"get", "post", "put", "patch", "delete", "head", "options", "request"}
)
_OPTIONS = frozenset({"headers", "data", "json", "params", "timeout"})


def _literal(node: ast.AST, depth: int = 0) -> object:
    if depth > MAX_DEPTH:
        raise ValueError("too_deep")
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict) and len(node.keys) <= MAX_ITEMS:
        out = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                raise ValueError("unpacking")
            name = scalar(_literal(key, depth + 1))
            out[name] = _literal(value, depth + 1)
        return out
    if isinstance(node, ast.List | ast.Tuple) and len(node.elts) <= MAX_ITEMS:
        return [_literal(item, depth + 1) for item in node.elts]
    raise ValueError("dynamic_literal")


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _name(node.value) + "." + node.attr
    return ""


def _keywords(call: ast.Call) -> dict[str, object]:
    out = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg not in _OPTIONS or keyword.arg in out:
            raise ValueError("unsupported_keyword")
        out[keyword.arg] = _literal(keyword.value)
    return out


def _carriers(options: dict[str, object]) -> list[Carrier]:
    out = []
    for name, value in options.items():
        if name in {"headers", "params"}:
            if not isinstance(value, dict) or not all(
                isinstance(item, str) for item in value.values()
            ):
                raise ValueError("unsupported_pairs")
            role = Role.HEADER if name == "headers" else Role.QUERY
            out.extend(pairs(value, role))
        elif name in {"data", "json"}:
            out.extend(body(value))
    return out


def _http_call(call: ast.Call, at: Locator, bindings: set[str]) -> Requests:
    name = _name(call.func)
    module, _, operation = name.rpartition(".")
    if module not in bindings:
        raise ValueError("unbound_or_unknown_call")
    if (module == "urllib.request" and operation != "urlopen") or (
        module != "urllib.request" and operation not in _HTTP_OPERATIONS
    ):
        raise ValueError("unknown_http_operation")
    args = list(call.args)
    if operation == "request" and module in {"requests", "httpx"} and args:
        method = scalar(_literal(args.pop(0))).upper()
    elif module == "urllib.request" and operation == "urlopen":
        method = "POST" if len(args) > 1 or any(k.arg == "data" for k in call.keywords) else "GET"
    else:
        method = operation.upper()
    if not args or len(args) > (2 if operation == "urlopen" else 1):
        raise ValueError("unsupported_arguments")
    target = scalar(_literal(args[0]))
    options = _keywords(call)
    if len(args) == _URL_OPEN_ARGS:
        options["data"] = _literal(args[1])
    fact = request(at, target, method, _carriers(options), "python_literal")
    return Requests((fact,))


def _bind_import(statement: ast.Import, bindings: set[str]) -> bool:
    complete = True
    for alias in statement.names:
        root = (alias.asname or alias.name).split(".")[0]
        bindings.difference_update(m for m in tuple(bindings) if m.split(".")[0] == root)
        if alias.name in _MODULES and alias.asname is None:
            bindings.add(alias.name)
        else:
            complete = False
    return complete


def python_requests(source: str, at: Locator) -> Requests:
    """Literal calls only, with explicit invalidation after unsupported statements."""
    if len(source) > MAX_CHARS:
        return Requests(complete=False)
    try:
        tree = parse_module(source)
        if sum(1 for _ in ast.walk(tree)) > MAX_ITEMS * MAX_DEPTH:
            return Requests(complete=False)
    except (SyntaxError, ValueError, RecursionError):
        return Requests(complete=False)
    return _module_requests(tree, at)


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    value = (
        statement.value if isinstance(statement, ast.Expr | ast.Assign | ast.AnnAssign) else None
    )
    return value if isinstance(value, ast.Call) else None


def _inert(statement: ast.stmt) -> bool:
    # Definitions with decorators/defaults can execute calls: do not infer inertness.
    return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)


def _module_requests(tree: ast.Module, at: Locator) -> Requests:
    bindings: set[str] = set()
    facts = []
    complete = True
    for statement in tree.body[:MAX_ITEMS]:
        try:
            if isinstance(statement, ast.Import):
                complete &= _bind_import(statement, bindings)
            elif _inert(statement):
                continue
            else:
                call = _statement_call(statement)
                if call is None:
                    raise ValueError("unsupported_statement")
                facts.extend(_http_call(call, at, bindings).requests)
                # Even a request result may rebind requests or an attribute on it.
                if not isinstance(statement, ast.Expr):
                    bindings.clear()
        except (ValueError, UnicodeError, RecursionError):
            complete = False
            break
    return Requests(tuple(facts), complete and len(tree.body) <= MAX_ITEMS)
