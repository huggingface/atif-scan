"""Architecture: package layering and no import cycles (function-level imports included).

Layers, lowest first. A layer may import itself and the layers listed for it:

- data      parsing and per-trial facts (stdlib only): the data package (model, loader,
            jsonval, shell, jslit, credentials, facts, accounting, paths, web_*)
- sources   finding and fetching inputs: the sources package (inputs, sync, layout, harbor)
- analysis  checks and their evaluation: checks, rules, policy, engine, access, detectors,
            packs

- evidence  masked excerpts and history for people and judges: the evidence package
- review    questions and labels: the review package
- output    the report document, views, the run brief and judge bundles: the output package

The CLI and the result cache sit above everything and are not constrained. Imports under
`if TYPE_CHECKING:` don't count: they never run.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "atif_scan"
DATA = {"data"}
SOURCES = {"sources"}
ANALYSIS = {"checks", "rules", "policy", "engine", "access"}


LAYER_OF = {
    **dict.fromkeys(DATA, "data"),
    **dict.fromkeys(SOURCES, "sources"),
    **dict.fromkeys((*ANALYSIS, "detectors", "packs"), "analysis"),
    "evidence": "evidence",
    "review": "review",
    "output": "output",
}


def layer(module: str) -> str | None:
    return LAYER_OF.get(module.split(".", maxsplit=1)[0])


ALLOWED = {
    "data": {"data"},
    "sources": {"data", "sources"},
    "analysis": {"data", "analysis"},
    "evidence": {"data", "analysis", "evidence"},
    "review": {"data", "analysis", "evidence", "review"},
    "output": {"data", "sources", "analysis", "evidence", "review", "output"},
}


def _module(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    if rel.parts == ("__init__",):
        return "__init__"  # the package root
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


def _type_checking_lines(tree: ast.Module) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            for child in node.body:
                lines.update(range(child.lineno, (child.end_lineno or child.lineno) + 1))
    return lines


def _target(module: str, is_package: bool, node: ast.ImportFrom) -> list[str]:
    """Package-internal modules an import names (relative or `atif_scan.` absolute)."""
    if node.level == 0:
        name = node.module or ""
        if not name.startswith("atif_scan"):
            return []
        base = name.removeprefix("atif_scan").lstrip(".")
    else:
        parts = [] if module == "__init__" else module.split(".")
        package = parts if is_package else parts[:-1]
        package = package[: len(package) - (node.level - 1)] if node.level > 1 else package
        base = ".".join([*package, *([node.module] if node.module else [])])
    if node.module is None or not base:  # `from . import x` / `from .. import x`
        return [f"{base}.{a.name}".strip(".") for a in node.names]
    return [base]


def imports() -> dict[str, set[str]]:
    modules = {_module(p): p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts}
    graph: dict[str, set[str]] = {m: set() for m in modules}
    for module, path in modules.items():
        tree = ast.parse(path.read_text())
        skip = _type_checking_lines(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.lineno not in skip:
                for named in _target(module, path.name == "__init__.py", node):
                    # `from .detectors.integrity import x` and `from .checks import Status`
                    # both name modules: keep the longest known prefix.
                    found = named
                    while found and found not in modules:
                        found = found.rpartition(".")[0]
                    if found and found != module:
                        graph[module].add(found)
    return graph


def test_import_graph_is_found():
    graph = imports()
    assert "engine" in graph["__init__"]
    assert "detectors.builtin" in graph["detectors"]


def test_no_import_cycles():
    graph = imports()
    # Packages import their own submodules in __init__; a submodule importing a sibling
    # through the package is the cycle that matters, so drop package -> own child edges.
    edges = {m: {t for t in ts if not t.startswith(f"{m}.")} for m, ts in graph.items()}
    seen: set[str] = set()
    stack: list[str] = []
    cycles: list[list[str]] = []

    def visit(node: str) -> None:
        if node in stack:
            cycles.append([*stack[stack.index(node) :], node])
            return
        if node in seen:
            return
        seen.add(node)
        stack.append(node)
        for nxt in sorted(edges.get(node, ())):
            visit(nxt)
        stack.pop()

    for node in sorted(edges):
        visit(node)
    assert not cycles, cycles


def test_layers_only_import_downwards():
    violations = []
    for module, targets in imports().items():
        own = layer(module)
        if own is None:
            continue
        for target in targets:
            other = layer(target)
            if other is not None and other not in ALLOWED[own]:
                violations.append(f"{module} ({own}) -> {target} ({other})")
            elif other is None and target != "":
                violations.append(f"{module} ({own}) -> {target} (above the layers)")
    assert not violations, violations
