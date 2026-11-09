"""Synthetic source strings only: no submitted code is executed."""

from dataclasses import FrozenInstanceError

import pytest

from atif_scan.data.python_source_access import (
    MAX_CODE,
    MAX_DEPTH,
    MAX_LOOP_ITERATIONS,
    MAX_NODES,
    ModuleSource,
    PythonSourceAccess,
    python_source_access,
)


def test_public_frozen_contract() -> None:
    assert PythonSourceAccess() == PythonSourceAccess((), (), True)
    assert ModuleSource("demo.widget") == ModuleSource("demo.widget", ())
    result = python_source_access("open('demo.py')")
    attribute = "complete"
    with pytest.raises(FrozenInstanceError):
        setattr(result, attribute, False)


@pytest.mark.parametrize("mode", ["r", "rb", "rt", "r+", "rb+", "r+b", "rt+", "r+t"])
def test_builtin_open_read_modes(mode: str) -> None:
    result = python_source_access(f"open('demo.py', {mode!r}).read()")
    assert result == PythonSourceAccess(paths=("demo.py",))


@pytest.mark.parametrize("mode", ["w", "wb", "a", "x", "w+", "a+", "x+"])
def test_write_modes_not_original_source_requests(mode: str) -> None:
    assert python_source_access(f"open('demo.py', {mode!r})") == PythonSourceAccess()


def test_literal_assigned_paths_keywords_and_deduplication() -> None:
    code = """
root = '/synthetic/'
name = 'widget' + '.py'
path = root + name
mode = 'rb'
open(file=path, mode=mode)
open(path)
"""
    assert python_source_access(code) == PythonSourceAccess(paths=("/synthetic/widget.py",))


@pytest.mark.parametrize(
    "code",
    [
        "from pathlib import Path; Path('/synthetic/widget.py').read_text()",
        "import pathlib as p; p.Path('/synthetic', 'widget.py').read_bytes()",
        "from pathlib import Path as P; f=P('/synthetic')/'widget.py'; f.read_text()",
        "from pathlib import Path; root=Path('/synthetic'); "
        "root.joinpath('widget.py').read_bytes()",
        "from pathlib import Path; f='/synthetic/widget.py'; Path(f).read_text()",
        "import os.path as p; open(p.join('/synthetic', 'widget.py'))",
        "from os.path import join as j; open(j('/synthetic', 'widget.py'))",
        "from pathlib import Path; open(Path('/synthetic') / 'widget.py')",
        "from builtins import open as read; read('/synthetic/widget.py')",
    ],
)
def test_path_constructors_and_simple_joins(code: str) -> None:
    assert python_source_access(code) == PythonSourceAccess(paths=("/synthetic/widget.py",))


@pytest.mark.parametrize(
    ("imports", "expression", "canonical"),
    [
        ("import inspect; import demo", "inspect.getsource(demo)", "demo"),
        ("import inspect as i; import demo.widget as w", "i.getsourcelines(w)", "demo.widget"),
        (
            "from inspect import getsource as src; from demo.widget import Widget as W",
            "src(W)",
            "demo.widget.Widget",
        ),
        (
            "import inspect; import demo.widget",
            "inspect.getsource(demo.widget.Widget)",
            "demo.widget.Widget",
        ),
        (
            "from inspect import getsourcelines; from demo import widget",
            "getsourcelines(object=widget.Widget.run)",
            "demo.widget.Widget.run",
        ),
    ],
)
def test_import_aliases_and_canonical_objects(
    imports: str, expression: str, canonical: str
) -> None:
    assert python_source_access(imports + "; " + expression) == PythonSourceAccess(
        modules=(ModuleSource(canonical),)
    )


def test_import_time_search_paths_not_retroactive() -> None:
    code = """
import sys, inspect
import demo.before as old
sys.path.insert(0, '/synthetic/site')
from demo.after import Widget as New
inspect.getsource(old)
inspect.getsource(New)
inspect.getsourcelines(old)
"""
    assert python_source_access(code) == PythonSourceAccess(
        modules=(
            ModuleSource("demo.before"),
            ModuleSource("demo.after.Widget", ("/synthetic/site",)),
        )
    )


def test_unaliased_dotted_imports_keep_member_timestamps() -> None:
    code = """
import sys, inspect
import demo.before
sys.path.insert(0, '/synthetic/site')
import demo.after
inspect.getsource(demo.before.Widget)
inspect.getsource(demo.after.Widget)
inspect.getsource(demo)
"""
    assert python_source_access(code) == PythonSourceAccess(
        modules=(
            ModuleSource("demo.before.Widget"),
            ModuleSource("demo.after.Widget", ("/synthetic/site",)),
            ModuleSource("demo"),
        )
    )


def test_search_path_aliases_assignments_and_order_are_candidates() -> None:
    code = """
import sys as s
from inspect import getsource
site = '/synthetic/site'
s.path.append('/synthetic/fallback')
s.path.insert(0, site)
import demo as d
getsource(d)
"""
    assert python_source_access(code) == PythonSourceAccess(
        modules=(ModuleSource("demo", ("/synthetic/site", "/synthetic/fallback")),)
    )


def test_inert_text_and_ordinary_introspection_are_not_source_requests() -> None:
    code = """
# open('/synthetic/comment.py')
text = "Path('/synthetic/string.py').read_text()"
import inspect, demo
print('/synthetic/literal.py')
print(demo.__file__, demo.__version__)
inspect.signature(demo.Widget)
def inert():
    open('/synthetic/function.py')
    inspect.getsource(demo)
thunk = lambda: open('/synthetic/lambda.py')
"""
    assert python_source_access(code) == PythonSourceAccess()


def test_with_and_nested_expression_calls_are_executed_requests() -> None:
    code = """
from pathlib import Path
with open('/synthetic/a.py') as stream:
    print(Path('/synthetic/b.py').read_text())
    with open('/synthetic/c.py', 'rb'):
        pass
"""
    assert python_source_access(code) == PythonSourceAccess(
        paths=("/synthetic/a.py", "/synthetic/b.py", "/synthetic/c.py")
    )


def test_function_defaults_not_function_bodies() -> None:
    code = """
def inert(x=open('/synthetic/default.py')):
    open('/synthetic/body.py')
"""
    assert python_source_access(code) == PythonSourceAccess(paths=("/synthetic/default.py",))


@pytest.mark.parametrize(
    "code",
    [
        "open(dynamic)",
        "open('demo.py', mode=dynamic)",
        "open('demo.py', **options)",
        "open(*args)",
        "from pathlib import Path; Path(dynamic).read_text()",
        "import inspect; inspect.getsource(dynamic)",
        "import inspect; inspect.getsource('demo.widget')",
        "import inspect, demo; inspect.getsource(demo.__file__)",
        "import inspect, demo; demo=other; inspect.getsource(demo)",
        "import inspect, demo; inspect=other; inspect.getsource(demo)",
        "from inspect import getsource as src; import demo; src=other; src(demo)",
        "open=custom; open('demo.py')",
        "from builtins import open as read; read=custom; read('demo.py')",
        "from pathlib import Path; Path=custom; Path('demo.py').read_text()",
        "import sys; sys.path.insert(0, dynamic)",
        "import sys; sys.path.insert(1, '/synthetic/site')",
        "import sys; sys.path.insert(False, '/synthetic/site')",
        "from demo import *",
        "from . import widget",
    ],
)
def test_dynamic_and_invalidated_targets_are_unknown_not_facts(code: str) -> None:
    result = python_source_access(code)
    assert not result.complete
    assert result.paths == ()
    assert result.modules == ()


@pytest.mark.parametrize(
    "statement",
    [
        "if flag:\n    path = '/synthetic/new.py'",
        "for path in values:\n    pass",
        "try:\n    path = '/synthetic/new.py'\nexcept Exception:\n    pass",
        "path += suffix",
        "del path",
        "with custom() as path:\n    pass",
    ],
)
def test_unsupported_or_rebinding_does_not_keep_stale_paths(statement: str) -> None:
    code = f"path = '/synthetic/old.py'\n{statement}\nopen(path)"
    result = python_source_access(code)
    assert result.paths == ()
    assert not result.complete


@pytest.mark.parametrize(
    "code",
    [
        "if flag:\n    open('/synthetic/conditional.py')",
        "flag and open('/synthetic/conditional.py')",
        "open('/synthetic/a.py') if flag else open('/synthetic/b.py')",
        "[open('/synthetic/comprehension.py') for x in values]",
        "(open('/synthetic/generator.py') for x in values)",
        "class Inert:\n    def method(self):\n        open('/synthetic/method.py')",
    ],
)
def test_unsupported_control_flow_never_collects_body_facts(code: str) -> None:
    result = python_source_access(code)
    assert result.paths == ()
    assert not result.complete


def test_prior_facts_survive_unknown_and_subsequent_explicit_imports_work() -> None:
    code = """
open('/synthetic/known.py')
if flag:
    pass
from pathlib import Path
Path('/synthetic/later.py').read_text()
"""
    assert python_source_access(code) == PythonSourceAccess(
        paths=("/synthetic/known.py", "/synthetic/later.py"), complete=False
    )


def test_attribute_rebinding_invalidates_import_provenance() -> None:
    result = python_source_access("""
import inspect, demo
demo.Widget = replacement
inspect.getsource(demo.Widget)
""")
    assert result == PythonSourceAccess(complete=False)


@pytest.mark.parametrize("code", ["def (", "\x00", "x = " + " " * MAX_CODE])
def test_invalid_or_oversized_parse_is_typed_unknown(code: str) -> None:
    assert python_source_access(code) == PythonSourceAccess(complete=False)


def test_node_and_depth_budgets() -> None:
    wide = "x=0\n" * MAX_NODES
    deep = "x=" + "+".join(["1"] * (MAX_DEPTH + 1))
    assert python_source_access(wide) == PythonSourceAccess(complete=False)
    assert python_source_access(deep) == PythonSourceAccess(complete=False)


def test_exponential_literal_expansion_is_bounded() -> None:
    code = "p='x'\n" + "p=p+p\n" * 30 + "open(p)"
    assert python_source_access(code) == PythonSourceAccess(complete=False)


def test_lambda_defaults_execute_but_body_is_inert() -> None:
    code = "thunk = lambda x=open('/synthetic/default.py'): open('/synthetic/body.py')"
    assert python_source_access(code) == PythonSourceAccess(paths=("/synthetic/default.py",))


@pytest.mark.parametrize(
    "definition",
    ["def helper():\n    open('/synthetic/body.py')", "helper=lambda: open('/synthetic/body.py')"],
)
def test_called_local_bodies_are_unknown_not_traversed(definition: str) -> None:
    code = "from pathlib import Path\n" + definition + "\nhelper()\nPath('stale.py').read_text()"
    assert python_source_access(code) == PythonSourceAccess(complete=False)


@pytest.mark.parametrize("call", ["exec(dynamic)", "eval(dynamic)", "__import__(dynamic)"])
def test_dynamic_execution_is_not_executed_or_claimed_complete(call: str) -> None:
    assert python_source_access(call) == PythonSourceAccess(complete=False)


def test_unsupported_search_path_mutations_do_not_leave_stale_candidates() -> None:
    code = """
import sys, inspect
sys.path.insert(0, '/synthetic/site')
sys.path.clear()
import demo
inspect.getsource(demo)
"""
    assert python_source_access(code) == PythonSourceAccess(
        modules=(ModuleSource("demo"),), complete=False
    )


@pytest.mark.parametrize(
    "code",
    [
        "import custom; custom.read_text(dynamic)",
        "from pathlib import Path; Path.read_text(dynamic)",
        "import custom; custom.getsource(dynamic)",
    ],
)
def test_unsupported_source_shaped_imported_calls_are_unknown(code: str) -> None:
    assert python_source_access(code) == PythonSourceAccess(complete=False)


def test_literal_loop_reads_after_separate_exists_stat_probe() -> None:
    code = """
from pathlib import Path
p = Path('/synthetic/site') / 'demo'
for f in ['events.py', '_example.py']:
    print((p / f).exists(), (p / f).stat().st_size)
for f in ['events.py', '_example.py']:
    s = (p / f).read_text()
    print(s.find(needle))
"""
    assert python_source_access(code) == PythonSourceAccess(
        paths=("/synthetic/site/demo/events.py", "/synthetic/site/demo/_example.py")
    )


def test_literal_loop_exists_stat_alone_are_not_reads() -> None:
    code = """
from pathlib import Path
p = Path('/synthetic/site') / 'demo'
for f in ['events.py', '_example.py']:
    print((p / f).exists(), (p / f).stat().st_size)
"""
    assert python_source_access(code) == PythonSourceAccess()


def test_literal_loop_builtin_open_and_final_target_binding() -> None:
    code = """
for f in ('a.py', 'b.py'):
    open(f, 'rb')
open(f)
"""
    assert python_source_access(code) == PythonSourceAccess(paths=("a.py", "b.py"))


def test_bounded_nested_literal_loops() -> None:
    code = """
from pathlib import Path
for folder in ['a', 'b']:
    for name in ['one.py', 'two.py']:
        (Path('/synthetic') / folder / name).read_bytes()
"""
    assert python_source_access(code) == PythonSourceAccess(
        paths=(
            "/synthetic/a/one.py",
            "/synthetic/a/two.py",
            "/synthetic/b/one.py",
            "/synthetic/b/two.py",
        )
    )


def test_literal_loop_rebinding_keeps_sequential_path_values() -> None:
    code = """
from pathlib import Path
p = Path('/synthetic/a')
for name in ['one.py', 'two.py']:
    (p / name).read_text()
    p = Path('/synthetic/b')
(p / name).read_text()
"""
    assert python_source_access(code) == PythonSourceAccess(
        paths=("/synthetic/a/one.py", "/synthetic/b/two.py")
    )


@pytest.mark.parametrize("target", ["Path", "p"])
def test_literal_loop_shadows_prior_path_or_constructor(target: str) -> None:
    code = f"""
from pathlib import Path
p = Path('/synthetic')
for {target} in ['one.py', 'two.py']:
    (p / 'not-proven.py').read_text()
"""
    result = python_source_access(code)
    # Path shadowing leaves the already-constructed p intact; p shadowing does not.
    expected = ("/synthetic/not-proven.py",) if target == "Path" else ()
    assert result.paths == expected
    assert result.complete is (target == "Path")


def test_literal_loop_shadows_imported_module_without_stale_facts() -> None:
    code = """
import inspect, demo
for demo in ['one.py', 'two.py']:
    inspect.getsource(demo)
"""
    assert python_source_access(code) == PythonSourceAccess(complete=False)


def test_literal_loop_import_time_search_path_snapshots() -> None:
    code = """
import sys, inspect
import demo.before as before
for site in ['/synthetic/one', '/synthetic/two']:
    sys.path.insert(0, site)
    import demo.after as after
    inspect.getsource(before)
    inspect.getsource(after)
"""
    assert python_source_access(code) == PythonSourceAccess(
        modules=(
            ModuleSource("demo.before"),
            ModuleSource("demo.after", ("/synthetic/one",)),
            ModuleSource("demo.after", ("/synthetic/two", "/synthetic/one")),
        )
    )


@pytest.mark.parametrize("exit_statement", ["break", "continue", "raise RuntimeError", "return"])
def test_literal_loop_unknown_exit_stops_later_body_and_iterations(exit_statement: str) -> None:
    code = f"""
for name in ['one.py', 'two.py']:
    open(name)
    {exit_statement}
    open('not-proven.py')
"""
    assert python_source_access(code) == PythonSourceAccess(paths=("one.py",), complete=False)


def test_unknown_exit_in_nested_with_stops_loop_body() -> None:
    code = """
for name in ['one.py', 'two.py']:
    with open(name):
        break
        open('not-proven-inside.py')
    open('not-proven-outside.py')
"""
    assert python_source_access(code) == PythonSourceAccess(paths=("one.py",), complete=False)


@pytest.mark.parametrize(
    "iterable",
    ["dynamic", "['one.py', dynamic]", "['one.py', 3]", "range(2)", "'one.py'", "[*dynamic]"],
)
def test_nonliteral_loop_iterables_remain_unknown(iterable: str) -> None:
    code = f"for name in {iterable}:\n    open(name)"
    assert python_source_access(code) == PythonSourceAccess(complete=False)


def test_empty_literal_loop_keeps_existing_target_and_runs_else_only() -> None:
    code = """
name = 'before.py'
for name in []:
    open('inert.py')
else:
    open(name)
"""
    assert python_source_access(code) == PythonSourceAccess(paths=("before.py",))


def test_literal_loop_else_runs_after_completed_iterations() -> None:
    code = """
for name in ['one.py']:
    open(name)
else:
    open('else.py')
"""
    assert python_source_access(code) == PythonSourceAccess(paths=("one.py", "else.py"))


def test_literal_loop_after_unknown_program_still_captures_explicit_reads() -> None:
    code = """
if flag:
    pass
from pathlib import Path
for name in ['one.py', 'two.py']:
    Path(name).read_text()
"""
    assert python_source_access(code) == PythonSourceAccess(
        paths=("one.py", "two.py"), complete=False
    )


def test_literal_loop_iteration_limit() -> None:
    values = repr([f"{n}.py" for n in range(MAX_LOOP_ITERATIONS + 1)])
    code = f"for name in {values}:\n    open(name)"
    assert python_source_access(code) == PythonSourceAccess(complete=False)


def test_nested_literal_loop_total_budget() -> None:
    code = """
for root in ['a', 'b']:
    for name in ['one.py', 'two.py']:
        for suffix in ['x', 'y', 'z', 'w', 'v', 'u', 't', 's']:
            open(root + '/' + name + suffix)
"""
    result = python_source_access(code)
    assert not result.complete
    assert 0 < len(result.paths) <= MAX_LOOP_ITERATIONS
    assert "b/two.pys" not in result.paths


def test_literal_loop_unknown_control_flow_does_not_guess_later_reads() -> None:
    code = """
for name in ['one.py', 'two.py']:
    open(name)
    if flag:
        break
    open('not-proven.py')
"""
    assert python_source_access(code) == PythonSourceAccess(paths=("one.py",), complete=False)


def test_literal_loop_exact_iteration_limit_is_complete() -> None:
    values = repr([f"{n}.py" for n in range(MAX_LOOP_ITERATIONS)])
    code = f"for name in {values}:\n    open(name)"
    result = python_source_access(code)
    assert result.complete
    assert len(result.paths) == MAX_LOOP_ITERATIONS


def test_literal_loop_does_not_interpret_function_or_lambda_bodies() -> None:
    code = """
def inert():
    for name in ['body.py']:
        open(name)
thunk = lambda: open('lambda.py')
for name in ['one.py']:
    open(name)
"""
    assert python_source_access(code) == PythonSourceAccess(paths=("one.py",))


def test_literal_loop_path_constructor_rebinding_stays_invalidated() -> None:
    code = """
from pathlib import Path
for name in ['one.py', 'two.py']:
    Path(name).read_text()
    Path = replacement
"""
    assert python_source_access(code) == PythonSourceAccess(paths=("one.py",), complete=False)


def test_literal_loop_nested_unknown_exit_stops_outer_iterations() -> None:
    code = """
for folder in ['a', 'b']:
    for name in ['one.py', 'two.py']:
        open(folder + '/' + name)
        continue
    open('not-proven.py')
"""
    assert python_source_access(code) == PythonSourceAccess(paths=("a/one.py",), complete=False)


@pytest.mark.parametrize(
    "header",
    ["for a, b in ['one.py']:", "async for name in ['one.py']:"],
)
def test_unsupported_loop_forms_are_unknown_without_body_facts(header: str) -> None:
    code = header + "\n    open('not-proven.py')"
    assert python_source_access(code) == PythonSourceAccess(complete=False)
