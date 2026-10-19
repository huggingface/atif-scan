"""Synthetic source only: no submitted Python is executed or paths opened."""

import ast
from dataclasses import FrozenInstanceError

import pytest

from atif_scan.data import python_activity as activity_module
from atif_scan.data.python_activity import PythonActivity, PythonOperation, python_activity


@pytest.mark.parametrize("reader", ["open", "io.open"])
@pytest.mark.parametrize("arguments", ["'synthetic.txt'", "file='synthetic.txt'"])
def test_default_open_is_a_read(reader, arguments):
    assert python_activity(f"{reader}({arguments})") == PythonActivity(
        (PythonOperation("synthetic.txt", False, False),), True
    )


@pytest.mark.parametrize("reader", ["open", "io.open"])
@pytest.mark.parametrize("mode", ["r", "rb", "rt", "w", "wb", "a", "ab", "x", "r+", "rb+"])
@pytest.mark.parametrize("keyword", [False, True])
def test_literal_open_modes(reader, mode, keyword):
    arguments = f"file='synthetic.txt', mode={mode!r}" if keyword else f"'synthetic.txt', {mode!r}"
    assert python_activity(f"{reader}({arguments})") == PythonActivity(
        (PythonOperation("synthetic.txt", any(c in mode for c in "wax+"), False),), True
    )


@pytest.mark.parametrize("constructor", ["Path", "pathlib.Path"])
@pytest.mark.parametrize(
    ("method", "arguments", "writes"),
    [
        ("read_text", "encoding='utf-8'", False),
        ("read_bytes", "", False),
        ("write_text", "'synthetic contents'", True),
        ("write_bytes", "b'synthetic contents'", True),
        ("unlink", "missing_ok=True", True),
    ],
)
def test_path_methods(constructor, method, arguments, writes):
    result = python_activity(f"{constructor}('synthetic.txt').{method}({arguments})")
    assert result == PythonActivity((PythonOperation("synthetic.txt", writes, False),), True)


@pytest.mark.parametrize(
    ("name", "writes"),
    [("remove", True), ("unlink", True), ("rmdir", True), ("listdir", False), ("scandir", False)],
)
@pytest.mark.parametrize("arguments", ["'synthetic'", "path='synthetic'"])
def test_os_calls(name, writes, arguments):
    assert python_activity(f"os.{name}({arguments})") == PythonActivity(
        (PythonOperation("synthetic", writes, False),), True
    )


@pytest.mark.parametrize("name", ["copy", "copyfile", "copy2", "move"])
@pytest.mark.parametrize(
    "arguments", ["'source', 'destination'", "dst='destination', src='source'"]
)
def test_copy_and_move_have_two_operations(name, arguments):
    assert python_activity(f"shutil.{name}({arguments})") == PythonActivity(
        (
            PythonOperation("source", name == "move", False),
            PythonOperation("destination", True, False),
        ),
        True,
    )


@pytest.mark.parametrize("condition", ["True", "1", "False", "0", "1.0", "2", "ready", "bool(1)"])
def test_only_literal_forever_loops_mark_their_body(condition):
    result = python_activity(
        f"open('before')\nwhile {condition}:\n open('inside', 'w')\nelse:\n open('elsewhere')\n"
        "open('after')"
    )
    assert result == PythonActivity(
        (
            PythonOperation("before", False, False),
            PythonOperation("inside", True, condition in {"True", "1"}),
            PythonOperation("elsewhere", False, False),
            PythonOperation("after", False, False),
        ),
        True,
    )


def test_nested_loop_context_is_restored_not_global():
    result = python_activity(
        "for item in items:\n"
        " open('for')\n"
        " while True:\n"
        "  if ready:\n"
        "   with open('nested') as stream:\n"
        "    while ready:\n"
        "     os.unlink('deep')\n"
        "  break\n"
        " open('outside')\n"
        "while open('condition'):\n"
        " open('conditional-body')\n"
    )
    assert result.operations == (
        PythonOperation("for", False, False),
        PythonOperation("nested", False, True),
        PythonOperation("deep", True, True),
        PythonOperation("outside", False, False),
        PythonOperation("condition", False, False),
        PythonOperation("conditional-body", False, False),
    )
    assert result.complete


@pytest.mark.parametrize(
    "source",
    [
        "def dormant():\n while True:\n  open('hidden', 'w')",
        "async def dormant():\n open('hidden', 'w')",
        "class Ignored:\n open('hidden', 'w')\n def method(self):\n  os.unlink('hidden')",
        "callback = lambda: open('hidden')",
        "text = \"open('hidden', 'w')\"\n# os.unlink('hidden')",
        "def called():\n open('hidden')\ncalled()",
        "while True:\n def dormant():\n  open('hidden')\n break",
    ],
)
def test_dormant_bodies_and_mentions_are_not_operations(source):
    assert python_activity(source) == PythonActivity((), True)


def test_executed_definition_headers_are_separate_from_bodies():
    source = (
        "@decorate(open('decorator'))\n"
        "def dormant(arg=open('default'), *, other=open('keyword')):\n"
        " open('body')\n"
        "callback = lambda arg=open('lambda-default'): open('lambda-body')\n"
        "class Ignored(base(open('base'))):\n"
        " open('class-body')\n"
    )
    assert python_activity(source) == PythonActivity(
        tuple(
            PythonOperation(path, False, False)
            for path in ("decorator", "default", "keyword", "lambda-default", "base")
        ),
        True,
    )


def test_deferred_generator_is_not_executed():
    assert python_activity(
        "g = (open('hidden') for x in os.listdir('iterator'))"
    ) == PythonActivity((PythonOperation("iterator", False, False),), False)


@pytest.mark.parametrize(
    "source",
    [
        "open(path)",
        "path = 'synthetic'; open(path)",
        "open('prefix' + suffix)",
        "open(f'{directory}/synthetic')",
        "open(b'synthetic')",
        "open(3)",
        "open('synthetic', mode)",
        "io.open(file='synthetic', mode=mode)",
        "open('synthetic', **options)",
        "open(*arguments)",
        "open('first', file='second')",
        "open('synthetic', 'w', mode='r')",
        "open()",
        "os.remove(path)",
        "os.listdir()",
        "os.scandir(path=directory)",
        "Path(path).read_text()",
        "pathlib.Path('directory', 'file').write_text('contents')",
        "Path(*parts).unlink()",
        "Path(root='synthetic').read_bytes()",
        "shutil.copy(source, destination)",
    ],
)
def test_unresolved_recognized_arguments_are_unknown(source):
    assert python_activity(source) == PythonActivity((), False)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("shutil.copy('known', destination)", PythonOperation("known", False, False)),
        ("shutil.move(source, 'known')", PythonOperation("known", True, False)),
        ("open(path); open('known')", PythonOperation("known", False, False)),
    ],
)
def test_partial_operations_survive_unknown_arguments(source, expected):
    assert python_activity(source) == PythonActivity((expected,), False)


def test_actual_calls_in_assignments_and_nested_expressions_are_found():
    assert python_activity("text = open('synthetic').read(); print(open('synthetic').read())") == (
        PythonActivity((PythonOperation("synthetic", False, False),) * 2, True)
    )


def test_names_and_unrelated_methods_are_not_calls():
    assert python_activity(
        "open; os.remove; obj.read_text(); Path('unused'); print('synthetic')"
    ) == (PythonActivity((), True))


@pytest.mark.parametrize(
    "source", ["def broken(", "open('synthetic')\nthis is not valid python", "\0"]
)
def test_malformed_module_is_unknown_not_a_partial_scan(source):
    assert python_activity(source) == PythonActivity((), False)


@pytest.mark.parametrize("error", [SyntaxError, ValueError, RecursionError])
def test_parser_failure_preserves_unknown(monkeypatch, error):
    def fail(_source):
        raise error("synthetic parse failure")

    monkeypatch.setattr(activity_module, "parse_module", fail)
    assert python_activity("synthetic") == PythonActivity((), False)


def test_traversal_recursion_limit_preserves_known_operations_and_unknown(monkeypatch):
    tree = ast.parse("open('known')")
    nested = ast.UnaryOp(op=ast.Not(), operand=ast.Constant(value=True))
    for _ in range(10_000):
        nested = ast.UnaryOp(op=ast.Not(), operand=nested)
    tree.body.append(ast.Expr(value=nested))
    monkeypatch.setattr(activity_module, "parse_module", lambda _source: tree)
    assert python_activity("synthetic") == PythonActivity(
        (PythonOperation("known", False, False),), False
    )


@pytest.mark.parametrize(
    ("value", "attribute"),
    [
        (PythonOperation("synthetic", False, False), "writes"),
        (PythonActivity((), True), "complete"),
    ],
)
def test_result_is_immutable(value, attribute):
    with pytest.raises(FrozenInstanceError):
        setattr(value, attribute, False)


def test_empty_module_is_complete_within_documented_scope():
    assert python_activity("") == PythonActivity((), True)


def test_unknown_file_operation_retains_loop_scope():
    assert python_activity("while True: open(target, 'w')").repeated_unread
    assert not python_activity("open(target, 'w')").repeated_unread
