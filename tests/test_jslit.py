"""Static reading of code-mode programs: robustness, linear time, lexing and shadowing."""

from __future__ import annotations

import pytest

from atif_scan.jslit import UNREAD, tool_calls

UNKNOWN = [("?", UNREAD)]


@pytest.mark.parametrize(
    "program, expected",
    [
        # Regression: peek() returned "" at end of input and `"" in "\"'`"` is True, so
        # string() indexed past the end and IndexError escaped the whole scan.
        ("tools.a(", [("a", UNREAD)]),
        ("tools.a(x /* c */", [("a", UNREAD)]),
        ("tools.a({}); const x =", [("a", {})]),
        ("tools.a({k: ", [("a", UNREAD)]),
        ("tools.a('x\n')", [("a", UNREAD)]),  # quoted strings can't span lines
        ("tools.a(f(]))", [("a", UNREAD)]),
        ("const {", UNKNOWN[:0]),
        ("tools", UNKNOWN),
    ],
)
def test_truncated_input_never_raises(program, expected):
    assert tool_calls(program) == expected


def test_every_prefix_of_a_program_is_safe():
    program = (
        'const p = "a\\x41", {q} = o; for (let i of [1]) try { await tools.x({cmd: `c ${p}`,'
        " n: [1, -2e3, null], ...r}); } catch (e) { tools['y'](/re\"/g) } // tools.z()"
    )
    for end in range(len(program) + 1):
        assert isinstance(tool_calls(program[:end]), list)


SIZE = 50_000  # x4 in the linear check: 200 KB programs


@pytest.mark.parametrize(
    "program",
    [
        # Regression: every `tools.x(` inside an already-read string was rescanned (3.6s).
        lambda n: 'tools.exec_command({cmd: "' + "tools.x(" * (n // 8) + '"})',
        # Regression: each nested call re-skipped the rest of the program.
        lambda n: "tools.a(" * (n // 9) + ")" * (n // 9),
        lambda n: "tools.a([" * (n // 9),
        lambda n: "tools.a('" * (n // 9),
        lambda n: "tools.a(`${" * (n // 12),
        lambda n: "tools.a({k: tools.b(1)}); " * (n // 26),
        # Regression: one regex search per declared name, plus a slice per declaration (2s).
        lambda n: "".join(f'const v{i} = "x";\n' for i in range(n // 18)) + "tools.a(v1)",
        lambda n: "function f(" * (n // 11),
        lambda n: "a" * n + " => tools.a(1)",
        lambda n: "/* " * (n // 3),
        lambda n: "x / " * (n // 4) + "tools.a(1)",
    ],
    ids=lambda p: p(40)[:12],
)
def test_large_programs_are_read_in_linear_time(linear, program):
    linear(lambda n: tool_calls(program(n)), SIZE)


@pytest.mark.parametrize(
    "program, expected",
    [
        # Calls in comments and strings aren't calls (they were reported as evidence).
        ('// tools.a({cmd: "x"})\ntools.b(1)', [("b", 1)]),
        ('/* tools.a() */ tools.b("tools.c(2)")', [("b", "tools.c(2)")]),
        ("const r = /tools.a(/; tools.b(1)", [("b", 1)]),  # a regex literal
        ("x = a / tools.b(1) / 2", [("b", 1)]),  # division: still code
        (
            "t = `tools.a( ${await tools.b({q: 'z'})} ${`${tools.c(3)}`}`",
            [("b", {"q": "z"}), ("c", 3)],
        ),
        ('text("no tools.x( here")', UNKNOWN),  # `tools` only outside code: unknown, not []
        ("text(1)", []),
        # Other uses of `tools` can't be read: unknown, never silently dropped.
        ('tools["exec_command"]({cmd: "ls"})', UNKNOWN),
        ('tools.exec_command?.({cmd: "ls"}); tools.a(1)', [*UNKNOWN, ("a", 1)]),
        ('const t = tools; t.exec_command({cmd: "ls"})', UNKNOWN),
        ("const f = tools.a; f(1)", UNKNOWN),
        ("mytools.a(1); $tools.b(2); tools_c(3)", []),
        # Escapes and line continuation.
        (r'tools.a("\x41\u{1F600}\u0042\q")', [("a", "A\U0001f600Bq")]),
        ('tools.a("a\\\nb", "second argument ignored")', [("a", "ab")]),
    ],
)
def test_only_calls_in_code_are_read(program, expected):
    assert tool_calls(program) == expected


@pytest.mark.parametrize(
    "shadow",
    [
        'for (const c of ["curl x"]) ',
        "const {c} = obj; ",
        "const [a, c] = xs; ",
        "let a = 1, c; ",
        "[a, c] = xs; ",
        "({c} = obj); ",
        "try {} catch (c) {} ",
        "function f(a, c = 1) {} ",
        "const f = (c) => 1; ",
        "const f = c => 1; ",
        "const o = {m(c) { return 1 }}; ",
        "function c() {} ",
        "function* c() {} ",
        "class c {} ",
        "c++; ",
        "c += 'x'; ",
    ],
)
def test_shadowed_or_rebound_names_are_unread(shadow):
    # Regression: a second binding of `c` resolved to the outer literal (a false negative).
    program = f'const c = "ls"; {shadow}tools.exec_command({{cmd: c}})'
    assert tool_calls(program) == [("exec_command", {"cmd": UNREAD})]


def test_a_name_bound_once_is_still_read():
    program = 'const c = "ls", d = `x`; if (c) { tools.exec_command({cmd: c, d}) }'
    assert tool_calls(program) == [("exec_command", {"cmd": "ls", "d": UNREAD})]
