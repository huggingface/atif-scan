"""Static comprehension scope regressions; synthetic programs, never executed."""

import pytest
from test_python_programs import calls, program_trace

from atif_scan import Channel
from atif_scan.data.jslit import UNREAD
from atif_scan.data.loader import parse_trace


@pytest.mark.parametrize(
    "expression",
    [
        "[shell(command=f'echo {n}') for n in [1, 2, 3]]",
        "(shell(command=f'echo {n}') for n in [1, 2, 3])",
        "{shell(command=f'echo {n}') for n in [1, 2, 3]}",
        "{n: shell(command=f'echo {n}') for n in [1, 2, 3]}",
    ],
)
def test_comprehension_target_is_independent_of_other_scopes(expression):
    program = (
        "def helper(n):\n    return n\n"
        f"results = await asyncio.gather(*{expression})\n"
        "for n in results:\n    print(n)\n"
    )
    assert calls(program) == [("shell", {"command": f"echo {n}"}) for n in (1, 2, 3)]
    parsed = parse_trace(program_trace(program))
    commands = [
        field
        for _, call in parsed.agent_calls()
        if call.tool == "shell"
        for channel, field in call.fields
        if channel == Channel.COMMAND
    ]
    assert [field.text for field in commands] == ["echo 1", "echo 2", "echo 3"]
    assert all(field.understood and not field.computed for field in commands)


def test_separate_comprehensions_reuse_a_target_without_leaking_it():
    program = (
        "[shell(command=f'echo {n}') for n in [1, 2]]\n"
        "[shell(command=f'echo {n}') for n in [3]]\n"
        "await shell(command=f'echo {n}')\n"
    )
    assert calls(program) == [
        ("shell", {"command": "echo 1"}),
        ("shell", {"command": "echo 2"}),
        ("shell", {"command": "echo 3"}),
        ("shell", {"command": UNREAD}),
    ]


@pytest.mark.parametrize(
    "program",
    [
        "for n in [1, 2]:\n    n = other\n    await shell(command=f'echo {n}')",
        "for n in [1, 2]:\n    await shell(command=f'echo {n}')\nn = other",
        "values = [1, 2]\nvalues = other\n[shell(command=f'echo {n}') for n in values]",
        "[shell(command=f'echo {n}') for n in [1, 2] for n in runtime_values]",
        "[shell(command=f'echo {n}') for n in runtime_values]",
        "[shell(command=f'echo {n}') for n in range(3)]",
        "[shell(command=f'echo {n}') for n in [1, 2] if (n := runtime_value)]",
        "n = 1\n[shell(command=f'echo {n}') for n in runtime_values]",
        "[shell(command=f'echo {n}') for n in [1, 2] if (cmd := runtime_value)]\n"
        "await shell(command=cmd)",
    ],
)
def test_reassignment_and_dynamic_sources_stay_unread(program):
    assert calls(program)[-1] == ("shell", {"command": UNREAD})


def test_dynamic_source_is_not_executed(tmp_path):
    marker = tmp_path / "must-not-exist"
    program = (
        f"[shell(command=f'echo {{n}}') for n in open({str(marker)!r}, 'w').write('synthetic')]"
    )
    assert calls(program) == [("shell", {"command": UNREAD})]
    assert not marker.exists()
