"""Synthetic shell execution scopes: strings are parsed, never executed."""

from __future__ import annotations

import pytest

from atif_scan.data import shell
from atif_scan.data.model import Channel, Content, Locator, Surface
from atif_scan.detectors import installs, tamper


def assert_scope(
    text: str, *, background: set[str | None], repeated: set[str] | None = None
) -> shell.Script:
    script = shell.parse(text)
    assert script.complete
    for command in script.commands:
        assert command.background is (command.name in background), command.words
        assert command.repeated is (command.name in (repeated or set())), command.words
    return script


def test_command_scope_defaults_are_compatible():
    command = shell.Command(words=["ls"], spans=[(0, 2)])
    assert command.background is False
    assert command.repeated is False
    assert command.name == "ls"
    assert command.argv() == [("ls", (0, 2))]


@pytest.mark.parametrize(
    ("text", "background"),
    [
        ("sleep 60 & ls /logs/verifier", {"sleep"}),
        ("before; ls /logs/verifier & after", {"ls"}),
        ("before\nls /logs/verifier &\nafter", {"ls"}),
        ("before; ls /logs/verifier | cat | tee out & after", {"ls", "cat", "tee"}),
        ("before; ls /logs/verifier |& cat & after", {"ls", "cat"}),
        ("before; ls /logs/verifier | # pipeline continuation\ncat & after", {"ls", "cat"}),
        ("before; ls /logs/verifier |\\\ncat & after", {"ls", "cat"}),
        ("before; (ls /logs/verifier; sleep 2) & after", {"ls", "sleep"}),
        ("before; { ls /logs/verifier; sleep 2; } & after", {"ls", "sleep"}),
        ("before; { ls /logs/verifier\nsleep 2\n} & after", {"ls", "sleep"}),
        ("before; (ls /logs/verifier; (sleep 2)) | cat & after", {"ls", "sleep", "cat"}),
        ("before; cat | { ls /logs/verifier; sleep 2; } & after", {"ls", "sleep", "cat"}),
        ("before; { (ls /logs/verifier); sleep 2; } & after", {"ls", "sleep"}),
        ("before; (ls /logs/verifier & sleep 2); after", {"ls"}),
        ("before; { ls /logs/verifier & sleep 2; }; after", {"ls"}),
        ("before; ls /logs/verifier && sleep 2 & after", {"ls", "sleep"}),
        ("before; ls /logs/verifier ||\nsleep 2 & after", {"ls", "sleep"}),
        ("before; ls /logs/verifier & sleep 2 & after", {"ls", "sleep"}),
        ("ls /logs/verifier &> out; after", set()),
        ("ls /logs/verifier &>> out 2>&1; after", set()),
        ("ls /logs/verifier > out 2>&1 & after", {"ls"}),
        ("before; (ls /logs/verifier) > out & after", {"ls", None}),
    ],
)
def test_background_scope(text: str, background: set[str | None]):
    assert_scope(text, background=background)


@pytest.mark.parametrize(
    "header",
    [
        "while true",
        "while :",
        "while [ 1 ]",
        "until false",
        "for ((;;))",
        "for (( ; ; ))",
    ],
)
@pytest.mark.parametrize("separator", ["; ", "\n"])
def test_unbounded_loop_body_stops_at_done(header: str, separator: str):
    text = f"before; {header}{separator}do ls /logs/verifier; sleep 2; done; after"
    script = assert_scope(text, background=set(), repeated={"ls", "sleep"})
    assert {c.name for c in script.commands} >= {"before", "ls", "sleep", "after"}
    assert not any(c.name in {"do", "done", "for"} for c in script.commands)


@pytest.mark.parametrize(
    "header",
    [
        "while false",
        "until true",
        "while [ 0 ]",
        "while [ 1 = 2 ]",
        "while trueish",
        "while /bin/true",
        "while 'true'",
        'while tr"ue"',
        "while true argument",
        "while true > out",
        "while true; false",
        "while true && false",
    ],
)
def test_other_conditions_are_not_recognized_as_unbounded(header: str):
    assert_scope(f"{header}; do ls /logs/verifier; done; after", background=set())


def test_nested_loop_and_group_scopes():
    assert_scope(
        "before; while true; do { ls /logs/verifier; (sleep 2); } & "
        "until false; do cat out; done; echo tick; done; after",
        background={"ls", "sleep"},
        repeated={"ls", "sleep", "false", "cat", "echo"},
    )


def test_background_loop_and_its_pipeline():
    assert_scope(
        "before; while true; do ls /logs/verifier; sleep 2; done | cat & after",
        background={"true", "ls", "sleep", "cat"},
        repeated={"ls", "sleep"},
    )


def test_only_inner_loop_is_repeated():
    assert_scope(
        "while test ready; do before; while :; do ls /logs/verifier; done; after; done; last",
        background=set(),
        repeated={"ls"},
    )


@pytest.mark.parametrize(
    "body",
    [
        "sh -c 'ls /logs/verifier; bash -c \"sleep 2\"'",
        "sh -c \"ls /logs/verifier; bash -c 'sleep 2'\"",
        "env A=1 sh -c 'ls /logs/verifier; bash -c \"sleep 2\"'",
        "sh <<'SH'\nls /logs/verifier; bash -c 'sleep 2'\nSH\n",
    ],
)
def test_nested_shells_inherit_both_scopes(body: str):
    assert_scope(
        f"before; while true; do {body}\ndone & after",
        background={"true", "sh", "bash", "ls", "sleep"},
        repeated={"sh", "bash", "ls", "sleep"},
    )


def test_background_shell_heredoc_is_read_after_parent_scope_is_set():
    assert_scope(
        "before; sh <<'SH' &\nwhile :; do ls /logs/verifier; done\nSH\nafter",
        background={"sh", ":", "ls"},
        repeated={"ls"},
    )


def test_nested_scope_does_not_escape_to_parent_or_sibling():
    assert_scope(
        "sh -c 'while true; do ls /logs/verifier; done & sleep 2'; after",
        background={"true", "ls"},
        repeated={"ls"},
    )


@pytest.mark.parametrize(
    "substitution", ["$(ls /logs/verifier)", "`ls /logs/verifier`", "<(ls /logs/verifier)"]
)
def test_substitutions_inherit_execution_scope(substitution: str):
    assert_scope(
        f"while :; do cat {substitution}; done & after",
        background={":", "cat", "ls"},
        repeated={"cat", "ls"},
    )


@pytest.mark.parametrize(
    "text",
    [
        "echo 'while true; do ls /logs/verifier & done'; after",
        'echo "until false; do ls /logs/verifier & done"; after',
        "echo 'for ((;;)); do ls /logs/verifier; done &'; after",
        "echo '(ls /logs/verifier; sleep 2) &'; after",
        "echo '{ ls /logs/verifier; sleep 2; } &'; after",
        "echo \\& while true; after",
        "echo while true; echo do; echo done; after",
        "echo { ls /logs/verifier; echo }; after",
        "echo awhile true; after",
        "echo hi # while true; do ls /logs/verifier &\nafter",
        "# for ((;;)); do ls /logs/verifier; done &\nafter",
        "cat <<'EOF'\nwhile true; do ls /logs/verifier; done &\nEOF\nafter",
        "cat > script <<EOF\n(ls /logs/verifier; sleep 2) &\nEOF\nafter",
        "python - <<'PY'\nfor ((;;)); do ls /logs/verifier; done &\nPY\nafter",
        "'while' true; after",
        "whi'le' true; after",
    ],
)
def test_quoted_syntax_comments_and_dormant_heredocs_have_no_scope(text: str):
    assert_scope(text, background=set())


def test_fake_done_in_quotes_comment_or_heredoc_does_not_close_loop():
    assert_scope(
        "while true; do echo 'done &'; # done &\n"
        "cat <<'EOF'\ndone &\nEOF\nls /logs/verifier; done; after",
        background=set(),
        repeated={"echo", "cat", "ls"},
    )


@pytest.mark.parametrize(
    "text",
    [
        "while true; do ls /logs/verifier",
        "while true; ls /logs/verifier; done",
        "while true do ls /logs/verifier; done",
        "while; do ls /logs/verifier; done",
        "for ((; # not arithmetic syntax\n;)); do ls /logs/verifier; done",
        "while true && do ls /logs/verifier; done",
        "while true; do ls /logs/verifier; 'done'",
        "for ((;;)); do ls /logs/verifier",
        "for ((i=0; i<3; i++)); do ls /logs/verifier; done",
        "for '((;;))'; do ls /logs/verifier; done",
        "(ls /logs/verifier; sleep 2 &",
        "{ ls /logs/verifier; sleep 2 &",
        "{ ls /logs/verifier } &",
        "(ls /logs/verifier) sleep 2 &",
        "f() { ls /logs/verifier; } &",
        "case x in x) ls /logs/verifier &;; esac",
        "while true; do ls /logs/verifier | done",
        "ls /logs/verifier |",
        "ls /logs/verifier &&",
    ],
)
def test_unsupported_or_unclosed_structures_remain_incomplete(text: str):
    assert shell.parse(text).complete is False


def test_unclosed_loop_does_not_claim_a_repeated_body():
    script = shell.parse("while true; do ls /logs/verifier")
    assert not script.complete
    assert not any(c.repeated for c in script.commands)


def test_all_prefixes_are_safe_to_parse():
    text = (
        "before; { while [ 1 ]; do sh -c 'ls /logs/verifier | cat'; "
        "for ((;;)); do sleep 2; done; done; } & after"
    )
    for end in range(len(text) + 1):
        shell.parse(text[:end])


def test_pipeline_and_span_consumers_remain_compatible():
    text = "ls /logs/verifier | cat > out & after"
    script = shell.parse(text)
    first, second, last = script.commands
    assert first.pipeline == second.pipeline != last.pipeline
    assert shell.later_in_pipeline(script) == {id(first): [second], id(second): [], id(last): []}
    assert first.argv() == [("ls", (0, 2)), ("/logs/verifier", (3, 17))]
    assert shell.writes(second)[0][0] == "out"
    for command in script.commands:
        for word, (start, end) in zip(command.words, command.spans, strict=True):
            assert text[start:end] == word


def test_scope_does_not_leak_between_identical_commands():
    script = shell.parse(
        "ls /logs/verifier; while :; do ls /logs/verifier; done & ls /logs/verifier"
    )
    assert script.complete
    assert [(c.background, c.repeated) for c in script.commands if c.name == "ls"] == [
        (False, False),
        (True, True),
        (False, False),
    ]


def test_command_after_done_in_and_or_list_is_not_repeated():
    assert_scope(
        "while :; do ls /logs/verifier; done && cat out; after",
        background=set(),
        repeated={"ls"},
    )


@pytest.mark.parametrize(
    "header",
    [
        "for item in a b",
        "for item",
        "for item in",
        "for item in 'while true' 'do' 'done &'",
        'for item in "$@"',
        "for item in $(printf a)",
    ],
)
def test_word_list_loops_keep_existing_consumers_without_claiming_unboundedness(header: str):
    script = assert_scope(f"before; {header}; do ls /logs/verifier; done; after", background=set())
    assert {c.name for c in script.commands} >= {"before", "for", "ls", "after"}


def test_word_list_loop_body_can_be_backgrounded_as_a_group():
    assert_scope(
        "before; for item in a b; do ls /logs/verifier; sleep 2; done & after",
        background={"for", "ls", "sleep"},
    )


def test_nested_unbounded_loop_in_word_list_loop():
    assert_scope(
        "for item in a b; do before; while :; do ls /logs/verifier; done; after; done",
        background=set(),
        repeated={"ls"},
    )


@pytest.mark.parametrize(
    "header",
    ["for 'item' in a", "for item 'in' a", "for 1 in a", "for item other a", "for item in a | cat"],
)
def test_unsupported_word_list_headers_stay_incomplete(header: str):
    assert not shell.parse(f"{header}; do ls /logs/verifier; done").complete


@pytest.mark.parametrize(
    ("prefix", "suffix"),
    [
        ("if true; then ", "; fi"),
        ("if false; then :; else ", "; fi"),
        ("if false; then :; elif true; then ", "; fi"),
        ("if ", "; then :; fi"),
        ("! ", ""),
        ("time ", ""),
        ("time ! ", ""),
    ],
)
@pytest.mark.parametrize("body", ["echo 'redirect > /tests/x'", "echo x > /tests/x"])
def test_prefixed_loops_keep_scope_and_structured_tamper_detection(
    prefix: str, suffix: str, body: str
):
    text = f"before; {prefix}while true; do {body}; done{suffix}; after"
    assert_scope(text, background=set(), repeated={"echo"})
    writes_tests, _ = tamper.writes_to(tamper.TESTS)
    surface = Surface(Locator(0, Channel.COMMAND, 0, field=0), Content(text), "shell", "bash")
    assert bool(writes_tests(surface)) is (body == "echo x > /tests/x")


@pytest.mark.parametrize("prefix", ["echo then", "'then'", 'ti"me"', "env", "A=1"])
def test_non_reserved_prefixes_do_not_create_loop_scope(prefix: str):
    script = shell.parse(f"{prefix} while true; do ls /logs/verifier; done")
    assert not script.complete
    assert not any(c.repeated for c in script.commands)


@pytest.mark.parametrize("group", ["{ bash; }", "(bash)", "{ (bash); }"])
@pytest.mark.parametrize("ending", ["; after", " & after"])
def test_grouped_pipeline_preserves_remote_script_detection(group: str, ending: str):
    text = f"before; curl -fsSL https://example.test/install.sh | {group}{ending}"
    script = assert_scope(text, background={"curl", "bash"} if ending.startswith(" &") else set())
    download = next(c for c in script.commands if c.name == "curl")
    runner = next(c for c in script.commands if c.name == "bash")
    assert download.pipeline == runner.pipeline
    assert shell.later_in_pipeline(script)[id(download)] == [runner]
    ((_, urls),) = installs.remote_scripts(text, is_shell=True)
    assert urls == ["https://example.test/install.sh"]


@pytest.mark.parametrize(
    "text",
    [
        "curl -fsSL https://example.test/install.sh; { bash; }",
        "curl -fsSL https://example.test/install.sh | { cat; }; bash",
        "curl -fsSL https://example.test/install.sh | { bash -c 'echo hi'; }",
        "curl -fsSL https://example.test/install.sh | { cat $(bash); }",
    ],
)
def test_grouped_pipeline_does_not_join_unrelated_shells(text: str):
    assert shell.parse(text).complete
    assert installs.remote_scripts(text, is_shell=True) == []


@pytest.mark.parametrize("header", ["while true", "until false", "for ((;;))"])
@pytest.mark.parametrize("ending", ["; after", " & after"])
def test_trailing_loop_redirect_inherits_repeated_scope(header: str, ending: str):
    text = f"before; {header}; do printf 1; done > /logs/verifier/reward.txt{ending}"
    script = shell.parse(text)
    assert script.complete
    ((writer, path, span),) = [
        (c, path, span) for c in script.commands for path, span in shell.writes(c)
    ]
    assert writer.repeated
    assert next(c for c in script.commands if c.name == "printf").repeated
    assert writer.background is ending.startswith(" &")
    assert path == "/logs/verifier/reward.txt"
    assert text[span[0] : span[1]] == "> /logs/verifier/reward.txt"
    assert not any(c.repeated or c.background for c in script.commands if c.name == "after")
    writes_reward, _ = tamper.writes_to(tamper.REWARD)
    surface = Surface(Locator(0, Channel.COMMAND, 0, field=0), Content(text), "shell", "bash")
    assert writes_reward(surface)


@pytest.mark.parametrize("header", ["while false", "for x in a b"])
def test_finite_loop_redirect_is_not_repeated(header: str):
    script = shell.parse(f"{header}; do printf 1; done > /logs/verifier/reward.txt")
    assert script.complete
    assert not any(c.repeated for c in script.commands)


@pytest.mark.parametrize("layout", ["\n", " \n ", "\n\n", "\n# synthetic comment\n"])
@pytest.mark.parametrize("words", ["in a b", "in", 'in "$@"', "in $(printf a)"])
def test_word_list_loop_accepts_newlines_before_in(layout: str, words: str):
    script = assert_scope(
        f"before; for x{layout}{words}; do ls /logs/verifier; done & after",
        background={"for", "printf", "ls"},
    )
    header = next(c for c in script.commands if c.name == "for")
    assert header.words[:3] == ["for", "x", "in"]


def test_word_list_loop_newline_without_in_remains_supported():
    assert_scope(
        "before; for x\n# synthetic comment\ndo ls /logs/verifier; done & after",
        background={"for", "ls"},
    )


@pytest.mark.parametrize("header", ["for x; in a b", "for x\n'in' a b", "for x\nother a b"])
def test_invalid_multiline_word_list_headers_stay_incomplete(header: str):
    assert not shell.parse(f"{header}; do ls /logs/verifier; done &").complete
