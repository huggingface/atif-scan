"""Synthetic scheduler text only: nothing submitted here is executed."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from atif_scan.data import shell
from atif_scan.data.scheduled_commands import ScheduledCommands, scheduled_commands


def read(text: str, name: str = "at") -> ScheduledCommands | None:
    script = shell.parse(text)
    command = next(c for c in script.commands if c.name == name)
    return scheduled_commands(command, script, text)


@pytest.mark.parametrize("text", ["echo at now", "atq", "atrm 12", "cat schedule", "batch"])
def test_non_schedulers(text: str):
    script = shell.parse(text)
    assert scheduled_commands(script.commands[0], script, text) is None


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("crontab -l", "crontab"),
        ("crontab -u alice -l", "crontab"),
        ("crontab -r", "crontab"),
        ("crontab -e", "crontab"),
        ("at -l", "at"),
        ("at -r 12", "at"),
        ("at -d 12", "at"),
        ("at -c 12", "at"),
    ],
)
def test_inspection_and_removal(text: str, name: str):
    assert read(text, name) is None


@pytest.mark.parametrize("delimiter", ["EOF", "'EOF'", '"EOF"'])
def test_at_heredoc_is_plain_shell_source(delimiter: str):
    body = "echo synthetic\nfor item in one two; do echo item; done\n"
    assert read(f"at now <<{delimiter}\n{body}EOF") == ScheduledCommands((body,), True)


def test_quoted_heredoc_keeps_future_shell_expansions():
    body = 'echo "$HOME"; echo `date`; echo "$(date)"\n'
    assert read(f"at now <<'EOF'\n{body}EOF") == ScheduledCommands((body,), True)


@pytest.mark.parametrize("body", ['echo "$HOME"', "echo `date`", "echo $(date)", "echo a\\b"])
def test_unquoted_heredoc_expansions_are_unknown(body: str):
    assert read(f"at now <<EOF\n{body}\nEOF") == ScheduledCommands((), False)


def test_heredoc_owner_does_not_leak():
    text = "cat <<EOF\nnot a job\nEOF\nat now <<EOF\necho job\nEOF\nat tomorrow"
    script = shell.parse(text)
    jobs = [scheduled_commands(c, script, text) for c in script.commands if c.name == "at"]
    assert jobs == [ScheduledCommands(("echo job\n",), True), ScheduledCommands((), False)]


def test_tab_stripping_heredoc():
    assert read("at now <<-EOF\n\techo job\n\tEOF") == ScheduledCommands(("echo job\n",), True)


@pytest.mark.parametrize("text", ["at now <<EOF\necho job", "at now <<EOF"])
def test_missing_heredoc_delimiter_is_incomplete(text: str):
    assert read(text) == ScheduledCommands((), False)


@pytest.mark.parametrize(
    "producer", ["echo 'echo synthetic'", 'echo "echo" "synthetic"', "echo echo synthetic"]
)
def test_echo_joins_literal_tokens(producer: str):
    assert read(f"{producer} | at now") == ScheduledCommands(("echo synthetic\n",), True)


@pytest.mark.parametrize(
    ("producer", "expected"),
    [
        ("printf '%s' 'echo ' 'synthetic'", "echo synthetic"),
        (r"printf '%s\n' 'echo one' 'echo two'", "echo one\necho two\n"),
        ('printf "%s" "echo synthetic"', "echo synthetic"),
    ],
)
def test_printf_supported_formats(producer: str, expected: str):
    assert read(f"{producer} | at now") == ScheduledCommands((expected,), True)


@pytest.mark.parametrize(
    "producer",
    [
        "echo $JOB",
        'echo "$JOB"',
        "echo `date`",
        "echo $(date)",
        r"echo echo\ job",
        r"echo 'echo\njob'",
        "echo *.sh",
        "echo ~/job",
        "echo {one,two}",
        "echo -e 'echo job'",
        "printf 'echo job'",
        "printf '%b' 'echo job'",
        "printf '%s %s' echo job",
        'printf "$FORMAT" job',
        "printf '%s' $JOB",
    ],
)
def test_computed_or_unsupported_producers_are_incomplete(producer: str):
    assert read(f"{producer} | at now") == ScheduledCommands((), False)


def test_single_quoted_dollar_is_literal_source():
    assert read("echo 'echo $HOME' | at now") == ScheduledCommands(("echo $HOME\n",), True)


@pytest.mark.parametrize(
    "text",
    [
        "echo 'echo job'; at now",
        "echo 'echo job' && at now",
        "echo 'echo job' | cat | at now",
        "echo 'echo job' > out | at now",
        "echo 'echo job' | at now < input",
        "echo 'echo job' |& at now",
        "cat input | at now",
    ],
)
def test_only_direct_literal_pipeline_input(text: str):
    assert read(text) == ScheduledCommands((), False)


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("at -f job.sh now", "at"),
        ("at -fjob.sh now", "at"),
        ("at now < job.sh", "at"),
        ("crontab schedule", "crontab"),
        ("crontab -u alice schedule", "crontab"),
        ("echo '@daily echo job' | crontab schedule", "crontab"),
        ("at -f job.sh now <<EOF\necho other\nEOF", "at"),
    ],
)
def test_external_files_are_not_opened(text: str, name: str, monkeypatch: pytest.MonkeyPatch):
    def forbidden(*args: object, **kwargs: object) -> str:
        raise AssertionError("scheduler files must not be opened")

    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", forbidden)
        patch.setattr(Path, "read_text", forbidden)
        result = read(text, name)
    assert result == ScheduledCommands((), False)


def test_crontab_lines_ignore_comments_and_environment():
    body = (
        "\n # comment\nSHELL=/bin/sh\nMAILTO = 'nobody'\n"
        "*/5 0-4 * * 1,3 echo one\n@daily echo two\n"
    )
    assert read(f"crontab - <<'EOF'\n{body}EOF", "crontab") == ScheduledCommands(
        ("echo one", "echo two"), True
    )


@pytest.mark.parametrize(
    "macro", ["reboot", "yearly", "annually", "monthly", "weekly", "daily", "midnight", "hourly"]
)
def test_crontab_macros_via_printf(macro: str):
    assert read(f"printf '%s\\n' '@{macro} echo job' | crontab -", "crontab") == ScheduledCommands(
        ("echo job",), True
    )


@pytest.mark.parametrize(
    "entry",
    [
        "bad syntax",
        "* * * * echo job",
        "@sometimes echo job",
        "@daily",
        "* * * * *",
        "* * * * ? echo job",
        "* * * xyz * echo job",
        "* * * * * echo x%y",
        "@daily echo x\\y",
    ],
)
def test_unsupported_cron_lines_preserve_known_entries(entry: str):
    text = f"crontab - <<'EOF'\n@reboot echo known\n{entry}\nEOF"
    assert read(text, "crontab") == ScheduledCommands(("echo known",), False)


def test_named_cron_months_and_weekdays():
    text = "echo '0 9 * JAN,Feb MON-FRI echo job' | crontab -"
    assert read(text, "crontab") == ScheduledCommands(("echo job",), True)


@pytest.mark.parametrize("text", ["at now", "at now <<< job", "at now 2<<EOF\necho job\nEOF"])
def test_unavailable_or_non_stdin_source_is_unknown(text: str):
    assert read(text) == ScheduledCommands((), False)


def test_empty_crontab_is_known_empty():
    assert read("crontab - <<EOF\n# empty\nSHELL=/bin/sh\nEOF", "crontab") == ScheduledCommands(
        (), True
    )


@pytest.mark.parametrize("installer", ["sudo /usr/bin/crontab -u alice -", "crontab -ualice -"])
def test_scheduler_wrappers_and_user_options(installer: str):
    assert read(f"echo '@daily echo job' | {installer}", "crontab") == ScheduledCommands(
        ("echo job",), True
    )


def test_outer_parse_incompleteness_propagates():
    assert read("echo 'echo job' | at now; (") == ScheduledCommands(("echo job\n",), False)


@pytest.mark.parametrize("attribute", ["scripts", "complete"])
def test_typed_result_is_immutable(attribute: str):
    result = ScheduledCommands(("echo job",), True)
    with pytest.raises(FrozenInstanceError):
        setattr(result, attribute, False)
