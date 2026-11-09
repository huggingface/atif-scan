"""Synthetic local edit payloads versus actual DeepSWE upstream requests."""

from __future__ import annotations

import pytest
from test_pack_deepswe import diff, evaluate, shell_steps

from atif_scan.checks import Status
from atif_scan.packs.deepswe import upstream_command_spans, upstream_request

URL = "https://github.com/abs-lang/abs"


def python_edit(source: str) -> str:
    return "python - <<'PY'\n" + source + "\nPY"


@pytest.mark.parametrize(
    "command",
    [
        python_edit(f"from pathlib import Path\np = Path('CHANGELOG.md')\np.write_text('{URL}')"),
        python_edit(
            f"from pathlib import Path\np = Path('docs/usage.md')\ns = p.read_text()\n"
            f"s = s.replace('old', '{URL}')\np.write_text(s)"
        ),
        python_edit(
            f"import json\nfrom pathlib import Path\n"
            f"report = {{'helpUri': '{URL}'}}\nPath('example.sarif').write_text(json.dumps(report))"
        ),
        python_edit(
            f"from pathlib import Path\n"
            f"fixture = {f'requests.get({URL!r})'!r}\nPath('tests/fixture.txt').write_text(fixture)"
        ),
        python_edit(f"with open('CHANGELOG.md', 'w') as f:\n    f.write('{URL}')"),
        python_edit(f"# Documentation: {URL}\nprint('local edit')"),
        f"""python -c 'from pathlib import Path; Path("CHANGELOG.md").write_text("{URL}")'""",
        f"sed -i 's|old|{URL}|g' CHANGELOG.md",
        f"sed -i -e 's|old|{URL}|' -e 's|a|b|' src/test.go",
        f"cat > CHANGELOG.md <<'EOF'\n{URL}\nEOF",
        f"cat > tests/fixture.txt <<'EOF'\n$(curl {URL})\nEOF",
    ],
)
def test_local_payload_is_not_an_upstream_request(command):
    result = evaluate(diff("src/a.go", "x"), shell_steps(command))["deepswe.upstream_lookup"].result
    assert result.status == Status.NO_MATCH
    assert result.complete


@pytest.mark.parametrize(
    "command",
    [
        f"curl -I {URL}",
        f"sudo -u worker timeout 10 curl {URL}",
        f"sed -i 's|old|{URL}|' CHANGELOG.md; curl {URL}/pulls",
        f'sed -i "s|old|$(curl {URL})|" CHANGELOG.md',
        f"echo $(curl {URL})",
        f"echo `curl {URL}`",
        f"bash -lc 'curl {URL}'",
        python_edit(f"import requests\nrequests.get('{URL}')"),
        python_edit(f"import requests as r\nurl = '{URL}'\nr.get(url)"),
        python_edit(f"from urllib.request import urlopen as fetch\nurl = '{URL}'\nfetch(url)"),
        python_edit(f"import httpx\nurl = '{URL}'\nclient = httpx.Client()\nclient.get(url)"),
        python_edit(
            f"import requests\nfrom pathlib import Path\nu = '{URL}'\n"
            "Path('CHANGELOG.md').write_text(u)\nrequests.get(u)"
        ),
        f"cat > fetch.sh <<'EOF'\ncurl {URL}\nEOF\nbash fetch.sh",
        f"""python -c 'import requests; requests.get("{URL}")'""",
    ],
)
def test_real_requests_survive_payload_exclusions(command):
    result = evaluate(diff("src/a.go", "x"), shell_steps(command))["deepswe.upstream_lookup"].result
    assert result.status == Status.MATCH
    assert result.evidence


@pytest.mark.parametrize(
    "command",
    [
        python_edit(f"consume('{URL}')"),
        python_edit(f"import socket\nsock = socket.socket()\nsock.write('{URL}')"),
        python_edit(f"def fetch():\n    requests.get('{URL}')"),
        python_edit(f"from pathlib import Path\nPath('a').write_text('{URL}'"),
        f"python - <<'PY'\nurl = '{URL}'\nthis is invalid python!\nPY",
        f"cat > CHANGELOG.md <<EOF\n$(curl {URL})\nEOF",
        python_edit(f"from pathlib import Path\nPath('fetch.sh').write_text('curl {URL}')")
        + "\nbash fetch.sh",
    ],
)
def test_unresolved_execution_is_unknown_not_clean(command):
    result = evaluate(diff("src/a.go", "x"), shell_steps(command))["deepswe.upstream_lookup"].result
    assert result.status == Status.UNKNOWN
    assert not result.complete
    assert result.unread


@pytest.mark.parametrize(
    "command",
    [
        python_edit(f"open('CHANGELOG.md', 'w').write('{URL}')"),
        python_edit(
            "import collections\nimport json\nfrom pathlib import Path\n"
            f"rules = collections.OrderedDict([('synthetic', {{'helpUri': '{URL}'}})])\n"
            "Path('example.sarif').write_text(json.dumps({'rules': rules}))"
        ),
        f"sed -i '1s|old|{URL}|' CHANGELOG.md; head CHANGELOG.md",
        f"sed -i 's|old|{URL}|' CHANGELOG.md; grep github CHANGELOG.md",
        python_edit(f"open('CHANGELOG.md', 'w').write('{URL}')") + "\nhead CHANGELOG.md",
        f"cat > CHANGELOG.md <<'EOF'\n{URL}\nEOF\nhead CHANGELOG.md",
        "python -c 'this is not valid Python!'",
        "echo 'unfinished quote",
    ],
)
def test_additional_proven_local_cases(command):
    result = evaluate(diff("src/a.go", "x"), shell_steps(command))["deepswe.upstream_lookup"].result
    assert result.status == Status.NO_MATCH
    assert result.complete


@pytest.mark.parametrize(
    "source",
    [
        f"def print(value):\n    consume(value)\nprint('{URL}')",
        f"def open(value):\n    consume(value)\nopen('{URL}')",
        f"class Path:\n    def __init__(self, value):\n        consume(value)\nPath('{URL}')",
        f"from mystery import Path\nPath('{URL}')",
        f"from pathlib import Path\np = Path('doc')\np = unknown_object\np.write_text('{URL}')",
        "from pathlib import Path\np = Path('doc')\np.write_text = unknown_function\n"
        f"p.write_text('{URL}')",
        f"f = open('doc', 'w')\nf = unknown_object\nf.write('{URL}')",
        "from pathlib import Path\np = Path('doc')\nsetattr(p, 'write_text', fetch)\n"
        f"p.write_text('{URL}')",
        f"class Text:\n    pass\ns = Text()\nopen('doc', 'w').write(s.replace('old', '{URL}'))",
    ],
)
def test_shadowed_or_rebound_local_sinks_are_not_excused(source):
    result = evaluate(diff("src/a.go", "x"), shell_steps(python_edit(source)))[
        "deepswe.upstream_lookup"
    ].result
    assert result.status == Status.UNKNOWN
    assert not result.complete


def test_local_name_bound_to_request_is_still_a_request():
    source = f"import requests\nopen = requests.get\nopen('{URL}')"
    result = evaluate(diff("src/a.go", "x"), shell_steps(python_edit(source)))[
        "deepswe.upstream_lookup"
    ].result
    assert result.status == Status.MATCH


@pytest.mark.parametrize(
    "command",
    [
        f"cat <<'DATA' <<EOF > fixture.txt\nlocal data\nDATA\n$(curl {URL})\nEOF",
        f"cat > fetch.sh <<'EOF'\ncurl {URL}\nEOF\nsed -n '1e' fetch.sh",
        f"cat > fetch.sh <<'EOF'\ncurl {URL}\nEOF\nrg --pre=bash curl fetch.sh",
        f"cat > fetch.sh <<'EOF'\ncurl {URL}\nEOF\ncat fetch.sh | bash",
        f"sed -i 's|old|curl {URL}|' fetch.sh; bash fetch.sh",
        f"sed -i 's|old|curl {URL}|e' CHANGELOG.md",
    ],
)
def test_executable_or_ambiguous_shell_payloads_are_never_clean(command):
    result = evaluate(diff("src/a.go", "x"), shell_steps(command))["deepswe.upstream_lookup"].result
    assert result.status in {Status.MATCH, Status.UNKNOWN}


def test_mixed_python_payload_and_request_have_exact_request_span():
    command = python_edit(
        f"import requests\nfrom pathlib import Path\n"
        f"Path('CHANGELOG.md').write_text('{URL}')\nrequests.get('{URL}/pulls')"
    )
    spans, uncertain = upstream_command_spans(command, upstream_request("abs-lang/abs"))
    assert not uncertain
    assert len(spans) == 1
    assert spans[0][0] == command.rindex(URL)
    assert command[slice(*spans[0])] == URL


def test_non_ascii_python_offsets_are_character_offsets():
    command = python_edit(
        f"import requests\nfrom pathlib import Path\n"
        f"Path('CHANGELOG.md').write_text('café {URL}')\nrequests.get('{URL}/pulls')"
    )
    spans, uncertain = upstream_command_spans(command, upstream_request("abs-lang/abs"))
    assert not uncertain
    assert spans == [(command.rindex(URL), command.rindex(URL) + len(URL))]


def test_local_edit_and_assert_string_membership_are_inert():
    source = (
        "from pathlib import Path\np = Path('CHANGELOG.md')\ns = p.read_text()\n"
        f"old = '{URL}/old'\nnew = '{URL}/new'\nassert old in s\n"
        "s = s.replace(old, new)\nassert new in s\np.write_text(s)"
    )
    result = evaluate(diff("src/a.go", "x"), shell_steps(python_edit(source)))[
        "deepswe.upstream_lookup"
    ].result
    assert result.status == Status.NO_MATCH


@pytest.mark.parametrize(
    "write",
    [
        f"cat > fetch.sh <<'EOF'\ncurl {URL}\nEOF",
        python_edit(f"from pathlib import Path\nPath('fetch.sh').write_text('curl {URL}')"),
        python_edit(f"open('fetch.sh', 'w').write('curl {URL}')"),
        f"sed -i 's|old|curl {URL}|' fetch.sh",
    ],
)
@pytest.mark.parametrize(
    "execute",
    [
        "bash fetch.sh",
        'p=fetch.sh; bash "$p"',
        "eval 'bash fetch.sh'",
        """python -c 'import subprocess; subprocess.run(["sh", "fetch.sh"])'""",
    ],
)
def test_later_step_executes_saved_payload(write, execute):
    result = evaluate(diff("src/a.go", "x"), shell_steps(write, execute))[
        "deepswe.upstream_lookup"
    ].result
    assert result.status in {Status.MATCH, Status.UNKNOWN}


def test_later_step_benign_reader_does_not_invalidate_local_edit():
    write = python_edit(f"open('CHANGELOG.md', 'w').write('{URL}')")
    result = evaluate(diff("src/a.go", "x"), shell_steps(write, "head CHANGELOG.md"))[
        "deepswe.upstream_lookup"
    ].result
    assert result.status == Status.NO_MATCH


def test_python_process_call_can_execute_saved_payload():
    source = (
        "import subprocess\nfrom pathlib import Path\n"
        f"Path('fetch.sh').write_text('curl {URL}')\n"
        "subprocess.run(['sh', 'fetch.sh'])"
    )
    result = evaluate(diff("src/a.go", "x"), shell_steps(python_edit(source)))[
        "deepswe.upstream_lookup"
    ].result
    assert result.status == Status.UNKNOWN


def test_direct_python_request_survives_deferred_execution_uncertainty():
    source = (
        "import requests\nimport subprocess\nfrom pathlib import Path\n"
        f"Path('fetch.sh').write_text('curl {URL}')\n"
        "subprocess.run(['sh', 'fetch.sh'])\n"
        f"requests.get('{URL}/pulls')"
    )
    result = evaluate(diff("src/a.go", "x"), shell_steps(python_edit(source)))[
        "deepswe.upstream_lookup"
    ].result
    assert result.status == Status.MATCH
    assert not result.complete


@pytest.mark.parametrize("same_step", [True, False])
def test_local_sed_edit_with_ambiguous_later_consumer_is_unknown(same_step):
    write = f"sed -i '1s|old|{URL}|' src/comment.go"
    consumer = "inspect_file src/comment.go"
    commands = [write + "; " + consumer] if same_step else [write, consumer]
    result = evaluate(diff("src/a.go", "x"), shell_steps(*commands))[
        "deepswe.upstream_lookup"
    ].result
    assert result.status == Status.UNKNOWN
    assert not result.complete


def test_real_request_survives_ambiguous_sed_destination_use():
    command = f"sed -i 's|old|{URL}|' src/comment.go; inspect_file src/comment.go; curl {URL}/pulls"
    result = evaluate(diff("src/a.go", "x"), shell_steps(command))["deepswe.upstream_lookup"].result
    assert result.status == Status.MATCH
    assert not result.complete
    spans, uncertain = upstream_command_spans(command, upstream_request("abs-lang/abs"))
    assert uncertain
    assert spans == [(command.rindex(URL), command.rindex(URL) + len(URL))]


def test_incomplete_later_executor_does_not_clear_saved_payload():
    write = python_edit(f"open('fetch.sh', 'w').write('curl {URL}')")
    result = evaluate(diff("src/a.go", "x"), shell_steps(write, "bash 'fetch.sh"))[
        "deepswe.upstream_lookup"
    ].result
    assert result.status == Status.UNKNOWN
    assert not result.complete


def test_known_request_survives_incomplete_following_surface():
    command = f"curl {URL}"
    spans, uncertain = upstream_command_spans(
        command, upstream_request("abs-lang/abs"), "bash 'fetch.sh"
    )
    assert uncertain
    assert spans == [(command.index(URL), len(command))]
