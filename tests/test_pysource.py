"""Untrusted Python is parsed without warnings. Regression: a DeepSWE (Datacurve, DeepSeek V4
Flash) scan printed ~30 `SyntaxWarning: invalid escape sequence` lines about agent scripts
to stderr, from every check that reads Python syntax. Synthetic fixtures only."""

from __future__ import annotations

import warnings

import pytest

from atif_scan.data.pysource import parse_expression, parse_module
from atif_scan.data.python_file_requests import python_paths
from atif_scan.data.search_requests import command_requests

ESCAPES = "import re\nPATTERN = '\\d+\\/x'\nprint(open('/app/a.txt').read())\n"


def quiet(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    assert [str(w.message) for w in caught] == []
    return result


def test_parsing_invalid_escapes_is_silent_and_still_parses():
    assert quiet(lambda: parse_module(ESCAPES)).body
    assert quiet(lambda: parse_expression("['\\d']")).body


def test_parse_errors_still_raise():
    with pytest.raises(SyntaxError):
        quiet(lambda: parse_module("def ("))


def test_checks_reading_python_are_silent():
    quiet(lambda: command_requests(f"python3 - <<'PY'\n{ESCAPES}PY"))
    assert quiet(lambda: python_paths(ESCAPES))
