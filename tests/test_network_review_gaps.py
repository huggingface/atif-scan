"""Synthetic regressions for network evidence gaps found during blind review.

These strict expected failures record unresolved payload attribution, not cleared
network findings. Detector integration is deferred; never execute the source.
"""

import pytest
from test_network_discovery import result

from atif_scan import Status


@pytest.mark.parametrize(
    "source",
    [
        (
            "from pathlib import Path\n"
            'example = \'httpx.stream("GET", "https://docs.example.invalid/data")\'\n'
            "Path('docs/guide.md').write_text(example)\n"
        ),
        (
            "p = 'docs/api.md'\n"
            "text = open(p).read()\n"
            "example = 'httpx.get(\"https://docs.example.invalid/parts\")'\n"
            "text = text.replace('EXAMPLE', example)\n"
            "open(p, 'w').write(text)\n"
        ),
    ],
)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Known evidence gap: Python documentation payloads count as HTTP commands.",
)
def test_documentation_example_is_not_a_live_http_command(source):
    command = "python - <<'PY'\n" + source + "PY"
    assert result(command).status is Status.NO_MATCH
