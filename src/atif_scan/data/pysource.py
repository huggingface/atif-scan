"""Parse untrusted Python source (trace scripts, task files) for its syntax tree, silently.

`ast.parse` warns about the source it reads (`SyntaxWarning: invalid escape sequence`),
and those warnings would otherwise print once per script to the scanner's stderr: noise
about the agent's code that buries the scanner's own messages. Parsing never runs the
code. Errors still raise, so callers keep deciding what unparsable source means.

Not thread-safe (`warnings.catch_warnings` swaps process-wide state): scans evaluate
traces on one thread.
"""

from __future__ import annotations

import ast
import warnings


def parse_module(source: str) -> ast.Module:
    """`ast.parse(source)` without warnings; raises SyntaxError/ValueError/RecursionError."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ast.parse(source)


def parse_expression(source: str) -> ast.Expression:
    """`ast.parse(source, mode="eval")` without warnings; raises like `parse_module`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ast.parse(source, mode="eval")
