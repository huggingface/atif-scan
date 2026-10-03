"""Optional rich renderers, imported only when used: rich is an optional dependency.

`report` builds documents and plain-text views; `rich_view` renders them with rich and
reads `report`'s helpers. These entry points keep that one-way: nothing below imports the
rich view, and without rich installed callers fall back to text.
"""

from __future__ import annotations

from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from ..jsonval import Doc


def render_rich(doc: Doc, file: IO[str] | None = None) -> None:
    """Rich view; requires the optional `pretty` extra (ImportError without it)."""
    from .rich import render  # noqa: PLC0415 - rich is optional: import it only here

    render(doc, file)


def render_summary_rich(s: Doc, file: IO[str] | None = None) -> bool:
    """Rich summary on a terminal; False (nothing printed) when rich is missing or the
    output isn't a terminal, so the caller prints `summary_text`."""
    try:
        from .rich import render_summary  # noqa: PLC0415 - rich is optional
    except ImportError:
        return False
    return render_summary(s, file)
