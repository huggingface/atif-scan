"""Rich renderers, imported only when used so the core never loads rich.

`document` and `text` build documents and plain-text views; `rich` renders them with rich
and reads their helpers. These entry points keep that one-way: nothing below imports the
rich view, and if rich can't be imported callers fall back to text.
"""

from __future__ import annotations

from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.jsonval import Doc


def render_rich(doc: Doc, file: IO[str] | None = None) -> None:
    """Rich view; ImportError if rich is unavailable."""
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
