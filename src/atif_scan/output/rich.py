"""The rich (coloured, tabular) view of a per-trace report. Imported only by
`views.render_rich`, so rich stays optional: without it, callers fall back to text."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING

from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from .document import (
    STYLE,
    _cited,
    citation_lines,
    counts,
    detail_heading,
    detail_spread,
    footer,
    headline,
    reward_label,
    sections,
    summary_head_lines,
    summary_tail_lines,
    where,
)
from .document import filter_notice as _filter_notice
from .document import tally as _tally
from .document import unresolved as _unresolved

if TYPE_CHECKING:
    from ..data.jsonval import Doc


def _findings_table(group: dict[str, list[Doc]]) -> Table:
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("status")
    table.add_column("check")
    table.add_column("evidence / reason")
    for a in group["findings"]:
        sev = a["severity"]
        reason = where(a["evidence"])
        if explanation := a.get("explanation"):
            reason += f"\n{explanation}"
        table.add_row(Text(sev, style=STYLE[sev]), a["id"], reason)
    for a in group["expected"]:
        table.add_row(Text("expected", style="green"), a["id"], "by " + ", ".join(a["expected_by"]))
    return table


NEWLINE = "⏎"  # _flat/_one_line's line-break marker, dimmed so the prose reads through
MATCH = "bold reverse"


def _prose(text: str, style: str = "") -> Text:
    out = Text(text, style=style)
    out.highlight_words([NEWLINE], "dim")
    return out


def _cell(label: str, text: str | tuple[str, str, str]) -> Text:
    if isinstance(text, tuple):  # the match row: mark the match itself, no brackets
        before, match, after = text
        return Text.assemble(_prose(before), (match, MATCH), _prose(after))
    if label == "@":
        return Text(text, style="bold")
    return _prose(text, "yellow" if label == "warning" else "dim")


def citation_grid(cites: list[Doc], indent: int) -> Padding:
    """Citations as a two-column grid (label, text). Long text wraps inside its own
    column, so the indent survives the terminal width; a blank row separates citations."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True, style="dim", justify="right")  # why / match / result …
    grid.add_column(overflow="fold")
    for n, c in enumerate(cites):
        if n:
            grid.add_row("", "")
        for label, text in citation_lines(c):
            tag = {"@": "", ">": "match"}.get(label, label)
            grid.add_row(tag, _cell(label, text))
    return Padding(grid, (0, 0, 0, indent), expand=False)


def _print_heading(console: Console, item: Doc) -> None:
    style = STYLE.get(item["severity"] or "", "green")
    if item["input_status"] != "available":
        style = "bold magenta"
    console.print()
    console.print(Text(item["input_id"], style="bold"), Text(headline(item), style=style))
    if shape := counts(item):
        console.print(Text(shape, style="dim"))


def _print_citations(console: Console, item: Doc, group: dict[str, list[Doc]]) -> None:
    for a in group["findings"]:
        cited = _cited(item, a["id"])
        if cited:
            console.print(Text(f"  {a['id']}", style=STYLE[a["severity"]]))
        if cited:
            console.print(citation_grid(cited, 4))


def _print_item(console: Console, item: Doc) -> None:
    group = sections(item)
    _print_heading(console, item)
    table = _findings_table(group)
    if table.row_count:
        console.print(table)
    _print_citations(console, item, group)
    for line in _unresolved(item):
        console.print(Text(line, style="magenta"))
    if item["input_status"] == "available":
        console.print(Text(_tally(group), style="dim"))


def render(doc: Doc, file: IO[str] | None = None) -> None:
    console = Console(file=file, highlight=False)
    console.print(f"[bold]atif-scan[/] {doc['scanner_version']}")
    if notice := _filter_notice(doc):
        console.print(Text(notice, style="dim"))
    for item in doc["inputs"]:
        _print_item(console, item)
    console.print()
    console.print(Text(footer(doc), style="dim"))


# --- Summary ----------------------------------------------------------------------------


def _print_lines(console: Console, lines: list[str]) -> None:
    for line in lines:
        text = Text(line)
        for severity, style in STYLE.items():
            text.highlight_regex(rf"^\s+{severity}\b", style)  # check-table rows
        console.print(text)


def _print_trace(console: Console, t: Doc) -> None:
    reward = reward_label(t)
    style = "green" if t.get("reward") else "dim"
    console.print(
        Padding(
            Text.assemble(
                (t["input_id"], "bold"),
                "  ",
                (reward, style),
                "  " if reward else "",
                (where(t["evidence"]), "dim"),
            ),
            (0, 0, 0, 4),
        )
    )
    if t.get("citations"):
        console.print(citation_grid(t["citations"], 6))


def _print_details(console: Console, s: Doc) -> None:
    if not s["details"]:
        return
    console.print()
    console.print(Text(detail_heading(s), style="dim"))
    for d in s["details"]:
        console.print()
        console.print(
            Text.assemble(
                "  ",
                (f"{d['severity']:<6}", STYLE[d["severity"]]),
                " ",
                (d["check"], "bold"),
                (f" · {detail_spread(s, d)}", "dim"),
            )
        )
        for t in d["traces"]:
            _print_trace(console, t)


def render_summary(s: Doc, file: IO[str] | None = None) -> bool:
    """The summary with rich citations; False (nothing printed) off a terminal."""
    console = Console(file=file, highlight=False)
    if not console.is_terminal:
        return False
    _print_lines(console, summary_head_lines(s))
    _print_details(console, s)
    _print_lines(console, summary_tail_lines(s))
    return True
