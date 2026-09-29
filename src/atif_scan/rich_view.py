"""The rich (coloured, tabular) view of a per-trace report. Imported only by
`report.render_rich`, so rich stays optional: without it, callers fall back to text."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .report import STYLE, _cited, citation_lines, counts, footer, headline, sections, where
from .report import filter_notice as _filter_notice
from .report import tally as _tally
from .report import unresolved as _unresolved

if TYPE_CHECKING:
    from .jsonval import Doc


def _findings_table(group: dict[str, list[Doc]]) -> Table:
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("status")
    table.add_column("check")
    table.add_column("evidence / reason")
    for a in group["findings"]:
        sev = a["severity"]
        table.add_row(Text(sev, style=STYLE[sev]), a["id"], where(a["evidence"]))
    for a in group["expected"]:
        table.add_row(Text("expected", style="green"), a["id"], "by " + ", ".join(a["expected_by"]))
    return table


def _print_citation(console: Console, c: Doc) -> None:
    for label, text in citation_lines(c):
        if isinstance(text, tuple):  # the match row (`>`): mark the match itself
            before, match, after = text
            console.print(
                Text("    │ ", style="dim"),
                Text(before),
                Text(match, style="bold reverse"),
                Text(after),
                sep="",
                soft_wrap=True,
            )
        else:
            tag = "┌ " if label == "@" else f"│ {label}: "
            console.print(Text(f"    {tag}{text}", style="dim"), soft_wrap=True)


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
        for c in cited:
            _print_citation(console, c)


def _print_item(console: Console, item: Doc) -> None:
    group = sections(item)
    _print_heading(console, item)
    table = _findings_table(group)
    if table.row_count:
        console.print(table)
    _print_citations(console, item, group)
    for line in _unresolved(item, group):
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
