"""Allowlisted CSV/JSON exports from a completed offline harness audit database."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import stat
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from harness_corpus import CorpusError, _absolute, _directory

from atif_scan.data.jsonval import as_object

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import TextIO

    from atif_scan.data.jsonval import Doc

HARNESSES = """
WITH a AS (SELECT family,COUNT(*) traces,SUM(scope='primary') primary_traces,
 SUM(scope='nested') nested_traces,COUNT(DISTINCT run_id) runs,
 SUM(json_extract(data,'$.recorded_calls')) recorded_calls,
 SUM(json_extract(data,'$.static_call_sites')) static_call_sites,
 SUM(json_extract(data,'$.inventory_status')='known') declared_inventory_traces,
 SUM(json_extract(data,'$.dropped_steps')) dropped_steps,
 SUM(json_extract(data,'$.shell_calls_missing_results')) shell_calls_missing_results
 FROM traces GROUP BY family),
 b AS (SELECT family,COUNT(DISTINCT tool) cumulative_tools,
 COUNT(DISTINCT CASE WHEN declared=1 THEN tool END) declared_tools_seen,
 COUNT(DISTINCT CASE WHEN recorded_calls>0 THEN tool END) directly_used_tools,
 COUNT(DISTINCT CASE WHEN static_call_sites>0 THEN tool END) statically_referenced_tools
 FROM tools JOIN traces USING(trace_id) GROUP BY family)
SELECT a.*,b.cumulative_tools,b.declared_tools_seen,b.directly_used_tools,
 b.statically_referenced_tools FROM a LEFT JOIN b USING(family) ORDER BY family
"""
TOOLS = """
WITH totals AS (SELECT family,COUNT(*) family_traces,
 SUM(json_extract(data,'$.recorded_calls')) family_recorded_calls FROM traces GROUP BY family)
SELECT t.family,u.tool,SUM(u.recorded_calls) recorded_calls,
 SUM(u.static_call_sites) static_call_sites,
 SUM(u.recorded_calls>0) traces_with_recorded_use,
 SUM(u.static_call_sites>0) traces_with_static_reference,
 SUM(u.declared=1) declared_offered_traces,
 SUM(u.offered_unused=1) confirmed_offered_unused_traces,
 GROUP_CONCAT(DISTINCT u.inventory_source) inventory_sources,
 z.family_traces,z.family_recorded_calls,
 CAST(SUM(u.recorded_calls) AS REAL)/NULLIF(z.family_recorded_calls,0) recorded_call_share
FROM tools u JOIN traces t USING(trace_id) JOIN totals z USING(family)
GROUP BY t.family,u.tool ORDER BY t.family,u.tool
"""
VARIANTS = """
WITH a AS (SELECT family,COALESCE(signature,'unknown') signature,COUNT(*) traces,
 GROUP_CONCAT(DISTINCT version) versions,
 GROUP_CONCAT(DISTINCT json_extract(data,'$.inventory_source')) sources
 FROM traces GROUP BY family,signature),
 b AS (SELECT family,COALESCE(signature,'unknown') signature,COUNT(DISTINCT tool) cumulative_tools,
 COUNT(DISTINCT CASE WHEN declared=1 THEN tool END) declared_tools,
 SUM(recorded_calls) recorded_calls,SUM(static_call_sites) static_call_sites
 FROM traces JOIN tools USING(trace_id) GROUP BY family,signature)
SELECT a.*,b.cumulative_tools,b.declared_tools,b.recorded_calls,b.static_call_sites
FROM a LEFT JOIN b USING(family,signature) ORDER BY family,signature
"""
VARIANT_TOOLS = """
SELECT family,COALESCE(signature,'unknown') signature,tool,
 SUM(recorded_calls) recorded_calls,SUM(static_call_sites) static_call_sites,
 SUM(declared=1) declared_offered_traces,SUM(offered_unused=1) confirmed_offered_unused_traces
FROM traces JOIN tools USING(trace_id) GROUP BY family,signature,tool ORDER BY family,signature,tool
"""
RUN_TOOLS = """
WITH r AS (SELECT family,COALESCE(run_id,'unassigned') run_id,COUNT(*) traces
 FROM traces GROUP BY family,run_id),
 names AS (SELECT DISTINCT family,tool FROM traces JOIN tools USING(trace_id)),
 usage AS (SELECT family,COALESCE(run_id,'unassigned') run_id,tool,
 SUM(recorded_calls) recorded_calls,SUM(static_call_sites) static_call_sites,
 SUM(declared=1) declared_offered_traces,SUM(offered_unused=1) offered_unused
 FROM traces JOIN tools USING(trace_id) GROUP BY family,run_id,tool)
SELECT r.family,r.run_id,names.tool,r.traces,COALESCE(usage.recorded_calls,0) recorded_calls,
 COALESCE(usage.static_call_sites,0) static_call_sites,
 usage.declared_offered_traces,usage.offered_unused confirmed_offered_unused_traces
FROM r JOIN names USING(family) LEFT JOIN usage USING(family,run_id,tool)
ORDER BY r.family,r.run_id,names.tool
"""
SHELL = """
WITH events AS (SELECT family,context,COUNT(*) output_records,
 COUNT(truncation) assessed_output_records,
 SUM(truncation) recognized_truncations,
 CAST(SUM(truncation) AS REAL)/NULLIF(COUNT(truncation),0) recognized_truncation_rate,
 SUM(json_extract(o.data,'$.kind')='possible_signal') possible_signals,
 SUM(json_extract(o.data,'$.kind')='unsupported') unsupported_records,
 SUM(json_extract(o.data,'$.kind')='terminal_screen') terminal_screen_records,
 SUM(truncation=1 AND retention='file') file_spool_events,
 SUM(truncation=1 AND retention='file' AND followup_observed=1) file_followed_events,
 SUM(truncation=1 AND retention='file' AND followup_observed IS NOT NULL) file_followup_eligible,
 SUM(truncation=1 AND retention='process_handle') handle_events,
 SUM(truncation=1 AND retention='process_handle' AND followup_observed=1) handle_followed_events,
 SUM(truncation=1 AND retention='process_handle' AND followup_observed IS NOT NULL)
 handle_followup_eligible,
 SUM(truncation=1 AND retention='not_indicated') truncated_without_retention_notice,
 SUM(truncation=1 AND retention='unresolved_retention') unresolved_retention_events,
 AVG(returned_chars) mean_returned_chars,NULLIF(p50(returned_chars),-1) p50_returned_chars,
 NULLIF(p95(returned_chars),-1) p95_returned_chars,MAX(returned_chars) max_returned_chars,
 AVG(CASE WHEN truncation=1 THEN returned_chars END) truncated_mean_chars,
 NULLIF(p50(CASE WHEN truncation=1 THEN returned_chars END),-1) truncated_p50_chars,
 NULLIF(p95(CASE WHEN truncation=1 THEN returned_chars END),-1) truncated_p95_chars,
 NULLIF(p99(CASE WHEN truncation=1 THEN returned_chars END),-1) truncated_p99_chars,
 MAX(CASE WHEN truncation=1 THEN returned_chars END) truncated_max_chars,
 AVG(CASE WHEN truncation=1 THEN returned_bytes END) truncated_mean_bytes,
 AVG(CASE WHEN truncation=1 THEN json_extract(o.data,'$.payload_chars') END)
 truncated_mean_payload_chars,
 COUNT(CASE WHEN truncation=1 THEN json_extract(o.data,'$.payload_chars') END)
 known_truncated_payload_records
FROM outputs o JOIN traces USING(trace_id) GROUP BY family,context)
SELECT events.*,
 CAST(file_followed_events AS REAL)/NULLIF(file_followup_eligible,0) file_followup_rate,
 CAST(handle_followed_events AS REAL)/NULLIF(handle_followup_eligible,0) handle_followup_rate
FROM events ORDER BY family,context
"""
VISIBILITY = """
SELECT family,context,
 COALESCE(json_extract(o.data,'$.visibility_basis'),'unknown') visibility_basis,
 truncation,COUNT(*) output_records,
 COUNT(json_extract(o.data,'$.model_visible_chars')) model_visible_records,
 AVG(json_extract(o.data,'$.model_visible_chars')) mean_model_visible_chars,
 NULLIF(p50(json_extract(o.data,'$.model_visible_chars')),-1) p50_model_visible_chars,
 NULLIF(p95(json_extract(o.data,'$.model_visible_chars')),-1) p95_model_visible_chars,
 MAX(json_extract(o.data,'$.model_visible_chars')) max_model_visible_chars,
 AVG(json_extract(o.data,'$.model_visible_bytes')) mean_model_visible_bytes,
 AVG(json_extract(o.data,'$.recorded_chars')) mean_recorded_chars,
 NULLIF(p50(json_extract(o.data,'$.recorded_chars')),-1) p50_recorded_chars,
 AVG(json_extract(o.data,'$.payload_chars')) mean_payload_chars,
 COUNT(json_extract(o.data,'$.payload_chars')) known_payload_records
FROM outputs o JOIN traces USING(trace_id)
GROUP BY family,context,visibility_basis,truncation
ORDER BY family,context,visibility_basis,truncation
"""
BOUNDARIES = """
SELECT family,context,marker,json_extract(o.data,'$.limit') observed_limit,
 json_extract(o.data,'$.limit_unit') limit_unit,
 json_extract(o.data,'$.requested_max_output_tokens') requested_max_output_tokens,
 retention,COUNT(*) output_records,AVG(returned_chars) mean_returned_chars,
 NULLIF(p50(returned_chars),-1) p50_returned_chars,
 NULLIF(p95(returned_chars),-1) p95_returned_chars,
 MIN(returned_chars) min_returned_chars,MAX(returned_chars) max_returned_chars
FROM outputs o JOIN traces USING(trace_id) WHERE truncation=1
GROUP BY family,context,marker,observed_limit,limit_unit,requested_max_output_tokens,retention
ORDER BY family,context,marker,observed_limit
"""
TRACES = """
SELECT trace_id,family,signature,version,scope,parent_id,run_id,
 json_extract(data,'$.inventory_source') inventory_source,
 json_extract(data,'$.inventory_status') inventory_status,
 json_extract(data,'$.declared_tools') declared_tools,
 json_extract(data,'$.recorded_calls') recorded_calls,
 json_extract(data,'$.static_call_sites') static_call_sites,
 json_extract(data,'$.agent_steps') agent_steps,
 json_extract(data,'$.dropped_steps') dropped_steps,
 json_extract(data,'$.redacted_values') redacted_values,
 json_extract(data,'$.redacted_values_basis') redacted_values_basis,
 json_extract(data,'$.shell_calls_missing_results') shell_calls_missing_results,
 json_extract(data,'$.unclassified_observations') unclassified_observations,
 json_extract(data,'$.shell_output_records') shell_output_records,
 json_extract(data,'$.truncated_output_records') truncated_output_records,
 json_extract(data,'$.issues') issues FROM traces ORDER BY family,trace_id
"""
TABLES = {
    "harnesses": HARNESSES,
    "tools": TOOLS,
    "variants": VARIANTS,
    "variant_tools": VARIANT_TOOLS,
    "run_tools": RUN_TOOLS,
    "shell_summary": SHELL,
    "visibility": VISIBILITY,
    "boundaries": BOUNDARIES,
    "traces": TRACES,
    "trace_tools": (
        "SELECT trace_id,tool,recorded_calls,static_call_sites,declared,offered_unused,"
        "inventory_source FROM tools"
    ),
    "followups": (
        "SELECT trace_id,origin_step,origin_observation,step,call,tool,access_kind,"
        "partial_requested,outcome,delay_calls FROM followups"
    ),
    "copies": "SELECT trace_id,source_id,root,run_id,raw_sha256 FROM copies",
    "native_sources": "SELECT trace_id,definition_sha256,status FROM native_sources",
    "failures": "SELECT trace_id,code FROM failures",
}
EVENT_FIELDS = (
    "step",
    "observation",
    "tool",
    "context",
    "pairing",
    "text_understood",
    "has_media",
    "truncation",
    "marker",
    "kind",
    "spool_indicated",
    "spool_complete",
    "reported_original_chars",
    "reported_original_bytes",
    "reported_original_lines",
    "reported_original_tokens",
    "estimated_original_tokens",
    "omitted_chars",
    "omitted_bytes",
    "limit",
    "limit_unit",
    "payload_chars",
    "payload_bytes",
    "returned_chars",
    "returned_bytes",
    "recorded_chars",
    "recorded_bytes",
    "model_visible_chars",
    "model_visible_bytes",
    "visibility_basis",
    "returned_basis",
    "requested_max_output_tokens",
    "retention",
    "followup_accesses",
    "followup_observed",
    "first_followup_delay_calls",
    "content_returned_accesses",
    "partial_read_accesses",
)


class Quantile:
    fraction = 0.5

    def __init__(self) -> None:
        self.counts: Counter[int] = Counter()

    def step(self, value: object) -> None:
        if type(value) is int and value >= 0:
            self.counts[value] += 1

    def finalize(self) -> int:
        """Return -1 for no lengths; every SQL projection maps this sentinel to NULL."""
        target = math.ceil(sum(self.counts.values()) * self.fraction)
        if not target:
            return -1
        seen = 0
        for value, number in sorted(self.counts.items()):
            seen += number
            if seen >= target:
                return value
        return -1


class P95(Quantile):
    fraction = 0.95


class P99(Quantile):
    fraction = 0.99


def rows(db: sqlite3.Connection, query: str) -> Iterator[Doc]:
    cursor = db.execute(query)
    names = [column[0] for column in cursor.description]
    for row in cursor:
        yield dict(zip(names, row, strict=True))


def _csv(db: sqlite3.Connection, output: int, name: str, query: str) -> int:
    cursor = db.execute(query)
    with _file(output, f"{name}.csv") as stream:
        writer = csv.writer(stream)
        writer.writerow(column[0] for column in cursor.description)
        count = 0
        for row in cursor:
            writer.writerow(row)
            count += 1
    return count


def _events(db: sqlite3.Connection, output: int) -> int:
    query = """SELECT o.trace_id,family,signature,o.data FROM outputs o JOIN traces USING(trace_id)
    WHERE truncation=1 OR json_extract(o.data,'$.kind') IN ('possible_signal','terminal_screen')"""
    count = 0
    with _file(output, "truncations.jsonl") as stream:
        for trace, family, signature, text in db.execute(query):
            data = as_object(json.loads(text))
            row: Doc = {"trace_id": trace, "family": family, "signature": signature}
            row.update({key: data.get(key) for key in EVENT_FIELDS})
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def _methodology() -> str:
    return """
## Definitions and limitations

- Family is the harness, not the version. `variants.csv` splits known declared
  tool/schema interfaces only. `unknown` inventory is not assigned to a known variant.
  Version lists are provenance. Interface variants are not independent harnesses.
- Cumulative tool counts combine declarations, recorded uses, and static code-mode
  references. They describe evidenced names, not a verified full inventory when
  declarations are missing. Unresolved requests/references need not establish runtime
  availability; source columns preserve that distinction. The family union may exceed
  any single configuration's tool count.
- `recorded_calls` counts trace-recorded tool invocations. `static_call_sites` counts
  statically recovered code-mode references, NOT executions: branches, loops and runtime
  dispatch prevent exact invocation accounting. Never add the two columns as calls.
- A zero `offered_unused` opportunity is established only by a known declaration and
  no recorded/static use with no dropped steps. SQL/CSV NULL (blank) means unknown;
  zero recorded calls in a cumulative run grid does NOT prove the tool was offered.
- Both primary and embedded benchmark subagent sessions are analyzed. Counts remain
  separate in harnesses/traces; nested sessions inherit their parent's run association.
  Analysis-tool judge sessions are excluded. Copied context is excluded.
- Exact-byte and canonical-JSON copies are identified separately. Session IDs do not
  deduplicate executions. Shared identical traces join mirrored run locations; an
  archive copy does not create a run. `copies.csv` links opaque source IDs to the
  separate private inventory database, which alone contains local paths.
- Truncation frequencies use recorded eligible output observations, not commands.
  Shared terminal, code-mode shared and command-or-subagent contexts remain separate.
  No recognized marker means no recognized evidence, NOT proof of complete output.
  Unsupported/possible notices, missing results and dropped steps are coverage gaps.
  Unsupported/possible assessments are NULL and excluded from `assessed_output_records`,
  the rate denominator; an entirely unsupported group has an unknown rate, not zero.
  Redacted input conservatively prevents offered-unused certification; per-trace counts
  record whether redaction coverage belongs to the document or an inherited parent.
  File/handle follow-up rates divide followed events by eligible events only; unknown
  follow-up eligibility is excluded, and no eligible events gives NULL, not zero.
  `truncated_without_retention_notice` means no advertised spool for that family/context,
  not proof that the harness has no spooling implementation.
- `returned_*` measures the selected evidence text: verified native tool-result
  content where available, otherwise decoded ATIF content as a proxy. `returned_basis`
  identifies that choice. `recorded_*` preserves the exported text measurement;
  `model_visible_*` is populated only for verified native content and otherwise NULL.
  Claude exports can append stdout and metadata that was not part of the model's
  tool-result block; these extras are excluded from verified model-visible measurements.
  `visibility.csv` separates native measurements from unknown/proxy coverage.
  Lengths are Unicode characters and UTF-8 bytes, NOT token counts or a reconstruction
  of the full provider prompt. Payload lengths appear only where supported. Units of
  thresholds remain distinct; Claude KB is not converted to bytes. Quantiles use nearest rank.
- Explicitly rejected nonexistent/unavailable tool requests are excluded from harness
  usage and inventory evidence; they belong in the separate MODEL tool-request report.
  Unreadable tool names are coverage gaps, not an invented inventory entry. Declaration
  conflicts retain the declaration but make the offered-unused assessment unknown.
  Other invalid-argument errors still count as attempts to use the named existing tool.
- File and process-handle retention are separate. `not_indicated` means the recognized
  notice did not advertise retention, not that no spool implementation ever exists.
  Any later literal-path access in the same trace counts; content reads and metadata
  access are distinguished. No same-step or cross-trace accesses count. A followed
  event can have multiple accesses; one access can follow multiple notices of a reused
  spool path. Static code-mode references are not observed follow-ups. Exact lexical
  paths and supported readers only: variables, aliases,
  unsupported commands and missing continuation evidence can hide accesses.
- Follow-up negatives mean no matching recorded access was found. Attempted access is
  separate from returned-content evidence, explicit failure, and partial-read requests.
  Full recovery of omitted output is not established. Process status/wait is not a spool
  read; matching process.read_output is. Spool files themselves are never opened.
- ATIF declarations are primary inventory evidence. Native Grok definitions are used
  only with exact session matching beside a pinned downloaded copy; digests and source
  statuses are in native_sources.csv. Local harness source validates marker mechanics,
  not unrecorded historical availability or default settings.

## Files and queries

`analysis.sqlite3` (in the analysis directory) holds allowlisted per-output details;
these CSVs/JSONL are convenience projections. The database schema is in harness_audit.py.
Use `tools.csv` for family use, `variant_tools.csv` for declared interfaces,
`run_tools.csv` for a complete run-by-cumulative-tool grid, `trace_tools.csv` for
recorded/declared per-trace tools, and `boundaries.csv` for threshold/returned-size groups.
`truncations.jsonl` also includes possible notices and terminal-screen events; filter
`truncation == true` for the explicit recognized numerator.

```sql
SELECT t.family, o.marker, COUNT(*) AS events,
       AVG(o.returned_chars) AS mean_returned_chars
FROM outputs o JOIN traces t USING(trace_id)
WHERE o.truncation = 1 GROUP BY t.family, o.marker;
```
"""


def _number(value: object) -> str:
    return str(value) if isinstance(value, int | float) else "unknown"


def _report(output: int, harnesses: list[Doc], shell: list[Doc]) -> None:
    lines = [
        "# Benchmark harness tool/output audit",
        "",
        "| Harness | Primary | Nested | Cumulative tools | Recorded calls | Static call sites |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in harnesses:
        values = [
            row[k]
            for k in (
                "family",
                "primary_traces",
                "nested_traces",
                "cumulative_tools",
                "recorded_calls",
                "static_call_sites",
            )
        ]
        lines.append("| " + " | ".join(str(v) for v in values) + " |")
    lines.extend(
        [
            "",
            "## Recognized shell-output truncation",
            "",
            "Lengths use verified native content where available, otherwise recorded text. "
            "See visibility.csv for coverage; "
            "unsupported truncation detection is not a zero result.",
            "",
            "| Harness | Context | Outputs | Truncated | Truncated chars p50 / p95 |"
            " File followed/eligible | Handle followed/eligible |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in shell:
        lines.append(
            f"| {row['family']} | {row['context']} | {row['output_records']} | "
            f"{_number(row['recognized_truncations'])} | "
            f"{_number(row['truncated_p50_chars'])} / {_number(row['truncated_p95_chars'])} | "
            f"{row['file_followed_events']}/{row['file_followup_eligible']} | "
            f"{row['handle_followed_events']}/{row['handle_followup_eligible']} |"
        )
    with _file(output, "REPORT.md") as stream:
        stream.write("\n".join(lines) + "\n" + _methodology())


@contextmanager
def _source(database: Path) -> Iterator[sqlite3.Connection]:
    """Pin a regular input, refusing symlink ancestors and incomplete selections."""
    path = _absolute(database)
    repo = Path(__file__).resolve().parents[2]
    if path.is_relative_to(repo):
        raise CorpusError("input_inside_repository")
    try:
        with _directory(path.parent) as parent:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise CorpusError("invalid_analysis")
            db = sqlite3.connect(f"file:/proc/self/fd/{fd}?mode=ro", uri=True)
            try:
                db.execute("PRAGMA query_only=ON")
                db.execute("BEGIN")
                if db.execute("SELECT complete FROM state").fetchall() != [(1,)]:
                    raise CorpusError("analysis_incomplete")
                yield db
            finally:
                db.close()
        finally:
            os.close(fd)
    except (OSError, sqlite3.Error):
        raise CorpusError("invalid_analysis") from None


@contextmanager
def _output(output: Path) -> Iterator[int]:
    """Create and pin a new private directory; never resolve symlink parents."""
    path = _absolute(output)
    repo = Path(__file__).resolve().parents[2]
    if path.is_relative_to(repo):
        raise CorpusError("output_inside_repository")
    try:
        with _directory(path.parent) as parent:
            os.mkdir(path.name, mode=0o700, dir_fd=parent)
            fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            os.fchmod(fd, 0o700)
            yield fd
        finally:
            os.close(fd)
    except OSError:
        raise CorpusError("private_output_unavailable") from None


@contextmanager
def _file(folder: int, name: str) -> Iterator[TextIO]:
    """Only fixed exporter filenames enter here; exclusive no-follow creation."""
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=folder)
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        yield stream


def _export(db: sqlite3.Connection, output: int) -> Doc:
    db.create_aggregate("p50", 1, Quantile)
    db.create_aggregate("p95", 1, P95)
    db.create_aggregate("p99", 1, P99)
    counts: Doc = {name: _csv(db, output, name, query) for name, query in TABLES.items()}
    counts["truncations"] = _events(db, output)
    harnesses = list(rows(db, HARNESSES))
    shell = list(rows(db, SHELL))
    summary: Doc = {
        "schema_version": 2,
        "tables": counts,
        "harnesses": harnesses,
        "shell": shell,
    }
    with _file(output, "summary.json") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    _report(output, harnesses, shell)
    return summary


def export(database: Path, output: Path) -> Doc:
    with _source(database) as db, _output(output) as folder:
        return _export(db, folder)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = export(args.database.absolute(), args.output.absolute())
    print(json.dumps(summary["tables"], indent=2))


if __name__ == "__main__":
    main()
