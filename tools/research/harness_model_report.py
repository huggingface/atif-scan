"""Private, offline model-request report from a completed audit database.

Usage: uv run python tools/research/harness_model_report.py --database PRIVATE_SQLITE
       --output NEW_PRIVATE_DIRECTORY
Only audited identifiers, counts, codes and event locators are exported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from harness_report import _csv, _file, _output, _source, rows

if TYPE_CHECKING:
    import sqlite3

    from atif_scan.data.jsonval import Doc

MODEL_USAGE = """
SELECT model,COUNT(DISTINCT trace_id) traces,
 COUNT(DISTINCT CASE WHEN rejected_requests>0 THEN trace_id END) affected_traces,
 SUM(recorded_requests) recorded_requests,SUM(rejected_requests) rejected_requests,
 SUM(recorded_requests) rejection_rate_denominator,
 CAST(SUM(rejected_requests) AS REAL)/NULLIF(SUM(recorded_requests),0) rejection_rate,
 SUM(unknown_tool_labels) unknown_tool_label_requests,
 SUM(CASE WHEN model_source='step' THEN recorded_requests ELSE 0 END) step_requests,
 SUM(CASE WHEN model_source='root' THEN recorded_requests ELSE 0 END) root_requests,
 SUM(CASE WHEN model_source='unknown' THEN recorded_requests ELSE 0 END) unknown_source_requests,
 COUNT(DISTINCT CASE WHEN model_source='step' THEN trace_id END) step_traces,
 COUNT(DISTINCT CASE WHEN model_source='root' THEN trace_id END) root_traces,
 COUNT(DISTINCT CASE WHEN model_source='unknown' THEN trace_id END) unknown_source_traces
FROM model_usage GROUP BY model ORDER BY model
"""
MODEL_TOOLS = """
WITH totals AS (
 SELECT model,COUNT(DISTINCT trace_id) model_traces,
 SUM(recorded_requests) model_recorded_requests,
 SUM(rejected_requests) model_rejected_requests FROM model_usage GROUP BY model
), events AS (
 SELECT model,tool,reason,COUNT(*) rejected_requests,
 COUNT(DISTINCT trace_id) affected_traces
 FROM rejected_tools GROUP BY model,tool,reason
)
SELECT e.model,e.tool,e.reason,e.rejected_requests,e.affected_traces,
 t.model_traces,t.model_recorded_requests,t.model_rejected_requests,
 t.model_recorded_requests rejection_rate_denominator,
 CAST(e.rejected_requests AS REAL)/NULLIF(t.model_recorded_requests,0) rejection_rate
FROM events e LEFT JOIN totals t ON e.model IS t.model
ORDER BY e.model,e.tool,e.reason
"""
REJECTED_REQUESTS = """
SELECT trace_id,step,call,tool,model,model_source,observation,reason,
 declaration_conflict,pairing
FROM rejected_tools ORDER BY trace_id,step,call,observation
"""
ATTRIBUTION = """
SELECT model,model_source,COUNT(DISTINCT trace_id) traces,
 COUNT(DISTINCT CASE WHEN rejected_requests>0 THEN trace_id END) affected_traces,
 SUM(recorded_requests) recorded_requests,SUM(rejected_requests) rejected_requests,
 SUM(recorded_requests) rejection_rate_denominator,
 CAST(SUM(rejected_requests) AS REAL)/NULLIF(SUM(recorded_requests),0) rejection_rate,
 SUM(unknown_tool_labels) unknown_tool_label_requests
FROM model_usage GROUP BY model,model_source ORDER BY model,model_source
"""
TABLES = {
    "model_usage": MODEL_USAGE,
    "model_tools": MODEL_TOOLS,
    "rejected_requests": REJECTED_REQUESTS,
    "attribution": ATTRIBUTION,
}
CORPUS = """
SELECT (SELECT COUNT(*) FROM traces) traces,
 COUNT(DISTINCT trace_id) traces_with_recorded_requests,
 COALESCE(SUM(recorded_requests),0) recorded_requests,
 COALESCE(SUM(rejected_requests),0) rejected_requests,
 COALESCE(SUM(unknown_tool_labels),0) unknown_tool_label_requests,
 COALESCE(SUM(CASE WHEN model IS NULL THEN recorded_requests ELSE 0 END),0)
 unknown_model_requests,
 (SELECT COUNT(*) FROM failures) failures,
 (SELECT SUM(json_extract(data,'$.dropped_steps')) FROM traces) dropped_steps,
 (SELECT COUNT(*) FROM traces WHERE json_type(data,'$.dropped_steps') IS NOT 'integer')
 unknown_dropped_steps_traces,
 (SELECT COUNT(*) FROM traces WHERE json_extract(data,'$.redacted_values')>0)
 traces_with_redacted_input,
 (SELECT COUNT(*) FROM traces WHERE json_type(data,'$.redacted_values') IS NOT 'integer')
 unknown_redaction_coverage_traces
FROM model_usage
"""


def _validate_schema(db: sqlite3.Connection) -> None:
    """Require the new audited tables before creating an output directory."""
    for query in (*TABLES.values(), CORPUS):
        db.execute(query + " LIMIT 0")


def _methodology() -> str:
    return """
## Definitions and limitations

- This is a descriptive audit of recorded model requests across the explicitly
  downloaded benchmark selection, NOT a causal model ranking. Task-mix differences
  are unknown; environments, configurations and observation coverage may differ.
- Model identity is the exact recorded identifier. Aliases remain separate; no
  equivalent models are guessed. An explicit step model is authoritative; root
  fallback is used only as audited. `model_source` is `step`, `root` or `unknown`.
  SQL/JSON NULL models form a separate unknown group, not an identifier named unknown.
- `recorded_requests` includes rejected requests, missing results and unknown tool
  labels. Derived code-mode call sites are excluded: they are not recorded requests.
  Counts cover the already deduplicated audited traces, including embedded sessions;
  no further selection or deduplication is applied here.
- `rejected_requests` counts only explicit supported tool-dispatch error envelopes
  paired to a recorded request: `not_found` or `unavailable`. Invalid arguments,
  shell command-not-found errors, quoted mentions and unpaired observations are not
  dispatch rejections. These events describe MODEL requests, not proven nonexistence
  or a diagnosis of why dispatch rejected them. No raw error text is exported.
- In model_usage.csv and attribution.csv, `rejection_rate` is rejected requests /
  recorded requests in that group. `rejection_rate_denominator` states this explicitly.
  In model_tools.csv, the numerator is the model/tool/reason event count and the
  denominator is ALL recorded requests for that model, NOT requests for that tool
  (tool-specific attempt denominators are unavailable). Denominators are aggregated
  before joining events, across all attribution sources, to avoid multiplication.
- `traces` counts distinct traces in each usage group; `affected_traces` counts
  distinct traces with a rejection in that group. Source trace counts can overlap;
  source request counts partition requests. Unknown tool labels are counted, not
  dropped. A withheld event tool label is NULL (blank in CSV).
- A positive denominator with no recognized rejections gives rate zero, meaning
  no recognized recorded events, NOT clean absence. A zero denominator gives NULL,
  never zero. CSV NULL is blank; JSON NULL is null. Models with no usage rows cannot
  be enumerated from this schema and are not assigned zero rates.
- Failures, unknown models, dropped steps, missing observations and unsupported
  envelopes are evidence gaps, not clean results or proof of tool availability.
  Corpus coverage counts remain separate from rate denominators. The exporter
  requires a completed database; it cannot assess missing downloads or task mix.
- rejected_requests.csv exports only opaque trace IDs, numeric step/call/observation
  locators, audited model/tool labels, source and reason codes, declaration_conflict
  and pairing. Declaration conflict records known declared membership at rejection;
  NULL means unknown. No trace content, paths, URLs, call IDs or raw arguments appear.
"""


def _report(output: int, models: list[Doc], corpus: Doc) -> None:
    lines = [
        "# Benchmark model tool-request audit",
        "",
        f"Recorded requests: {corpus['recorded_requests']}; "
        f"recognized dispatch rejections: {corpus['rejected_requests']}.",
        f"Coverage: {corpus['unknown_model_requests']} unknown-model requests; "
        f"{corpus['unknown_tool_label_requests']} unknown-tool-label requests; "
        f"{corpus['failures']} failures; {corpus['dropped_steps']} dropped steps; "
        f"{corpus['unknown_dropped_steps_traces']} traces with unknown dropped-step counts.",
        "",
        "| Model | Traces | Affected traces | Recorded requests (denominator) |"
        " Rejected requests | Rejection rate | Step / root / unknown requests |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in models:
        model = row["model"] if row["model"] is not None else "(unknown model)"
        rate = "NULL" if row["rejection_rate"] is None else str(row["rejection_rate"])
        lines.append(
            f"| {model} | {row['traces']} | {row['affected_traces']} | "
            f"{row['rejection_rate_denominator']} | {row['rejected_requests']} | {rate} | "
            f"{row['step_requests']} / {row['root_requests']} / {row['unknown_source_requests']} |"
        )
    with _file(output, "REPORT.md") as stream:
        stream.write("\n".join(lines) + "\n" + _methodology())


def _export(db: sqlite3.Connection, output: int) -> Doc:
    counts: Doc = {name: _csv(db, output, name, query) for name, query in TABLES.items()}
    models = list(rows(db, MODEL_USAGE))
    corpus = next(rows(db, CORPUS))
    summary: Doc = {
        "schema_version": 1,
        "selection": "Downloaded benchmark selection; descriptive, not a causal model ranking",
        "tables": counts,
        "corpus": corpus,
        "models": models,
        "attribution": list(rows(db, ATTRIBUTION)),
    }
    with _file(output, "summary.json") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    _report(output, models, corpus)
    return summary


def export(database: Path, output: Path) -> Doc:
    """Export allowlisted model evidence to a new private no-follow directory."""
    with _source(database) as db:
        _validate_schema(db)
        with _output(output) as folder:
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
