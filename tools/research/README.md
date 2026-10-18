# Research scripts (TB2.1, TB4)

Campaign scripts behind the Terminal-Bench 2.1 and 4 evaluations, kept so the label sets
can be rebuilt. They are not product tooling: they have no tests yet, they need your own
`gh` and `harbor` logins, and most of them use the network (GitHub, Harbor Hub).

| Script | Does |
|---|---|
| `tb21_inventory.py` | TB2.1 leaderboard PRs, Hub rows, their jobs and maintainer rulings → `OUT/inventory.json` |
| `tb21_labels.py` | Rulings mapped onto public trials → `INV/labels.json` (the input of `atif-scan labels import-tb21`) |
| `tb21_judge.py` | The TB2.1 judge's per-trial verdicts, downloaded and matched to public trials |
| `tb21_eval.py` | Scan every labelled job, recall against rulings, and pick a blind hunt pilot |
| `tb4_inventory.py` | TB4 leaderboard rows, judge reports and cheat trials (imports `tb21_inventory`) |
| `tb4_hunt.py` | A blind hunt pilot over TB4 scans, and its answer counts per group |

Everything they write names real runs, and `tb21_judge.py` and `tb4_inventory.py`
download real trajectories and artifacts. Write outputs outside the repository, under
the atif-scan home, with `umask 077`.

The pilot builders overlap `atif-scan labels disagreements` (and their keys use `row`
where it uses `run`); `recall` and `score` overlap `labels import-*` followed by `labels eval`.
Folding them into `atif-scan labels` is the way forward; see
[docs/improvement-loop.md](../../docs/improvement-loop.md).

## Offline harness audit

Unlike the campaign scripts above, `harness_audit.py` is stdlib-only and offline.
First build a complete private inventory with `harness_corpus.build_inventory`,
explicitly selecting the corpus roots. Then use a **new**, private output directory:

```bash
uv run python tools/research/harness_audit.py \
  --inventory /external/private/inventory.sqlite3 \
  --output /external/private/new-analysis
```

`--limit N` selects a pilot; even if it exhausts the inventory, it remains incomplete.
Input SQLite is read-only and must be complete before any output is created. Linux
descriptor-pinned I/O rejects symlinks; outputs must be outside the repository and
have an existing parent. The SQLite analysis and JSON summary contain allowlisted
counts, digests, fixed statuses and labels, never paths or trace snippets.

Canonical trace aliases deduplicate metrics and connect run copies transitively;
session IDs never merge divergent traces, and archive-only traces create no job.
Nested subagents are measured separately; a physical primary supersedes nested
attribution. Native definitions require an exact session-directory match and a
digest-pinned Harbor-full trial copy. Conflicting definitions remain unknown.
Record failures roll back atomically and leave completion false, as do interruptions.

Changed audit semantics do not update an already-imported runner or old outputs.
Rerun into a new private directory to obtain corrected attribution, native provenance,
and completion/failure accounting; never treat an older completion flag as repaired.

Export separate harness and model reports from the completed analysis:

```bash
uv run python tools/research/harness_report.py \
  --database /external/private/new-analysis/analysis.sqlite3 \
  --output /external/private/new-harness-report
uv run python tools/research/harness_model_report.py \
  --database /external/private/new-analysis/analysis.sqlite3 \
  --output /external/private/new-model-report
```

Both contain CSV tables, JSON summaries and `REPORT.md`. Harness reports also include
allowlisted truncation events as JSONL. Exact model identifiers and step/root attribution
are retained; no model-alias equivalence is guessed.

Explicit nonexistent/unavailable-tool rejections belong only to the model report,
not harness tool inventories. Model denominators include all recorded requests;
harness usage excludes these rejected requests and unknown-name placeholders.
Declared availability conflicts remain unknown rather than an offered-unused zero.

Returned-content measurements distinguish verified native model-visible blocks from
the exported ATIF text. In particular, Claude exports may append stdout/metadata that
was not in the model's tool-result block. `visibility.csv` and event fields keep these
measurements separate; unavailable model-visible lengths are null, not zero.
