# Security and evidence handling

- Treat trajectories as untrusted and possibly credential-bearing. Keep traces, reports,
  auth and env files out of Git. Use only synthetic fixtures, and never copy benchmark
  solutions into this repository.
- Built-ins never execute commands, visit URLs, run models or fetch files.
- **`--image-model MODEL`** is the scan's one model call, and it is opt-in. It sends
  only the images in trials where an image leaves a check unknown (decoded from the
  trajectory, written to a `0700` temporary folder and deleted afterwards) to MODEL for
  transcription, through `fast-agent go --isolated --no-shell --no-subagents` with
  fixed arguments and no shell (see *fast-agent calls* below). Images may show anything
  on the agent's screen, credentials included, so only use a provider you may send the
  traces to. Transcripts are trace content: they are stored in the atif-scan home's
  `images/` folder (`0700`/`0600`) and never reported (reports carry counts). The loader
  keeps an inline payload's sha256, never the payload.
- Reports are an explicit field allowlist: no raw text, paths, URLs, argument values or
  exception messages. The one report field with trace text is opt-in `--cite`: bounded,
  secret-masked excerpts of the trace. Masking is best-effort, so treat cited output
  like the trace itself and keep it out of Git and issues. OpenAI account identifiers
  (`observation.account_*`) are masked the same way; a finding is a scrub reminder, and
  its absence does not make a trace safe to publish. Text reports render only that allowlisted document. Manifest IDs,
  check IDs and check titles must be non-sensitive labels.
- **Review bundles** (`--questions DIR`, formerly `--judge-prompts`) contain trace text
  (bounded, masked excerpts) and local paths. Treat them like the trace itself: keep them
  private and outside Git, in the atif-scan home's `bundles/` folder
  (`~/.cache/atif-scan/bundles/` by default), not the source tree or the expendable
  result cache. Use `umask 077` when generating or answering (directories `0700`, files
  `0600`), and a fresh directory per selection and judge model, so stale questions can't
  mix into a new review. Generating a bundle never sends data anywhere.
- **Answering** is the user's decision: whoever answers (e.g. `atif-scan hunt`) sends the
  prompts to a model provider. Prompts frame trace text as untrusted data. `hunt` disables
  shell and subagents; the optional `--inspect-tool` grants read-only MCP tools bound to
  one local trajectory, never arbitrary paths or execution. `--answers` keeps only the
  validated answer, confidence, mechanism and steps, never the free-text reason.
- **fast-agent calls** (`--image-model`, `hunt`) run `fast-agent go --isolated`
  (fast-agent 0.10.43+): config, secrets and model aliases are read from the fast-agent
  home, but nothing is written there (no session history, file logs or telemetry) and no
  skills, agent cards, plugins, hooks, or shell, filesystem or subagent tools load. Only
  the MCP server `hunt --inspect-tool` passes is started. An older fast-agent rejects the
  flag: atif-scan then warns once and runs without it (`--no-shell --no-subagents`,
  session history off through the environment), where the home's plugins and hooks may
  still load. Upgrade rather than rely on that.
- **Companion history.** Full Harbor archives (`--full`) may include Grok compaction
  segments and task artifacts. Companion-history tools inventory only fixed local
  session locations beside the bound trajectory and never follow paths supplied by a
  summary. Numeric file IDs, size limits, symlink rejection and whole-file masking
  precede bounded reads and searches. Markdown remains untrusted data, not instructions
  or validated ATIF reconstruction. Availability does not prove completeness. Only Grok
  Markdown layouts are checked; other formats, including fast-agent JSON snapshots, are
  unchecked, not absent or uncollected.
- `atif-inspect` prints masked trace text: treat its output like the trace.
- When a directory or `hf://` prefix is expanded, each input is labelled by its path
  *relative to the root you passed* (e.g. `trial-1/agent/trajectory.json`). The root
  itself never appears. Don't scan roots whose sub-paths are sensitive; use a manifest.
- Remote inputs are synced by default to the atif-scan home, `~/.cache/atif-scan` (or
  `$ATIF_SCAN_HOME`, `$XDG_CACHE_HOME/atif-scan`; `$ATIF_SCAN_SYNC_DIR` or `--sync-dir` for
  the copies alone). The home also holds the label store and gold snapshots, which name
  real runs: keep all of it private. Those copies are real, possibly
  credential-bearing traces: keep the directory private, never inside a repository, and
  delete it when done (`--no-sync` streams without keeping files). The sync layout is
  confined to that directory (path traversal and symlink destinations are rejected).
  Owned download directories are made `0700` and files `0600`, including existing copies.
  Hugging Face mirrors retain a private `.atif-sync.json` inventory (relative paths and
  content identities, no trace text). Keep it with the copy: it preserves missing inputs
  on offline rescans. Failed sync files are reported, never silently dropped or replaced
  with an old copy; any sync failure returns exit 2.
- The result cache (`--cache`, default `<sync dir>/results`) stores a subset of the
  allowlisted JSON report (the trace-derived results; run facts are re-read each scan),
  never citations or trace text.
- Besides trajectories, the scanner reads only small, size-capped run files: reward files
  (`verifier/reward.{json,txt}`, at most 4 KiB), Harbor's `result.json`/`config.json`,
  `trials.jsonl`, and harbor-hf's `run.json` (declared prices) and
  `attempt-costs/*.json` (at most 4 KiB). Only allowlisted numbers, codes and labels are
  extracted from them. `--inspect` reads no file contents.
- A trial's submitted patch (`<trial>/artifacts/model.patch` beside
  `<trial>/agent/trajectory.json`, as DeepSWE/Pier record it; at most 8 MiB, symlinks
  refused) is read for its `diff --git` file paths, change kinds and added lines. They
  stay in memory for fixed predicates (the DeepSWE pack); reports carry only check IDs,
  counts and trace step locators, never paths, patch lines or patch text. A missing patch
  is unknown, never an empty submission. Patch text is the agent's code: treat it like
  the trace.
- Harbor Hub jobs are read through the user's own `harbor` CLI: no shell, fixed
  arguments, a validated UUID, stderr withheld. Downloaded trajectories are real traces;
  `--sync-to` puts them where you choose, otherwise they go to a temporary folder that's
  removed afterwards.
- Network access happens only for `hf://` / huggingface.co inputs you name, via
  `huggingface_hub` and your saved token. Other URLs are refused. Remote error messages
  (which may contain URLs) are withheld; files over the size cap are rejected.
- Bench-run discovery reads only capped cohort catalogs, receipts and receipt-pinned
  configs. It never imports or executes bench-run code, follows auth/config paths, or
  contacts a provider. Codex diagnostic receipts are recognized by `kind` and a
  top-level `run_id` only; `plan_path` is not opened. Job selection is confined to the
  chosen root's `jobs/` directory; paths escaping the root are rejected. Listing exports
  only run IDs, allowlisted lifecycle statuses, counts and fixed issue codes.
  Replacement receipts (`runs/replacements/*/receipt.json`) and release manifests are
  read the same way: capped, digest-checked against the parent run, confined to the
  root, failing closed when one names the run but doesn't verify. Reports carry only
  their codes (error types, phases), IDs and trial folder names, never operator `reason`
  text.
- For an errored trial only, fast-agent's `fast-agent-results.json` (beside the
  trajectory, capped) is read for its provider safety details: `provider`, `reason` and
  `category`, each kept only as an identifier-like code. The provider's explanation and
  every message in that file are never reported.
- Plugins and rule files are **trusted** code/configuration, not a sandbox. A plugin can
  read raw text and do its own I/O. Only load modules you have reviewed. Plugins are
  never discovered automatically. The only code loaded without `--plugin` is this
  package's own bundled packs (`atif_scan.packs`), for runs they recognise; `--packs none`
  disables that.
- The loader has a size cap, but there is no isolation for JSON depth, regex runtime or
  plugin CPU/filesystem use. Use a restricted worker for hostile inputs.
- `--browse` is private inspection, not report export: a token-authenticated IPv4
  loopback server with strict Host/Origin checks, fixed assets/routes, CSP, no request
  logging and no-store responses. Never publish or tunnel its port or share its launch
  URL. Whole-field masking precedes paging/search; text is rendered literally and no
  trace commands or URLs are executed. Masking is best effort. Local sources are pinned
  by digest before scanning, checked again before serving, and revalidated on access;
  changed sources require a new scan. Browser inputs are opaque IDs, not client paths.
  Finding-level feedback is separate from reports and trial labels, stored privately in
  `<atif-scan home>/feedback/` (or `--feedback-dir`). Notes are unmasked reviewer text:
  keep them and the append-only journal out of Git. Feedback is bound to source/task,
  trace content and assessment identity; it cannot clear coverage gaps. Latest append
  wins: this is a single-reviewer POSIX tool, not a multi-user adjudication service.
- `--viewer DIR` is the one output that deliberately contains trace text: a static
  folder (fixed HTML/CSS/JS plus `data.js`) for publishing an individually reviewed
  trajectory. Every field is masked as a whole before export, but masking is best
  effort, so read the export before publishing it; it is not a clearance. The data is
  allowlisted (labels, task, reward, coverage, check IDs/titles/priorities, field
  locators and masked field text) and never includes source paths, citations, feedback
  or raw metadata; media is not exported. Highlights are offsets proven against the
  masked text, otherwise the whole field is marked as the evidence. Sources are pinned
  by digest before scanning and must still match when exported. The page loads nothing
  outside its folder (CSP, no fonts, images or network), renders trace text literally
  and executes nothing. The folder is created `0700` with `0600` files and must be new
  or empty.
- `--viewer DIR --review QUESTION` writes a blind review export (same masking, CSP and
  file rules; no findings, scores, judge answers or scanner run sections). Verdicts stay
  in the browser's local storage, keyed by a random export ID, and leave only as a JSON
  file the reviewer downloads. Notes in it are the reviewer's own unmasked text: keep the
  file private. `atif-scan labels import-review` keeps answer, mechanism and reward only,
  never the note, and maps opaque export IDs to trials through the private key.
- A finding doesn't authorize accusation, disqualification or exclusion. Keep the
  coverage information and get human review.

Report security concerns privately to the repository owner. Don't post sensitive
trajectories in an issue.
