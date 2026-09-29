# Security and evidence handling

- Treat trajectories as untrusted and possibly credential-bearing. Keep traces, reports,
  auth and env files out of Git. Use only synthetic fixtures, and never copy benchmark
  solutions into this repository.
- Built-ins never execute commands, visit URLs, run models or fetch files.
- Reports are an explicit field allowlist: no raw text, paths, URLs, argument values or
  exception messages. The single exception is opt-in `--cite`, which adds bounded,
  secret-masked excerpts of the trace. Masking is best-effort, so treat cited output
  like the trace itself and keep it out of Git and issues. Text reports render only that allowlisted document. Manifest IDs,
  check IDs and check titles must be non-sensitive labels.
- `--judge-prompts DIR` (alias `--judge DIR`), `--questions DIR` and `atif-inspect` also write trace text: bounded, masked excerpts in
  prompt files, and masked step dumps. Treat both like the trace itself. atif-scan never
  sends prompts anywhere. Whoever answers them (e.g. `tools/ask-fast-agent.sh`) sends them
  to a model provider, and that's the user's decision. Prompts frame trace text as
  untrusted data. The answering script disables shell/subagents; optional `--inspect-tool`
  grants read-only MCP tools bound to one local trajectory, never arbitrary paths or execution.
  Review bundles contain local paths as well as masked excerpts; keep them private and outside Git.
  Generating prompts never sends data to a provider. Fresh judge directories prevent mixing
  stale questions into a new selection. `--answers` keeps
  only the validated answer, confidence and steps, never the free-text reason.
- When a directory or `hf://` prefix is expanded, each input is labelled by its path
  *relative to the root you passed* (e.g. `trial-1/agent/trajectory.json`). The root
  itself never appears. Don't scan roots whose sub-paths are sensitive; use a manifest.
- Remote inputs are synced by default to `~/.cache/atif-scan` (or `$ATIF_SCAN_SYNC_DIR`,
  `$XDG_CACHE_HOME/atif-scan`, `--sync-dir`). Those copies are real, possibly
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
- Harbor Hub jobs are read through the user's own `harbor` CLI: no shell, fixed
  arguments, a validated UUID, stderr withheld. Downloaded trajectories are real traces;
  `--sync-to` puts them where you choose, otherwise they go to a temporary folder that's
  removed afterwards.
- Network access happens only for `hf://` / huggingface.co inputs you name, via
  `huggingface_hub` and your saved token. Other URLs are refused. Remote error messages
  (which may contain URLs) are withheld; files over the size cap are rejected.
- Plugins and rule files are **trusted** code/configuration, not a sandbox. A plugin can
  read raw text and do its own I/O. Only load modules you have reviewed. Plugins are
  never discovered automatically. The only code loaded without `--plugin` is this
  package's own bundled packs (`atif_scan.packs`), for runs they recognise; `--packs none`
  disables that.
- The loader has a size cap, but there is no isolation for JSON depth, regex runtime or
  plugin CPU/filesystem use. Use a restricted worker for hostile inputs.
- A finding doesn't authorize accusation, disqualification or exclusion. Keep the
  coverage information and get human review.

Report security concerns privately to the repository owner. Don't post sensitive
trajectories in an issue.
