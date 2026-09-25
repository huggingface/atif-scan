# Security and evidence handling

- Treat trajectories as untrusted and possibly credential-bearing. Keep traces, reports,
  auth and env files out of Git. Use only synthetic fixtures, and never copy benchmark
  solutions into this repository.
- Built-ins never execute commands, visit URLs, run models or fetch files.
- Reports are an explicit field allowlist: no raw text, paths, URLs, argument values or
  exception messages. Text reports render only that allowlisted document. Manifest IDs
  and check IDs must be non-sensitive labels.
- When a directory or `hf://` prefix is expanded, each input is labelled by its path
  *relative to the root you passed* (e.g. `trial-1/agent/trajectory.json`). The root
  itself never appears. Don't scan roots whose sub-paths are sensitive; use a manifest.
- `hf://` reads use `huggingface_hub` and your saved token. Remote error messages
  (which may contain URLs) are withheld; files over the size cap are rejected.
- Plugins and rule files are **trusted** code/configuration, not a sandbox. A plugin can
  read raw text and do its own I/O. Only load modules you have reviewed. Plugins are
  never discovered automatically.
- The loader has a size cap, but there is no isolation for JSON depth, regex runtime or
  plugin CPU/filesystem use. Use a restricted worker for hostile inputs.
- A finding doesn't authorize accusation, disqualification or exclusion. Keep the
  coverage information and get human review.

Report security concerns privately to the repository owner. Don't post sensitive
trajectories in an issue.
