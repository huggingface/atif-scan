# PyPI publication

`atif-scan` and `atif-inspect` are console entry points in the same distribution.
After publication:

```bash
uvx atif-scan@latest --help
uvx atif-scan@0.8.15 --help
uvx --from atif-scan@latest atif-inspect --help
```

## One-time configuration

The PyPI project owner configures a GitHub Trusted Publisher with owner
`huggingface`, repository `atif-scan`, workflow filename `publish.yml` and environment
`pypi`. No long-lived PyPI token is stored in GitHub. The GitHub `pypi` environment
allows only the `main` branch and `v*` tags; manual dispatch is accepted only on main.
Actions are pinned to commits and maintained by Dependabot.

## Release flow

1. Update the project version and lockfile, run all four development checks, build
   and inspect synthetic viewer screenshots, and merge to main.
2. Create and push an annotated stable `vMAJOR.MINOR.PATCH` tag whose version
   matches `pyproject.toml`. Release commits must be ancestors of main.
3. `publish.yml` validates the tag, tests the exact release commit on Python 3.11
   and 3.14, then builds one wheel and one source archive from a clean checkout.
4. Archive validation rejects paths outside the package namespace, links and
   untracked source files. Only the two validated distributions are uploaded.
   Isolated wheel smoke tests exercise both console scripts and a synthetic scan;
   fresh base and MCP-extra resolutions must meet the recorded dependency security
   floors.
5. A separate, source-free `pypi` job downloads the immutable build artifact and
   uploads it using OIDC Trusted Publishing and PyPI attestations. Only that job
   has `id-token: write`. It does not install or execute the release package.

CI jobs must succeed in this workflow: unrelated concurrently running checks do not
authorize publication. Publishing is serialized and never cancels an in-progress
upload. Existing versions are not skipped or overwritten; investigate partial
uploads before retrying. PyPI releases and published Git tags are immutable.

## First publication / manual recovery

The existing `v0.8.15` tag predates this workflow. Publish it without moving the tag:

```bash
gh workflow run publish.yml --ref main -f release_tag=v0.8.15
```

The workflow/validator checkout remains separate from the pinned release checkout.
Do not publish from an arbitrary branch or introduce a replacement package version
under an existing tag.

The main lockfile includes patched `certifi` and `filelock` versions. Historical tags
retain their original lockfiles. Wheels do not enforce `uv.lock`: `uvx` resolves
runtime dependencies from package metadata. Smoke checks therefore also validate
fresh dependency resolutions, including the optional MCP dependency set.

After upload, verify the PyPI file hashes against the workflow artifacts and run
`uvx atif-scan@0.8.15 --help` and `uvx atif-scan@latest --help` from outside the
checkout. A successful GitHub tag/build alone does not mean PyPI was updated.
