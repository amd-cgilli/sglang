# qwen38-submission-v2: changes on top of upstream main

Branch `qwen38-submission-v2` (amd-cgilli/sglang) = upstream `sgl-project/sglang` main + the PRs below, for
Qwen3.8-Flash-Next on MI355X. AITER = the commit upstream SGLang bundles (`AITER_COMMIT_DEFAULT` in
`docker/rocm.Dockerfile`) + the AITER patches below.

## Layout

- `NN-name/` one folder per PR, applied in NN order:
  - `meta.yaml`: title, status (`local` / `pr #N` / `merged`), description, measured impact (with date, config,
    base commits), changes per repo (files / insertions / deletions, generated), base commits (generated).
  - `sglang.patch` and/or `aiter.patch`: `git format-patch` mbox of the PR's commits (generated).
- `BASE`: the upstream main commit the branch was last rebuilt on.
- `tools/export.sh <NN-name> --sglang <range> [--aiter <dir> <range>]`: write a PR's patches + stats from commits.
- `tools/apply.sh [--sync] [--aiter <dir>] [--dry-run]`: rebuild the branch from upstream main with every patch
  (`--sync` fetches the latest main first; patches already upstream are skipped; stops at the first conflict).
- `tools/summary.py`: regenerate the table below from the `meta.yaml` files.

## Summary

<!-- summary:begin -->
| PR | What | Status | Changes | Measured impact | Base (sglang main / aiter) |
|---|---|---|---|---|---|
<!-- summary:end -->
