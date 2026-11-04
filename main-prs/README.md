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
- `recipes/recipe_cN.sh`: the exact server command (env + flags) for AgentX C12/C24/C48/C80 with every PR here.
- `rejected.md`: candidates that were measured and not kept (what, measured, why).
- `tools/export.sh <NN-name> --sglang <range> [--aiter <dir> <range>]`: write a PR's patches + stats from commits.
- `tools/apply.sh [--sync] [--aiter <dir>] [--dry-run]`: rebuild the branch from upstream main with every patch
  (`--sync` fetches the latest main first; patches already upstream are skipped; stops at the first conflict).
- `tools/summary.py`: regenerate the table below from the `meta.yaml` files.

## Summary

<!-- summary:begin -->
| PR | What | Status | Changes | Measured impact | Base (sglang main / aiter) |
|---|---|---|---|---|---|
| [03-aiter-causal-conv1d-prefill](03-aiter-causal-conv1d-prefill/) | [FlyDSL] AITER: varlen causal conv1d prefill kernel (causal_conv1d_fn contract) | local | aiter: 3 files, +587 / -0 | kernel only: 1.27-1.72x vs Triton at D=10240 W=4 (16k tokens 315 -> 184 us); no standalone e2e gate, end-to-end effect is inside 07-gdn-aiter-prefill (2026-11-03, gfx950, AITER pin 285579b62) | - / 285579b621 |
| [04-aiter-gdn-fused-prefill](04-aiter-gdn-fused-prefill/) | [FlyDSL] AITER: single-kernel GDN fused chunked prefill for gfx950 | local | aiter: 4 files, +1319 / -1 | kernel only: 1.9-2.4x vs SGLang Triton chunk_gated_delta_rule at Qwen3.8 TP1 shapes (16k tokens 1694 -> 878 us); no standalone e2e gate, end-to-end effect is inside 07-gdn-aiter-prefill (2026-11-03, gfx950, AITER pin 285579b62) | - / 285579b621 |
| [05-qsa-flydsl-decode](05-qsa-flydsl-decode/) | [Perf] [AMD] QSA: fused FlyDSL sparse decode on gfx950 | local | sglang: 12 files, +1260 / -0 | t1 c24 +4.6% tput / +7.3% p90; t1 c80 +0.7% / +7.2%; 16k c24 +6.2% / +6.7%; 16k c80 +6.4% / +7.0% (B vs mean of 2 base runs) (2026-11-03, MI355X TP1 HIP 2/0/5/7, sglang 99fd01047d (upstream main) + main-prs scaffold; AITER 285579b62 (pin) + PRs 03, 04 (stack 975b92d83)) | 99fd01047d / - |
| [07-gdn-aiter-prefill](07-gdn-aiter-prefill/) | [Perf] [AMD] GDN: AITER FlyDSL prefill conv and fused chunked delta rule | local | sglang: 5 files, +261 / -2 | t1 c24 +4.2% tput / +2.1% p90; t1 c80 +5.1% / +2.6%; 16k c24 +4.3% / +4.6%; 16k c80 +5.8% / +5.6% (with AITER 03+04; B vs mean of 2 base runs) (2026-11-03, MI355X TP1 HIP 2/0/5/7, sglang 99fd01047d (upstream main) + main-prs scaffold; AITER 285579b62 (pin) + PRs 03, 04 (stack 975b92d83)) | 99fd01047d / - |
| [08-fused-hc-decode](08-fused-hc-decode/) | [Perf] [AMD] Qwen4-Exp HC: fuse decode-sized hyper-connection norm and inject into FlyDSL mix | local | sglang: 4 files, +693 / -0 | t1 c24 +5.3% tput / +9.5% p90; 16k c24 +5.2% / +6.5% (both vs mean of 2 base runs); t1 c80 +0.6% / +2.5%; 16k c80 neutral (noisy, +3.0% / +3.1% on the rerun) (2026-11-03, MI355X TP1 HIP 2/5/7, sglang 99fd01047d (upstream main) + main-prs scaffold; AITER 285579b62 (pin) + PRs 03, 04 (stack 975b92d83)) | 99fd01047d / - |
| [11-qsa-indexer-q-pad](11-qsa-indexer-q-pad/) | [Perf] [AMD] QSA: pad decode index Q to the ROCm MFMA width and use 256 threads | local | sglang: 2 files, +12 / -5 | t1 c24 +2.8% tput / +5.9% p90 (vs mean of 2 base runs); t1 c80 +2.6% / +1.5%; 16k c24 +0.5% / +0.7%; 16k c80 +0.2% / +0.2% (2026-11-03, MI355X TP1 HIP 2/0/5/7, sglang 99fd01047d (upstream main) + main-prs scaffold; AITER 285579b62 (pin) + PRs 03, 04 (stack 975b92d83)) | 99fd01047d / - |
| [13-gdn-prefill-views](13-gdn-prefill-views/) | [Perf] [AMD] Qwen3.5 GDN: use zero-copy prefill projection views on HIP | local | sglang: 1 files, +1 / -1 | 16k c80 +6.0% tput / +7.2% p90 (vs mean of 2 base runs); t1 c24 +1.8% / +1.6%; t1 c80 +2.0% / +1.4%; 16k c24 +1.6% / +1.5% (2026-11-03, MI355X TP1 HIP 2/5, sglang 99fd01047d (upstream main) + main-prs scaffold; AITER 285579b62 (pin) + PRs 03, 04 (stack 975b92d83)) | 99fd01047d / - |
<!-- summary:end -->
