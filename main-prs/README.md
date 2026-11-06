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
- `recipes/recipe_cN.sh`: the exact server command (env + flags) for AgentX C12/C24/C48/C80 with every PR here and the
  launch flags below.
- `rejected.md`: candidates that were measured and not kept (what, measured, why).
- `REPORT.md` + `pareto.png`: latest agentic results vs the fork and NVIDIA (2026-11-05).
- `tools/export.sh <NN-name> --sglang <range> [--aiter <dir> <range>]`: write a PR's patches + stats from commits.
- `tools/apply.sh [--sync] [--aiter <dir>] [--dry-run]`: rebuild the branch from upstream main with every patch
  (`--sync` fetches the latest main first; patches already upstream are skipped; stops at the first conflict).
- `tools/summary.py [--refresh]`: regenerate the tables below from the `meta.yaml` files. When a PR is opened, add
  `prs: {sglang: <url>}` (or `aiter:`) to its `meta.yaml` and run it with `--refresh`: the row gets the link and a state
  badge (draft gray, open yellow, merged green, closed red).

## Stack result

Agentic (AgentX agentic-coding traces, 3600 s, one MI355X, TP1, NEXTN, acceptance pinned 2.32, `recipes/` params), 2026-11-05.
Tree = this branch (PRs 05-21; SGLang 4772396f44 + 16, 17, 19-21 uncommitted); AITER 285579b62 + 03, 04, 18.
C24 = mean of 2 runs of this branch (`perf/runs/agentic_blue_G{4,5}_D5noC1*_c24`, HIP 5 / 4). C48 / C80 = one run each of
this branch + C1 (`agentic_blue_F{2,3}_D5_v2aiter_c*`, HIP 0 / 5; C1 is within noise, see `rejected.md`). VRAM sampled,
no eviction stalls.
P90 = P90 interactivity (tok/s/user); tput = total tok/s/GPU.

| Conc | bare v2 P90 / tput | this branch P90 / tput | vs bare v2 (P90 / tput) | fork (2026-11-03) P90 / tput | NVIDIA B300 P90 / tput | vs NVIDIA (P90 / tput) |
|---|---|---|---|---|---|---|
| 24 | 55.7 / 109.5k | **85.0 / 123.3k** | +53% / +13% | 90.6 / 125.1k | 85.6 / 123.7k | -0.7% / -0.3% |
| 48 | 30.6 / 140.2k | **45.9 / 171.5k** | +50% / +22% | 45.6 / 172.7k | 47.6 / 171.7k | -3.6% / -0.1% |
| 80 | 16.2 / 157.9k | **20.8 / 200.9k** | +28% / +27% | 21.8 / 204.3k | 24.8 / 218.7k | -16% / -8.1% |

bare v2 = upstream 99fd01047d + AITER pin 285579b62. fork = amd-nprotaso/sglang qwen_3_8_improvement consolidated stack.
NVIDIA = InferenceX PR #4683 (B300 SGLang AgentX, run 41117066222). GSM8K (1314 scored, real MTP verification, C24 params):
96.80% (bare v2 and the PR 05-13 stack: 96.88%).

### Gate tests (PRs 05-13 only)

Measured 2026-11-04 on one MI355X per test (TP1, NEXTN, acceptance pinned 2.32, `recipes/` C24/C80 server params), base -> stack -> base
on the same GPU. A = bare v2: sglang upstream main 99fd01047d (+ main-prs scaffold e8f7aa0cc2) with AITER 285579b62 (the Dockerfile pin).
B = this branch (code head 116d3c9710, PRs 05, 07, 08, 11, 13) with AITER 285579b62 + PRs 03, 04 (975b92d83) and the PRs' env.
tput = total tok/s; P90 = P90 interactivity (tok/s/user; 64k-prefix multi-turn: rounds >= 2). Gain = B vs mean(A1, A2).

| Test | GPU (HIP) | A1 / A2 tput | B tput | tput gain | A1 / A2 P90 | B P90 | P90 gain | A spread (tput / P90) |
|---|---|---|---|---|---|---|---|---|
| 64k-prefix multi-turn c24 | 2 | 79,124 / 79,349 | 95,450 | **+20.5%** | 61.36 / 62.52 | 79.41 | **+28.2%** | 0.28% / 1.87% |
| 64k-prefix multi-turn c80 | 0 | 102,528 / 102,656 | 119,458 | **+16.4%** | 26.78 / 26.71 | 32.33 | **+20.9%** | 0.12% / 0.26% |
| 16k/1k c24 | 5 | 17,926 / 18,000 | 21,559 | **+20.0%** | 47.84 / 47.85 | 58.37 | **+22.0%** | 0.42% / 0.02% |
| 16k/1k c80 | 7 | 24,268 / 24,216 | 28,197 | **+16.3%** | 19.32 / 19.30 | 22.59 | **+17.0%** | 0.22% / 0.10% |

The 64k c80 row is a rerun: the first sequence had a 4.1% A1-A2 P90 spread (+14.9% tput / +16.3% P90).
Median TTFT also drops (16k c80 19.7 -> 17.6 s, 16k c24 6.6 -> 5.8 s).
GSM8K (1319, real MTP verification, C24 params): 96.88%, the same as bare v2 (96.88%). Mean accept length 3.09 vs 3.25 on bare v2,
because 05's fused kernel attends the MTP drafted tail that upstream's HIP fallback drops at < 2051 valid tokens (see 05's meta.yaml).

PR 15 (HC prefill fusion) was added after this measurement (gated against this stack, see its meta.yaml); the table above
does not include it.

## Launch flags

Two server flags in `recipes/` (C12-C80) that are not code changes. Both come from the NVIDIA B300 recipe.
Gate = A/B on one MI355X, B vs mean of 2 base runs (CONFIRM=1), tput / P90 interactivity, on this branch's stack
(sglang 2da35881db, AITER 32ac35904), 2026-11-05, acceptance pinned 2.32. Agentic = AgentX 3600 s, one run per arm
on different GPUs (agentic noise is about +-10%), on the InferenceX #4832 stack (sglang 1ced9a9eb0 + #4832 patches,
AITER 285579b62 + #4832 patches, #4832 recipe), 2026-11-06, acceptance pinned 2.32.

| Flag | What it does | Gate (64k-prefix multi-turn / 16k cold) | Agentic C12 | Agentic C80 | Cost |
|---|---|---|---|---|---|
| `--cuda-graph-backend-prefill breakable` | Runs prefill (and extend) as breakable CUDA graphs, 74 token buckets 4-16384. Default on CUDA, off on ROCm. | c24 +6.6% / +3.0%, c80 +8.4% / +1.0%; 16k c24 +6.5% / +5.7%, c80 +8.2% / +7.4%. TTFT of 256 new tokens on a 64k cached prefix 114 -> 42 ms | +2.3% P90 / +1.5% tput, TTFT p90 -3.5% | +1.6% P90 / +4.0% tput, TTFT p90 -5.8% | 17-20 GB graph pool from the headroom after the KV pool (KV pool unchanged), ~45 s startup. Serving at C80 left 0.04-0.11 GiB free (no OOM in 1 h). |
| `--mamba-track-interval 262144` | No Mamba state checkpoints during decode (256 = one checkpoint every 256 generated tokens). AgentX never re-sends model output, so those checkpoints are never reused; without them the GPU cache keeps prompt prefixes instead of spilling them to HiCache. | c24 +5.1% / +8.4%, c80 +7.8% / +19.5%; 16k flat | +0.1% P90 / +0.6% tput (cache already ~95% GPU hits) | +13.6% P90 / +8.3% tput, TTFT p90 -21.7%; GPU-tier prefix hits 84.0% -> 93.2% (all-tier hits 92.4% -> 93.8%) | none measured; loses decode-time prefix reuse for clients that re-send model output |

Both flags together were not measured. GSM8K (1319, real MTP verification, C24): 96.3% with the prefill graph flag,
97.0% with 262144 (base 96.6%). Files: `results/gates/{bpg-breakable-prefill,mamba-track-interval-nv}.json`, agentic
`results/runs/ix4832_*` and `ix4832mti_*` in the glab experiment.

## Summary

<!-- summary:begin -->
| PR | What | GitHub PR | Related upstream (open) | Changes | Measured impact | Base (sglang main / aiter) |
|---|---|---|---|---|---|---|
| [03-aiter-causal-conv1d-prefill](03-aiter-causal-conv1d-prefill/) | [FlyDSL] AITER: varlen causal conv1d prefill kernel (causal_conv1d_fn contract) | no PR | alternative: [aiter#7028](https://github.com/ROCm/aiter/pull/7028) ![open](https://img.shields.io/badge/-open-yellow) | aiter: 3 files, +587 / -0 | kernel only: 1.27-1.72x vs Triton at D=10240 W=4 (16k tokens 315 -> 184 us); no standalone e2e gate, end-to-end effect is inside 07-gdn-aiter-prefill (2026-11-03, gfx950, AITER pin 285579b62) | - / 285579b621 |
| [04-aiter-gdn-fused-prefill](04-aiter-gdn-fused-prefill/) | [FlyDSL] AITER: single-kernel GDN fused chunked prefill for gfx950 | no PR | same change: [aiter#6496](https://github.com/ROCm/aiter/pull/6496) ![open](https://img.shields.io/badge/-open-yellow)<br>alternative: [aiter#5732](https://github.com/ROCm/aiter/pull/5732) ![open](https://img.shields.io/badge/-open-yellow)<br>alternative: [aiter#5543](https://github.com/ROCm/aiter/pull/5543) ![open](https://img.shields.io/badge/-open-yellow) | aiter: 4 files, +1319 / -1 | kernel only: 1.9-2.4x vs SGLang Triton chunk_gated_delta_rule at Qwen3.8 TP1 shapes (16k tokens 1694 -> 878 us); no standalone e2e gate, end-to-end effect is inside 07-gdn-aiter-prefill (2026-11-03, gfx950, AITER pin 285579b62) | - / 285579b621 |
| [05-qsa-flydsl-decode](05-qsa-flydsl-decode/) | [Perf] [AMD] QSA: fused FlyDSL sparse decode on gfx950 | no PR | same change: [sglang#45196](https://github.com/sgl-project/sglang/pull/45196) ![open](https://img.shields.io/badge/-open-yellow)<br>alternative: [aiter#5996](https://github.com/ROCm/aiter/pull/5996) ![open](https://img.shields.io/badge/-open-yellow) | sglang: 12 files, +1260 / -0 | t1 c24 +4.6% tput / +7.3% p90; t1 c80 +0.7% / +7.2%; 16k c24 +6.2% / +6.7%; 16k c80 +6.4% / +7.0% (B vs mean of 2 base runs) (2026-11-03, MI355X TP1 HIP 2/0/5/7, sglang 99fd01047d (upstream main) + main-prs scaffold; AITER 285579b62 (pin) + PRs 03, 04 (stack 975b92d83)) | 99fd01047d / - |
| [07-gdn-aiter-prefill](07-gdn-aiter-prefill/) | [Perf] [AMD] GDN: AITER FlyDSL prefill conv and fused chunked delta rule | no PR | same change: [sglang#44103](https://github.com/sgl-project/sglang/pull/44103) ![open](https://img.shields.io/badge/-open-yellow)<br>same change: [sglang#41879](https://github.com/sgl-project/sglang/pull/41879) ![open](https://img.shields.io/badge/-open-yellow)<br>prerequisite: [aiter#6496](https://github.com/ROCm/aiter/pull/6496) ![open](https://img.shields.io/badge/-open-yellow)<br>alternative: [sglang#39595](https://github.com/sgl-project/sglang/pull/39595) ![open](https://img.shields.io/badge/-open-yellow) | sglang: 5 files, +261 / -2 | t1 c24 +4.2% tput / +2.1% p90; t1 c80 +5.1% / +2.6%; 16k c24 +4.3% / +4.6%; 16k c80 +5.8% / +5.6% (with AITER 03+04; B vs mean of 2 base runs) (2026-11-03, MI355X TP1 HIP 2/0/5/7, sglang 99fd01047d (upstream main) + main-prs scaffold; AITER 285579b62 (pin) + PRs 03, 04 (stack 975b92d83)) | 99fd01047d / - |
| [08-fused-hc-decode](08-fused-hc-decode/) | [Perf] [AMD] Qwen4-Exp HC: fuse decode-sized hyper-connection norm and inject into FlyDSL mix | [sglang#48648](https://github.com/sgl-project/sglang/pull/48648) ![open](https://img.shields.io/badge/-open-yellow) | alternative: [sglang#46881](https://github.com/sgl-project/sglang/pull/46881) ![draft](https://img.shields.io/badge/-draft-lightgrey)<br>alternative: [aiter#5913](https://github.com/ROCm/aiter/pull/5913) ![merged](https://img.shields.io/badge/-merged-brightgreen)<br>alternative: [sglang#40552](https://github.com/sgl-project/sglang/pull/40552) ![open](https://img.shields.io/badge/-open-yellow) | sglang: 4 files, +693 / -0 | t1 c24 +5.3% tput / +9.5% p90; 16k c24 +5.2% / +6.5% (both vs mean of 2 base runs); t1 c80 +0.6% / +2.5%; 16k c80 neutral (noisy, +3.0% / +3.1% on the rerun) (2026-11-03, MI355X TP1 HIP 2/5/7, sglang 99fd01047d (upstream main) + main-prs scaffold; AITER 285579b62 (pin) + PRs 03, 04 (stack 975b92d83)) | 99fd01047d / - |
| [11-qsa-indexer-q-pad](11-qsa-indexer-q-pad/) | [Perf] [AMD] QSA: pad decode index Q to the ROCm MFMA width and use 256 threads | [sglang#48649](https://github.com/sgl-project/sglang/pull/48649) ![open](https://img.shields.io/badge/-open-yellow) | alternative: [aiter#5996](https://github.com/ROCm/aiter/pull/5996) ![open](https://img.shields.io/badge/-open-yellow) | sglang: 2 files, +12 / -5 | t1 c24 +2.8% tput / +5.9% p90 (vs mean of 2 base runs); t1 c80 +2.6% / +1.5%; 16k c24 +0.5% / +0.7%; 16k c80 +0.2% / +0.2% (2026-11-03, MI355X TP1 HIP 2/0/5/7, sglang 99fd01047d (upstream main) + main-prs scaffold; AITER 285579b62 (pin) + PRs 03, 04 (stack 975b92d83)) | 99fd01047d / - |
| [13-gdn-prefill-views](13-gdn-prefill-views/) | [Perf] [AMD] Qwen3.5 GDN: use zero-copy prefill projection views on HIP | no PR | same change: [sglang#47128](https://github.com/sgl-project/sglang/pull/47128) ![open](https://img.shields.io/badge/-open-yellow)<br>same change: [sglang#41683](https://github.com/sgl-project/sglang/pull/41683) ![open](https://img.shields.io/badge/-open-yellow)<br>alternative: [sglang#38806](https://github.com/sgl-project/sglang/pull/38806) ![open](https://img.shields.io/badge/-open-yellow) | sglang: 1 files, +1 / -1 | 16k c80 +6.0% tput / +7.2% p90 (vs mean of 2 base runs); t1 c24 +1.8% / +1.6%; t1 c80 +2.0% / +1.4%; 16k c24 +1.6% / +1.5% (2026-11-03, MI355X TP1 HIP 2/5, sglang 99fd01047d (upstream main) + main-prs scaffold; AITER 285579b62 (pin) + PRs 03, 04 (stack 975b92d83)) | 99fd01047d / - |
| [15-hc-prefill-fusion](15-hc-prefill-fusion/) | [Perf] [AMD] Qwen4-Exp HC: fused prefill hc_combine kernel | no PR | none found | sglang: 5 files, +399 / -9 | t1 c80 +2.4% tput / +4.7% p90 (vs mean of 2 base runs, base spread 2.3%); 16k c80 +3.7% / +3.7%; t1 c24 +0.4% / +0.2% (3 sequences pooled); 16k c24 +0.0% / +1.9% (2 sequences pooled). Kept by a small margin (gain ~2x noise); integrated on user request to track the HC prefill fusion (2026-11-04 09:28-10:20 UTC, MI355X TP1 HIP 5/4/0/2/6, A = v2 stack (sglang d8ff624648 = PRs 05, 07, 08, 11, 13; AITER 285579b62 + PRs 03, 04), B = A + this PR (e65d0985f8)) | 99fd01047d / - |
| [16-qsa-decode-indexer-launch](16-qsa-decode-indexer-launch/) | [Perf] [AMD] QSA: group-strided decode MQA logits launch on ROCm | [sglang#48649](https://github.com/sgl-project/sglang/pull/48649) ![open](https://img.shields.io/badge/-open-yellow) | alternative: [sglang#48251](https://github.com/sgl-project/sglang/pull/48251) ![open](https://img.shields.io/badge/-open-yellow)<br>alternative (NVIDIA): [sglang#48250](https://github.com/sgl-project/sglang/pull/48250) ![open](https://img.shields.io/badge/-open-yellow) | sglang: 1 files, +30 / -5 | agentic C24 leave-one-out: +5.0 P90 (2.5 sd) / +1.2k tput (FULL 84.0 / 122.2k vs 79.0 / 121.0k without it); 1.2-1.7x decode MQA logits at 50k-200k context (2026-11-05, MI355X TP1, AgentX agentic C24 3600 s, acceptance 2.32; FULL = 4772396f44 + 06/09/10 + C1 + this, AITER 285579b62 + 03, 04, 01 (runs/agentic_loo_{L0_full,L0b_full_repeat,LnoD2}_c24, HIP 7/7/0)) | 99fd01047d / - |
| [17-fused-qk-norm-rope-gate-hip](17-fused-qk-norm-rope-gate-hip/) | [Perf] [AMD] Qwen3.5: use the fused QK GemmaRMSNorm + RoPE + gate kernel on HIP | no PR | same change: [sglang#45529](https://github.com/sgl-project/sglang/pull/45529) ![open](https://img.shields.io/badge/-open-yellow)<br>follow-up: [sglang#47779](https://github.com/sgl-project/sglang/pull/47779) ![draft](https://img.shields.io/badge/-draft-lightgrey) | sglang: 2 files, +11 / -5 | agentic C24 add-one: +3.7 P90 / +1.9k tput on V2 AITER (87.7 / 124.1k vs FULL mean 84.0 / 122.2k, single run), +3.4 P90 on the fork AITER pair (88.9 vs 85.5, 2 runs each); C48 +0.8 / +0.6%, C80 -0.0 / +0.1% (neutral) (2026-11-04/05, MI355X TP1, AgentX agentic 3600 s, acceptance 2.32; A = 4772396f44 + 06/09/10 + C1 + 16, B = A + this; AITER 285579b62 + 03, 04, 01 (runs/agentic_blue_F{1,2,3}_D5_v2aiter_c*, E3/E4/E5 D2 runs)) | 99fd01047d / - |
| [18-aiter-gemm-tunes](18-aiter-gemm-tunes/) | [Config] AITER: gfx950 tuned GEMM configs for Qwen3.8-Flash-Next | no PR | alternative: [aiter#5870](https://github.com/ROCm/aiter/pull/5870) ![open](https://img.shields.io/badge/-open-yellow) | aiter: 4 files, +418 / -2 | alone: +0.3% to +2.3% tput / P90 on the 2026-11-03 gates (below 4%); kept as part of the bundle 18-21, agentic C24 leave-one-out +2.9 P90 / +1.1k tput (within noise), see 19 (2026-11-03 gates and 2026-11-05 agentic, MI355X TP1, AITER pin 285579b62) | - / 285579b621 |
| [19-packed-fp8-gdn-in-proj](19-packed-fp8-gdn-in-proj/) | [Perf] [AMD] Qwen4-Exp: enable packed FP8 GDN in_proj on ROCm | no PR | alternative: [sglang#39912](https://github.com/sgl-project/sglang/pull/39912) ![draft](https://img.shields.io/badge/-draft-lightgrey) | sglang: 2 files, +18 / -5 | bundle 18-21 agentic C24 leave-one-out: +2.9 P90 (1.5 sd, within noise) / +1.1k tput (+0.9%); 64k-prefix multi-turn on the stack + 15: c48 +8.4% tput / +4.2% P90, c80 +4.5% / +0.9% (2 sequences); alone (with 18) +0.05% to +1.8% tput, +1.1% to +3.0% P90 (2026-11-04/05, MI355X TP1; agentic arm runs/agentic_loo_Lnobundle_c24 (HIP 4) vs FULL L0/L0b (HIP 7), AITER 285579b62 + 03, 04 (+ 01 in FULL)) | 99fd01047d / - |
| [20-lm-head-aiter-gemm](20-lm-head-aiter-gemm/) | [Perf] [AMD] lm_head: Opt-in aiter tgemm for the bf16 lm_head on gfx950 | [sglang#47485](https://github.com/sgl-project/sglang/pull/47485) ![open](https://img.shields.io/badge/-open-yellow) | none found | sglang: 2 files, +8 / -1 | alone -1.1% to +1.75% (2026-11-03 gates); part of the bundle 18-21, agentic C24 leave-one-out +2.9 P90 / +1.1k tput (within noise), see 19 (2026-11-03 gates and 2026-11-05 agentic, MI355X TP1) | 99fd01047d / - |
| [21-shared-attn-fp8-quant](21-shared-attn-fp8-quant/) | [Perf] [AMD] Qwen4-Exp QSA: share the per-token FP8 quant of qkv_proj and index_qk_proj | no PR | alternative: [sglang#34502](https://github.com/sgl-project/sglang/pull/34502) ![open](https://img.shields.io/badge/-open-yellow)<br>alternative: [sglang#29723](https://github.com/sgl-project/sglang/pull/29723) ![open](https://img.shields.io/badge/-open-yellow) | sglang: 5 files, +63 / -13 | alone +0.2% to +2.75% (2026-11-03 gates, ~15 us per QSA layer); part of the bundle 18-21, agentic C24 leave-one-out +2.9 P90 / +1.1k tput (within noise), see 19 (2026-11-03 gates and 2026-11-05 agentic, MI355X TP1) | 99fd01047d / - |
<!-- summary:end -->

## Ranking

P90 / tput = agentic C24 contribution (`perf/runs/agentic_loo_*`, 2026-11-05): FULL (all PRs here except 17, + C1) minus
the run without that change (env kill switch, or a tree with it reverted), one run per arm; FULL = mean of 2 runs
(84.0 / 122.2k). 17 is add-one (FULL + 17 vs FULL). Within noise = |P90| < 2 sd = 3.9 (single-run sd 1.6). Individually
within-noise rows are not jointly removable: removing 15 + C1 together cost 2.9 P90 (2 runs). A change's prerequisites are
folded into its row (07 needs AITER 03 + 04; 19 is the bundle 18-21). "Same change upstream" = an open upstream PR that
already does it. Generated by `tools/summary.py` from each PR's `ranking:`; badges refresh with `--refresh`.

<!-- ranking:begin -->
| Rank | Change | P90 (tok/s/user) | tput (tok/s/GPU) | Within noise | Lines (+/-) | GitHub PR | Same change upstream |
|---|---|---|---|---|---|---|---|
| 1 | [08-fused-hc-decode](08-fused-hc-decode/) | +8.7 | +4.9k | **no** | 693 | [sglang#48648](https://github.com/sgl-project/sglang/pull/48648) ![open](https://img.shields.io/badge/-open-yellow) | none |
| 2 | [05-qsa-flydsl-decode](05-qsa-flydsl-decode/) | +6.7 | +3.3k | **no** | 1260 | no PR | [sglang#45196](https://github.com/sgl-project/sglang/pull/45196) ![open](https://img.shields.io/badge/-open-yellow) |
| 3 | [16-qsa-decode-indexer-launch](16-qsa-decode-indexer-launch/) | +5.0 | +1.2k | **no** | 35 | [sglang#48649](https://github.com/sgl-project/sglang/pull/48649) ![open](https://img.shields.io/badge/-open-yellow) | none |
| 4 | [17-fused-qk-norm-rope-gate-hip](17-fused-qk-norm-rope-gate-hip/) | +3.7<br>add-one on the full stack (not leave-one-out), single run (~2 sd); +3.4 on the fork AITER pair | +1.9k | yes | 16 | no PR | [sglang#45529](https://github.com/sgl-project/sglang/pull/45529) ![open](https://img.shields.io/badge/-open-yellow) |
| 5 | [11-qsa-indexer-q-pad](11-qsa-indexer-q-pad/) | +3.1 | +1.5k | yes | 17 | [sglang#48649](https://github.com/sgl-project/sglang/pull/48649) ![open](https://img.shields.io/badge/-open-yellow) | none |
| 6 | [19-packed-fp8-gdn-in-proj](19-packed-fp8-gdn-in-proj/) + [18](18-aiter-gemm-tunes/) + [20](20-lm-head-aiter-gemm/) + [21](21-shared-attn-fp8-quant/) | +2.9<br>bundle 18-21 measured together | +1.1k | yes | 528 | [sglang#47485](https://github.com/sgl-project/sglang/pull/47485) ![open](https://img.shields.io/badge/-open-yellow) | none |
| 7 | [07-gdn-aiter-prefill](07-gdn-aiter-prefill/) + [03](03-aiter-causal-conv1d-prefill/) + [04](04-aiter-gdn-fused-prefill/) | +2.6 | +0.9k | yes | 2170 | no PR | [sglang#44103](https://github.com/sgl-project/sglang/pull/44103) ![open](https://img.shields.io/badge/-open-yellow)<br>[sglang#41879](https://github.com/sgl-project/sglang/pull/41879) ![open](https://img.shields.io/badge/-open-yellow)<br>[aiter#6496](https://github.com/ROCm/aiter/pull/6496) ![open](https://img.shields.io/badge/-open-yellow) |
| 8 | [13-gdn-prefill-views](13-gdn-prefill-views/) | +1.9 | +0.2k | yes | 2 | no PR | [sglang#47128](https://github.com/sgl-project/sglang/pull/47128) ![open](https://img.shields.io/badge/-open-yellow)<br>[sglang#41683](https://github.com/sgl-project/sglang/pull/41683) ![open](https://img.shields.io/badge/-open-yellow) |
| 9 | [15-hc-prefill-fusion](15-hc-prefill-fusion/) | -1.7<br>removing 15 and C1 together costs 2.9 P90 (2 runs), so 15 stays | -0.2k | yes | 408 | no PR | none |
<!-- ranking:end -->

Not on the branch: C1 (`_CHUNK_CONFIGS` in qsa/sparse_attn.py + FP8 K/V gather), dropped (`rejected.md`).
