# Rejected candidates (not on the branch)

Gate rule: keep a PR if tput or P90 interactivity improves >= 4% in any of {64k-prefix multi-turn, 16k/1k} x {c24, c80}
and nothing regresses > 2%. Measured 2026-11-03 on MI355X TP1 against bare v2 (sglang 99fd01047d + scaffold; AITER pin
285579b62, + PRs 03/04 for the SGLang candidates). Patches are kept in the eval workspace (`perf/v2/patches/`) for the record.

- **01-aiter-gemm-tunes** (AITER: gfx950 tuned a8w8/bf16 GEMM tables for Qwen3.8, incl. FlyDSL rows N=16512 K=2560): +0.3% to +2.3% tput/P90, no regression, GSM8K 96.50%; below 4% alone, and the PRs it enables (06, 09, 10) were not kept either.
- **02-aiter-moe-sort-quant** (AITER MoE sort + quant fusion): skipped by user decision, not ported.
- **06-packed-fp8-gdn-in-proj** (pack Quark FP8 in_proj_qkvz + in_proj_ba into one N=16512 GEMM for qwen4_exp_text, judged with 01): +0.05% to +1.8% tput, +1.1% to +3.0% P90, bit-exact, GSM8K 96.27%; below 4%.
- **09-lm-head-aiter-gemm** (opt-in AITER tgemm for the bf16 lm_head, upstream #47485): -1.1% to +1.75%; lm_head is ~0.2 ms of a 16+ ms step and tgemm is ~9% slower at verify M=72-128; GSM8K 96.73%.
- **10-shared-attn-fp8-quant** (one per-token FP8 quant shared by qkv_proj and the QSA indexer's index_qk_proj): +0.2% to +2.75%, bit-exact, GSM8K 96.35%; ~15 us per QSA layer is too small.
- **12-qsa-chunk-prefill-rocm-config** (ROCm launch table for sparse-GQA chunk prefill): dropped at microbench, 0.95-1.10x vs upstream's config (upstream now packs K/V to BF16, which removed the fork's FP8-pool win); no gate run.
- **14a-gdn-norm-quant** (AITER gated RMSNorm + per-token FP8 quant feeding the GDN out_proj) and **14b-moe-fused-shared-topk** (router + shared-expert gate as one N=528 GEMM + AITER topk_gating): joint gate max +0.82%, -1.07% P90 at worst; GSM8K 96.27% each; below 4%.
