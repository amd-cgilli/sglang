# Rejected candidates (not on the branch)

Gate rule: keep a PR if tput or P90 interactivity improves >= 4% in any of {64k-prefix multi-turn, 16k/1k} x {c24, c80}
and nothing regresses > 2%. Measured 2026-11-03 on MI355X TP1 against bare v2 (sglang 99fd01047d + scaffold; AITER pin
285579b62, + PRs 03/04 for the SGLang candidates). Patches are kept in the eval workspace (`perf/v2/patches/`) for the record.

- **01, 06, 09, 10** (AITER GEMM tunes, packed FP8 GDN in_proj, lm_head AITER GEMM, shared attn FP8 quant): each below 4% alone (+0.05% to +3.0%, 2026-11-03); later kept together as PRs 18-21 (agentic C24 leave-one-out of the bundle +2.9 P90 / +1.1k tput, 2026-11-05).
- **02-aiter-moe-sort-quant** (AITER MoE sort + quant fusion): skipped by user decision, not ported.
- **12-qsa-chunk-prefill-rocm-config** (ROCm launch table for sparse-GQA chunk prefill): dropped at microbench, 0.95-1.10x vs upstream's config (upstream now packs K/V to BF16, which removed the fork's FP8-pool win); no gate run.
- **14a-gdn-norm-quant** (AITER gated RMSNorm + per-token FP8 quant feeding the GDN out_proj) and **14b-moe-fused-shared-topk** (router + shared-expert gate as one N=528 GEMM + AITER topk_gating): joint gate max +0.82%, -1.07% P90 at worst; GSM8K 96.27% each; below 4%.
- **C1 qsa-chunk-config-fp8-gather** (fork `_CHUNK_CONFIGS` for sparse-GQA chunk prefill in qsa/sparse_attn.py + keep packed prefill K/V FP8 via `pack_qsa_prefill_kv` output_dtype, +17/-2): dropped 2026-11-05. Agentic C24 (3600 s, acceptance 2.32): with C1 (D5) 87.7 / 85.1 / 85.8, mean 86.2 P90 / 124.2k tput; without C1 84.4 / 85.5, mean 85.0 / 123.3k; -1.2 P90 / -0.85k, below the ~2 P90 keep bar (2 sd = 3.9). Leave-one-out -0.1 P90 / +0.3k.
