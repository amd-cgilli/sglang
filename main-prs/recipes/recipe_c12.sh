#!/usr/bin/env bash
# Qwen3.8-Flash-Next (amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8) on 1x MI355X (gfx950), TP1, AgentX agentic C12
#
# Software:
#   SGLang : amd-cgilli/sglang branch qwen38-submission-v2 (upstream main + main-prs/ 05, 07, 08, 11, 13, 15, 16, 17, 19, 20, 21)
#   AITER  : ROCm/aiter 285579b62 (the commit SGLang's ROCm Dockerfile pins) + main-prs/ 03, 04, 18 aiter.patch
#            (main-prs/tools/apply.sh --aiter <aiter-dir> builds it)
# Server params: perf/params/c12.sh (max-running 24, graph bs 24). Generated; flags deduplicated (params win).

export PYTHONNOUSERSITE=1
export SGLANG_USE_AITER=1
export SGLANG_USE_AITER_UNIFIED_ATTN=1
# keep --mem-fraction-static as given (the aiter path otherwise scales it)
export SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1
export SGLANG_TIMEOUT_KEEP_ALIVE=1800
# 05-qsa-flydsl-decode (default on; explicit)
export SGLANG_ENABLE_QSA_FLYDSL_DECODE=1
# 07-gdn-aiter-prefill (default on; needs AITER 03)
export SGLANG_AITER_GDN_CONV=1
# 07-gdn-aiter-prefill (default on; needs AITER 04)
export SGLANG_AITER_GDN_PREFILL=1
# 08-fused-hc-decode (opt-in)
export SGLANG_OPT_QWEN4_FUSED_HC=1
# 15-hc-prefill-fusion (default on for HIP; explicit)
export SGLANG_ENABLE_HC_COMBINE_NORM_FUSION=1
# 20-lm-head-aiter-gemm (opt-in)
export SGLANG_OPT_AITER_LM_HEAD_GEMM=1
# 21-shared-attn-fp8-quant (opt-in)
export SGLANG_OPT_QWEN4_SHARED_ATTN_QUANT=1
# InferenceX AgentX pins MTP acceptance to the golden AL 2.32 (all vendors); drop the 3 SIMULATE vars for real verification
export SGLANG_SIMULATE_ACC_LEN=2.32
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token

# Launch flags beyond the PRs (measured; see README "Launch flags"):
#   --cuda-graph-backend-prefill breakable: prefill runs as CUDA graphs (default on CUDA, off on ROCm).
#     Gate (64k-prefix multi-turn, 16k cold) +6.5..+8.4% tput; agentic C12 +2.3% P90 / +1.5% tput, C80 +1.6% / +4.0%.
#     Costs a ~17-20 GB graph pool and ~45 s of startup; the KV pool is unchanged.
#   --mamba-track-interval 262144: no Mamba state checkpoints during decode (AgentX never re-sends model output),
#     so the cache keeps prompt prefixes on the GPU instead of spilling them to HiCache. Gate 64k-prefix multi-turn
#     c24 +5.1% tput / +8.4% P90, c80 +7.8% / +19.5%; agentic C80 +8.3% tput / +13.6% P90, C12 neutral.
#   No HiCache at C12/C24 (GPU-only, as NVIDIA): agentic on/off on this stack 2026-11-06, C12 +1.7% P90 / -0.1% tput,
#     C24 +1.2% / -0.25% with it off (noise); host-tier hits 0.03% / 0.2%, so the 200 GB host pool bought nothing.
python -m sglang.launch_server \
  --host 0.0.0.0 --port ${PORT:-8000} \
  --model-path amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8 \
  --served-model-name qwen3.8-flash-next \
  --trust-remote-code \
  --tp-size 1 \
  --quantization quark \
  --attention-backend aiter \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.88 \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --watchdog-timeout 1200 \
  --chunked-prefill-size 16384 \
  --mamba-ssm-dtype bfloat16 \
  --ple-offload-embedding \
  --json-model-override-args '{"text_config": {"ple_embedding_dtype": "float8_e4m3fn"}}' \
  --scheduler-recv-interval 10 \
  --stream-interval 50 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-metrics \
  --enable-cache-report \
  --max-running-requests 24 \
  --cuda-graph-max-bs-decode 24 \
  --tokenizer-worker-num 6 \
  --enable-linear-replayssm-spec \
  --speculative-draft-kv-cache-dtype fp8_e4m3 \
  --max-mamba-cache-size 960 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-track-interval 262144 \
  --cuda-graph-backend-prefill breakable \

# Client (InferenceX inferencex-e2e AgentX replay, as in the official runs):
#   agentx profile --url http://127.0.0.1:${PORT:-8000} --model qwen3.8-flash-next --tokenizer amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8 \
#     --concurrency 12 --benchmark-duration 3600 --tokenizer-trust-remote-code --server-metrics http://127.0.0.1:${PORT:-8000}/metrics \
#     --public-dataset semianalysis_cc_traces_weka_062126_256k --apply-chat-template
