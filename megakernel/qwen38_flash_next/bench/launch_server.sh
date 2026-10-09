#!/bin/bash
# Cookbook MI350X "balanced" cell, verbatim except host/port.
# Defaults are the BF16 TP8 cell. Overrides (not cookbook cells):
#   MODEL=Qwen/Qwen3.8-Flash-Next-FP8 TP=1 ./gpu_run.sh ./launch_server.sh
# Run through gpu_run.sh so the process has access to /dev/kfd.
MODEL=${MODEL:-Qwen/Qwen3.8-Flash-Next}
TP=${TP:-8}
export HF_HOME=/sgl-workspace/sglang/hf_cache
export HF_HUB_OFFLINE=1
# /sgl-workspace/aiter is read-only for devuser, so aiter JIT-builds into ~/.aiter/jit
# but would import from the package dir; pointing both at ~/.aiter/jit fixes that.
export AITER_JIT_DIR=/home/devuser/.aiter/jit
# Same for FlyDSL: aiter defaults its compile cache (and lock files) to the read-only package dir.
export FLYDSL_RUNTIME_CACHE_DIR=/home/devuser/.aiter/jit/flydsl_cache
exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size "$TP" \
  --attention-backend aiter \
  --page-size 32 \
  --kv-cache-dtype auto \
  --chunked-prefill-size 16384 \
  --watchdog-timeout 1200 \
  --mem-fraction-static 0.9 \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 30000 \
  "$@"
