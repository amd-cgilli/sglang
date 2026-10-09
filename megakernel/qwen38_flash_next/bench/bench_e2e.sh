#!/bin/bash
# End-to-end serving benchmark against the running server (launch_server.sh).
# Usage: [MODEL=...] ./gpu_run.sh ./bench_e2e.sh <tag>   (MODEL must match the server's)
set -e
TAG=${1:-bf16_tp8}
MODEL=${MODEL:-Qwen/Qwen3.8-Flash-Next}
ISL=${ISL:-1024}
OSL=${OSL:-512}
# Online: the random dataset samples its text from ShareGPT on first use.
export HF_HOME=/sgl-workspace/sglang/hf_cache
mkdir -p results
for CONC in 1 4 16 64; do
  case $CONC in 1) N=8;; 4) N=16;; 16) N=64;; 64) N=256;; esac
  python3 -m sglang.bench_serving \
    --backend sglang-oai --host 127.0.0.1 --port 30000 \
    --model "$MODEL" \
    --dataset-name random --random-input-len $ISL --random-output-len $OSL --random-range-ratio 1 \
    --num-prompts $N --max-concurrency $CONC --request-rate inf \
    --warmup-requests 2 --flush-cache \
    --output-file results/${TAG}_isl${ISL}_osl${OSL}.jsonl
done
