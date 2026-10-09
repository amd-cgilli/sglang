#!/bin/bash
# Before/after for the GDN megakernel on the MXFP4 TP1 server, concurrency 1. Runs the server
# script twice (megakernel off, then on), the same bench_serving client against each, and prints
# both summaries. SERVER picks the script: run_server_nomtp.sh (default) or run_server_mtp.sh.
# BEFORE_ENV / AFTER_ENV override the two servers' env (default: the megakernel off / on), e.g.
#   BEFORE_ENV="SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL=1 SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL_FFN=0"
# ISL / OSL / NUM_PROMPTS / WARMUP / CONC set the client (default 1024 / 512 / 8 / 2 / 1).
#   HIP_VISIBLE_DEVICES=7 ./ab_c1.sh
#   HIP_VISIBLE_DEVICES=7 SERVER=run_server_mtp.sh ISL=32768 OSL=32768 NUM_PROMPTS=2 WARMUP=1 ./ab_c1.sh
# (here, where the shell can't open /dev/kfd: bash gpu_run.sh env HIP_VISIBLE_DEVICES=7 ... ./ab_c1.sh)
set -u
cd "$(dirname "$(readlink -f "$0")")"
M=$(ls -d /sgl-workspace/sglang/hf_cache/hub/models--amd--Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8/snapshots/1ad36fa866c05631c14adc6a74d7e9908c4e5888 \
      /data/hf_cache/hub/models--amd--Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8/snapshots/1ad36fa866c05631c14adc6a74d7e9908c4e5888 2>/dev/null | head -1)
M=${M:-amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8}  # the tokenizer
mkdir -p results
SERVER=${SERVER:-run_server_nomtp.sh}
NAME=$(basename "$SERVER" .sh | sed 's/^run_server_//')  # nomtp | mtp

run() {  # $1 = before | after, $2 = the server's env assignments
  local tag=${NAME}_$1 log=results/ab_server_${NAME}_$1.log
  echo "=== $tag: starting server ($2)"
  setsid env $2 ./"$SERVER" > "$log" 2>&1 &
  local pgid=$!
  for _ in $(seq 1 180); do
    grep -q "The server is fired up" "$log" && break
    grep -q "Initialization failed\|Scheduler hit an exception" "$log" && { echo "server failed, see $log"; kill -9 -$pgid; return 1; }
    sleep 5
  done
  grep -q "The server is fired up" "$log" || { echo "server not up after 15 min, see $log"; kill -9 -$pgid; return 1; }
  grep -o "Compiling the Qwen4 GDN megakernel for [a-z_]*\|Qwen4 GDN megakernel not used: .*" "$log" | sort | uniq -c
  python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 18888 \
    --model qwen3.8-flash-next --tokenizer "$M" --tokenize-prompt --seed 42 \
    --dataset-name random --random-input-len "${ISL:-1024}" --random-output-len "${OSL:-512}" --random-range-ratio 1 \
    --num-prompts "${NUM_PROMPTS:-8}" --max-concurrency "${CONC:-1}" --request-rate inf \
    --warmup-requests "${WARMUP:-2}" --flush-cache \
    --output-file "results/mxfp4_tp1_${tag}_isl${ISL:-1024}_osl${OSL:-512}_c${CONC:-1}.jsonl" > results/ab_client_$tag.log 2>&1
  echo "client exit $?"
  kill -TERM -$pgid; sleep 15; kill -9 -$pgid 2>/dev/null; sleep 5
}

export HF_HOME=/sgl-workspace/sglang/hf_cache
run before "${BEFORE_ENV:-SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL=0}"
run after "${AFTER_ENV:-SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL=1}"
for tag in before after; do
  echo "=== $NAME $tag"
  grep -E "Successful requests|Output token throughput|Median TTFT|Median TPOT|Median ITL|Mean TPOT|Accept length" results/ab_client_${NAME}_$tag.log
done
