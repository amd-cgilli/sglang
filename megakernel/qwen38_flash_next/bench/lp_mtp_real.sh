#!/bin/bash
# Real-MTP logprob check: the MTP server with real acceptance, stock then the megakernel, two
# greedy dumps each (server_logprobs.py), on GPU $HIP_VISIBLE_DEVICES and port $PORT.
#   bash gpu_run.sh env HIP_VISIBLE_DEVICES=6 PORT=18890 ./lp_mtp_real.sh
set -u
cd "$(dirname "$(readlink -f "$0")")"
export HF_HOME=/sgl-workspace/sglang/hf_cache HF_HUB_OFFLINE=1 REAL_MTP=1
for mk in 0 1; do
  log=results/lp_mtp_server_mk$mk.log
  setsid env SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL=$mk ./run_server_mtp.sh > "$log" 2>&1 &
  pgid=$!
  for _ in $(seq 1 180); do grep -q "The server is fired up\|Initialization failed" "$log" && break; sleep 5; done
  grep -o "Compiling the Qwen4 GDN megakernel for [a-z_]*" "$log" | sort | uniq -c
  for i in 1 2; do python3 server_logprobs.py --port "$PORT" --out results/lp_mtp_mk${mk}_$i.json; done
  kill -TERM -$pgid; sleep 15; kill -9 -$pgid 2>/dev/null; sleep 5
done
