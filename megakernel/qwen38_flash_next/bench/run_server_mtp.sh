#!/bin/bash
# The AMD MXFP4 TP1 server (colleague's run_server.sh) with its MTP config: NEXTN, 3 steps,
# 4 draft tokens, topk 1, simulated acceptance. Every decode step is then a 4-token verify pass,
# which the megakernel runs as gdn_block_verify. Added to the colleague's flags: a draft config that
# lists the MTP experts in quark's exclude list. This checkpoint stores them in bf16 without listing
# them, so this branch builds them as MXFP4 and fails to load (1280 vs 2560); listed, SGLang's own
# _mtp_quant_config builds the whole draft in bf16, which is all the MTP module holds.
#   ./run_server_mtp.sh                                         # before
#   SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL=1 ./run_server_mtp.sh   # after
# With the megakernel on, the log shows "Compiling the Qwen4 GDN megakernel" during the batch-1
# graph-capture warmup; if it doesn't, every layer fell back to the original path.
# Pick a GPU with HIP_VISIBLE_DEVICES=N (check rocm-smi for a free one).

# A shell without the group that owns /dev/kfd sees no GPUs: rerun through gpu_run.sh, which
# drops the environment, so forward the variables this script reads.
if [ ! -w /dev/kfd ]; then
  fwd=(SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL="${SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL:-0}")
  [ -n "${HIP_VISIBLE_DEVICES:-}" ] && fwd+=(HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES")
  exec bash "$(dirname "$(readlink -f "$0")")/gpu_run.sh" env "${fwd[@]}" "$(readlink -f "$0")"
fi

# The colleague's machine keeps the cache in /data/hf_cache; this one in the repo's hf_cache.
export HF_HOME=/data/hf_cache/
[ -d "$HF_HOME" ] || export HF_HOME=/sgl-workspace/sglang/hf_cache/
export HF_HUB_CACHE="${HF_HOME}hub"
export AITER_GDR_FLYDSL=1
export SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=0.1
export SGLANG_EXACT_CHUNK_FILL=1
export FLYDSL_RUNTIME_ENABLE_CACHE=0
if [ -z "${REAL_MTP:-}" ]; then  # REAL_MTP=1: the draft's real acceptance (for accuracy checks)
  export SGLANG_SIMULATE_ACC_LEN=2.32
  export SGLANG_SIMULATE_ACC_METHOD=match-expected
  export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi
# Where aiter's package dir is read-only (this machine), aiter JIT-builds into ~/.aiter/jit but imports
# from the package; point both there. Same for FlyDSL's cache and lock files (as launch_server.sh).
if [ ! -w "$(/opt/venv/bin/python -c 'import aiter, os; print(os.path.dirname(aiter.__file__))' 2>/dev/null)/jit" ]; then
  export AITER_JIT_DIR=${AITER_JIT_DIR:-$HOME/.aiter/jit}
  export FLYDSL_RUNTIME_CACHE_DIR=${FLYDSL_RUNTIME_CACHE_DIR:-$HOME/.aiter/jit/flydsl_cache}
fi
MK=${SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL:-0}

MODEL_PATH=${HF_HUB_CACHE}/models--amd--Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8/snapshots/1ad36fa866c05631c14adc6a74d7e9908c4e5888
[ -d "$MODEL_PATH" ] || { echo "checkpoint not found: $MODEL_PATH" >&2; exit 1; }
# The draft reads the checkpoint through a directory of symlinks whose config.json is patched (an
# override file alone didn't reach the quantization path).
DRAFT_DIR=$(dirname "$(readlink -f "$0")")/configs/mxfp4_draft_bf16_mtp
if [ ! -f "$DRAFT_DIR/config.json" ]; then
  mkdir -p "$DRAFT_DIR"
  for f in "$MODEL_PATH"/*; do [ "$(basename "$f")" = config.json ] || ln -sf "$(readlink -f "$f")" "$DRAFT_DIR/$(basename "$f")"; done
  /opt/venv/bin/python - "$MODEL_PATH/config.json" "$DRAFT_DIR/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
q = c.get("quantization_config") or c["text_config"]["quantization_config"]
q["exclude"] = q["exclude"] + ["mtp.layers.0.mlp.experts"]
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
fi
echo "GPUs: ${HIP_VISIBLE_DEVICES:-all}  megakernel: $MK" >&2

/opt/venv/bin/python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name qwen3.8-flash-next \
  --host 127.0.0.1 \
  --port "${PORT:-18888}" \
  --trust-remote-code \
  --quantization quark \
  --attention-backend aiter \
  --tp-size 1 \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 16384 \
  --page-size 64 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  --speculative-eagle-topk 1 \
  --speculative-draft-model-path "$DRAFT_DIR" \
  --max-running-requests 16 \
  --cuda-graph-max-bs-decode 64 \
  --scheduler-recv-interval 10 \
  --stream-interval 50 \
  --enable-metrics \
  --tokenizer-worker-num 6 \
  2>&1 | tee server_mtp_mk${MK}.log
