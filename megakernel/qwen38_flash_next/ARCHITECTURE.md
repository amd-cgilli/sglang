# Qwen3.8-Flash-Next: decode-step architecture reference

Working reference for the megakernel. All dimensions are from the real checkpoint
config (`Qwen/Qwen3.8-Flash-Next`, `text_config`). Paths are under `python/sglang/`.
Executable version (TP=1 FP8, validated against SGLang): `reference/qwen38_ref.py`.

## 1. Where it lives in SGLang

| Piece | File |
|---|---|
| Entry class `Qwen4ExpForConditionalGeneration` | `srt/models/qwen4_exp.py:2005` |
| Text stack `Qwen4ExpModel.forward` | `srt/models/qwen4_exp.py:1869` |
| GDN decoder layer | `srt/models/qwen4_exp.py:1596` (base `Qwen3_5LinearDecoderLayer`, `srt/models/qwen3_5.py:1080`) |
| QSA decoder layer | `srt/models/qwen4_exp.py:1637` (base `Qwen3_5AttentionDecoderLayer`, `srt/models/qwen3_5.py:1209`) |
| GDN token mixer `Qwen3_5GatedDeltaNet` | `srt/models/qwen3_5.py:327` |
| Gated residual (hyper-connection) | `srt/layers/hyperconnection.py:115`, wiring in `srt/layers/layer_boundary/residual/gated.py` |
| N-gram embedding (PLE) | `srt/models/qwen4_exp.py:545` (hash), `:1030` (layer) |
| QSA indexer / backend | `srt/layers/attention/qsa/qsa_indexer.py`, `srt/layers/attention/qwen_sparse_attn_backend.py` |
| MoE block | `srt/models/qwen2_moe.py:255` (`Qwen2MoeSparseMoeBlock`) |
| MTP draft | `srt/models/qwen4_exp_mtp.py` |
| Config | `srt/configs/qwen4_exp.py` (subclass of `Qwen3NextConfig`) |

## 2. Dimensions (global, TP=1)

| Symbol | Value | Config field |
|---|---|---|
| H (hidden) | 2560 | `hidden_size` |
| L (layers) | 48 = 36 GDN + 12 QSA, pattern `[GDN,GDN,GDN,QSA] x 12` | `layer_types` |
| hc (residual streams) | 4, so the residual is `[T, 10240]` | `hc_count` |
| hc_lowrank | 320 | `hc_lowrank` |
| vocab | 248320, untied lm_head | `vocab_size` |
| rms eps | 1e-6 | `rms_norm_eps` |
| GDN heads | 16 qk heads x 128, 48 v heads x 128, conv width 4 | `linear_*` |
| QSA heads | 24 q / 2 kv, head_dim 256, rotary dim 64 (factor 0.25), mrope interleaved [11,11,10], theta 1e7 | |
| QSA indexer | 4 heads x 128, 1 k head, compress ratio 4, budget 2048 tokens (512 blocks) | `indexer_*` |
| MoE | 512 experts, top-10, expert I=640, shared expert I=640 | |
| PLE | on layer index 1 only (`ple_layer_ids=[2]` is 1-based), ngram 3, 8 heads per ngram order, 16 heads x 160 dims | |
| GDN out gate | `output_gate_type = "sigmoid"` | |
| MTP | 1 layer, QSA attention + MoE | `mtp_num_hidden_layers` |

Parameter budget check: experts 2.52B/layer x 48 = 121B, N-gram table 320,001,536 rows x 160 = 51.2B.
Active per token is about 6.5B: routed+shared experts 2.6B, GDN 2.1B, QSA 0.6B, gated residual 0.63B, lm_head 0.64B.

## 3. Top-level decode step

```
R = repeat(embed_tokens[tok], 4)                  # [T, 4*2560], streams start identical
ple_batch = prepare_ple_batch()                   # n-gram history (2 prev tokens) per request
for l in 0..47:
    if l == 1: R = R + PLE(R)                     # added BEFORE the attention read
    x, n_a = mix_attn[l](R)                       # [T,2560], n_a = per-stream-normed R
    a = GDN[l](x)  or  QSA[l](x)                  # TP: partial sum -> all-reduce
    R = combine_attn[l](a, R, n_a)
    x, n_f = mix_ffn[l](R)
    y = MoE[l](x)                                 # TP/EP: all-reduce
    R = combine_ffn[l](y, R, n_f)
commit_ple_batch()                                # shift n-gram history by one token
h = final_mixer.mix(R)                            # NO final RMSNorm; this is the terminal read
logits = h @ lm_head.T                            # [T, 248320]
# MTP consumes R (the widened [T,10240] residual), not h.
```

There is no `input_layernorm` or `post_attention_layernorm`. The gated mix does the normalization.

## 4. Gated residual (one `GatedResidual` per stage, 2 per layer + 1 final)

Weights: `hc_norm.weight [10240]`, `input_mix_weight_down [320, 10240]`,
`input_mix_weight_up [10240, 320]`, `block_inject_weight [4, 10240]` (all bf16).

```
n      = gemma_rmsnorm_per_stream(R)              # group size 2560, x*rsqrt(mean(x^2)+eps)*(1+w), fp32 math
g      = sigmoid(W_up @ silu((W_down @ n) / 4))   # [10240]
x      = mean_over_4_streams(g * n)               # [2560], stage input
...stage computes out...
c      = 2 * sigmoid((W_inj @ n) / 4)             # [4], uses the SAME n as the read
R'_s   = R_s + c_s * out                          # per stream s
```

The final mixer has only the mix half (`use_combine=False`).
Current CUDA kernels: `hc_mix` (SM100, T<=24), `fused_hc_mix`, `hc_combine_split` (T<=32), `hc_combine`.

## 5. GDN token mixer (36 layers)

Weights: fused `in_proj` = `[qkvz (16384) | ba (96)] x 2560`; `conv1d [10240, 4]` channel order `[q 2048 | k 2048 | v 6144]`;
`A_log [48]` fp32; `dt_bias [48]`; `norm.weight [128]` (plain weight, no 1+w); `out_proj [2560, 6144]`.

Per request, per token:
```
qkvz, b, a = in_proj(x)
qkv = silu(causal_conv1d_update(conv_state[10240,3], qkv))   # fp32 acc, shift-left state
q,k (16 heads x 128), v (48 heads x 128), z (48 x 128)
q = l2norm(q) * 128**-0.5 ; k = l2norm(k)                    # eps 1e-6 inside sqrt
g    = -exp(A_log) * softplus(a + dt_bias)                   # softplus threshold 20
beta = sigmoid(b), rounded to bf16 then back to fp32
for each v head hv (k head = hv // 3):
    S = S * exp(g)                                           # S: [V=128, K=128], K-last, fp32 compute
    S = S + outer((v - S@k) * beta, k)
    o = S @ q
o = rmsnorm(o) * w * sigmoid(z)                              # per head over 128; sigmoid, not silu
out = out_proj(o)
```

State per request per layer: SSM `[48,128,128]` (3 MiB fp32, 1.5 MiB bf16 with `--mamba-ssm-dtype bfloat16`), conv `[10240,3]` bf16.
Indexed by `req_to_token_pool.get_mamba_indices(req_pool_indices)`; slot 0 or index < 0 is padding.
Kernels: `fused_qkvzba_causal_conv1d_update_contiguous` (`kernels/ops/attention/triton_gdn_fused_proj.py`),
`fused_recurrent_gated_delta_rule_packed_decode` (`kernels/ops/attention/fla/fused_recurrent.py:187`),
FlashInfer `gated_delta_rule_decode_pretranspose` on the flashinfer backend.

## 6. QSA attention (12 layers)

Weights: `qkv_proj [13312, 2560]` where the q part is per-head interleaved `[q(256) | gate(256)] x 24`, then k (2x256), v (2x256);
`q_norm`, `k_norm` Gemma `[256]`; `o_proj [2560, 6144]`;
indexer `index_qk_proj [640, 2560]` (4 q heads + 1 k head, x128), `q_layernorm`, `k_layernorm` Gemma `[128]`. No indexer head weights.

```
q, gate, k, v = split(qkv_proj(x))
q, k = gemma_rmsnorm(q), gemma_rmsnorm(k); neox rope on first 64 dims (mrope)
write k, v to paged KV cache (page 64)

# indexer
qi, ki = split(index_qk_proj(x)); qi = rope(gemma_norm(qi)); ki stays raw
push ki into the per-request pending ring (4 slots)
if seq_len % 4 == 0:                      # this token closes a group
    kc = rope(k_layernorm(mean_fp32(ring)), pos = group's first token)
    store kc at compressed slot full_slot // 4
score[j] = sum_h relu(qi_h . kc_j) / sqrt(128)   for j < seq_len // 4
blocks = topk(score, 512)
tokens = expand(blocks, 4) + tail [floor(seq_len/4)*4, seq_len)     # <= 2051 tokens

o = softmax_gqa(q, K[tokens], V[tokens]) * 256**-0.5 scale
o = o * sigmoid(gate)
out = o_proj(o)
```

Below about 2052 tokens every block is selected, so QSA equals dense attention.
When `seq_len % 4 == 0` the tail is empty, so the current token is attended only if its own block is selected.
MTP draft steps reuse the draft-extend top-k (`index_share_for_mtp_iteration`) and skip the indexer.

## 7. MoE (all 48 layers + MTP)

Weights: router `gate [512, 2560]` bf16 (never quantized, no bias); `w13 [512, 1280, 2560]` (gate then up);
`w2 [512, 2560, 640]`; shared expert `gate_up [1280, 2560]`, `down [2560, 640]`; `shared_expert_gate [1, 2560]`.

```
p = softmax_fp32(gate @ x); top10; renormalize
y = sum_k w_k * down_k(silu(gate_k(x)) * up_k(x)) + sigmoid(x . w_sg) * shared(x)
```

No routed scaling factor, no correction bias, no dense layers.

FP8 checkpoint: only routed experts (128x128 block scales) and the PLE table are FP8. On ROCm with aiter,
SGLang loads the BF16 shared expert into FP8 expert slot 512 with scale 1.0 (lossy); see `reference/README.md`.

## 8. N-gram embedding / PLE (layer index 1 only)

Weights: table `[320001536, 160]` (bf16, or fp8 e4m3 with a per-tensor `weight_scale`), usually in pinned host memory;
`key_proj [10240, 2560]`; `value_proj [2560, 2560]`; three Gemma norms `[10240]` (per stream);
depthwise `conv1d [10240, 1, 4]`, dilation 3.

Per request, per token, with context `c = [t-2, t-1, t]` (EOS resets the window; earlier tokens become the EOS id):
```
for order n in {2,3}:  mix = XOR_{i<n} (c[t-i] * mult[i])            # int64, splitmix-derived odd multipliers
    for head h in 8:   id = mix mod prime_h + offset_h                # 16 ids total
emb   = table[ids] * weight_scale -> [2560]                            # 16 x 160
key   = key_proj(emb) -> [4, 2560];  value = value_proj(emb) -> [2560]
s     = sum(norm_key(key)_s * norm_query(R)_s) / sqrt(2560)           # [4]
gate  = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))
gv    = gate_s * value                                                 # [4, 2560]
conv  = silu(dilated_depthwise_conv(norm_conv(gv)))                    # state [10240, 9]
R     = R + gv + conv
```

The gather is issued one layer early on a side stream (`start_prefetch`) so the host read overlaps layer 0.
Side state per request: n-gram history (2 tokens) and short-conv state `[10240, 9]`, both in the mamba pool.

## 9. Recommended serving shapes (cookbook)

From `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx`: BF16/FP8 at TP4 (+EP4 for throughput) on H200/B200/B300/GB300;
NVFP4 at TP1 on B200/B300/GB300 and RTX PRO 6000; TP2 across two DGX Sparks. MTP NEXTN 3/1/4 for low latency.
`--mamba-ssm-dtype bfloat16` and FlashInfer GDN on the datacenter cells.

## 10. Megakernel notes

- The decode step is weight-bandwidth bound: about 6.5B active params per token, dominated by 11 of 513 experts per layer.
- Data-dependent work inside the step: MoE expert selection, QSA top-k block selection, and the PLE host gather.
- Numerics that differ from Qwen3-Next: sigmoid GDN output gate, Gemma (1+w) norms everywhere except the GDN gated norm, beta rounded to bf16, no final norm.
- Non-model state the kernel must update: GDN conv + SSM state, QSA KV cache, the indexer ring and compressed cache, PLE n-gram history and short-conv state.
