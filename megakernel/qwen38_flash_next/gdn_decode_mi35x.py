"""Qwen3.8-Flash-Next GDN (Gated DeltaNet) decode step, as SGLang runs it on MI35X.

Self-contained: depends only on torch and triton (aiter is optional, used for the
GEMMs when importable, exactly as SGLang routes them). No sglang import outside
the optional `--check-sglang` parity test in __main__.

Deployment this reproduces (cookbook MI350X/MI355X BF16 cell, verified against the
server log in bench/logs/server_bf16_tp8.log):
  --tp-size 8, --attention-backend aiter, SGLANG_USE_AITER=1, no speculative decoding,
  no --mamba-ssm-dtype (checkpoint mamba_ssm_dtype=float32 -> fp32 SSM state).
  Log: "Linear attention kernel backend: decode=triton ..." and
       "GDN kernel dispatcher: decode=TritonGDNKernel ... packed_decode=True".

Per-rank decode step for one GDN layer (B requests, one token each):
  1. qkvz = x @ W_qkvz^T  [B, 2048]   aiter tgemm.mm (K=2560 > 2048 rules out Triton a16w16)
     ba   = x @ W_ba^T    [B, 12]     aiter tgemm.mm
     The packed qkvz|ba GEMM is Qwen3.5-only on ROCm (_QWEN3_5_ROCM_PACKED_MODEL_TYPES).
  2. Unfused split + cat (fix_query_key_value_ordering): head ratio 48/16 = 3 is not in the
     aiter fused-unpack ratios (1, 2, 4, 8), and the fused unpack+conv decode is CUDA-only.
  3. Triton causal_conv1d_update, width 4, SiLU, indexed conv state [slots, 1280, 3] bf16.
  4. Triton packed recurrent decode: l2norm(q,k), q*=128^-0.5, g=-exp(A_log)*softplus(a+dt_bias),
     beta=sigmoid(b) rounded to bf16, S=S*exp(g); S+=k(x)((v-S k)*beta); o=S q.
     SSM state [slots, 6, 128(V), 128(K)] fp32, updated in place.
  5. Triton gated RMSNorm per head (eps 1e-6, plain weight, sigmoid(z) gate after the norm).
  6. out_proj [B, 768] -> [B, 2560]: aiter Triton gemm_a16w16 when B <= 64 (K=768 <= 2048),
     else tgemm.mm. Result is a TP partial sum; the all-reduce runs in the layer boundary.

Sources (python/sglang/): srt/models/qwen3_5.py (Qwen3_5GatedDeltaNet.forward),
srt/layers/attention/linear/gdn_backend.py (GDNAttnBackend.forward_decode),
srt/layers/attention/linear/kernels/gdn_triton.py, kernels/ops/mamba/causal_conv1d_triton.py,
kernels/ops/attention/fla/fused_recurrent.py, kernels/ops/attention/fla/layernorm_gated.py,
srt/layers/quantization/unquant.py (UnquantizedLinearMethod.apply).

Run the self-test on an MI35X node (needs /dev/kfd, see bench/gpu_run.sh):
  python gdn_decode_mi35x.py                 # random weights vs fp32 torch reference
  python gdn_decode_mi35x.py --real-weights  # layer-0 weights from the local HF cache
  python gdn_decode_mi35x.py --check-sglang  # bitwise parity with SGLang's own kernels
"""

import argparse
import os
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ===== Model constants (Qwen3.8-Flash-Next text_config) =====

HIDDEN_SIZE = 2560
NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_KERNEL = 4
RMS_EPS = 1e-6
SOFTPLUS_THRESHOLD = 20.0
TP_SIZE = 8

# Padded rows (CUDA-graph padding, idle DP ranks) carry this slot index.
PAD_SLOT_ID = -1


class GDNShape(NamedTuple):
    """Per-rank GDN dimensions for a given TP degree."""

    tp_size: int
    num_k_heads: int  # per rank
    num_v_heads: int  # per rank
    key_dim: int  # per rank, q and k each
    value_dim: int  # per rank, v and z each
    conv_dim: int  # q + k + v

    @classmethod
    def for_tp(cls, tp_size: int = TP_SIZE) -> "GDNShape":
        assert NUM_K_HEADS % tp_size == 0 and NUM_V_HEADS % tp_size == 0
        nk = NUM_K_HEADS // tp_size
        nv = NUM_V_HEADS // tp_size
        key_dim = nk * HEAD_K_DIM
        value_dim = nv * HEAD_V_DIM
        return cls(tp_size, nk, nv, key_dim, value_dim, 2 * key_dim + value_dim)


class GDNWeights(NamedTuple):
    """One rank's GDN layer weights in SGLang's in-memory layout."""

    in_proj_qkvz: torch.Tensor  # [2*key_dim + 2*value_dim, 2560] bf16, rows q|k|v|z
    in_proj_ba: torch.Tensor  # [2*num_v_heads, 2560] bf16, rows b|a
    conv_weight: torch.Tensor  # [conv_dim, 4] bf16, channels q|k|v
    A_log: torch.Tensor  # [num_v_heads] fp32
    dt_bias: torch.Tensor  # [num_v_heads] bf16 (model default dtype)
    norm_weight: torch.Tensor  # [128] bf16
    out_proj: torch.Tensor  # [2560, value_dim] bf16


# ===== Triton kernel 1: causal_conv1d_update (decode, seqlen=1) =====
# Specialized from _causal_conv1d_update_kernel with IS_SPEC_DECODING, SAVE_INTERMEDIATE
# and the EAGLE-tree branch off, KERNEL_WIDTH=4. The bf16 products (x * w) are rounded to
# bf16 before the fp32 accumulate, as in the original; keep the expressions unchanged.


@triton.jit()
def _causal_conv1d_update_kernel(
    x_ptr,  # (batch, dim, 1)
    w_ptr,  # (dim, width)
    conv_state_ptr,  # (num_slots, dim, state_len)
    conv_state_indices_ptr,
    o_ptr,  # (batch, dim, 1)
    batch: int,
    dim: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    pad_slot_id: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seqlen: tl.constexpr = 1
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    conv_state_batch_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices
    ).to(tl.int64)
    if conv_state_batch_coord == pad_slot_id:
        return

    # Read the 3-token window before the shift.
    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim
    col0 = tl.load(conv_states_base, mask_w, 0.0)
    col1 = tl.load(conv_states_base + 1 * stride_conv_state_tok, mask_w, 0.0)
    col2 = tl.load(conv_states_base + 2 * stride_conv_state_tok, mask_w, 0.0)

    # Shift the window left by one and append x.
    idx_tokens = tl.arange(0, NP2_STATELEN)
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + seqlen) * stride_conv_state_tok)[:, None]
    )
    mask = (
        (conv_state_batch_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)
    x_ptrs = x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)
    conv_state_ptrs_target = (
        conv_states_base[None, :] + (idx_tokens * stride_conv_state_tok)[:, None]
    )
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    w_base = w_ptr + (idx_feats * stride_w_dim)
    w_col0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
    w_col1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
    w_col2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
    w_col3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)

    acc = acc_preload
    acc += col0 * w_col0
    acc += col1 * w_col1
    acc += col2 * w_col2
    matrix_x = tl.load(x_base, mask=mask_w)
    acc += matrix_x * w_col3

    acc = acc / (1 + tl.exp(-acc))  # SiLU
    o_ptrs = o_ptr + idx_seq * stride_o_seq + idx_feats * stride_o_dim
    tl.store(o_ptrs, acc, mask=mask_w)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    conv_state_indices: torch.Tensor,
) -> torch.Tensor:
    """x [B, dim] -> silu(conv(window ++ x)) [B, dim]; conv_state [slots, dim, 3] in place."""
    x = x.unsqueeze(-1)
    batch, dim, _ = x.shape
    width = weight.shape[1]
    assert width == CONV_KERNEL
    num_cache_lines, _, _ = conv_state.size()
    state_len = width - 1
    # Rows skipped as padding keep this uninitialized, same as SGLang.
    out = torch.empty_like(x)
    block_n = 256
    grid = (batch, triton.cdiv(dim, block_n))
    _causal_conv1d_update_kernel[grid](
        x,
        weight,
        conv_state,
        conv_state_indices,
        out,
        batch,
        dim,
        state_len,
        num_cache_lines,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_state_indices.stride(0),
        out.stride(0),
        out.stride(1),
        PAD_SLOT_ID,
        NP2_STATELEN=triton.next_power_of_2(state_len),
        BLOCK_N=block_n,
    )
    return out.squeeze(-1)


# ===== Triton kernel 2: packed recurrent gated-delta-rule decode =====
# Verbatim from fused_recurrent_gated_delta_rule_packed_decode_kernel.


@triton.jit
def _packed_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v

    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    b_h *= tl.exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_ht = ht + state_idx * stride_final_state_token
    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
) -> torch.Tensor:
    """mixed_qkv [B, q|k|v] post-conv -> o [B, HV, V]; ssm_state [slots, HV, V, K] in place."""
    B = mixed_qkv.shape[0]
    HV, V, K = ssm_state.shape[-3:]
    H = (mixed_qkv.shape[1] - HV * V) // 2 // K
    assert mixed_qkv.stride(-1) == 1 and a.stride(-1) == 1 and b.stride(-1) == 1
    assert ssm_state.stride(-1) == 1
    out = mixed_qkv.new_empty(B, 1, HV, V)
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    grid = (triton.cdiv(V, BV), B * HV)
    _packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=ssm_state,
        ht=ssm_state,
        ssm_state_indices=ssm_state_indices,
        scale=K**-0.5,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_init_state_token=ssm_state.stride(0),
        stride_final_state_token=ssm_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=SOFTPLUS_THRESHOLD,
        USE_QK_L2NORM_IN_KERNEL=True,
        num_warps=1,
        num_stages=3,
    )
    return out.view(B, HV, V)


# ===== Triton kernel 3: gated RMSNorm, norm(o) * w * sigmoid(z) =====
# Specialized from _layer_norm_fwd_1pass_kernel: IS_RMS_NORM, NORM_BEFORE_GATE,
# ACTIVATION="sigmoid", no bias, 2-D z, no FP8 quant (BF16 out_proj).

_MAX_ROWS_PER_BLOCK = 4


@triton.jit
def _gated_rmsnorm_kernel(
    X,
    Y,
    W,
    Z,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)
    col_offsets = cols[None, :]
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    x = tl.load(X + rows[:, None] * stride_x_row + col_offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = tl.rsqrt(var + eps)

    w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
    y = x * rstd[:, None] * w[None, :]

    z = tl.load(Z + rows[:, None] * stride_z_row + col_offsets, mask=mask, other=0.0)
    y *= tl.sigmoid(z.to(tl.float32))
    tl.store(Y + rows[:, None] * stride_y_row + col_offsets, y, mask=mask)


_CU_COUNT: dict = {}


def gated_rmsnorm(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x, z [M, 128] -> rmsnorm(x) * weight * sigmoid(z), bf16."""
    M, N = x.shape
    assert x.stride(-1) == 1 and z.shape == (M, N) and z.stride(-1) == 1
    out = torch.empty_like(x)
    block_n = triton.next_power_of_2(N)
    num_warps = min(max(block_n // 256, 1), 8)
    # Same rows-per-block rule as calc_rows_per_block; it changes which rows share a
    # program, not the per-row math.
    dev = x.device.index or 0
    if dev not in _CU_COUNT:
        _CU_COUNT[dev] = torch.cuda.get_device_properties(x.device).multi_processor_count
    rows_per_block = min(
        triton.next_power_of_2(triton.cdiv(M, 2 * _CU_COUNT[dev])), _MAX_ROWS_PER_BLOCK
    )
    grid = (triton.cdiv(M, rows_per_block),)
    _gated_rmsnorm_kernel[grid](
        x,
        out,
        weight,
        z,
        x.stride(0),
        out.stride(0),
        z.stride(0),
        M,
        N,
        RMS_EPS,
        BLOCK_N=block_n,
        ROWS_PER_BLOCK=rows_per_block,
        num_warps=num_warps,
    )
    return out


# ===== BF16 GEMMs: aiter routing from UnquantizedLinearMethod.apply =====

# Triton a16w16 covers decode shapes absent from the tuned table (unquant.py).
_A16W16_TRITON_MAX_K = 2048
_A16W16_TRITON_MAX_M = 64
_A16W16_TRITON_NARROW_K = 512
_A16W16_TRITON_NARROW_K_MAX_M = 512

_aiter_gemm = None  # (tgemm, gemm_a16w16) or False once the import has failed


def _load_aiter_gemm():
    global _aiter_gemm
    if _aiter_gemm is None:
        if os.environ.get("GDN_DISABLE_AITER", "0") == "1":
            _aiter_gemm = False
        else:
            try:
                from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
                from aiter.tuned_gemm import tgemm

                _aiter_gemm = (tgemm, gemm_a16w16)
            except Exception:
                _aiter_gemm = False
    return _aiter_gemm


def _is_gfx95(device: torch.device) -> bool:
    return "gfx95" in torch.cuda.get_device_properties(device).gcnArchName


def _prefer_triton_a16w16(x: torch.Tensor, weight: torch.Tensor) -> bool:
    k = weight.shape[1]
    max_m = (
        _A16W16_TRITON_NARROW_K_MAX_M
        if k <= _A16W16_TRITON_NARROW_K
        else _A16W16_TRITON_MAX_M
    )
    return (
        x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and 0 < k <= _A16W16_TRITON_MAX_K
        and 0 < x.shape[0] <= max_m
        and x.is_contiguous()
        and weight.is_contiguous()
    )


def linear_bf16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x @ weight^T. Falls back to F.linear (hipBLASLt) when aiter is unavailable."""
    gemm = _load_aiter_gemm()
    if not gemm:
        return F.linear(x, weight)
    tgemm, gemm_a16w16 = gemm
    if _is_gfx95(x.device) and _prefer_triton_a16w16(x, weight):
        return gemm_a16w16(x, weight, None, dtype=x.dtype)
    return tgemm.mm(x, weight, None, otype=x.dtype)


# ===== The decode step =====


def gdn_decode(
    hidden_states: torch.Tensor,
    weights: GDNWeights,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    cache_indices: torch.Tensor,
    shape: GDNShape,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """One GDN layer decode step on one TP rank.

    hidden_states: [B, 2560] bf16, the gated-residual mix output for this layer.
    conv_state: [slots, conv_dim, 3] bf16; ssm_state: [slots, HV, V, K] fp32 (both in place).
    cache_indices: [B] int32 mamba slot per request, PAD_SLOT_ID for padded rows.
    Returns [B, 2560] bf16; all-reduced over tp_group when given, else the rank's partial.
    """
    if hidden_states.shape[0] == 0:
        return hidden_states.new_zeros(0, HIDDEN_SIZE)

    qkvz = linear_bf16(hidden_states, weights.in_proj_qkvz)
    ba = linear_bf16(hidden_states, weights.in_proj_ba)

    query, key, value, z = qkvz.split(
        [shape.key_dim, shape.key_dim, shape.value_dim, shape.value_dim], dim=-1
    )
    b, a = ba.split([shape.num_v_heads, shape.num_v_heads], dim=-1)
    b = b.contiguous()
    a = a.contiguous()
    mixed_qkv = torch.cat((query, key, value), dim=-1)

    mixed_qkv = causal_conv1d_update(mixed_qkv, conv_state, weights.conv_weight, cache_indices)

    core_attn_out = packed_decode(
        mixed_qkv, a, b, weights.A_log, weights.dt_bias, ssm_state, cache_indices
    )

    core_attn_out = core_attn_out.reshape(-1, HEAD_V_DIM)
    z = z.reshape(-1, HEAD_V_DIM)
    normed = gated_rmsnorm(core_attn_out, z, weights.norm_weight)
    normed = normed.reshape(-1, shape.value_dim)

    out = linear_bf16(normed, weights.out_proj)
    if tp_group is not None:
        torch.distributed.all_reduce(out, group=tp_group)
    return out


# ===== Weights: HF checkpoint -> per-rank layout =====


def shard_checkpoint_weights(
    ckpt: dict, tp_rank: int, shape: GDNShape, device: torch.device
) -> GDNWeights:
    """ckpt keys: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, conv1d, A_log, dt_bias,
    norm, out_proj (the `...linear_attn.<name>.weight` tensors of one layer)."""
    key_full = NUM_K_HEADS * HEAD_K_DIM
    value_full = NUM_V_HEADS * HEAD_V_DIM

    def rows(t: torch.Tensor, per_rank: int) -> torch.Tensor:
        return t[tp_rank * per_rank : (tp_rank + 1) * per_rank]

    q, k, v = ckpt["in_proj_qkv"].split([key_full, key_full, value_full], dim=0)
    qkvz = torch.cat(
        [
            rows(q, shape.key_dim),
            rows(k, shape.key_dim),
            rows(v, shape.value_dim),
            rows(ckpt["in_proj_z"], shape.value_dim),
        ]
    )
    ba = torch.cat(
        [rows(ckpt["in_proj_b"], shape.num_v_heads), rows(ckpt["in_proj_a"], shape.num_v_heads)]
    )
    cq, ck, cv = ckpt["conv1d"].reshape(-1, CONV_KERNEL).split(
        [key_full, key_full, value_full], dim=0
    )
    conv = torch.cat(
        [rows(cq, shape.key_dim), rows(ck, shape.key_dim), rows(cv, shape.value_dim)]
    )
    out_proj = ckpt["out_proj"][
        :, tp_rank * shape.value_dim : (tp_rank + 1) * shape.value_dim
    ]

    def bf16(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=device, dtype=torch.bfloat16).contiguous()

    return GDNWeights(
        in_proj_qkvz=bf16(qkvz),
        in_proj_ba=bf16(ba),
        conv_weight=bf16(conv),
        A_log=rows(ckpt["A_log"], shape.num_v_heads).to(device, torch.float32).contiguous(),
        dt_bias=bf16(rows(ckpt["dt_bias"], shape.num_v_heads)),
        norm_weight=bf16(ckpt["norm"]),
        out_proj=bf16(out_proj),
    )


def allocate_state(
    num_slots: int, shape: GDNShape, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zeroed per-layer (conv_state, ssm_state) pools, matching MambaPool's default layout."""
    conv = torch.zeros(
        num_slots, shape.conv_dim, CONV_KERNEL - 1, dtype=torch.bfloat16, device=device
    )
    ssm = torch.zeros(
        num_slots, shape.num_v_heads, HEAD_V_DIM, HEAD_K_DIM, dtype=torch.float32, device=device
    )
    return conv, ssm


# ===== fp32 torch reference (math only, not bit-exact) =====


def gdn_decode_reference(
    hidden_states: torch.Tensor,
    weights: GDNWeights,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    cache_indices: torch.Tensor,
    shape: GDNShape,
) -> torch.Tensor:
    x = hidden_states.float()
    qkvz = x @ weights.in_proj_qkvz.float().T
    ba = x @ weights.in_proj_ba.float().T
    q, k, v, z = qkvz.split(
        [shape.key_dim, shape.key_dim, shape.value_dim, shape.value_dim], dim=-1
    )
    b, a = ba.split([shape.num_v_heads, shape.num_v_heads], dim=-1)
    mixed = torch.cat([q, k, v], dim=-1)

    out = torch.zeros(x.shape[0], HIDDEN_SIZE, device=x.device)
    group = shape.num_v_heads // shape.num_k_heads
    for i, slot in enumerate(cache_indices.tolist()):
        if slot < 0:
            continue
        window = torch.cat([conv_state[slot].float(), mixed[i, :, None]], dim=-1)
        conv_state[slot] = window[:, 1:].to(conv_state.dtype)
        y = F.silu((window * weights.conv_weight.float()).sum(-1))
        yq, yk, yv = y.split([shape.key_dim, shape.key_dim, shape.value_dim])
        yq = F.normalize(yq.view(-1, HEAD_K_DIM), dim=-1, eps=1e-6) * HEAD_K_DIM**-0.5
        yk = F.normalize(yk.view(-1, HEAD_K_DIM), dim=-1, eps=1e-6)
        yv = yv.view(-1, HEAD_V_DIM)
        g = -weights.A_log.float().exp() * F.softplus(
            a[i] + weights.dt_bias.float(), threshold=SOFTPLUS_THRESHOLD
        )
        beta = torch.sigmoid(b[i])
        o = torch.empty(shape.num_v_heads, HEAD_V_DIM, device=x.device)
        for hv in range(shape.num_v_heads):
            hk = hv // group
            S = ssm_state[slot, hv] * g[hv].exp()
            S = S + torch.outer((yv[hv] - S @ yk[hk]) * beta[hv], yk[hk])
            ssm_state[slot, hv] = S
            o[hv] = S @ yq[hk]
        zz = z[i].view(-1, HEAD_V_DIM)
        o = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + RMS_EPS)
        o = o * weights.norm_weight.float() * torch.sigmoid(zz)
        out[i] = o.reshape(-1) @ weights.out_proj.float().T
    return out


# ===== Self-test =====


def _random_weights(shape: GDNShape, device: torch.device, seed: int = 0) -> GDNWeights:
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*size, std=1.0):
        return (torch.randn(*size, generator=gen) * std).to(device, torch.bfloat16)

    return GDNWeights(
        in_proj_qkvz=rnd(2 * shape.key_dim + 2 * shape.value_dim, HIDDEN_SIZE, std=0.02),
        in_proj_ba=rnd(2 * shape.num_v_heads, HIDDEN_SIZE, std=0.02),
        conv_weight=rnd(shape.conv_dim, CONV_KERNEL, std=0.5),
        A_log=torch.empty(shape.num_v_heads).uniform_(-2, 1, generator=gen).to(device),
        dt_bias=rnd(shape.num_v_heads),
        norm_weight=(1 + torch.randn(HEAD_V_DIM, generator=gen) * 0.1).to(
            device, torch.bfloat16
        ),
        out_proj=rnd(HIDDEN_SIZE, shape.value_dim, std=0.02),
    )


def _load_real_weights(layer: int, tp_rank: int, shape: GDNShape, device) -> GDNWeights:
    import glob
    import json

    from safetensors import safe_open

    root = os.environ.get(
        "HF_HOME", os.path.join(os.path.dirname(__file__), "..", "..", "hf_cache")
    )
    snap = glob.glob(f"{root}/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/*")[0]
    weight_map = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
    prefix = f"model.language_model.layers.{layer}.linear_attn."
    names = {
        "in_proj_qkv": "in_proj_qkv.weight",
        "in_proj_z": "in_proj_z.weight",
        "in_proj_b": "in_proj_b.weight",
        "in_proj_a": "in_proj_a.weight",
        "conv1d": "conv1d.weight",
        "A_log": "A_log",
        "dt_bias": "dt_bias",
        "norm": "norm.weight",
        "out_proj": "out_proj.weight",
    }
    ckpt = {}
    for key, suffix in names.items():
        full = prefix + suffix
        with safe_open(f"{snap}/{weight_map[full]}", framework="pt") as f:
            ckpt[key] = f.get_tensor(full)
    return shard_checkpoint_weights(ckpt, tp_rank, shape, device)


def _sglang_decode(
    hidden_states, weights, conv_state, ssm_state, cache_indices, shape
) -> torch.Tensor:
    """The same step through SGLang's own kernels, for the bitwise parity check."""
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
    from sglang.kernels.ops.mamba.causal_conv1d_triton import (
        causal_conv1d_update as sgl_conv_update,
    )

    qkvz = linear_bf16(hidden_states, weights.in_proj_qkvz)
    ba = linear_bf16(hidden_states, weights.in_proj_ba)
    q, k, v, z = qkvz.split(
        [shape.key_dim, shape.key_dim, shape.value_dim, shape.value_dim], dim=-1
    )
    b, a = (t.contiguous() for t in ba.split([shape.num_v_heads] * 2, dim=-1))
    mixed = sgl_conv_update(
        torch.cat((q, k, v), dim=-1),
        conv_state,
        weights.conv_weight,
        None,
        "silu",
        conv_state_indices=cache_indices,
    )
    o = mixed.new_empty(mixed.shape[0], 1, shape.num_v_heads, HEAD_V_DIM)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed,
        a=a,
        b=b,
        A_log=weights.A_log,
        dt_bias=weights.dt_bias,
        scale=HEAD_K_DIM**-0.5,
        initial_state=ssm_state,
        out=o,
        ssm_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
    )
    norm = RMSNorm(HEAD_V_DIM, eps=RMS_EPS, activation="sigmoid").to(o.device)
    norm.weight.data = weights.norm_weight
    normed = norm(o.reshape(-1, HEAD_V_DIM), z.reshape(-1, HEAD_V_DIM))
    return linear_bf16(normed.reshape(-1, shape.value_dim), weights.out_proj)


def _run_self_test(args) -> None:
    device = torch.device("cuda", 0)
    shape = GDNShape.for_tp(args.tp)
    if args.real_weights:
        weights = _load_real_weights(args.layer, args.tp_rank, shape, device)
    else:
        weights = _random_weights(shape, device)
    gemm = _load_aiter_gemm()
    print(
        f"device={torch.cuda.get_device_properties(device).gcnArchName} "
        f"tp={args.tp} rank={args.tp_rank} gemm={'aiter' if gemm else 'torch'} "
        f"weights={'layer %d' % args.layer if args.real_weights else 'random'}"
    )

    num_slots = 2 * args.batch + 1
    conv, ssm = allocate_state(num_slots, shape, device)
    gen = torch.Generator(device="cpu").manual_seed(1)
    slots = (torch.randperm(num_slots - 1, generator=gen)[: args.batch] + 1).to(torch.int32)
    slots[-1] = PAD_SLOT_ID  # one CUDA-graph padding row
    slots = slots.to(device)

    conv_ref, ssm_ref = conv.clone(), ssm.clone()
    conv_sgl, ssm_sgl = conv.clone(), ssm.clone()
    valid = slots >= 0
    worst = 0.0
    for step in range(args.steps):
        x = (torch.randn(args.batch, HIDDEN_SIZE, generator=gen) * 0.5).to(
            device, torch.bfloat16
        )
        out = gdn_decode(x, weights, conv, ssm, slots, shape)
        if args.check_sglang:
            out_sgl = _sglang_decode(x, weights, conv_sgl, ssm_sgl, slots, shape)
            assert torch.equal(out[valid], out_sgl[valid]), f"out differs at step {step}"
            assert torch.equal(conv, conv_sgl), f"conv state differs at step {step}"
            assert torch.equal(ssm, ssm_sgl), f"ssm state differs at step {step}"
        expected = gdn_decode_reference(x, weights, conv_ref, ssm_ref, slots, shape)
        err = (out[valid].float() - expected[valid]).abs().max().item()
        scale = expected[valid].abs().max().item()
        worst = max(worst, err / max(scale, 1e-6))
    ssm_err = (ssm - ssm_ref).abs().max().item()
    if args.check_sglang:
        print("bitwise equal to SGLang kernels: out, conv_state, ssm_state")
    print(
        f"{args.steps} decode steps, batch {args.batch}: "
        f"max |out - ref| / max|ref| = {worst:.3e}, max |ssm - ref| = {ssm_err:.3e}"
    )
    assert worst < 2e-2, "output deviates from the fp32 reference"
    assert torch.equal(conv[0], conv_ref[0]), "slot 0 must stay untouched"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=TP_SIZE)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--real-weights", action="store_true")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--check-sglang", action="store_true")
    _run_self_test(parser.parse_args())
