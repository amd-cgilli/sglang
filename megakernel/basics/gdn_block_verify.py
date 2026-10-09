"""gdn_block for MTP target verify: a GDN layer's attention half for T draft tokens in one launch.

    R [T, 4 * 2560]  ->  R_out = R + c * out_proj(gdn(in_proj(hc_mix(R))))       per token row

gdn_block.py's schedule with every GEMV taking T tokens (each weight is read once for all T) and
gdn_verify_op in place of gdn_core: the slot's conv and SSM state are read, not written, and the
state after each token goes to SGLang's verify scratch (conv windows, SSM snapshots) at row
win_idx[0]. Head blocks recompute the T steps to store the snapshots after publishing o.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T

import buffer_ops
from gdn_block import BA, BA_LIGHT, CONV_SLOT_BYTES, DEPTH, HEADS, HIDDEN_V, QKVZ, QKVZ_LIGHT, SSM_SLOT_BYTES
from gdn_core_op import CONV_CH, D, IN_DIM, STAGE_STRIDE, V_HEADS, gdn_core_prefetch
from gdn_verify_op import MARKS as VERIFY_MARKS, emit_gdn_verify, gdn_verify_snapshots
from gemv_op import MARKS as GEMV_MARKS, NBLK, RED_WORDS, emit_gemv, gemv_prefetch
from hc_op import (HC, LOW, H, emit_hc_norm, emit_hc_up, emit_inject, finish_epoch, hc_norm_load, hc_up_prefetch,
                   inject_prefetch, next_epoch_tag, publish_combine, publish_low)
from timeline import mark

T_VERIFY = 4  # --speculative-num-draft-tokens 4
N_WORDS = T_VERIFY * HC // 2  # n of T tokens, bf16 pairs: 80 KB
# 24 of out_proj's 42 loads a wave: all 42 plus 4 tokens of o fragments exceed 256 VGPRs.
OUT_DEPTH = 24
STAGE_FLOATS = D * STAGE_STRIDE
assert STAGE_FLOATS <= N_WORDS

M_NORM, M_DOWN, M_UP, M_IN, M_BA, M_CORE, M_STATE, M_OUT = 0, 2, 7, 9, 14, 19, 23, 24
SPANS = ([("hc_norm", 0, 1)]
         + [(f"hc_down {n}", M_DOWN + i, M_DOWN + i + 1) for i, n in
            enumerate(["issue", "poll", "dot + reduce", "publish low"])]
         + [("hc_up", M_DOWN + 4, M_UP + 1)]
         + [(f"in_proj {n}", M_IN + i, M_IN + i + 1) for i, n in
            enumerate(["issue weights", "poll x", "dot + reduce", "publish"])]
         + [(f"gdn_verify {n}", M_CORE + i, M_CORE + i + 1) for i, n in
            enumerate(["poll + conv + norms", "recurrence x T", "norm + gate + publish o"])]
         + [("gdn_verify snapshots", M_CORE + VERIFY_MARKS - 1, M_STATE)]
         + [(f"out_proj {n}", M_OUT + i, M_OUT + i + 1) for i, n in
            enumerate(["issue weights", "poll o", "dot + reduce", "combine into R"])])
LAST_MARK = M_OUT + GEMV_MARKS - 1


@fx.struct
class VerifyBlockSmem:
    partial: fx.Array[fx.Float32, RED_WORDS, 16]  # emit_gemv
    qkv: fx.Array[fx.Float32, T_VERIFY * 3 * D, 16]  # gdn_verify_op (as VerifySmem)
    z: fx.Array[fx.Float32, T_VERIFY * D, 16]
    out: fx.Array[fx.Float32, T_VERIFY * D, 16]
    scal: fx.Array[fx.Float32, 64, 16]
    red: fx.Array[fx.Float32, 64, 16]  # hc_op
    x: fx.Array[fx.Float32, 64, 16]
    c: fx.Array[fx.Float32, 16, 16]
    # n ([T, HC / 2] words) until hc_up and inject; then, on head blocks, the snapshot stage.
    big: fx.Array[fx.Int32, N_WORDS, 16]


def mailboxes(device="cuda", tokens=T_VERIFY):
    import torch

    i32 = lambda n: torch.zeros(n, dtype=torch.int32, device=device)
    return {"x_mb": i32(tokens * H), "low_mb": i32(tokens * LOW), "in_mb": i32(tokens * IN_DIM),
            "o_mb": i32(tokens * HIDDEN_V), "sync": i32(2)}


@functools.cache
def build(traced, tokens=T_VERIFY, win_dedup=False):
    """win_dedup: the window scratch is SGLang's deduplicated layout (see gdn_verify_op)."""
    win_row_bytes = CONV_CH * ((tokens + 2) if win_dedup else 3 * tokens) * 2

    @flyc.kernel(known_block_size=[256, 1, 1])
    def gdn_block_verify(r: fx.Tensor, r_out: fx.Tensor, hc_norm_w: fx.Tensor, w_down: fx.Tensor, w_up: fx.Tensor,
                         w_inj: fx.Tensor, qkvz_w: fx.Tensor, ba_w: fx.Tensor, out_w: fx.Tensor, conv_w: fx.Tensor,
                         a_log: fx.Tensor, dt_bias: fx.Tensor, norm_w: fx.Tensor, conv_states: fx.Tensor,
                         ssm_states: fx.Tensor, cache_idx: fx.Tensor, win_scratch: fx.Tensor, ssm_scratch: fx.Tensor,
                         scratch_idx: fx.Tensor, x_mb: fx.Tensor, low_mb: fx.Tensor, in_mb: fx.Tensor,
                         o_mb: fx.Tensor, sync: fx.Tensor, trace: fx.Tensor):
        rs = lambda t, nbytes, base=None: buffer_ops.create_buffer_resource(
            t, max_size=False, num_records_bytes=nbytes, base_byte_offset=base)
        smem = fx.SharedAllocator().allocate(VerifyBlockSmem).peek()
        n_ptr, stage = smem.big.ptr, fx.recast_iter(fx.Float32, smem.big.ptr)
        bid = fx.Int32(fx.block_idx.x)
        tag = next_epoch_tag(rs(sync, 8))
        slot = fx.Int32(buffer_ops.buffer_load(rs(cache_idx, 4), 0, vec_width=1, dtype=T.i32))
        row = fx.Int32(buffer_ops.buffer_load(rs(scratch_idx, 4), 0, vec_width=1, dtype=T.i32))
        conv_rs = rs(conv_states, CONV_SLOT_BYTES, slot * CONV_SLOT_BYTES)
        ssm_rs = rs(ssm_states, SSM_SLOT_BYTES, slot * SSM_SLOT_BYTES)
        win_rs = rs(win_scratch, win_row_bytes, row * win_row_bytes)
        snap_rs = rs(ssm_scratch, tokens * SSM_SLOT_BYTES, row * (tokens * SSM_SLOT_BYTES))
        r_rs, x_rs, low_rs = rs(r, tokens * HC * 2), rs(x_mb, tokens * H * 4), rs(low_mb, tokens * LOW * 4)
        in_rs, o_rs = rs(in_mb, tokens * IN_DIM * 4), rs(o_mb, tokens * HIDDEN_V * 4)
        w_down_rs, w_up_rs, w_inj_rs = rs(w_down, LOW * HC * 2), rs(w_up, HC * LOW * 2), rs(w_inj, 4 * HC * 2)
        out_w_rs = rs(out_w, H * HIDDEN_V * 2)
        mark(trace, 4, M_NORM, traced)

        hn = hc_norm_load(r_rs, rs(hc_norm_w, HC * 2), tokens)
        down_w = gemv_prefetch(w_down_rs, LOW, HC, 4, 4, DEPTH, light=HEADS)
        emit_hc_norm(hn, smem, n_ptr)
        up_w = hc_up_prefetch(w_up_rs)
        mark(trace, 4, M_NORM + 1, traced)

        emit_gemv(low_rs, w_down_rs, low_rs, smem.partial.ptr, trace, tag, tag, LOW, HC, 4, 4, DEPTH, True, M_DOWN,
                  traced, x_lds=n_ptr, inflight=down_w, light=HEADS, x_lds_ready=True,
                  publish=functools.partial(publish_low, low_rs, tag), tokens=tokens)
        emit_hc_up(up_w, low_rs, tag, x_rs, tag, smem, tokens, n_ptr)
        mark(trace, 4, M_UP + 1, traced)
        core = gdn_core_prefetch(bid, rs(conv_w, CONV_CH * 4 * 2), conv_rs, ssm_rs, rs(norm_w, D * 2))

        emit_gemv(x_rs, rs(qkvz_w, QKVZ * H * 2), in_rs, smem.partial.ptr, trace, tag, tag, QKVZ, H, 4, 1, DEPTH,
                  True, M_IN, traced, light=QKVZ_LIGHT, tokens=tokens, y_row_pairs=IN_DIM // 2)
        if bid < V_HEADS:
            ba_rs = rs(in_mb, ((tokens - 1) * IN_DIM + BA) * 4, QKVZ * 4)
            emit_gemv(x_rs, rs(ba_w, BA * H * 2), ba_rs, smem.partial.ptr, trace, tag, tag, BA, H, 4, 1, DEPTH, True,
                      M_BA, traced, light=BA_LIGHT, tokens=tokens, y_row_pairs=IN_DIM // 2)

        out_pre = gemv_prefetch(out_w_rs, H, HIDDEN_V, 4, 4, OUT_DEPTH, light=HEADS)
        inj = inject_prefetch(w_inj_rs, bid >= V_HEADS)
        if bid < V_HEADS:
            emit_gdn_verify(bid, core, in_rs, tag, o_rs, win_rs, rs(a_log, V_HEADS * 4), rs(dt_bias, V_HEADS * 4),
                            smem, tokens, trace, M_CORE, traced, win_dedup)
            gdn_verify_snapshots(bid, core, snap_rs, smem, stage, tokens, trace, M_STATE, traced)
        else:
            emit_inject(inj, smem, tokens, n_ptr)
            emit_gemv(o_rs, out_w_rs, o_rs, smem.partial.ptr, trace, tag, tag, H, HIDDEN_V, 4, 4, OUT_DEPTH, True,
                      M_OUT, traced, inflight=out_pre, light=HEADS, tokens=tokens,
                      publish=functools.partial(publish_combine, r_rs, rs(r_out, tokens * HC * 2), smem.c.ptr))
        finish_epoch(sync, tag)

    @flyc.jit
    def launch(r: fx.Tensor, r_out: fx.Tensor, hc_norm_w: fx.Tensor, w_down: fx.Tensor, w_up: fx.Tensor,
               w_inj: fx.Tensor, qkvz_w: fx.Tensor, ba_w: fx.Tensor, out_w: fx.Tensor, conv_w: fx.Tensor,
               a_log: fx.Tensor, dt_bias: fx.Tensor, norm_w: fx.Tensor, conv_states: fx.Tensor,
               ssm_states: fx.Tensor, cache_idx: fx.Tensor, win_scratch: fx.Tensor, ssm_scratch: fx.Tensor,
               scratch_idx: fx.Tensor, x_mb: fx.Tensor, low_mb: fx.Tensor, in_mb: fx.Tensor, o_mb: fx.Tensor,
               sync: fx.Tensor, trace: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        gdn_block_verify(r, r_out, hc_norm_w, w_down, w_up, w_inj, qkvz_w, ba_w, out_w, conv_w, a_log, dt_bias,
                         norm_w, conv_states, ssm_states, cache_idx, win_scratch, ssm_scratch, scratch_idx, x_mb,
                         low_mb, in_mb, o_mb, sync, trace).launch(grid=(NBLK, 1, 1), block=(256, 1, 1), stream=stream)

    return launch
