"""The attention half of a Qwen3.8 GDN layer in one launch: gated read -> GDN mixer -> gated write.

    R [1, 4 * 2560] bf16  ->  R_out = R + c * out_proj(gdn(in_proj(hc_mix(R))))

Every block runs (block 0..47 also own one GDN value head):

    loads       R, hc_norm.w and W_down, then W_up after hc_norm, gdn state (heads) after hc_up
    hc_norm     n = gemma_rmsnorm_per_stream(R), into LDS               all blocks, redundantly
    hc_down     n -> low [320] mailbox, bf16(silu(. / 4))               blocks 48..207
    hc_up       low -> x [2560] mailbox                                 all blocks, 10 outputs each
    in_proj     x -> in [16480] = qkvz | ba mailbox                      qkvz: all (fewer rows on
                                                                        0..47); ba: blocks 0..47
    blocks 0..47:    gdn_core in -> o [6144] mailbox, then the conv + SSM state
    blocks 48..255:  inject c [4]; out_proj o -> R_out, combined in the epilogue

Mailboxes take this launch's tag: sync[0] + 1, bumped by the last block to finish (hc_op), so a
graph can replay the launch. State is SGLang's layer view: conv [slots, 10240, 3] bf16, SSM
[slots, 48, 128 (V), 128 (K)] fp32; the slot is cache_idx[0], read on device.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T

import buffer_ops
from gdn_core_op import (CONV_CH, D, IN_DIM, STAGE_STRIDE, V_HEADS, ack_pairs, emit_gdn_core, gdn_core_prefetch,
                         gdn_core_write_state)
from gemv_op import NBLK, RED_WORDS, emit_gemv, gemv_prefetch
from hc_op import (HC, LOW, H, emit_hc_norm, emit_hc_up_mfma, emit_inject, finish_epoch, hc_norm_load, hc_up_prefetch_mfma,
                   inject_prefetch, next_epoch_tag, publish_combine, publish_low)
from timeline import mark

BA, HIDDEN_V = 2 * V_HEADS, V_HEADS * D  # b | a, o
QKVZ = IN_DIM - BA  # q | k | v | z = 16384
CONV_SLOT_BYTES = CONV_CH * 3 * 2
SSM_SLOT_BYTES = V_HEADS * D * D * 4
# Head blocks (0..47) also run gdn_core: 24 qkvz pairs keeps the others at 34 (17 rows a wave),
# as with an even split; no hc_down and no out_proj rows (step 4b).
HEADS = (V_HEADS, 0)
QKVZ_LIGHT, BA_LIGHT = (V_HEADS, 24), (V_HEADS, 1)
DEPTH, OUT_DEPTH = 16, 64

# Trace marks
M_NORM, M_DOWN, M_UP, M_IN, M_BA, M_CORE, M_STATE, M_OUT = 0, 2, 7, 9, 14, 19, 23, 24  # M_UP: end only
SPANS = ([("hc_norm", 0, 1)]
         + [(f"hc_down {n}", M_DOWN + i, M_DOWN + i + 1) for i, n in
            enumerate(["issue", "poll", "dot + reduce", "publish low"])]
         + [("hc_up", M_DOWN + 4, M_UP + 1)]
         + [(f"in_proj {n}", M_IN + i, M_IN + i + 1) for i, n in
            enumerate(["issue weights", "poll x", "dot + reduce", "publish"])]
         + [(f"gdn_core {n}", M_CORE + i, M_CORE + i + 1) for i, n in
            enumerate(["poll + conv + norms", "recurrence", "norm + gate + publish o"])]
         + [("gdn_core write state", M_CORE + 3, M_STATE)]
         + [(f"out_proj {n}", M_OUT + i, M_OUT + i + 1) for i, n in
            enumerate(["issue weights", "poll o", "dot + reduce", "combine into R"])])
LAST_MARK = M_OUT + 4


@fx.struct
class BlockSmem:
    partial: fx.Array[fx.Float32, RED_WORDS, 16]  # emit_gemv
    xin: fx.Array[fx.Float32, 3 * D, 16]  # emit_gdn_core (as GdnSmem)
    qkv: fx.Array[fx.Float32, 3 * D, 16]
    z: fx.Array[fx.Float32, D, 16]
    out: fx.Array[fx.Float32, D, 16]
    scal: fx.Array[fx.Float32, 8, 16]
    stage: fx.Array[fx.Float32, D * STAGE_STRIDE, 16]
    n: fx.Array[fx.Int32, HC // 2, 16]  # hc_op (as HcSmem)
    red: fx.Array[fx.Float32, 16, 16]
    x: fx.Array[fx.Float32, 16, 16]
    c: fx.Array[fx.Float32, 4, 16]


def mailboxes(device="cuda"):
    """Scratch shared by every launch (and layer): the tags keep launches apart."""
    import torch

    i32 = lambda n: torch.zeros(n, dtype=torch.int32, device=device)
    return {"x_mb": i32(H), "low_mb": i32(LOW), "in_mb": i32(IN_DIM), "o_mb": i32(HIDDEN_V),
            "ack": i32(ack_pairs() * 2), "sync": i32(2)}


@functools.cache
def build(traced):
    @flyc.kernel(known_block_size=[256, 1, 1])
    def gdn_block(r: fx.Tensor, r_out: fx.Tensor, hc_norm_w: fx.Tensor, w_down: fx.Tensor, w_up: fx.Tensor,
                  w_inj: fx.Tensor, qkvz_w: fx.Tensor, ba_w: fx.Tensor, out_w: fx.Tensor, conv_w: fx.Tensor,
                  a_log: fx.Tensor, dt_bias: fx.Tensor, norm_w: fx.Tensor, conv_states: fx.Tensor,
                  ssm_states: fx.Tensor, cache_idx: fx.Tensor, x_mb: fx.Tensor, low_mb: fx.Tensor,
                  in_mb: fx.Tensor, o_mb: fx.Tensor, ack: fx.Tensor, sync: fx.Tensor, trace: fx.Tensor):
        rs = lambda t, nbytes, base=None: buffer_ops.create_buffer_resource(
            t, max_size=False, num_records_bytes=nbytes, base_byte_offset=base)
        smem = fx.SharedAllocator().allocate(BlockSmem).peek()
        up_scratch = fx.recast_iter(fx.Int32, smem.partial.ptr)  # free between emit_gemvs
        bid = fx.Int32(fx.block_idx.x)
        tag = next_epoch_tag(rs(sync, 8))
        slot = fx.Int32(buffer_ops.buffer_load(rs(cache_idx, 4), 0, vec_width=1, dtype=T.i32))
        conv_rs = rs(conv_states, CONV_SLOT_BYTES, slot * CONV_SLOT_BYTES)
        ssm_rs = rs(ssm_states, SSM_SLOT_BYTES, slot * SSM_SLOT_BYTES)
        r_rs, x_rs, low_rs = rs(r, HC * 2), rs(x_mb, H * 4), rs(low_mb, LOW * 4)
        in_rs, o_rs, ack_rs = rs(in_mb, IN_DIM * 4), rs(o_mb, HIDDEN_V * 4), rs(ack, ack_pairs() * 8)
        w_down_rs, w_up_rs, w_inj_rs = rs(w_down, LOW * HC * 2), rs(w_up, HC * LOW * 2), rs(w_inj, 4 * HC * 2)
        out_w_rs = rs(out_w, H * HIDDEN_V * 2)
        mark(trace, 4, M_NORM, traced)

        # Loads return in order and stall issue once the queue is full, so each op's loads go out
        # just ahead of it: hc_norm only waits for R and W_down's 10 per lane.
        hn = hc_norm_load(r_rs, rs(hc_norm_w, HC * 2))
        down_w = gemv_prefetch(w_down_rs, LOW, HC, 4, 4, DEPTH, light=HEADS)
        emit_hc_norm(hn, smem)
        up_w = hc_up_prefetch_mfma(w_up_rs)
        mark(trace, 4, M_NORM + 1, traced)

        emit_gemv(low_rs, w_down_rs, low_rs, smem.partial.ptr, trace, tag, tag, LOW, HC, 4, 4, DEPTH, True, M_DOWN,
                  traced, x_lds=smem.n.ptr, inflight=down_w, light=HEADS, x_lds_ready=True,
                  publish=functools.partial(publish_low, low_rs, tag))
        emit_hc_up_mfma(up_w, low_rs, tag, x_rs, tag, smem, up_scratch, 1)
        mark(trace, 4, M_UP + 1, traced)
        # gdn_core's 64 KB state read (head blocks) goes out now: in flight during in_proj, ~15 us ahead.
        core = gdn_core_prefetch(bid, rs(conv_w, CONV_CH * 4 * 2), conv_rs, ssm_rs, rs(norm_w, D * 2))

        emit_gemv(x_rs, rs(qkvz_w, QKVZ * H * 2), in_rs, smem.partial.ptr, trace, tag, tag, QKVZ, H, 4, 1, DEPTH,
                  True, M_IN, traced, light=QKVZ_LIGHT)
        if bid < V_HEADS:
            emit_gemv(x_rs, rs(ba_w, BA * H * 2), rs(in_mb, BA * 4, QKVZ * 4), smem.partial.ptr, trace, tag, tag,
                      BA, H, 4, 1, DEPTH, True, M_BA, traced, light=BA_LIGHT)

        # Every out_proj and W_inj load of the writing blocks goes out now, before o exists.
        out_pre = gemv_prefetch(out_w_rs, H, HIDDEN_V, 4, 4, OUT_DEPTH, light=HEADS)
        inj = inject_prefetch(w_inj_rs, bid >= V_HEADS)
        if bid < V_HEADS:
            S = emit_gdn_core(bid, core, in_rs, tag, o_rs, tag, ack_rs, rs(a_log, V_HEADS * 4),
                              rs(dt_bias, V_HEADS * 4), smem, trace, M_CORE, traced)
            gdn_core_write_state(bid, S, core, ack_rs, tag, conv_rs, ssm_rs, smem, trace, M_STATE, traced)
        else:
            emit_inject(inj, smem)
            emit_gemv(o_rs, out_w_rs, o_rs, smem.partial.ptr, trace, tag, tag, H, HIDDEN_V, 4, 4, OUT_DEPTH, True,
                      M_OUT, traced, inflight=out_pre, light=HEADS,
                      publish=functools.partial(publish_combine, r_rs, rs(r_out, HC * 2), smem.c.ptr))
        finish_epoch(sync, tag)

    @flyc.jit
    def launch(r: fx.Tensor, r_out: fx.Tensor, hc_norm_w: fx.Tensor, w_down: fx.Tensor, w_up: fx.Tensor,
               w_inj: fx.Tensor, qkvz_w: fx.Tensor, ba_w: fx.Tensor, out_w: fx.Tensor, conv_w: fx.Tensor,
               a_log: fx.Tensor, dt_bias: fx.Tensor, norm_w: fx.Tensor, conv_states: fx.Tensor,
               ssm_states: fx.Tensor, cache_idx: fx.Tensor, x_mb: fx.Tensor, low_mb: fx.Tensor, in_mb: fx.Tensor,
               o_mb: fx.Tensor, ack: fx.Tensor, sync: fx.Tensor, trace: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        gdn_block(r, r_out, hc_norm_w, w_down, w_up, w_inj, qkvz_w, ba_w, out_w, conv_w, a_log, dt_bias, norm_w,
                  conv_states, ssm_states, cache_idx, x_mb, low_mb, in_mb, o_mb, ack, sync, trace).launch(
            grid=(NBLK, 1, 1), block=(256, 1, 1), stream=stream)

    return launch
