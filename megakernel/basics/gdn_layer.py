"""A whole GDN decoder layer for one token in one launch: gdn_block's attention half, then the FFN half.

    R' = R + c1 * out_proj(gdn(in_proj(hc_mix1(R))))           (gdn_block.py, R' to a mailbox)
    R_out = R' + c2 * moe(hc_mix2(R'))                          (hc_op + moe_op, MXFP4 experts)

FFN half, every block unless noted:

    hc_norm2   poll R' (20 KB), n2 into LDS
    hc_down2   emit_gemv -> low2 (no rows on head blocks: they are last out of the attention half)
    hc_up2     10 outputs of x2 per block; c2 (inject) on stage-2 blocks
    router     emit_gemv, 2 logits per block -> logits mailbox (bf16-rounded fp32)
    topk       every block: x2 into LDS, top 10 + shared weight, x2 MXFP4-rounded in place
    stage 1    blocks 36..255: one (slot, 32-group) of gate / up -> h mailbox
    stage 2    blocks 0..159: 16 output rows of all 11 slots, R_out = R' + bf16(y) c2
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T

import buffer_ops
from gdn_block import (BA, BA_LIGHT, CONV_SLOT_BYTES, DEPTH, HEADS, HIDDEN_V, M_BA, M_CORE, M_DOWN, M_IN, M_NORM,
                       M_OUT, M_STATE, M_UP, OUT_DEPTH, QKVZ, QKVZ_LIGHT, SPANS as ATTN_SPANS, SSM_SLOT_BYTES,
                       BlockSmem)
from gdn_core_op import CONV_CH, D, IN_DIM, V_HEADS, ack_pairs, emit_gdn_core, gdn_core_prefetch, gdn_core_write_state
from gemv_op import NBLK, emit_gemv, gemv_prefetch
from hc_op import (HC, LOW, H, combine_rows16, emit_hc_norm, emit_hc_up, emit_inject, finish_epoch, hc_norm_load,
                   hc_norm_poll, hc_up_prefetch, inject_prefetch, next_epoch_tag, publish_combine_mailbox,
                   publish_low)
from moe_op import (N_ROUTED, S1_BLOCKS, S2_BLOCKS, SLOTS, INTER, W13_BYTES, W13_GROUPS, W13_ROWS, W2_BYTES, W2_GROUPS,
                    MoeSmem, emit_stage1, emit_stage2, emit_topk, poll_to_lds, publish_f32, quantize_x_lds,
                    stage1_prefetch, stage2_prefetch)
from timeline import mark

E = N_ROUTED + 1  # + the fused shared expert
W13_SIZE, S13_SIZE = E * W13_ROWS * W13_BYTES, E * W13_ROWS * W13_GROUPS
W2_SIZE, S2_SIZE = E * H * W2_BYTES, E * H * W2_GROUPS
H_VALUES = SLOTS * INTER

M_NORM2, M_DOWN2, M_UP2, M_ROUTER, M_TOPK, M_S1, M_S2 = 29, 31, 36, 37, 42, 45, 46  # M_UP2, M_S1: end only
SPANS = (ATTN_SPANS
         + [("hc_norm2 (poll R')", M_NORM2, M_NORM2 + 1)]
         + [(f"hc_down2 {n}", M_DOWN2 + i, M_DOWN2 + i + 1) for i, n in
            enumerate(["issue", "poll", "dot + reduce", "publish low"])]
         + [("hc_up2 + inject", M_DOWN2 + 4, M_UP2)]
         + [(f"router {n}", M_ROUTER + i, M_ROUTER + i + 1) for i, n in
            enumerate(["issue", "poll x", "dot + reduce", "publish logits"])]
         + [("x2 to LDS", M_ROUTER + 4, M_TOPK), ("topk", M_TOPK, M_TOPK + 1),
            ("quantize x", M_TOPK + 1, M_TOPK + 2), ("stage 1", M_TOPK + 2, M_S1),
            ("stage 2 poll h", M_S1, M_S2), ("stage 2 + combine", M_S2, M_S2 + 1)])
LAST_MARK = M_S2 + 1


# BlockSmem's fields, then MoeSmem's, flat (the ops read fields by name): ~121 KB.
LayerSmem = fx.struct(type("LayerSmem", (), {"__annotations__": {**BlockSmem.__annotations__,
                                                                 **MoeSmem.__annotations__}}))


def mailboxes(device="cuda"):
    """Scratch shared by every launch (and layer): the tags keep launches apart; every handoff
    within a launch needs its own mailbox."""
    import torch

    i32 = lambda n: torch.zeros(n, dtype=torch.int32, device=device)
    return {"x_mb": i32(H), "low_mb": i32(LOW), "in_mb": i32(IN_DIM), "o_mb": i32(HIDDEN_V),
            "ack": i32(ack_pairs() * 2), "r1_mb": i32(HC), "low2_mb": i32(LOW), "x2_mb": i32(H),
            "logits_mb": i32(2 * N_ROUTED), "h_mb": i32(H_VALUES), "sync": i32(2)}


@functools.cache
def build(traced):
    @flyc.kernel(known_block_size=[256, 1, 1])
    def gdn_layer(r: fx.Tensor, r_out: fx.Tensor, hc_norm_w: fx.Tensor, w_down: fx.Tensor, w_up: fx.Tensor,
                  w_inj: fx.Tensor, qkvz_w: fx.Tensor, ba_w: fx.Tensor, out_w: fx.Tensor, conv_w: fx.Tensor,
                  a_log: fx.Tensor, dt_bias: fx.Tensor, norm_w: fx.Tensor, conv_states: fx.Tensor,
                  ssm_states: fx.Tensor, cache_idx: fx.Tensor, hc2_norm_w: fx.Tensor, w_down2: fx.Tensor,
                  w_up2: fx.Tensor, w_inj2: fx.Tensor, router_w: fx.Tensor, w_sg: fx.Tensor, w13: fx.Tensor,
                  s13: fx.Tensor, w2: fx.Tensor, s2: fx.Tensor, x_mb: fx.Tensor, low_mb: fx.Tensor,
                  in_mb: fx.Tensor, o_mb: fx.Tensor, ack: fx.Tensor, r1_mb: fx.Tensor, low2_mb: fx.Tensor,
                  x2_mb: fx.Tensor, logits_mb: fx.Tensor, h_mb: fx.Tensor, sync: fx.Tensor, trace: fx.Tensor):
        rs = lambda t, nbytes, base=None: buffer_ops.create_buffer_resource(
            t, max_size=False, num_records_bytes=nbytes, base_byte_offset=base)
        smem = fx.SharedAllocator().allocate(LayerSmem).peek()
        bid = fx.Int32(fx.block_idx.x)
        tag = next_epoch_tag(rs(sync, 8))
        slot = fx.Int32(buffer_ops.buffer_load(rs(cache_idx, 4), 0, vec_width=1, dtype=T.i32))
        conv_rs = rs(conv_states, CONV_SLOT_BYTES, slot * CONV_SLOT_BYTES)
        ssm_rs = rs(ssm_states, SSM_SLOT_BYTES, slot * SSM_SLOT_BYTES)
        r_rs, x_rs, low_rs = rs(r, HC * 2), rs(x_mb, H * 4), rs(low_mb, LOW * 4)
        in_rs, o_rs, ack_rs = rs(in_mb, IN_DIM * 4), rs(o_mb, HIDDEN_V * 4), rs(ack, ack_pairs() * 8)
        w_down_rs, w_up_rs, w_inj_rs = rs(w_down, LOW * HC * 2), rs(w_up, HC * LOW * 2), rs(w_inj, 4 * HC * 2)
        out_w_rs = rs(out_w, H * HIDDEN_V * 2)
        r1_rs, low2_rs, x2_rs = rs(r1_mb, HC * 4), rs(low2_mb, LOW * 4), rs(x2_mb, H * 4)
        logits_rs, h_rs = rs(logits_mb, 2 * N_ROUTED * 4), rs(h_mb, H_VALUES * 4)
        w_down2_rs = rs(w_down2, LOW * HC * 2)
        router_rs = rs(router_w, N_ROUTED * H * 2)
        mark(trace, 4, M_NORM, traced)

        # ===== Attention half: gdn_block.py's schedule, R' published to r1_mb =====
        hn = hc_norm_load(r_rs, rs(hc_norm_w, HC * 2))
        down_w = gemv_prefetch(w_down_rs, LOW, HC, 4, 4, DEPTH, light=HEADS)
        emit_hc_norm(hn, smem)
        up_w = hc_up_prefetch(w_up_rs)
        mark(trace, 4, M_NORM + 1, traced)

        emit_gemv(low_rs, w_down_rs, low_rs, smem.partial.ptr, trace, tag, tag, LOW, HC, 4, 4, DEPTH, True, M_DOWN,
                  traced, x_lds=smem.n.ptr, inflight=down_w, light=HEADS, x_lds_ready=True,
                  publish=functools.partial(publish_low, low_rs, tag))
        emit_hc_up(up_w, low_rs, tag, x_rs, tag, smem)
        mark(trace, 4, M_UP + 1, traced)
        core = gdn_core_prefetch(bid, rs(conv_w, CONV_CH * 4 * 2), conv_rs, ssm_rs, rs(norm_w, D * 2))

        emit_gemv(x_rs, rs(qkvz_w, QKVZ * H * 2), in_rs, smem.partial.ptr, trace, tag, tag, QKVZ, H, 4, 1, DEPTH,
                  True, M_IN, traced, light=QKVZ_LIGHT)
        if bid < V_HEADS:
            emit_gemv(x_rs, rs(ba_w, BA * H * 2), rs(in_mb, BA * 4, QKVZ * 4), smem.partial.ptr, trace, tag, tag,
                      BA, H, 4, 1, DEPTH, True, M_BA, traced, light=BA_LIGHT)

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
                      publish=functools.partial(publish_combine_mailbox, r_rs, r1_rs, smem.c.ptr, tag))

        # ===== FFN half: gated residual read, MoE, gated write =====
        mark(trace, 4, M_NORM2, traced)
        down2_w = gemv_prefetch(w_down2_rs, LOW, HC, 4, 4, DEPTH, light=HEADS)
        hn2 = hc_norm_poll(r1_rs, tag, rs(hc2_norm_w, HC * 2))
        emit_hc_norm(hn2, smem)
        up2_w = hc_up_prefetch(rs(w_up2, HC * LOW * 2))
        mark(trace, 4, M_NORM2 + 1, traced)

        emit_gemv(low2_rs, w_down2_rs, low2_rs, smem.partial.ptr, trace, tag, tag, LOW, HC, 4, 4, DEPTH, True,
                  M_DOWN2, traced, x_lds=smem.n.ptr, inflight=down2_w, light=HEADS, x_lds_ready=True,
                  publish=functools.partial(publish_low, low2_rs, tag))
        inj2 = inject_prefetch(rs(w_inj2, 4 * HC * 2), bid < S2_BLOCKS)
        emit_hc_up(up2_w, low2_rs, tag, x2_rs, tag, smem)
        emit_inject(inj2, smem)  # off the critical path: the router waits for every block's x2
        mark(trace, 4, M_UP2, traced)

        emit_gemv(x2_rs, router_rs, logits_rs, smem.partial.ptr, trace, tag, tag, N_ROUTED, H, 4, 1, DEPTH, True,
                  M_ROUTER, traced, publish=functools.partial(publish_f32, logits_rs, tag))
        poll_to_lds(x2_rs, tag, smem.xq.ptr, H)
        mark(trace, 4, M_TOPK, traced)
        emit_topk(logits_rs, tag, smem.xq.ptr, rs(w_sg, H * 2), smem)
        mark(trace, 4, M_TOPK + 1, traced)
        # The experts' 27 MB go out as soon as the ids are known; quantizing x hides under them.
        pre1 = stage1_prefetch(rs(w13, W13_SIZE), rs(s13, S13_SIZE), smem)
        pre2 = stage2_prefetch(rs(w2, W2_SIZE), rs(s2, S2_SIZE), smem)
        quantize_x_lds(smem.xq.ptr)
        mark(trace, 4, M_TOPK + 2, traced)

        if bid >= NBLK - S1_BLOCKS:
            emit_stage1(pre1, h_rs, tag, smem.xq.ptr, smem)
        mark(trace, 4, M_S1, traced)
        if bid < S2_BLOCKS:
            poll_to_lds(h_rs, tag, smem.h.ptr, H_VALUES)
            mark(trace, 4, M_S2, traced)
            emit_stage2(pre2, smem)
            combine_rows16(r1_rs, tag, rs(r_out, HC * 2), smem.rows.ptr, smem.c.ptr)
            mark(trace, 4, M_S2 + 1, traced)
        finish_epoch(sync, tag)

    @flyc.jit
    def launch(r: fx.Tensor, r_out: fx.Tensor, hc_norm_w: fx.Tensor, w_down: fx.Tensor, w_up: fx.Tensor,
               w_inj: fx.Tensor, qkvz_w: fx.Tensor, ba_w: fx.Tensor, out_w: fx.Tensor, conv_w: fx.Tensor,
               a_log: fx.Tensor, dt_bias: fx.Tensor, norm_w: fx.Tensor, conv_states: fx.Tensor,
               ssm_states: fx.Tensor, cache_idx: fx.Tensor, hc2_norm_w: fx.Tensor, w_down2: fx.Tensor,
               w_up2: fx.Tensor, w_inj2: fx.Tensor, router_w: fx.Tensor, w_sg: fx.Tensor, w13: fx.Tensor,
               s13: fx.Tensor, w2: fx.Tensor, s2: fx.Tensor, x_mb: fx.Tensor, low_mb: fx.Tensor, in_mb: fx.Tensor,
               o_mb: fx.Tensor, ack: fx.Tensor, r1_mb: fx.Tensor, low2_mb: fx.Tensor, x2_mb: fx.Tensor,
               logits_mb: fx.Tensor, h_mb: fx.Tensor, sync: fx.Tensor, trace: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        gdn_layer(r, r_out, hc_norm_w, w_down, w_up, w_inj, qkvz_w, ba_w, out_w, conv_w, a_log, dt_bias, norm_w,
                  conv_states, ssm_states, cache_idx, hc2_norm_w, w_down2, w_up2, w_inj2, router_w, w_sg, w13, s13,
                  w2, s2, x_mb, low_mb, in_mb, o_mb, ack, r1_mb, low2_mb, x2_mb, logits_mb, h_mb, sync,
                  trace).launch(grid=(NBLK, 1, 1), block=(256, 1, 1), stream=stream)

    return launch
