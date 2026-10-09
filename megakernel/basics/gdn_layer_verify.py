"""A whole GDN decoder layer for an MTP verify pass (T draft tokens) in one launch:
gdn_block_verify's attention half, then the FFN half on T tokens (moe_verify_op.py).

    R' = R + c1 * out_proj(gdn(in_proj(hc_mix1(R))))      per token row, R' to a mailbox
    R_out = R' + c2 * moe(hc_mix2(R'))                    per token row, MXFP4 experts

FFN half: hc read 2, router and their GEMVs take T tokens (each weight read once). Each block
then rebuilds the routing (top 10 + shared per token) and the table of unique experts; stage 1
runs over (expert, group) units on all blocks, stage 2 on blocks 0..159 (16 rows x T tokens).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T

import buffer_ops
from gdn_block import BA, BA_LIGHT, CONV_SLOT_BYTES, DEPTH, HEADS, HIDDEN_V, QKVZ, QKVZ_LIGHT, SSM_SLOT_BYTES
from gdn_block_verify import (M_BA, M_CORE, M_DOWN, M_IN, M_NORM, M_OUT, M_STATE, M_UP, N_WORDS, OUT_DEPTH,
                              SPANS as ATTN_SPANS, T_VERIFY, VerifyBlockSmem)
from gdn_core_op import CONV_CH, D, IN_DIM, V_HEADS, gdn_core_prefetch
from gdn_layer import S13_SIZE, S2_SIZE, W13_SIZE, W2_SIZE
from gdn_verify_op import emit_gdn_verify, gdn_verify_snapshots
from gemv_op import NBLK, emit_gemv, gemv_prefetch
from hc_op import (HC, LOW, H, combine_rows16, emit_hc_norm, emit_hc_up_mfma, emit_inject, finish_epoch, hc_norm_load,
                   hc_norm_poll, hc_up_prefetch_mfma, inject_prefetch, next_epoch_tag, publish_combine_mailbox,
                   publish_low)
from moe_op import N_ROUTED, S2_BLOCKS, poll_to_lds, publish_f32
from moe_verify_op import (H_GROUPS, H_LDS_WORDS, H_PAIRS, S1_PER_BLOCK, T_MAX, MoeVerifySmem, emit_expert_table,
                           emit_stage1, emit_stage2, emit_topk_tokens, poll_h_to_lds, quantize_x_tokens)
from timeline import mark

assert T_VERIFY <= T_MAX and H_LDS_WORDS <= N_WORDS  # smem.big holds n, the snapshot stage, then h

M_NORM2, M_DOWN2, M_UP2, M_ROUTER, M_X2, M_TOPK, M_QUANT, M_S1, M_H, M_S2 = 29, 31, 36, 37, 42, 43, 44, 45, 46, 47
SPANS = (ATTN_SPANS
         + [("hc_norm2 (poll R')", M_NORM2, M_NORM2 + 1)]
         + [(f"hc_down2 {n}", M_DOWN2 + i, M_DOWN2 + i + 1) for i, n in
            enumerate(["issue", "poll", "dot + reduce", "publish low"])]
         + [("hc_up2 + inject", M_DOWN2 + 4, M_UP2)]
         + [(f"router {n}", M_ROUTER + i, M_ROUTER + i + 1) for i, n in
            enumerate(["issue", "poll x", "dot + reduce", "publish logits"])]
         + [("x2 to LDS", M_ROUTER + 4, M_X2), ("topk + expert table", M_X2, M_TOPK),
            ("x2 quantized", M_TOPK, M_QUANT), ("stage 1", M_QUANT, M_S1),
            ("stage 2 poll h", M_S1, M_H), ("stage 2 + combine", M_H, M_S2)])
LAST_MARK = M_S2

# VerifyBlockSmem's fields, then MoeVerifySmem's, flat (the ops read fields by name).
LayerVerifySmem = fx.struct(type("LayerVerifySmem", (), {"__annotations__": {
    **VerifyBlockSmem.__annotations__, **MoeVerifySmem.__annotations__}}))


def mailboxes(device="cuda", tokens=T_VERIFY):
    import torch

    i32 = lambda n: torch.zeros(n, dtype=torch.int32, device=device)
    return {"x_mb": i32(tokens * H), "low_mb": i32(tokens * LOW), "in_mb": i32(tokens * IN_DIM),
            "o_mb": i32(tokens * HIDDEN_V), "r1_mb": i32(tokens * HC), "low2_mb": i32(tokens * LOW),
            "x2_mb": i32(tokens * H), "logits_mb": i32(tokens * 2 * N_ROUTED), "h_mb": i32(H_GROUPS * H_PAIRS * 2),
            "sync": i32(2)}


@functools.cache
def build(traced, tokens=T_VERIFY, win_dedup=False):
    win_row_bytes = CONV_CH * ((tokens + 2) if win_dedup else 3 * tokens) * 2

    @flyc.kernel(known_block_size=[256, 1, 1])
    def gdn_layer_verify(r: fx.Tensor, r_out: fx.Tensor, hc_norm_w: fx.Tensor, w_down: fx.Tensor, w_up: fx.Tensor,
                         w_inj: fx.Tensor, qkvz_w: fx.Tensor, ba_w: fx.Tensor, out_w: fx.Tensor, conv_w: fx.Tensor,
                         a_log: fx.Tensor, dt_bias: fx.Tensor, norm_w: fx.Tensor, conv_states: fx.Tensor,
                         ssm_states: fx.Tensor, cache_idx: fx.Tensor, win_scratch: fx.Tensor, ssm_scratch: fx.Tensor,
                         scratch_idx: fx.Tensor, hc2_norm_w: fx.Tensor, w_down2: fx.Tensor, w_up2: fx.Tensor,
                         w_inj2: fx.Tensor, router_w: fx.Tensor, w_sg: fx.Tensor, w13: fx.Tensor, s13: fx.Tensor,
                         w2: fx.Tensor, s2: fx.Tensor, x_mb: fx.Tensor, low_mb: fx.Tensor, in_mb: fx.Tensor,
                         o_mb: fx.Tensor, r1_mb: fx.Tensor, low2_mb: fx.Tensor, x2_mb: fx.Tensor,
                         logits_mb: fx.Tensor, h_mb: fx.Tensor, sync: fx.Tensor, trace: fx.Tensor):
        rs = lambda t, nbytes, base=None: buffer_ops.create_buffer_resource(
            t, max_size=False, num_records_bytes=nbytes, base_byte_offset=base)
        smem = fx.SharedAllocator().allocate(LayerVerifySmem).peek()
        n_ptr, stage = smem.big.ptr, fx.recast_iter(fx.Float32, smem.big.ptr)
        up_scratch = fx.recast_iter(fx.Int32, smem.partial.ptr)  # free between emit_gemvs
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
        r1_rs, low2_rs, x2_rs = rs(r1_mb, tokens * HC * 4), rs(low2_mb, tokens * LOW * 4), rs(x2_mb, tokens * H * 4)
        logits_rs, h_rs = rs(logits_mb, tokens * 2 * N_ROUTED * 4), rs(h_mb, H_GROUPS * H_PAIRS * 8)
        w_down2_rs = rs(w_down2, LOW * HC * 2)
        mark(trace, 4, M_NORM, traced)

        # ===== Attention half: gdn_block_verify.py's schedule, R' published to r1_mb =====
        hn = hc_norm_load(r_rs, rs(hc_norm_w, HC * 2), tokens)
        down_w = gemv_prefetch(w_down_rs, LOW, HC, 4, 4, DEPTH, light=HEADS)
        emit_hc_norm(hn, smem, n_ptr)
        up_w = hc_up_prefetch_mfma(w_up_rs)
        mark(trace, 4, M_NORM + 1, traced)

        emit_gemv(low_rs, w_down_rs, low_rs, smem.partial.ptr, trace, tag, tag, LOW, HC, 4, 4, DEPTH, True, M_DOWN,
                  traced, x_lds=n_ptr, inflight=down_w, light=HEADS, x_lds_ready=True,
                  publish=functools.partial(publish_low, low_rs, tag), tokens=tokens)
        emit_hc_up_mfma(up_w, low_rs, tag, x_rs, tag, smem, up_scratch, tokens, n_ptr)
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
                      publish=functools.partial(publish_combine_mailbox, r_rs, r1_rs, smem.c.ptr, tag))

        # ===== FFN half: gated read, MoE, gated write, T tokens =====
        mark(trace, 4, M_NORM2, traced)
        down2_w = gemv_prefetch(w_down2_rs, LOW, HC, 4, 4, DEPTH, light=HEADS)
        hn2 = hc_norm_poll(r1_rs, tag, rs(hc2_norm_w, HC * 2), tokens)
        emit_hc_norm(hn2, smem, n_ptr)
        up2_w = hc_up_prefetch_mfma(rs(w_up2, HC * LOW * 2))
        mark(trace, 4, M_NORM2 + 1, traced)

        emit_gemv(low2_rs, w_down2_rs, low2_rs, smem.partial.ptr, trace, tag, tag, LOW, HC, 4, 4, DEPTH, True,
                  M_DOWN2, traced, x_lds=n_ptr, inflight=down2_w, light=HEADS, x_lds_ready=True,
                  publish=functools.partial(publish_low, low2_rs, tag), tokens=tokens)
        inj2 = inject_prefetch(rs(w_inj2, 4 * HC * 2), bid < S2_BLOCKS)
        emit_hc_up_mfma(up2_w, low2_rs, tag, x2_rs, tag, smem, up_scratch, tokens, n_ptr)
        emit_inject(inj2, smem, tokens, n_ptr)
        mark(trace, 4, M_UP2, traced)

        emit_gemv(x2_rs, rs(router_w, N_ROUTED * H * 2), logits_rs, smem.partial.ptr, trace, tag, tag, N_ROUTED, H,
                  4, 1, DEPTH, True, M_ROUTER, traced, tokens=tokens,
                  publish=functools.partial(publish_f32, logits_rs, tag))
        poll_to_lds(x2_rs, tag, smem.xq.ptr, tokens * H)
        mark(trace, 4, M_X2, traced)
        emit_topk_tokens(logits_rs, tag, rs(w_sg, H * 2), smem, tokens)
        emit_expert_table(smem, tokens)
        mark(trace, 4, M_TOPK, traced)

        quantize_x_tokens(smem.xq.ptr, smem, tokens)
        mark(trace, 4, M_QUANT, traced)
        emit_stage1(rs(w13, W13_SIZE), rs(s13, S13_SIZE), h_rs, tag, smem, tokens)
        mark(trace, 4, M_S1, traced)

        if bid < S2_BLOCKS:
            poll_h_to_lds(h_rs, tag, smem.big.ptr, smem, tokens)
            mark(trace, 4, M_H, traced)
            emit_stage2(rs(w2, W2_SIZE), rs(s2, S2_SIZE), smem.big.ptr, smem, tokens)
            combine_rows16(r1_rs, tag, rs(r_out, tokens * HC * 2), smem.rows.ptr + T_MAX * 64, smem.c.ptr, tokens)
            mark(trace, 4, M_S2, traced)
        finish_epoch(sync, tag)

    @flyc.jit
    def launch(r: fx.Tensor, r_out: fx.Tensor, hc_norm_w: fx.Tensor, w_down: fx.Tensor, w_up: fx.Tensor,
               w_inj: fx.Tensor, qkvz_w: fx.Tensor, ba_w: fx.Tensor, out_w: fx.Tensor, conv_w: fx.Tensor,
               a_log: fx.Tensor, dt_bias: fx.Tensor, norm_w: fx.Tensor, conv_states: fx.Tensor,
               ssm_states: fx.Tensor, cache_idx: fx.Tensor, win_scratch: fx.Tensor, ssm_scratch: fx.Tensor,
               scratch_idx: fx.Tensor, hc2_norm_w: fx.Tensor, w_down2: fx.Tensor, w_up2: fx.Tensor,
               w_inj2: fx.Tensor, router_w: fx.Tensor, w_sg: fx.Tensor, w13: fx.Tensor, s13: fx.Tensor,
               w2: fx.Tensor, s2: fx.Tensor, x_mb: fx.Tensor, low_mb: fx.Tensor, in_mb: fx.Tensor, o_mb: fx.Tensor,
               r1_mb: fx.Tensor, low2_mb: fx.Tensor, x2_mb: fx.Tensor, logits_mb: fx.Tensor, h_mb: fx.Tensor,
               sync: fx.Tensor, trace: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        gdn_layer_verify(r, r_out, hc_norm_w, w_down, w_up, w_inj, qkvz_w, ba_w, out_w, conv_w, a_log, dt_bias,
                         norm_w, conv_states, ssm_states, cache_idx, win_scratch, ssm_scratch, scratch_idx,
                         hc2_norm_w, w_down2, w_up2, w_inj2, router_w, w_sg, w13, s13, w2, s2, x_mb, low_mb, in_mb,
                         o_mb, r1_mb, low2_mb, x2_mb, logits_mb, h_mb, sync, trace).launch(
            grid=(NBLK, 1, 1), block=(256, 1, 1), stream=stream)

    return launch
