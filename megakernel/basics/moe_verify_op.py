"""The MoE of a Qwen3.8 layer for T tokens (MTP verify), as monokernel ops. moe_op.py's math and
weight layout; the work is split by unique expert so each expert's weights are read once:

    topk     wave t: token t's top 10 + the shared expert (moe_op.emit_topk per token)
    table    every block: the unique experts of the T x 11 entries (<= 41) in first-occurrence
             order, with each token's weight for each (0 where the token didn't pick it)
    stage 1  units (expert, 32-group), spread over all blocks: gate / up for all T tokens;
             h published as MXFP4 (4 FP4 words + the scale), a quarter of bf16's size
    stage 2  blocks 0..159: 16 output rows x T tokens over all unique experts, h from LDS

Per token, every value is the decode op's; only stage 2's fp32 sum runs over experts in table
order (decode sums in slot order).
"""

import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from common import device, poll_pairs, put_pair
from moe_op import (GROUP, H, KEY_TAKEN, LOG2E, N_ROUTED, S1_GROUPS, SHARED_ID, SLOTS, TOPK, W2_REAL, _pow2,
                    mxfp4_exponent, stage1_loads, stage2_loads)
from skinny_bf16 import _dot2_f32_bf16, _pair, _wave_reduce_add_f32
import buffer_ops

T_MAX = 4
ENTRIES = T_MAX * SLOTS  # 44 routing entries (token, slot)
U_MAX = T_MAX * TOPK + 1  # 41 unique experts at most: the shared expert is in every token's entries
S1_PER_BLOCK = -(-U_MAX * S1_GROUPS // 256)  # 4 stage-1 units a block at most
S1_STEPS = 20  # K steps (MFMAs) of a stage-1 wave: 2560 / 128
FP4 = 4  # MFMA operand format code
H_PAIRS = 6  # h mailbox per (expert, token, group): 4 FP4 words, the scale, padding (3 x 16 B)
H_GROUPS = U_MAX * T_MAX * S1_GROUPS  # 3280 (expert, token, group) records
H_SCALES = H_GROUPS * 4  # LDS: FP4 words [record][4], then scales [record]
H_LDS_WORDS = H_GROUPS * 5
S2_AHEAD = 2  # stage 2: experts in flight per wave
S2_EXPERTS = -(-U_MAX // 4)  # 11 per wave at most
NO_EXPERT = SHARED_ID + 1  # loads out of bounds


@fx.struct
class MoeVerifySmem:
    xq: fx.Array[fx.Int32, T_MAX * H // 2, 16]  # raw x of T tokens, bf16 pairs (the router input)
    xf4: fx.Array[fx.Int32, T_MAX * H // 8, 16]  # x as MXFP4: [t][group][4] words of FP4
    xe: fx.Array[fx.Int32, T_MAX * H // GROUP, 16]  # [t][group] E8M0 (biased exponent)
    ids: fx.Array[fx.Int32, 64, 16]  # [t * 11 + slot] expert
    wts: fx.Array[fx.Float32, 64, 16]  # [t * 11 + slot] routing weight
    first: fx.Array[fx.Int32, 64, 16]  # entry is its expert's first occurrence
    uexp: fx.Array[fx.Int32, 64, 16]  # unique experts, first-occurrence order
    uw: fx.Array[fx.Float32, 64 * T_MAX, 16]  # [u * T + t] token t's weight for unique expert u
    nu: fx.Array[fx.Int32, 4, 16]
    hexp: fx.Array[fx.Int32, T_MAX, 16]  # stage 1: h's shared exponent per token
    rows: fx.Array[fx.Float32, T_MAX * 64 + T_MAX * 16, 16]  # stage-1 dots [t][64]; stage 2 [t][wave][16], y [t][16]


def _key_flip(b):
    return (b < 0).select(b ^ 0x7FFF0000, b)


@device
def emit_topk_tokens(logits_rs, tag, w_sg_rs, smem, tokens):
    """Wave t < T: token t's top 10 of 512 bf16 logits (lowest index on ties), softmax over them,
    and the shared expert's weight sigmoid(x_t . w_sg) from the raw x in smem.xq; into
    smem.ids / smem.wts [t * 11 + slot]."""
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    if wave < tokens:
        words = poll_pairs(logits_rs, [wave * N_ROUTED + lane * 8 + 2 * q for q in range(4)], tag)
        keys = [_key_flip(fx.Int32(w)) | (N_ROUTED - 1 - (lane * 8 + i)) for i, w in enumerate(words)]
        top, ids = [], []
        for r in range_constexpr(TOPK):
            best = keys[0]
            for i in range_constexpr(1, 8):
                best = fx.max(best, keys[i])
            m = fx.Int32(fx.coop.warp_reduce(best, fx.ReductionOp.MAX))
            top.append(_key_flip(m & -65536).bitcast(fx.Float32))
            ids.append(N_ROUTED - 1 - (m & 0xFFFF))
            keys = [(k == m).select(fx.Int32(KEY_TAKEN), k) for k in keys]
        ex = [fx.math.exp(top[r] - top[0]) for r in range(TOPK)]
        total = ex[0]
        for r in range_constexpr(1, TOPK):
            total = total + ex[r]
        acc = fx.Float32(0.0)
        x_lds = smem.xq.ptr + wave * (H // 2)
        for i in range_constexpr(H // 64 // 8):
            k = (lane + 64 * i) * 8
            xv = fx.Vector(fx.ptr_load(x_lds + k // 2, result_type=T.vec(4, T.i32))).bitcast(fx.BFloat16)
            wv = fx.Vector(buffer_ops.buffer_load(w_sg_rs, k // 2, vec_width=4, dtype=T.i32)).bitcast(fx.BFloat16)
            for q in range_constexpr(4):
                acc = fx.Float32(_dot2_f32_bf16(acc, _pair(xv, q), _pair(wv, q)))
        dot = fx.Float32(_wave_reduce_add_f32(acc))
        if lane == 0:
            base = wave * SLOTS
            for r in range_constexpr(TOPK):
                fx.ptr_store(ids[r], smem.ids.ptr + base + r)
                fx.ptr_store(ex[r] / total, smem.wts.ptr + base + r)
            fx.ptr_store(fx.Int32(SHARED_ID), smem.ids.ptr + base + TOPK)
            fx.ptr_store(fx.Float32(1.0) / (fx.Float32(1.0) + fx.math.exp(-dot)), smem.wts.ptr + base + TOPK)
    fx.gpu.barrier()


@device
def emit_expert_table(smem, tokens):
    """The unique experts of the T x 11 entries in first-occurrence order (smem.uexp, count in
    smem.nu[0]) and each token's weight for each (smem.uw; 0 past the count)."""
    tid = fx.Int32(fx.thread_idx.x)
    n = tokens * SLOTS
    if tid < 64:
        for t in range_constexpr(tokens):
            fx.ptr_store(fx.Float32(0.0), smem.uw.ptr + tid * tokens + t)
    if tid < n:
        e0 = fx.ptr_load(smem.ids.ptr + tid)
        first = tid >= 0
        for k in range_constexpr(n - 1):
            first = first & ((tid <= k) | (fx.ptr_load(smem.ids.ptr + k) != e0))
        fx.ptr_store(first.select(fx.Int32(1), fx.Int32(0)), smem.first.ptr + tid)
    fx.gpu.barrier()
    if tid < n:
        e = fx.ptr_load(smem.ids.ptr + tid)
        u = fx.Int32(0)
        for k in range_constexpr(n):
            u = u + (tid > k).select(fx.ptr_load(smem.first.ptr + k), fx.Int32(0))
        if fx.ptr_load(smem.first.ptr + tid) == 1:
            fx.ptr_store(e, smem.uexp.ptr + u)
            for t in range_constexpr(tokens):
                wt = fx.Float32(0.0)
                for sl in range_constexpr(SLOTS):
                    i = t * SLOTS + sl
                    wt = wt + (fx.ptr_load(smem.ids.ptr + i) == e).select(fx.ptr_load(smem.wts.ptr + i), fx.Float32(0.0))
                fx.ptr_store(wt, smem.uw.ptr + u * tokens + t)
        if tid == n - 1:
            fx.ptr_store(u + fx.ptr_load(smem.first.ptr + tid), smem.nu.ptr)
    fx.gpu.barrier()


@device
def quantize_x_tokens(x_lds, smem, tokens):
    """MXFP4 of the bf16 x of T tokens (x_lds) as stage 1's MFMA operand: FP4 words into
    smem.xf4, E8M0 exponents into smem.xe; one thread a 32-group."""
    tid = fx.Int32(fx.thread_idx.x)
    n = tokens * H // GROUP
    for k in range_constexpr(-(-n // 256)):
        g = tid + 256 * k
        if g < n:
            p = x_lds + g * (GROUP // 2)
            vals = []
            for q in range_constexpr(GROUP // 8):
                v = fx.Vector(fx.ptr_load(p + 4 * q, result_type=T.vec(4, T.i32))).bitcast(fx.BFloat16).to(fx.Float32)
                vals += [v[i] for i in range(8)]
            amax = fx.math.absf(vals[0])
            for i in range_constexpr(1, GROUP):
                amax = fx.max(amax, fx.math.absf(vals[i]))
            e = mxfp4_exponent(amax)
            words = [_fp4_word(vals[8 * q: 8 * q + 8], _pow2(e)) for q in range(GROUP // 8)]
            fx.ptr_store(fx.Vector.from_elements(words, fx.Int32), smem.xf4.ptr + g * 4)
            fx.ptr_store(fx.max(e + 127, 1), smem.xe.ptr + g)
    fx.gpu.barrier()


def _mfma_fp4(acc, a_words, a_scale, b_words, b_scale):
    """acc (4 f32) += A . B, v_mfma_scale_f32_16x16x128 on FP4: lane l holds A row l % 16 and B
    column l % 16, K block l / 16 (32 values, 4 words); scales: E8M0 in the low byte. The result
    is column l % 16, rows 4 (l / 16) .. + 3."""
    z = fx.Int32(0)
    a8 = fx.Vector.from_elements([a_words[0], a_words[1], a_words[2], a_words[3], z, z, z, z], fx.Int32)
    b8 = fx.Vector.from_elements([b_words[0], b_words[1], b_words[2], b_words[3], z, z, z, z], fx.Int32)
    return fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(
        T.vec(4, T.f32), [a8.ir_value(), b8.ir_value(), acc.ir_value(), FP4, FP4, 0, fx.Int32(a_scale).ir_value(), 0,
                          fx.Int32(b_scale).ir_value()]))


def _fp4_word(vals8, scale):
    """8 fp32 -> one dword of 8 FP4 (E2M1 of v / scale, round to nearest even)."""
    word = fx.Int32(0)
    for sel in range_constexpr(4):
        word = fx.Int32(rocdl.cvt_scalef32_pk_fp4_f32(T.i32, word.ir_value(), vals8[2 * sel].ir_value(),
                                                      vals8[2 * sel + 1].ir_value(), scale.ir_value(), sel))
    return word


# ===== Stage 1 =====


@device
def emit_stage1(w13_rs, s13_rs, h_rs, tag, smem, tokens):
    """This block's stage-1 units k = 0 .. 3 (bid + 256 k = expert u * 20 + group) in a device
    loop (unrolled, they spill)."""
    for k in range(S1_PER_BLOCK):
        active, u, grp, pre = stage1_unit_loads(w13_rs, s13_rs, smem, fx.Int32(k))
        if active:
            emit_stage1_unit(pre, u, grp, h_rs, tag, smem, tokens)


def stage1_unit_loads(w13_rs, s13_rs, smem, k):
    """Loads of this block's k-th unit; out of bounds past the table. Returns (active, u, grp,
    loads)."""
    unit = fx.Int32(fx.block_idx.x) + 256 * k
    active = unit < fx.ptr_load(smem.nu.ptr) * S1_GROUPS
    u, grp = fx.min(unit // S1_GROUPS, U_MAX - 1), unit % S1_GROUPS
    expert = active.select(fx.ptr_load(smem.uexp.ptr + u), fx.Int32(NO_EXPERT))
    return active, u, grp, stage1_loads(w13_rs, s13_rs, expert, grp)


@device
def emit_stage1_unit(pre, u, grp, h_rs, tag, smem, tokens):
    """Gate / up rows of one (expert, group) for T tokens: 20 FP4 MFMAs a wave (A: its 16 rows,
    straight from the loads; B: x of the tokens as columns), h = MXFP4(bf16(silu(g) u)) per token,
    published as FP4 words + scale at record (u * T + t) * 20 + grp of h_rs."""
    weights, scales, sbyte = pre
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    tok, blk = lane % 16, lane // 16
    valid = tok < tokens
    tc = fx.min(tok, tokens - 1)
    zero4 = [fx.Int32(0)] * 4
    acc = fx.Vector.from_elements([fx.Float32(0.0)] * 4, fx.Float32)
    for s in range_constexpr(S1_STEPS):
        g = tc * (H // GROUP) + s * 4 + blk  # token tc's group of K block blk of step s
        xw = fx.Vector(fx.ptr_load(smem.xf4.ptr + g * 4, result_type=T.vec(4, T.i32)))
        b = [valid.select(xw[i], zero4[i]) for i in range(4)]
        acc = _mfma_fp4(acc, fx.Vector(weights[s]), fx.Int32(scales[s]) >> (sbyte[s] * 8), b,
                        fx.ptr_load(smem.xe.ptr + g))
    if valid:
        for i in range_constexpr(4):
            fx.ptr_store(acc[i], smem.rows.ptr + tok * 64 + wave * 16 + 4 * blk + i)
    fx.gpu.barrier()
    if tid < 32 * tokens:
        ht, j = tid // 32, tid % 32
        g = fx.ptr_load(smem.rows.ptr + ht * 64 + j)
        up = fx.ptr_load(smem.rows.ptr + ht * 64 + 32 + j)
        ex = fx.Float32(rocdl.exp2(T.f32, (g * fx.Float32(-LOG2E)).ir_value()))
        sig = fx.Float32(rocdl.rcp(T.f32, (fx.Float32(1.0) + ex).ir_value()))
        hv = fx.Float32((g * sig * up).to(fx.BFloat16))
        amax = fx.Float32(fx.coop.warp_reduce(fx.math.absf(hv), fx.ReductionOp.MAX, width=32))
        fx.ptr_store(hv, smem.rows.ptr + ht * 64 + j)  # own slot: g was read by this thread only
        if j == 0:
            fx.ptr_store(mxfp4_exponent(amax), smem.hexp.ptr + ht)
    fx.gpu.barrier()
    if tid < 4 * tokens:
        pt, pw = tid // 4, tid % 4
        hscale = _pow2(fx.ptr_load(smem.hexp.ptr + pt))
        hvals = [fx.ptr_load(smem.rows.ptr + pt * 64 + 8 * pw + i) for i in range(8)]
        base = ((u * tokens + pt) * S1_GROUPS + grp) * H_PAIRS
        put_pair(h_rs, base + pw, _fp4_word(hvals, hscale), tag)
        if pw == 0:
            put_pair(h_rs, base + 4, hscale.bitcast(fx.Int32), tag)
            put_pair(h_rs, base + 5, fx.Int32(0), tag)
    fx.gpu.barrier()


# ===== Stage 2 =====


def _stage2_issue(w2_rs, s2_rs, smem, nu, i):
    """This wave's i-th unique expert's (weights, scale dwords); out of bounds past the table."""
    wave = fx.Int32(fx.thread_idx.x) // 64
    u = wave + 4 * i
    e = (u < nu).select(fx.ptr_load(smem.uexp.ptr + fx.min(u, U_MAX - 1)), fx.Int32(NO_EXPERT))
    weights, scales, _ = stage2_loads(w2_rs, s2_rs, e, fx.Int32(fx.block_idx.x))
    return [fx.Vector(w) for w in weights], [fx.Int32(sc) for sc in scales]


def _stage2_expert(weights, scales, sbyte, h_words, h_scales, part, wk, valid):
    """part += wk * (16 rows of this expert's down projection . h): 5 FP4 MFMAs (A: the rows, B:
    h of the tokens as columns, from LDS); nothing when not valid (past the table: h is stale
    LDS, maybe NaN, so it is selected out, not zeroed)."""
    acc = fx.Vector.from_elements([fx.Float32(0.0)] * 4, fx.Float32)
    for s in range_constexpr(W2_REAL // 64):
        hw = fx.Vector(fx.ptr_load(h_words + 4 * s * 4, result_type=T.vec(4, T.i32)))
        he = fx.Int32(fx.ptr_load(h_scales + 4 * s)) >> 23  # scale 2^e as f32 bits: the E8M0 byte
        acc = _mfma_fp4(acc, fx.Vector(weights[s]), scales[s] >> (sbyte[s] * 8), [hw[i] for i in range(4)], he)
    return [valid.select(part[i] + wk * acc[i], part[i]) for i in range(4)]


@device
def poll_h_to_lds(h_rs, tag, h_lds, smem, tokens):
    """The h records of the table's experts into LDS: FP4 words at [record * 4], scales at
    [H_SCALES + record]. Spins on one 16 B chunk a thread, then fetches the rest in batches."""
    tid = fx.Int32(fx.thread_idx.x)
    n = fx.ptr_load(smem.nu.ptr) * (tokens * S1_GROUPS * 3)  # 3 chunks a record
    per = -(-U_MAX * tokens * S1_GROUPS * 3 // 256)
    cs = [fx.min(tid + 256 * j, n - 1) for j in range(per)]  # past the end: the last chunk again
    words = poll_pairs(h_rs, [2 * cs[0]], tag)
    batch = 13
    for b in range_constexpr(1, per, batch):
        words = words + poll_pairs(h_rs, [2 * c for c in cs[b: b + batch]], tag)
    for j in range_constexpr(per):
        rec, part = cs[j] // 3, cs[j] % 3
        if part < 2:
            fx.ptr_store(words[2 * j], h_lds + rec * 4 + 2 * part)
            fx.ptr_store(words[2 * j + 1], h_lds + rec * 4 + 2 * part + 1)
        else:
            fx.ptr_store(words[2 * j], h_lds + H_SCALES + rec)
    fx.gpu.barrier()


@device
def emit_stage2(w2_rs, s2_rs, h_lds, smem, tokens):
    """y for the block's 16 rows and T tokens (fp32, smem.rows[T * 64 + t * 16 + i]): wave w takes
    unique experts w, w + 4, ... in a device loop (unrolled, LLVM interleaves them and spills),
    weights S2_AHEAD experts ahead as carried state; then the 4 wave sums in order."""
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    rt = fx.Int32(fx.block_idx.x)
    tok, blk = lane % 16, lane // 16
    tc = fx.min(tok, tokens - 1)
    nu = fx.ptr_load(smem.nu.ptr)
    # The scale's byte in its dword: ((g / 4) % 2) * 2 + (m / 16) % 2 = (s % 2) * 2 + rt % 2, the
    # same for every expert (H / 16 is even).
    sbyte = [(s % 2) * 2 + rt % 2 for s in range(W2_REAL // 64)]
    pending = [_stage2_issue(w2_rs, s2_rs, smem, nu, fx.Int32(i)) for i in range(S2_AHEAD)]
    part = [fx.Float32(0.0) for _ in range(4)]
    for i in range(S2_EXPERTS):
        weights, scales = pending[0]
        nxt = _stage2_issue(w2_rs, s2_rs, smem, nu, fx.Int32(i) + S2_AHEAD)
        u = wave + 4 * fx.Int32(i)
        uc = fx.min(u, U_MAX - 1)
        rec = (uc * tokens + tc) * S1_GROUPS + blk  # (u, token, group of step 0)
        wk = fx.ptr_load(smem.uw.ptr + uc * tokens + tc)
        part = _stage2_expert(weights, scales, sbyte, h_lds + rec * 4, h_lds + H_SCALES + rec, part, wk, u < nu)
        pending = pending[1:] + [nxt]
    if tok < tokens:
        for i in range_constexpr(4):
            fx.ptr_store(part[i], smem.rows.ptr + (tok * 4 + wave) * 16 + 4 * blk + i)
    fx.gpu.barrier()
    if tid < 16 * tokens:
        yt, yi = tid // 16, tid % 16
        r = smem.rows.ptr + yt * 64 + yi
        y = fx.ptr_load(r) + fx.ptr_load(r + 16)
        y = y + fx.ptr_load(r + 32) + fx.ptr_load(r + 48)
        fx.ptr_store(y, smem.rows.ptr + T_MAX * 64 + yt * 16 + yi)
    fx.gpu.barrier()
