"""The MoE of a Qwen3.8 layer for one token, as monokernel ops, on SGLang's MXFP4 expert layout.

SGLang (quark W4A4 MXFP4 on gfx950, aiter): 512 routed experts plus the shared expert fused as
expert 512; w13 [513, 1536, 1280 B] holds gate rows [0, 640) and up rows [768, 1408) (the
intermediate 640 is padded to 768), w2 [513, 2560, 384 B] holds 320 real bytes a row. Weights are
aiter's shuffle_weight(16, 16): w[e, n, b] at [e][n / 16][b / 32][(b % 32) / 16][n % 16][b % 16],
so a 16 B chunk is one row's 32 FP4 values, one scale group. Scales are e8m0_shuffle'd:
[rows, groups] (groups padded to 8) at [m / 32][g / 8][g % 4][m % 16][(g / 4) % 2][(m / 16) % 2].

Math (stock: aiter's flydsl_moe1/2_afp4_wfp4): activations are MXFP4-quantized before each GEMM
(aiter's RoundUp scale 2^ceil(log2(amax / 6)), round-to-nearest-even E2M1). Every such value
and every dequantized weight is exact in bf16, so the ops fake-quantize to bf16 and use bf16 dots:
the products are exact, as in FP4 MFMA; only the fp32 summation order differs.

    router   logits = x . W_r^T (512, bf16 weights)  -> fp32 mailbox          (emit_gemv + publish_f32)
    topk     top 10 logits, w = softmax over them; slot 10 = shared expert, w = sigmoid(x . w_sg)
    stage 1  h_k = MXFP4(bf16(silu(gate_k . q(x)) * up_k . q(x)))   per 32 of 640, one group per block
    stage 2  y = sum_k w_k down_k . h_k, 16 output rows per block, then the gated write

Stock's stage 2 rounds each w_k down_k . h_k to bf16 and sums the 11 with bf16 atomic adds in
arrival order (nondeterministic); here the sum is fp32 in a fixed order, rounded once.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

import buffer_ops
from common import CM_DEV, CM_NT, bf16x2, device, poll_pairs, put_pair
from skinny_bf16 import _dot2_f32_bf16, _pair, _permlane_swap_add_f32, _wave_reduce_add_f32

H = 2560
N_ROUTED, TOPK, SLOTS = 512, 10, 11  # routed experts, routed per token, + the shared expert
SHARED_ID = N_ROUTED
INTER, INTER_PAD = 640, 768
W13_ROWS, W13_BYTES, W13_GROUPS = 2 * INTER_PAD, H // 2, H // 32  # 1536, 1280, 80
W2_BYTES, W2_GROUPS = INTER_PAD // 2, INTER_PAD // 32  # 384, 24 (20 real)
W2_REAL = INTER // 2  # 320 real bytes a w2 row
GROUP = 32  # MXFP4 block size, in values
S1_GROUPS = INTER // GROUP  # 20: stage-1 work units per expert
S1_BLOCKS = SLOTS * S1_GROUPS  # 220
S2_ROWS = 16
S2_BLOCKS = H // S2_ROWS  # 160
KEY_TAKEN = -(2**31)  # below every top-k key
LOG2E = 1.4426950408889634


@fx.struct
class MoeSmem:
    xq: fx.Array[fx.Int32, H // 2, 16]  # x as bf16 pairs; MXFP4-rounded in place for stage 1
    h: fx.Array[fx.Int32, SLOTS * INTER // 2, 16]  # stage-1 outputs of all 11 slots, bf16 pairs
    ids: fx.Array[fx.Int32, 16, 16]  # expert id per slot
    wts: fx.Array[fx.Float32, 16, 16]  # routing weight per slot
    rows: fx.Array[fx.Float32, 128, 16]  # stage 1: [wave, 16 rows] dots; stage 2: y, then [wave, 16] sums


# ===== MXFP4 helpers =====


def e8m0_f32(byte):
    """E8M0 scale byte -> fp32 2^(byte - 127), the cvt_scalef32 operand."""
    return ((fx.Int32(byte) & 0xFF) << 23).bitcast(fx.Float32)


def fp4x8_to_bf16(word, scale):
    """One dword of 8 FP4 values (low nibble first) times `scale` -> 8 bf16 (exact)."""
    parts = []
    for sel in range_constexpr(4):
        pair = fx.Vector(rocdl.cvt_scalef32_pk_bf16_fp4(T.vec(2, T.bf16), fx.Int32(word).ir_value(),
                                                        fx.Float32(scale).ir_value(), sel))
        parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


def mxfp4_exponent(amax):
    """aiter's default (RoundUp, mx_quant_utils.h) shared exponent: ceil(log2(amax * f32(1 / 6)))."""
    u = (fx.Float32(amax) * (1.0 / 6.0)).bitcast(fx.Int32)
    return ((u >> 23) & 0xFF) + ((u & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0)) - 127


def _pow2(e):
    """2^e from exponent bits, e in [-126, 127]."""
    return ((fx.max(e + 127, 1)) << 23).bitcast(fx.Float32)


def mxfp4_round(vals, e):
    """fp32 values (an even count) rounded as aiter's MXFP4 quantizer does with shared exponent
    e, as bf16: the hardware E2M1 convert (round to nearest even, divides by 2^e) and back."""
    scale, out = _pow2(e), []
    for q in range_constexpr(len(vals) // 8 + (len(vals) % 8 > 0)):
        part = vals[8 * q: 8 * q + 8]
        word = fx.Int32(0)
        for sel in range_constexpr(len(part) // 2):
            word = fx.Int32(rocdl.cvt_scalef32_pk_fp4_f32(T.i32, word.ir_value(), fx.Float32(part[2 * sel]).ir_value(),
                                                          fx.Float32(part[2 * sel + 1]).ir_value(),
                                                          scale.ir_value(), sel))
        dec = fp4x8_to_bf16(word, scale)
        out += [dec[i] for i in range(len(part))]
    return out


def _w_offset(e, row, kt, k1, ni, rows_per_expert, tiles_per_row):
    """Byte offset of a lane's 16 B in a shuffled [E, rows, bytes] FP4 matrix: row tile `row`
    (16 rows), K tile kt (32 B), half k1, row ni of the tile."""
    return (((e * (rows_per_expert // 16) + row) * tiles_per_row + kt) * 2 + k1) * 256 + ni * 16


def _scale_offset(m, g, groups_padded):
    """Byte offset of scale (row m of the flattened [E * rows], group g) after e8m0_shuffle."""
    return (((((m // 32) * (groups_padded // 8) + g // 8) * 4 + g % 4) * 16 + m % 16) * 2 + (g // 4) % 2) * 2 + \
        (m // 16) % 2


def _row_sum16(x):
    """Sum over the 4 lanes l, l ^ 16, l ^ 32, l ^ 48 (the 4 K-chunks of a row in a wave load)."""
    x = fx.Float32(_permlane_swap_add_f32(x, rocdl.permlane16_swap))
    return fx.Float32(_permlane_swap_add_f32(x, rocdl.permlane32_swap))


# ===== Router and top-k =====


@device
def publish_f32(rs, tag, t, pair, lo, hi, valid):
    """emit_gemv publish for the router: rows 2 pair, 2 pair + 1 as two (f32 bits, tag) pairs,
    bf16-rounded (SGLang's gate is a bf16 linear)."""
    if valid:
        lo, hi = fx.Float32(lo.to(fx.BFloat16)), fx.Float32(hi.to(fx.BFloat16))
        buffer_ops.buffer_store(fx.Vector.from_elements([lo.bitcast(fx.Int32), tag, hi.bitcast(fx.Int32), tag],
                                                        fx.Int32), rs, (t * N_ROUTED + 2 * pair) * 2,
                                cache_modifier=CM_DEV)


@device
def emit_topk(logits_rs, tag, x_lds, w_sg_rs, smem):
    """Wave 0: the 10 largest of 512 router logits (lowest index on ties) and their weights,
    softmax over the 10 (= the renormalized softmax over 512). Wave 1: the shared expert's weight
    sigmoid(x . w_sg) from the raw x in x_lds. Writes smem.ids / smem.wts for slots 0..10."""
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    if wave == 0:
        # Logits are bf16-rounded, so a key (orderable bf16 bits << 16 | 511 - index) orders by
        # value, then lowest index: one int max reduction a round.
        words = poll_pairs(logits_rs, [lane * 8 + 2 * q for q in range(4)], tag)
        flip = lambda b: (b < 0).select(b ^ 0x7FFF0000, b)  # negative floats: reverse magnitude order
        keys = [flip(fx.Int32(w)) | (N_ROUTED - 1 - (lane * 8 + i)) for i, w in enumerate(words)]
        top, ids = [], []
        for r in range_constexpr(TOPK):
            best = keys[0]
            for i in range_constexpr(1, 8):
                best = fx.max(best, keys[i])
            m = fx.Int32(fx.coop.warp_reduce(best, fx.ReductionOp.MAX))
            top.append(flip(m & -65536).bitcast(fx.Float32))
            ids.append(N_ROUTED - 1 - (m & 0xFFFF))
            keys = [(k == m).select(fx.Int32(KEY_TAKEN), k) for k in keys]
        ex = [fx.math.exp(top[r] - top[0]) for r in range(TOPK)]
        total = ex[0]
        for r in range_constexpr(1, TOPK):
            total = total + ex[r]
        if lane == 0:
            for r in range_constexpr(TOPK):
                fx.ptr_store(ids[r], smem.ids.ptr + r)
                fx.ptr_store(ex[r] / total, smem.wts.ptr + r)
    if wave == 1:
        acc = fx.Float32(0.0)
        for i in range_constexpr(H // 64 // 8):  # 5 x 8 values a lane
            k = (lane + 64 * i) * 8
            xv = fx.Vector(fx.ptr_load(x_lds + k // 2, result_type=T.vec(4, T.i32))).bitcast(fx.BFloat16)
            wv = fx.Vector(buffer_ops.buffer_load(w_sg_rs, k // 2, vec_width=4, dtype=T.i32)).bitcast(fx.BFloat16)
            for q in range_constexpr(4):
                acc = fx.Float32(_dot2_f32_bf16(acc, _pair(xv, q), _pair(wv, q)))
        total = fx.Float32(_wave_reduce_add_f32(acc))
        if lane == 0:
            fx.ptr_store(fx.Int32(SHARED_ID), smem.ids.ptr + TOPK)
            fx.ptr_store(fx.Float32(1.0) / (fx.Float32(1.0) + fx.math.exp(-total)), smem.wts.ptr + TOPK)
    fx.gpu.barrier()


@device
def quantize_x_lds(x_lds):
    """MXFP4-round the bf16 x (H values) in x_lds in place: threads 2g, 2g + 1 own group g's
    halves."""
    tid = fx.Int32(fx.thread_idx.x)
    if tid < 2 * (H // GROUP):
        p = x_lds + tid * (GROUP // 4)
        vals = []
        for q in range_constexpr(GROUP // 16):
            v = fx.Vector(fx.ptr_load(p + 4 * q, result_type=T.vec(4, T.i32))).bitcast(fx.BFloat16).to(fx.Float32)
            vals += [v[i] for i in range(8)]
        amax = fx.math.absf(vals[0])
        for i in range_constexpr(1, GROUP // 2):
            amax = fx.max(amax, fx.math.absf(vals[i]))
        amax = fx.Float32(fx.coop.warp_reduce(amax, fx.ReductionOp.MAX, width=2))
        r = mxfp4_round(vals, mxfp4_exponent(amax))
        for q in range_constexpr(GROUP // 16):
            words = fx.Vector.from_elements([_pair_bits(r[8 * q + 2 * i], r[8 * q + 2 * i + 1]) for i in range(4)],
                                            fx.Int32)
            fx.ptr_store(words, p + 4 * q)
    fx.gpu.barrier()


def _pair_bits(lo, hi):
    """Two bf16 as one word (lo in the low half)."""
    return fx.Vector.from_elements([lo, hi], fx.BFloat16).bitcast(fx.Int32)[0]


# ===== Stage 1: gate / up of one 32-wide group of one slot's intermediate =====


def stage1_unit():
    """(slot, group) of this block's stage-1 unit; blocks NBLK - 220 .. NBLK - 1 own one each."""
    u = fx.Int32(fx.block_idx.x) - (256 - S1_BLOCKS)
    return u // S1_GROUPS, u % S1_GROUPS


def stage1_prefetch(w13_rs, s13_rs, smem):
    """Wave w loads 16 rows: gate rows (w < 2) or up rows (w >= 2) j0 + 16 (w % 2) .. + 16 of the
    block's group; 20 wave loads of 16 rows x 64 B. The expert is smem.ids[slot]; blocks without a
    stage-1 unit load out of bounds (no traffic). Returns (weight dwordx4s, scale dwords, bytes)."""
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    slot, grp = stage1_unit()
    active = fx.Int32(fx.block_idx.x) >= 256 - S1_BLOCKS
    expert = active.select(fx.ptr_load(smem.ids.ptr + fx.max(slot, 0)), fx.Int32(SHARED_ID + 1))
    ni, k1 = lane % 16, (lane // 16) % 2
    row_tile = (wave // 2) * (INTER_PAD // 16) + grp * 2 + wave % 2
    m = expert * W13_ROWS + row_tile * 16 + ni
    weights, scales = [], []
    for s in range(W13_BYTES // 64):
        kt = 2 * s + lane // 32
        off = _w_offset(expert, row_tile, kt, k1, ni, W13_ROWS, W13_BYTES // 32)
        weights.append(buffer_ops.buffer_load(w13_rs, off // 4, vec_width=4, dtype=T.i32, cache_modifier=CM_NT))
        scales.append(buffer_ops.buffer_load(s13_rs, _scale_offset(m, kt * 2 + k1, W13_GROUPS) // 4, vec_width=1,
                                             dtype=T.i32))
    byte = lambda m, g: _scale_offset(m, g, W13_GROUPS) % 4
    return weights, scales, [byte(m, 2 * (2 * s + lane // 32) + k1) for s in range(W13_BYTES // 64)]


@device
def emit_stage1(pre, h_rs, tag, x_lds, smem):
    """Dots of this block's 64 rows with the MXFP4-rounded x, silu(g) * u for its 32 values,
    MXFP4-rounded, published to h (pairs [slot * 320 + group * 16, + 16))."""
    weights, scales, sbyte = pre
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    slot, grp = stage1_unit()
    k1 = (lane // 16) % 2
    acc = fx.Float32(0.0)
    for s in range_constexpr(W13_BYTES // 64):
        kt = 2 * s + lane // 32
        scale = e8m0_f32(fx.Int32(scales[s]) >> (sbyte[s] * 8))
        w = fx.Vector(weights[s])
        xk = (kt * 2 + k1) * GROUP  # first x of this lane's group
        for q in range_constexpr(4):
            wv = fp4x8_to_bf16(w[q], scale)
            xv = fx.Vector(fx.ptr_load(x_lds + xk // 2 + 4 * q, result_type=T.vec(4, T.i32))).bitcast(fx.BFloat16)
            for p in range_constexpr(4):
                acc = fx.Float32(_dot2_f32_bf16(acc, _pair(xv, p), _pair(wv, p)))
    total = _row_sum16(acc)
    if lane < 16:
        fx.ptr_store(total, smem.rows.ptr + wave * 16 + lane)
    fx.gpu.barrier()
    if tid < 64:  # wave 0: value j = lane % 32 of the group (lanes 32..63 duplicate)
        j = lane % 32
        g = fx.ptr_load(smem.rows.ptr + j)  # gate rows: waves 0, 1 -> rows 0..31
        u = fx.ptr_load(smem.rows.ptr + 32 + j)  # up rows: waves 2, 3
        # aiter's stage-1 epilogue (act.py): hardware exp2 and rcp, then h is stored as bf16.
        e = fx.Float32(rocdl.exp2(T.f32, (g * fx.Float32(-LOG2E)).ir_value()))
        sig = fx.Float32(rocdl.rcp(T.f32, (fx.Float32(1.0) + e).ir_value()))
        hv = fx.Float32((g * sig * u).to(fx.BFloat16))
        amax = fx.Float32(fx.coop.warp_reduce(fx.math.absf(hv), fx.ReductionOp.MAX, width=32))
        r = mxfp4_round([hv, fx.Float32(0.0)], mxfp4_exponent(amax))[0]
        fx.ptr_store(fx.Float32(r), smem.rows.ptr + j)
    fx.gpu.barrier()
    if tid < GROUP // 2:
        lo, hi = fx.ptr_load(smem.rows.ptr + 2 * tid), fx.ptr_load(smem.rows.ptr + 2 * tid + 1)
        put_pair(h_rs, slot * (INTER // 2) + grp * (GROUP // 2) + tid, bf16x2(lo, hi), tag)
    fx.gpu.barrier()


# ===== Stage 2: down projections of all slots for 16 output rows =====


def stage2_prefetch(w2_rs, s2_rs, smem):
    """Wave w loads slots w, w + 4, w + 8 (< 11) for the block's 16 rows: 5 wave loads each (320
    real bytes a row). Expert ids come from smem.ids. Returns (weights, scale dwords, bytes)."""
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    rt = fx.Int32(fx.block_idx.x)
    active = rt < S2_BLOCKS  # the rest load out of bounds (no traffic)
    ni, k1 = lane % 16, (lane // 16) % 2
    weights, scales, sbyte = [], [], []
    for i in range(3):
        slot = fx.min(wave + 4 * i, SLOTS - 1)  # wave 3's third slot is a duplicate, weighted 0
        e = active.select(fx.ptr_load(smem.ids.ptr + slot), fx.Int32(SHARED_ID + 1))
        m = e * H + rt * 16 + ni
        for s in range(W2_REAL // 64):  # 5 wave loads: K tiles 0..9 hold the real bytes
            kt = 2 * s + lane // 32
            weights.append(buffer_ops.buffer_load(w2_rs, _w_offset(e, rt, kt, k1, ni, H, W2_BYTES // 32) // 4,
                                                  vec_width=4, dtype=T.i32, cache_modifier=CM_NT))
            g = kt * 2 + k1
            scales.append(buffer_ops.buffer_load(s2_rs, _scale_offset(m, g, W2_GROUPS) // 4, vec_width=1, dtype=T.i32))
            sbyte.append(_scale_offset(m, g, W2_GROUPS) % 4)
    return weights, scales, sbyte


@device
def poll_to_lds(rs, tag, lds, n_values):
    """A bf16 mailbox of n_values into LDS as pairs (word p = values 2p, 2p + 1). Thread t takes
    16 B chunks t, t + 256, ...; it spins on its first chunk, then fetches the rest at once."""
    tid = fx.Int32(fx.thread_idx.x)
    chunks = n_values // 4
    per = -(-chunks // 256)
    cs = [fx.min(tid + j * 256, chunks - 1) for j in range(per)]  # past the end: re-read the last
    words = poll_pairs(rs, [2 * cs[0]], tag)
    if const_expr(per > 1):
        words = words + poll_pairs(rs, [2 * c for c in cs[1:]], tag)
    for j in range_constexpr(per):
        fx.ptr_store(words[2 * j], lds + 2 * cs[j])
        fx.ptr_store(words[2 * j + 1], lds + 2 * cs[j] + 1)
    fx.gpu.barrier()


@device
def emit_stage2(pre, smem):
    """y for the block's 16 rows (fp32, in smem.rows[0..16)): per wave its slots in order,
    weighted; then the 4 wave sums in order."""
    weights, scales, sbyte = pre
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    k1 = (lane // 16) % 2
    part = fx.Float32(0.0)
    for i in range_constexpr(3):
        slot = fx.min(wave + 4 * i, SLOTS - 1)
        wk = (wave + 4 * i < SLOTS).select(fx.ptr_load(smem.wts.ptr + slot), fx.Float32(0.0))
        acc = fx.Float32(0.0)
        for s in range_constexpr(W2_REAL // 64):
            j = i * (W2_REAL // 64) + s
            kt = 2 * s + lane // 32
            scale = e8m0_f32(fx.Int32(scales[j]) >> (sbyte[j] * 8))
            w = fx.Vector(weights[j])
            xk = slot * INTER + (kt * 2 + k1) * GROUP
            for q in range_constexpr(4):
                wv = fp4x8_to_bf16(w[q], scale)
                xv = fx.Vector(fx.ptr_load(smem.h.ptr + xk // 2 + 4 * q, result_type=T.vec(4, T.i32))).bitcast(fx.BFloat16)
                for p in range_constexpr(4):
                    acc = fx.Float32(_dot2_f32_bf16(acc, _pair(xv, p), _pair(wv, p)))
        part = part + wk * _row_sum16(acc)
    if lane < 16:
        fx.ptr_store(part, smem.rows.ptr + 16 + wave * 16 + lane)
    fx.gpu.barrier()
    if tid < 16:
        r = smem.rows.ptr + 16 + tid
        y = fx.ptr_load(r) + fx.ptr_load(r + 16)
        y = y + fx.ptr_load(r + 32) + fx.ptr_load(r + 48)
        fx.ptr_store(y, smem.rows.ptr + tid)
    fx.gpu.barrier()
