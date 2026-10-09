"""gdn_core: the gated delta rule for one value head and one decode token, as a monokernel op.

Input is the in_proj mailbox: [q 2048 | k 2048 | v 6144 | z 6144 | b 48 | a 48] bf16, the output
order of qwen38_ref.gdn's four in_proj GEMVs. Value head h uses q, k of key head h // 3, its own
128 v and 128 z channels, b[h] and a[h]. A_log and dt_bias are fp32, as SGLang keeps them. Output is o[h] (128 bf16) in a [48 * 128] mailbox,
gated and normalized, ready for out_proj. Math and rounding follow qwen38_ref.gdn.

One block of 256 threads per head:
    thread t < 96      one 4-channel chunk of [q | k | v]: conv, SiLU, conv state
    thread 96..127     one 4-channel chunk of z
    thread 128         gates: a, b -> decay, beta
    every thread       row v = t // 2, half = t % 2 of the state: S[h, v, 64 * half : 64 * half + 64]

The conv state of q and k is shared by heads 3kh, 3kh + 1, 3kh + 2. Only head 3kh writes it, and
only after the other two publish an ack saying they no longer need the old values.

Three calls, so a megakernel can place them around other ops: gdn_core_prefetch issues the loads,
emit_gdn_core computes and publishes o, gdn_core_write_state stores the state. Marks mark0 ..
mark0 + 3 of emit_gdn_core: start, inputs + conv + norms done, recurrence done, o published; the
write marks `mark_end` when done.
"""

import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

import buffer_ops
from common import bf16x2, device, poll_pairs, put_pair
from skinny_bf16 import _DPP_SWAP_1, _dpp_add_f32, _wave_reduce_add_f32
from timeline import mark

QK_HEADS, V_HEADS, D, CONV_K = 16, 48, 128, 4
Q0, K0, V0 = 0, QK_HEADS * D, 2 * QK_HEADS * D  # offsets in the in_proj mailbox, in values
Z0 = V0 + V_HEADS * D
B0 = Z0 + V_HEADS * D
A0 = B0 + V_HEADS
IN_DIM = A0 + V_HEADS  # 16480
CONV_CH = Z0  # 10240 conv channels: q | k | v
EPS = 1e-6
Q_SCALE = D**-0.5
THREADS = 256
STAGE_STRIDE = D + 4  # padded LDS row: the 32 rows of a wave's write hit different banks
MARKS = 4  # emit_gdn_core's; gdn_core_write_state adds one more


@fx.struct
class GdnSmem:
    xin: fx.Array[fx.Float32, 3 * D, 16]  # this step's conv inputs, q | k | v
    qkv: fx.Array[fx.Float32, 3 * D, 16]  # conv + SiLU outputs, rounded to bf16
    z: fx.Array[fx.Float32, D, 16]
    out: fx.Array[fx.Float32, D, 16]
    scal: fx.Array[fx.Float32, 8, 16]  # rsqrt(|q|^2), rsqrt(|k|^2), decay, beta, 4 wave sums
    stage: fx.Array[fx.Float32, D * STAGE_STRIDE, 16]  # S on its way to HBM, rows padded


def ack_pairs():
    """Size of the ack mailbox: 2 pairs per key head (its two non-owner value heads)."""
    return 2 * QK_HEADS


def _unpack(word):
    """bf16x2 word -> (low f32, high f32)."""
    return (word << 16).bitcast(fx.Float32), (word & -65536).bitcast(fx.Float32)


def _bf16(x):
    return fx.Float32(fx.Float32(x).to(fx.BFloat16))


def _sigmoid(x):
    return fx.Float32(1.0) / (fx.Float32(1.0) + fx.math.exp(-x))


def _load_bf16_word(rs, i):
    """The dword holding element i of a bf16 buffer; issuing it doesn't wait for the data."""
    return fx.Int32(buffer_ops.buffer_load(rs, i // 2, vec_width=1, dtype=T.i32))


def _pick_bf16(word, i):
    lo, hi = _unpack(word)
    return (i % 2 == 1).select(hi, lo)


def _load_f32(rs, i):
    return fx.Float32(buffer_ops.buffer_load(rs, i, vec_width=1, dtype=T.f32))


def _load_bf16(rs, i):
    """Element i of a bf16 buffer, as f32 (waits for the load)."""
    return _pick_bf16(_load_bf16_word(rs, i), i)


def _poll_value(rs, i, tag):
    """Element i of a bf16 mailbox, once its pair carries `tag` (polls the 16 B chunk holding it)."""
    words = poll_pairs(rs, [(i // 4) * 2], tag)
    lo, hi = _unpack((i // 2 % 2 == 1).select(words[1], words[0]))
    return (i % 2 == 1).select(hi, lo)


def _pair_sum(x):
    """x + x of lane ^ 1: the two halves of a state row are neighbouring lanes."""
    return fx.Float32(_dpp_add_f32(x, _DPP_SWAP_1))


def _chunk(head):
    """This thread's conv chunk: region r (q, k, v, z), channels c0 .. c0 + 3. Threads >= 96 point
    at z or past it, and heads >= V_HEADS at channel CONV_CH; their conv loads are out of bounds."""
    tid = fx.Int32(fx.thread_idx.x)
    r, j, kh = tid // 32, tid % 32, head // 3
    base = (r == 0).select(Q0 + kh * D, (r == 1).select(K0 + kh * D, (r == 2).select(V0 + head * D, Z0 + head * D)))
    return r, j, (head < V_HEADS).select(base, CONV_CH) + 4 * j


def _state_offset(head):
    """Thread (v, half) holds S[head, v, 64 half : 64 half + 64]: row v = t // 2, half = t % 2."""
    tid = fx.Int32(fx.thread_idx.x)
    return (head * D + tid // 2) * D + (tid % 2) * (D // 2)


def gdn_core_prefetch(head, conv_w_rs, conv_rs, ssm_rs, norm_rs):
    """Issue every load gdn_core needs that doesn't depend on this step's input; returns the
    in-flight values for emit_gdn_core. Call it as early as possible: the 64 KB state read takes
    about 2 us on one CU. For head >= V_HEADS every big load is out of bounds and costs nothing."""
    _, _, c0 = _chunk(head)
    s_off = _state_offset(head)
    S = [fx.Vector(buffer_ops.buffer_load(ssm_rs, s_off + 4 * i, vec_width=4, dtype=T.f32)) for i in range(16)]
    # conv1d [10240, 4] bf16: 4 channels = 8 dwords; conv state [10240, 3] bf16: 4 channels = 6 dwords.
    cw = [fx.Vector(buffer_ops.buffer_load(conv_w_rs, c0 * 2 + 4 * q, vec_width=4, dtype=T.i32)) for q in range(2)]
    cs = [fx.Vector(buffer_ops.buffer_load(conv_rs, c0 * 3 // 2 + 2 * q, vec_width=2, dtype=T.i32)) for q in range(3)]
    norm_word = _load_bf16_word(norm_rs, fx.Int32(fx.thread_idx.x) // 2)  # unpacked at its use
    return S, cw, cs, norm_word


@device
def emit_gdn_core(head, pre, in_rs, in_tag, out_rs, out_tag, ack_rs, a_log_rs, dt_bias_rs, smem, trace, mark0,
                  traced):
    """`pre` is gdn_core_prefetch(head, ...); returns the new S for gdn_core_write_state."""
    S, cw, cs, norm_word = pre
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    kh, owner = head // 3, head % 3 == 0
    waves = THREADS // 64
    v_row, half = tid // 2, tid % 2
    r, j, c0 = _chunk(head)
    mark(trace, waves, mark0, traced)

    # ---- 1. Poll the inputs; conv + SiLU for q, k, v; gates. Branches only write LDS.
    if tid < 128:
        words = poll_pairs(in_rs, [c0 // 2], in_tag)
        x = [*_unpack(words[0]), *_unpack(words[1])]
        if tid < 96:
            w = [e for v in cw for e in v.bitcast(fx.BFloat16).to(fx.Float32)]  # w[4 * ch + tap]
            s = [e for v in cs for e in v.bitcast(fx.BFloat16).to(fx.Float32)]  # s[3 * ch + tap]
            for e in range_constexpr(4):
                acc = w[4 * e] * s[3 * e] + w[4 * e + 1] * s[3 * e + 1] + w[4 * e + 2] * s[3 * e + 2]
                acc = acc + w[4 * e + 3] * x[e]
                fx.ptr_store(_bf16(acc * _sigmoid(acc)), smem.qkv.ptr + r * D + 4 * j + e)
                fx.ptr_store(x[e], smem.xin.ptr + r * D + 4 * j + e)
        else:
            for e in range_constexpr(4):
                fx.ptr_store(x[e], smem.z.ptr + 4 * j + e)
    if tid == 128:
        a = _poll_value(in_rs, A0 + head, in_tag)
        b = _poll_value(in_rs, B0 + head, in_tag)
        u = a + _load_f32(dt_bias_rs, head)
        softplus = (u > 20.0).select(u, fx.math.log1p(fx.math.exp(u)))  # torch's threshold
        fx.ptr_store(fx.math.exp(-fx.math.exp(_load_f32(a_log_rs, head)) * softplus), smem.scal.ptr + 2)
        fx.ptr_store(_bf16(_sigmoid(b)), smem.scal.ptr + 3)
    fx.gpu.barrier()
    # The old conv state has been consumed (the conv used its values), so a non-owner can release it.
    if (tid == 0) & (head % 3 != 0):
        put_pair(ack_rs, 2 * kh + head % 3 - 1, 0, out_tag)

    # ---- 2. L2 norms of q and k: wave 0 does q, wave 1 does k, 2 values per lane.
    if wave < 2:
        p = smem.qkv.ptr + wave * D + 2 * lane
        q0, q1 = fx.ptr_load(p), fx.ptr_load(p + 1)
        total = fx.Float32(_wave_reduce_add_f32(q0 * q0 + q1 * q1))
        if lane == 0:
            fx.ptr_store(fx.math.rsqrt(total + EPS), smem.scal.ptr + wave)
    fx.gpu.barrier()
    mark(trace, waves, mark0 + 1, traced)

    # ---- 3. The recurrence on this thread's 64 columns of row v:
    #   S *= decay;  e = v - S k;  S += beta e k;  o = S q   (row dots summed over the lane pair)
    rq, rk = fx.ptr_load(smem.scal.ptr), fx.ptr_load(smem.scal.ptr + 1)
    decay, beta = fx.ptr_load(smem.scal.ptr + 2), fx.ptr_load(smem.scal.ptr + 3)
    vec4 = lambda p: fx.Vector(fx.ptr_load(p, result_type=T.vec(4, T.f32)))
    kn = [vec4(smem.qkv.ptr + D + half * 64 + 4 * i) * rk for i in range(16)]
    S = [S[i] * decay for i in range(16)]
    ks = fx.Float32(0.0)
    for i in range_constexpr(16):
        ks = ks + (S[i] * kn[i]).reduce(fx.ReductionOp.ADD)
    err = fx.ptr_load(smem.qkv.ptr + 2 * D + v_row) - _pair_sum(ks)
    c = err * beta
    S = [S[i] + kn[i] * c for i in range(16)]
    qs = fx.Float32(0.0)
    for i in range_constexpr(16):
        qn = vec4(smem.qkv.ptr + half * 64 + 4 * i) * rq * Q_SCALE
        qs = qs + (S[i] * qn).reduce(fx.ReductionOp.ADD)
    o = _bf16(_pair_sum(qs))
    mark(trace, waves, mark0 + 2, traced)

    # ---- 4. Per-head RMSNorm (plain weight), sigmoid(z) gate, publish o[head].
    sq = fx.Float32(_wave_reduce_add_f32((half == 0).select(o * o, fx.Float32(0.0))))
    if lane == 0:
        fx.ptr_store(sq, smem.scal.ptr + 4 + wave)
    fx.gpu.barrier()
    total = fx.ptr_load(smem.scal.ptr + 4) + fx.ptr_load(smem.scal.ptr + 5)
    total = total + fx.ptr_load(smem.scal.ptr + 6) + fx.ptr_load(smem.scal.ptr + 7)
    norm_w = _pick_bf16(norm_word, v_row)
    gated = _bf16(o * fx.math.rsqrt(total / D + EPS) * norm_w * _sigmoid(fx.ptr_load(smem.z.ptr + v_row)))
    if half == 0:
        fx.ptr_store(gated, smem.out.ptr + v_row)
    fx.gpu.barrier()
    if tid < D // 2:
        lo, hi = fx.ptr_load(smem.out.ptr + 2 * tid), fx.ptr_load(smem.out.ptr + 2 * tid + 1)
        put_pair(out_rs, head * (D // 2) + tid, bf16x2(lo, hi), out_tag)
    mark(trace, waves, mark0 + 3, traced)

    return S


@device
def gdn_core_write_state(head, S, pre, ack_rs, tag, conv_rs, ssm_rs, smem, trace, mark_end, traced):
    """Store emit_gdn_core's new S and the new conv state. Nothing in the same launch reads them, so
    call it last: on CDNA a load's wait also waits for earlier stores, so 64 KB of stores issued
    earlier would delay every later poll of this block. smem.xin must still hold this step's input.
    The q, k conv state is written only by its owner head, after both acks for this tag."""
    _, _, cs, _ = pre
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    kh, owner = head // 3, head % 3 == 0
    r, j, c0 = _chunk(head)
    # The ack poll goes first: a wait on a load also waits for every older store.
    if (tid < 64) & owner:
        poll_pairs(ack_rs, [2 * kh], tag)
    # S through LDS, so each wave store is one contiguous 1 KB (2 rows): straight from the
    # compute layout, a store puts 16 B in each of 64 rows and the CU issues them at ~20 GB/s.
    v_row, half = tid // 2, tid % 2
    for i in range_constexpr(16):
        fx.ptr_store(S[i], smem.stage.ptr + v_row * STAGE_STRIDE + half * 64 + 4 * i)
    fx.gpu.barrier()
    for c in range_constexpr(16):
        row, col = 2 * (16 * wave + c) + lane // 32, (lane % 32) * 4
        v = fx.Vector(fx.ptr_load(smem.stage.ptr + row * STAGE_STRIDE + col, result_type=T.vec(4, T.f32)))
        buffer_ops.buffer_store(v, ssm_rs, (head * D + row) * D + col)
    if tid < 96:
        if (r == 2) | owner:
            # New state per channel: taps 1, 2 of the old one, then this step's input.
            s = [e for v in cs for e in v.bitcast(fx.BFloat16).to(fx.Float32)]
            x = [fx.ptr_load(smem.xin.ptr + r * D + 4 * j + e) for e in range(4)]
            new = [t for e in range(4) for t in (s[3 * e + 1], s[3 * e + 2], x[e])]
            for q in range_constexpr(3):
                pair = fx.Vector.from_elements([bf16x2(new[4 * q], new[4 * q + 1]),
                                                bf16x2(new[4 * q + 2], new[4 * q + 3])], fx.Int32)
                buffer_ops.buffer_store(pair, conv_rs, c0 * 3 // 2 + 2 * q)
    mark(trace, THREADS // 64, mark_end, traced)
