"""gdn_verify: the gated delta rule for one value head over T draft tokens (MTP target verify).

The decode op (gdn_core_op.py) for T tokens in a row, with SGLang's verify contract
(GDNAttnBackend.forward_extend, target verify): it reads the conv and SSM state at the request's
slot, leaves them unchanged, and writes a snapshot after every draft token t into scratch row
`row`: the conv window [row, t, 10240, 3] (bf16, the last 3 conv inputs) and the SSM state
[row, t, 48, 128 (V), 128 (K)] (fp32). After verify SGLang copies the accepted step's snapshots
over the main state. Per-step math and rounding are the decode op's.

    gdn_core_prefetch     (shared) state, conv weights and window, norm weight
    emit_gdn_verify       poll T tokens, conv + windows, norms, gates; T recurrence steps; publish o
    gdn_verify_snapshots  recompute the T steps from the saved start state and store each through LDS

The SSM snapshots are recomputed after o is published, so their 4 x 64 KB of stores per head stay
off the critical path; the steps are deterministic, so the recomputed states are bit-identical.
Every head writes the windows of its v channels; the q, k windows only by head 3kh (they're equal).
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

import buffer_ops
from common import bf16x2, device, poll_pairs, poll_pairs_first_then_rest, put_pair
from gdn_core_op import (A0, B0, CONV_CH, D, EPS, IN_DIM, Q_SCALE, STAGE_STRIDE, V_HEADS, _bf16, _chunk, _load_f32,
                         _pair_sum, _pick_bf16, _sigmoid, _unpack)
from skinny_bf16 import _wave_reduce_add_f32
from timeline import mark

T_MAX = 4
MARKS = 4  # emit_gdn_verify: start, inputs + conv + norms, recurrence, o published; snapshots add one


@fx.struct
class VerifySmem:
    qkv: fx.Array[fx.Float32, T_MAX * 3 * D, 16]  # [t, q | k | v] conv + SiLU outputs (bf16-rounded)
    z: fx.Array[fx.Float32, T_MAX * D, 16]
    out: fx.Array[fx.Float32, T_MAX * D, 16]
    scal: fx.Array[fx.Float32, 64, 16]  # [rq t, rk t, decay t, beta t] x T, then [t, wave] sums at 16


def _poll_values(rs, indices, tag):
    """Elements of a bf16 mailbox, each from the 16 B chunk holding it, in one poll."""
    words = poll_pairs(rs, [(i // 4) * 2 for i in indices], tag)
    out = []
    for k, i in enumerate(indices):
        lo, hi = _unpack((i // 2 % 2 == 1).select(words[2 * k + 1], words[2 * k]))
        out.append((i % 2 == 1).select(hi, lo))
    return out


def _scal(smem, kind, t, tokens):
    """kind 0..3 = rsqrt(|q|^2), rsqrt(|k|^2), decay, beta of token t."""
    return smem.scal.ptr + kind * tokens + t


def _step(S, smem, t, tokens, v_row, half):
    """One decode step on this thread's 64 columns of row v (the decode op's statements)."""
    vec4 = lambda p: fx.Vector(fx.ptr_load(p, result_type=T.vec(4, T.f32)))
    base = smem.qkv.ptr + t * 3 * D
    rq, rk = fx.ptr_load(_scal(smem, 0, t, tokens)), fx.ptr_load(_scal(smem, 1, t, tokens))
    decay, beta = fx.ptr_load(_scal(smem, 2, t, tokens)), fx.ptr_load(_scal(smem, 3, t, tokens))
    kn = [vec4(base + D + half * 64 + 4 * i) * rk for i in range(16)]
    S = [S[i] * decay for i in range(16)]
    ks = fx.Float32(0.0)
    for i in range_constexpr(16):
        ks = ks + (S[i] * kn[i]).reduce(fx.ReductionOp.ADD)
    err = fx.ptr_load(base + 2 * D + v_row) - _pair_sum(ks)
    c = err * beta
    S = [S[i] + kn[i] * c for i in range(16)]
    qs = fx.Float32(0.0)
    for i in range_constexpr(16):
        qn = vec4(base + half * 64 + 4 * i) * rq * Q_SCALE
        qs = qs + (S[i] * qn).reduce(fx.ReductionOp.ADD)
    return S, _bf16(_pair_sum(qs))


@device
def emit_gdn_verify(head, pre, in_rs, tag, o_rs, win_rs, a_log_rs, dt_bias_rs, smem, tokens, trace, mark0, traced,
                    win_dedup=False):
    """in_rs: [T, 16480] mailbox; o_rs: [T, 6144] mailbox; win_rs: this request's window scratch,
    bf16: dense [T, 10240, 3], or with win_dedup SGLang's deduplicated [10240, T + 2], where step
    t's window is columns t .. t + 2 (consecutive windows overlap). Returns nothing: the snapshots
    are gdn_verify_snapshots'."""
    _, cw, cs, norm_word = pre
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    owner = head % 3 == 0
    v_row, half = tid // 2, tid % 2
    r, j, c0 = _chunk(head)
    mark(trace, 4, mark0, traced)

    # ---- 1. Poll T tokens; conv + SiLU over [old window | x_0 .. x_T-1]; windows; gates.
    if tid < 128:
        words = poll_pairs_first_then_rest(in_rs, [[(t * IN_DIM + c0) // 2] for t in range(tokens)], tag)
        xs = [[*_unpack(words[t][0]), *_unpack(words[t][1])] for t in range(tokens)]
        if tid < 96:
            w = [e for v in cw for e in v.bitcast(fx.BFloat16).to(fx.Float32)]  # w[4 * ch + tap]
            s = [e for v in cs for e in v.bitcast(fx.BFloat16).to(fx.Float32)]  # s[3 * ch + tap]
            seq = [[s[3 * e], s[3 * e + 1], s[3 * e + 2]] + [xs[t][e] for t in range(tokens)] for e in range(4)]
            for t in range_constexpr(tokens):
                for e in range_constexpr(4):
                    q = seq[e]
                    acc = w[4 * e] * q[t] + w[4 * e + 1] * q[t + 1] + w[4 * e + 2] * q[t + 2]
                    acc = acc + w[4 * e + 3] * q[t + 3]
                    fx.ptr_store(_bf16(acc * _sigmoid(acc)), smem.qkv.ptr + t * 3 * D + r * D + 4 * j + e)
            if (r == 2) | owner:
                if const_expr(win_dedup):
                    # Channel c's row is seq[1 .. T + 2]; 4 channels are contiguous: 4 (T + 2) bf16.
                    vals = [seq[e][k + 1] for e in range(4) for k in range(tokens + 2)]
                    packed = [bf16x2(vals[2 * q], vals[2 * q + 1]) for q in range(len(vals) // 2)]
                    base = c0 * (tokens + 2) // 2
                    for q in range_constexpr(len(packed) // 2):
                        buffer_ops.buffer_store(fx.Vector.from_elements(packed[2 * q: 2 * q + 2], fx.Int32), win_rs,
                                                base + 2 * q)
                else:
                    for t in range_constexpr(tokens):
                        win = [v for e in range(4) for v in (seq[e][t + 1], seq[e][t + 2], seq[e][t + 3])]
                        for q in range_constexpr(3):
                            pair = fx.Vector.from_elements([bf16x2(win[4 * q], win[4 * q + 1]),
                                                            bf16x2(win[4 * q + 2], win[4 * q + 3])], fx.Int32)
                            buffer_ops.buffer_store(pair, win_rs, (t * CONV_CH + c0) * 3 // 2 + 2 * q)
        else:
            for t in range_constexpr(tokens):
                for e in range_constexpr(4):
                    fx.ptr_store(xs[t][e], smem.z.ptr + t * D + 4 * j + e)
    if (tid >= 128) & (tid < 128 + tokens):
        t = tid - 128
        a, b = _poll_values(in_rs, [t * IN_DIM + A0 + head, t * IN_DIM + B0 + head], tag)
        u = a + _load_f32(dt_bias_rs, head)
        softplus = (u > 20.0).select(u, fx.math.log1p(fx.math.exp(u)))  # torch's threshold
        fx.ptr_store(fx.math.exp(-fx.math.exp(_load_f32(a_log_rs, head)) * softplus), _scal(smem, 2, t, tokens))
        fx.ptr_store(_bf16(_sigmoid(b)), _scal(smem, 3, t, tokens))
    fx.gpu.barrier()

    # ---- 2. L2 norms of q and k per token: wave 0 does q, wave 1 does k.
    if wave < 2:
        for t in range_constexpr(tokens):
            p = smem.qkv.ptr + t * 3 * D + wave * D + 2 * lane
            q0, q1 = fx.ptr_load(p), fx.ptr_load(p + 1)
            total = fx.Float32(_wave_reduce_add_f32(q0 * q0 + q1 * q1))
            if lane == 0:
                fx.ptr_store(fx.math.rsqrt(total + EPS), smem.scal.ptr + wave * tokens + t)
    fx.gpu.barrier()
    mark(trace, 4, mark0 + 1, traced)

    # ---- 3. T recurrence steps from the slot's state.
    S, o = pre[0], []
    for t in range_constexpr(tokens):
        S, o_t = _step(S, smem, t, tokens, v_row, half)
        o.append(o_t)
    mark(trace, 4, mark0 + 2, traced)

    # ---- 4. Per token: per-head RMSNorm (plain weight), sigmoid(z) gate; publish o.
    for t in range_constexpr(tokens):
        sq = fx.Float32(_wave_reduce_add_f32((half == 0).select(o[t] * o[t], fx.Float32(0.0))))
        if lane == 0:
            fx.ptr_store(sq, smem.scal.ptr + 16 + t * 4 + wave)
    fx.gpu.barrier()
    norm_w = _pick_bf16(norm_word, v_row)
    for t in range_constexpr(tokens):
        w4 = smem.scal.ptr + 16 + t * 4
        total = fx.ptr_load(w4) + fx.ptr_load(w4 + 1)
        total = total + fx.ptr_load(w4 + 2) + fx.ptr_load(w4 + 3)
        gated = _bf16(o[t] * fx.math.rsqrt(total / D + EPS) * norm_w * _sigmoid(fx.ptr_load(smem.z.ptr + t * D + v_row)))
        if half == 0:
            fx.ptr_store(gated, smem.out.ptr + t * D + v_row)
    fx.gpu.barrier()
    if tid < D // 2:
        for t in range_constexpr(tokens):
            lo, hi = fx.ptr_load(smem.out.ptr + t * D + 2 * tid), fx.ptr_load(smem.out.ptr + t * D + 2 * tid + 1)
            put_pair(o_rs, t * (V_HEADS * D // 2) + head * (D // 2) + tid, bf16x2(lo, hi), tag)
    mark(trace, 4, mark0 + 3, traced)


@device
def gdn_verify_snapshots(head, pre, ssm_snap_rs, smem, stage, tokens, trace, mark_end, traced):
    """Recompute the T steps from the start state in `pre` and store the state after each as
    ssm_snap_rs[t, head] ([T, 48, 128, 128] fp32 scratch of this request), through LDS (`stage`,
    D x STAGE_STRIDE floats) so each wave store is one contiguous 1 KB."""
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    v_row, half = tid // 2, tid % 2
    S = pre[0]
    for t in range_constexpr(tokens):
        S, _ = _step(S, smem, t, tokens, v_row, half)
        for i in range_constexpr(16):
            fx.ptr_store(S[i], stage + v_row * STAGE_STRIDE + half * 64 + 4 * i)
        fx.gpu.barrier()
        for c in range_constexpr(16):
            row, col = 2 * (16 * wave + c) + lane // 32, (lane % 32) * 4
            v = fx.Vector(fx.ptr_load(stage + row * STAGE_STRIDE + col, result_type=T.vec(4, T.f32)))
            buffer_ops.buffer_store(v, ssm_snap_rs, ((t * V_HEADS + head) * D + row) * D + col)
        fx.gpu.barrier()
    mark(trace, 4, mark_end, traced)
