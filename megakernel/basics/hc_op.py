"""The gated residual (hyper-connection) around a stage, as monokernel ops. Math and rounding
follow qwen38_ref.hc_mix and hc_combine; R is 4 streams of H = 2560 bf16.

Read:   n = bf16(x * rsqrt(mean_stream(x^2) + eps) * (1 + w))     hc_norm: every block, into LDS
        low = bf16(silu(W_down n / 4))       [320]                 hc_down: emit_gemv + publish_low
        x = bf16(mean_s sigmoid(W_up[s H + j] . low) * n[s H + j])  hc_up: 10 outputs per block
Write:  c = 2 sigmoid(W_inj n / 4)           [4]                   inject: on the blocks that write R
        R'[s H + j] = bf16(R + bf16(y[j]) c[s])                    publish_combine, out_proj's epilogue

Every block computes n itself (20 KB of R) rather than receiving it: a handoff costs more.
"""

import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

import buffer_ops
from common import CM_DEV, bf16x2, device, poll_pairs, poll_pairs_first_then_rest, put_pair
from skinny_bf16 import _dot2_f32_bf16, _pair, _wave_reduce_add_f32

S, H, LOW = 4, 2560, 320
HC = S * H  # 10240
EPS = 1e-6
UP_PER_BLOCK = 10  # H / NBLK outputs of x per block
UP_SLOTS = 3  # outputs per wave: wave w takes w, w + 4, w + 8 (< 10)
NBLK = 256
INJ_STEPS = HC // 512  # 20 x 16 B loads per lane for one W_inj row


@fx.struct
class HcSmem:
    n: fx.Array[fx.Int32, HC // 2, 16]  # n as bf16 pairs: word p = elements 2p, 2p + 1
    red: fx.Array[fx.Float32, 64, 16]  # [wave, token, stream] partial sums of squares
    x: fx.Array[fx.Float32, 64, 16]  # [token, output] this block's hc_up outputs
    c: fx.Array[fx.Float32, 16, 16]  # [token, stream] inject coefficients


def _unpack(word):
    return (word << 16).bitcast(fx.Float32), (word & -65536).bitcast(fx.Float32)


def _bf16(x):
    return fx.Float32(fx.Float32(x).to(fx.BFloat16))


def _sigmoid(x):
    return fx.Float32(1.0) / (fx.Float32(1.0) + fx.math.exp(-x))


# ===== hc_norm =====


def hc_norm_load(r_rs, w_rs, tokens=1):
    """Issue the loads hc_norm needs: thread t takes 16 B chunks t + 256 i (i < 5) of each token's
    R row and of w. Returns ([R chunks of token 0], ...), w chunks."""
    tid = fx.Int32(fx.thread_idx.x)
    load = lambda rs, c: fx.Vector(buffer_ops.buffer_load(rs, c * 4, vec_width=4, dtype=T.i32))
    chunks = [tid + 256 * i for i in range(HC // 8 // 256)]
    rows = [[load(r_rs, t * (HC // 8) + c) for c in chunks] for t in range(tokens)]
    return rows, [load(w_rs, c) for c in chunks]


@device
def emit_hc_norm(loaded, smem, n_ptr=None):
    """n = bf16(x * rstd_stream * (1 + w)) per token into n_ptr ([T, HC / 2] words; default
    smem.n). 320 chunks of 8 make one stream."""
    rows, wv = loaded
    tokens = len(rows)
    n_ptr = smem.n.ptr if n_ptr is None else n_ptr
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    chunks = [tid + 256 * i for i in range(len(wv))]
    for t in range_constexpr(tokens):
        xs = [v.bitcast(fx.BFloat16).to(fx.Float32) for v in rows[t]]
        part = [fx.Float32(0.0) for _ in range(S)]
        for i in range_constexpr(len(wv)):
            sq = (xs[i] * xs[i]).reduce(fx.ReductionOp.ADD)
            for s in range_constexpr(S):
                part[s] = part[s] + (chunks[i] // (H // 8) == s).select(sq, fx.Float32(0.0))
        for s in range_constexpr(S):
            total = fx.Float32(_wave_reduce_add_f32(part[s]))
            if lane == 0:
                fx.ptr_store(total, smem.red.ptr + (wave * tokens + t) * S + s)
    fx.gpu.barrier()
    for t in range_constexpr(tokens):
        xs = [v.bitcast(fx.BFloat16).to(fx.Float32) for v in rows[t]]
        red = lambda w, s: fx.ptr_load(smem.red.ptr + (w * tokens + t) * S + s)
        rstd = []
        for s in range_constexpr(S):
            total = red(0, s) + red(1, s)
            total = total + red(2, s) + red(3, s)
            rstd.append(fx.math.rsqrt(total / H + EPS))
        for i in range_constexpr(len(wv)):
            st = chunks[i] // (H // 8)
            r = (st == 0).select(rstd[0], (st == 1).select(rstd[1], (st == 2).select(rstd[2], rstd[3])))
            w = wv[i].bitcast(fx.BFloat16).to(fx.Float32)
            n = [(xs[i][e] * r) * (fx.Float32(1.0) + w[e]) for e in range(8)]
            words = fx.Vector.from_elements([bf16x2(n[2 * q], n[2 * q + 1]) for q in range(4)], fx.Int32)
            fx.ptr_store(words, n_ptr + t * (HC // 2) + chunks[i] * 4)
    fx.gpu.barrier()


# ===== hc_down epilogue =====


@device
def publish_low(low_rs, tag, t, pair, lo, hi, valid):
    """emit_gemv publish for hc_down: low[t] = bf16(silu(acc / 4))."""
    if valid:
        a, b = lo * 0.25, hi * 0.25
        put_pair(low_rs, t * (LOW // 2) + pair, bf16x2(a * _sigmoid(a), b * _sigmoid(b)), tag)


# ===== hc_up =====


def _up_row(j_local, s):
    bid = fx.Int32(fx.block_idx.x)
    return s * H + bid * UP_PER_BLOCK + j_local


def hc_up_prefetch(w_up_rs):
    """Lane l < 40 loads 8 bf16 of each of this wave's 12 W_up rows ([HC, 320] bf16, 160 dwords a
    row). Lanes >= 40 and the missing third output of waves 2, 3 load out of bounds (zeros)."""
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    loads = []
    for m in range(UP_SLOTS):
        j = wave + 4 * m
        for s in range(S):
            row = ((j < UP_PER_BLOCK) & (lane < LOW // 8)).select(_up_row(j, s), HC)
            loads.append(buffer_ops.buffer_load(w_up_rs, row * (LOW // 2) + lane * 4, vec_width=4, dtype=T.i32))
    return loads


@device
def emit_hc_up(inflight, low_rs, low_tag, x_rs, x_tag, smem, tokens=1, n_ptr=None):
    """x for this block's 10 outputs of each token; polls low ([T, 320]), reads n ([T, HC / 2]
    words at n_ptr, default smem.n)."""
    n_ptr = smem.n.ptr if n_ptr is None else n_ptr
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    p = fx.min(lane, LOW // 8 - 1) * 4  # this lane's 8 low values are pairs p .. p + 3
    words = poll_pairs_first_then_rest(low_rs, [[t * (LOW // 2) + p + h for h in (0, 2)] for t in range(tokens)],
                                       low_tag)
    low = [fx.Vector.from_elements(words[t], fx.Int32).bitcast(fx.BFloat16) for t in range(tokens)]
    for m in range_constexpr(UP_SLOTS):
        j = wave + 4 * m
        acc = [fx.Float32(0.0) for _ in range(tokens)]
        for s in range_constexpr(S):
            wv = fx.Vector(inflight[m * S + s]).bitcast(fx.BFloat16)
            row = _up_row(j, s)
            for t in range_constexpr(tokens):
                d = fx.Float32(0.0)
                for q in range_constexpr(4):
                    d = fx.Float32(_dot2_f32_bf16(d, _pair(low[t], q), _pair(wv, q)))
                d = fx.Float32(_wave_reduce_add_f32(d))
                n_lo, n_hi = _unpack(fx.ptr_load(n_ptr + (t * HC + row) // 2))
                acc[t] = acc[t] + _sigmoid(d) * (row % 2 == 1).select(n_hi, n_lo)
        for t in range_constexpr(tokens):
            if (lane == 0) & (j < UP_PER_BLOCK):
                fx.ptr_store(_bf16(acc[t] * 0.25), smem.x.ptr + t * UP_PER_BLOCK + j)
    fx.gpu.barrier()
    half = UP_PER_BLOCK // 2
    if tid < tokens * half:
        bid = fx.Int32(fx.block_idx.x)
        t, q = tid // half, tid % half
        lo, hi = fx.ptr_load(smem.x.ptr + t * UP_PER_BLOCK + 2 * q), fx.ptr_load(smem.x.ptr + t * UP_PER_BLOCK + 2 * q + 1)
        put_pair(x_rs, t * (H // 2) + bid * half + q, bf16x2(lo, hi), x_tag)


# ===== write: inject coefficients and the combine =====


def inject_prefetch(w_inj_rs, active):
    """Wave s loads row s of W_inj ([4, HC] bf16): 20 x 16 B per lane; out of bounds where not active."""
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    row = active.select(wave, S)
    return [buffer_ops.buffer_load(w_inj_rs, row * (HC // 2) + k * 256 + lane * 4, vec_width=4, dtype=T.i32)
            for k in range(INJ_STEPS)]


@device
def emit_inject(inflight, smem, tokens=1, n_ptr=None):
    """c[t][s] = 2 sigmoid(W_inj[s] . n_t / 4) into smem.c (wave s). Read by publish_combine after
    emit_gemv's barriers."""
    n_ptr = smem.n.ptr if n_ptr is None else n_ptr
    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    for t in range_constexpr(tokens):
        acc = [fx.Float32(0.0) for _ in range(4)]
        for k in range_constexpr(INJ_STEPS):
            wv = fx.Vector(inflight[k]).bitcast(fx.BFloat16)
            p = n_ptr + t * (HC // 2) + k * 256 + lane * 4
            nv = fx.Vector(fx.ptr_load(p, result_type=T.vec(4, T.i32))).bitcast(fx.BFloat16)
            for q in range_constexpr(4):
                acc[q] = fx.Float32(_dot2_f32_bf16(acc[q], _pair(nv, q), _pair(wv, q)))
        total = fx.Float32(_wave_reduce_add_f32((acc[0] + acc[1]) + (acc[2] + acc[3])))
        if lane == 0:
            fx.ptr_store(fx.Float32(2.0) * _sigmoid(total * 0.25), smem.c.ptr + t * S + wave)


@device
def publish_combine(r_rs, out_rs, c_ptr, t, pair, lo, hi, valid):
    """emit_gemv publish for the stage's last GEMV (N = H): rows 2 pair, 2 pair + 1 of y for token
    t, written into each stream: R'[t, s H + j] = bf16(R[t, s H + j] + bf16(y[j]) c[t][s]). Plain
    stores: R' is read after this launch."""
    if valid:
        y_lo, y_hi = _bf16(lo), _bf16(hi)
        for s in range_constexpr(S):
            c = fx.ptr_load(c_ptr + t * S + s)
            off = t * (HC // 2) + s * (H // 2) + pair
            word = fx.Int32(buffer_ops.buffer_load(r_rs, off, vec_width=1, dtype=T.i32))
            r_lo, r_hi = _unpack(word)
            buffer_ops.buffer_store(bf16x2(r_lo + y_lo * c, r_hi + y_hi * c), out_rs, off)


# ===== launch epoch =====


@device
def next_epoch_tag(sync_rs):
    """This launch's tag: sync[0] + 1, where sync[0] is the last launch's tag (0 at start)."""
    return fx.Int32(buffer_ops.buffer_load(sync_rs, 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)) + 1


@device
def finish_epoch(sync, tag):
    """The last block to finish stores this launch's tag in sync[0] and resets the block count in
    sync[1]. Every block read sync[0] at its start, before its own count, so none can see the new
    value; the next launch starts after this one ends and sees it. No per-launch host argument, so
    the launch can sit in a graph."""
    fx.gpu.barrier()
    if fx.Int32(fx.thread_idx.x) == 0:
        agent = fx.rocdl.SyncScope.Agent
        done = fx.atomic_add(fx.add_offset(sync.iter, 1), fx.Int32(1), syncscope=agent,
                             ordering=fx.AtomicOrdering.AcqRel)
        if fx.Int32(done) == NBLK - 1:
            fx.generic_store(fx.add_offset(sync.iter, 1), fx.Int32(0), memory_order=fx.AtomicOrdering.Monotonic,
                             syncscope=agent)
            fx.generic_store(fx.add_offset(sync.iter, 0), tag, memory_order=fx.AtomicOrdering.Release,
                             syncscope=agent)


# ===== The second stage of a layer: R' arrives as a mailbox =====


@device
def publish_combine_mailbox(r_rs, mb_rs, c_ptr, tag, t, pair, lo, hi, valid):
    """publish_combine, but R' goes to a [HC] mailbox (pairs) for a later stage of the same launch."""
    if valid:
        y_lo, y_hi = _bf16(lo), _bf16(hi)
        for s in range_constexpr(S):
            c = fx.ptr_load(c_ptr + t * S + s)
            off = t * (HC // 2) + s * (H // 2) + pair
            r_lo, r_hi = _unpack(fx.Int32(buffer_ops.buffer_load(r_rs, off, vec_width=1, dtype=T.i32)))
            put_pair(mb_rs, off, bf16x2(r_lo + y_lo * c, r_hi + y_hi * c), tag)


@device
def hc_norm_poll(r_mb_rs, tag, w_rs):
    """hc_norm_load's result from an R mailbox ([HC] pairs) instead of a plain tensor. Spins on one
    chunk a thread first, then fetches the rest: spinning on all reloads them on every retry."""
    tid = fx.Int32(fx.thread_idx.x)
    chunks = [tid + 256 * i for i in range(HC // 8 // 256)]  # 8 values = pairs 4c .. 4c + 3
    first = poll_pairs(r_mb_rs, [4 * chunks[0], 4 * chunks[0] + 2], tag)
    rest = poll_pairs(r_mb_rs, [4 * c + h for c in chunks[1:] for h in (0, 2)], tag)
    words = first + rest
    rows = [fx.Vector.from_elements(words[4 * i: 4 * i + 4], fx.Int32) for i in range(len(chunks))]
    load = lambda c: fx.Vector(buffer_ops.buffer_load(w_rs, c * 4, vec_width=4, dtype=T.i32))
    return [rows], [load(c) for c in chunks]


@device
def combine_rows16(r_mb_rs, tag, out_rs, y_ptr, c_ptr):
    """R''[s H + j] = bf16(R'[s H + j] + bf16(y[j]) c[s]) for the block's 16 rows j = 16 bid + i
    (y fp32 at y_ptr), R' from its mailbox; plain stores to out_rs."""
    tid = fx.Int32(fx.thread_idx.x)
    if tid < 32:
        s, q = tid // 8, tid % 8
        pair = s * (H // 2) + fx.Int32(fx.block_idx.x) * 8 + q
        words = poll_pairs(r_mb_rs, [(pair // 2) * 2], tag)
        r_lo, r_hi = _unpack((pair % 2 == 1).select(words[1], words[0]))
        y_lo, y_hi = _bf16(fx.ptr_load(y_ptr + 2 * q)), _bf16(fx.ptr_load(y_ptr + 2 * q + 1))
        c = fx.ptr_load(c_ptr + s)
        buffer_ops.buffer_store(bf16x2(r_lo + y_lo * c, r_hi + y_hi * c), out_rs, pair)
