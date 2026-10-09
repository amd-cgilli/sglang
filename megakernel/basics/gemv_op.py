"""The GEMV op, y = W x for one token, emitted inline into a kernel (steps 2 and 3).

x and y are tagged-pair mailboxes (common.py). The inner loop is skinny_bf16.py's: 16 B weight
loads, 64 lanes x 8 bf16 = 512 of K per load, v_dot2c_f32_bf16, DPP wave reduction. Work split
over a persistent grid of one block per CU:

    block   a balanced, contiguous range of output rows, in whole pairs
    wave    K-part (wave % k_split) of rows slot, slot + slots, ...  (slot = wave // k_split)
    lane    8 bf16 of K per load

x lives in registers (8 bf16 per lane per k-step), or with `x_lds` in LDS: the block polls the
whole mailbox into LDS once and every k-step reads its 16 B from there. Registers stop scaling
around K = 8K (4 VGPRs per k-step per lane); LDS holds K up to 36864.

A wave's loads over (row, k-step) are one flat unrolled list kept `depth` loads in flight; a
scheduling barrier after each step stops LLVM from reordering that pipeline.
With `prefetch`, the first `depth` loads are issued before x is polled; weights don't depend on
x, so they can be in flight while the producers of x finish.

Marks mark0 .. mark0 + 4 (timeline.py): start, first phase done, second phase done (prefetch:
issue then poll; else poll then issue), dot products done, y published.
"""

import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

import buffer_ops
from common import CM_NT, bf16x2, device, poll_pairs, put_pair
from skinny_bf16 import _as_vec, _dot2_f32_bf16, _pair, _wave_reduce_add_f32
from timeline import mark

NBLK = 256
K_STEP = 512  # 64 lanes x 8 bf16
RED_WORDS = 2048
X_WORDS = 18432  # bf16 pairs of x staged in LDS: K up to 36864, 72 KB
POLL_BATCH = 8  # 16 B mailbox loads in flight per thread while staging x into LDS
MARKS = 5


@fx.struct
class Smem:
    partial: fx.Array[fx.Float32, RED_WORDS, 16]  # [row in block, k-part] partial dot products


@fx.struct
class SmemXLds:
    partial: fx.Array[fx.Float32, RED_WORDS, 16]
    x: fx.Array[fx.Int32, X_WORDS, 16]  # word p = bf16 elements 2p, 2p + 1


def alloc_smem(x_in_lds):
    """(partial, x_lds) LDS pointers for one kernel; x_lds is None when x stays in registers."""
    if x_in_lds:
        smem = fx.SharedAllocator().allocate(SmemXLds).peek()
        return smem.partial.ptr, smem.x.ptr
    return fx.SharedAllocator().allocate(Smem).peek().partial.ptr, None


def _split(N, light):
    """Host side of the row split: (light blocks, pairs each, base, extra, max pairs per block).
    Blocks [0, n) get `light = (n, pairs_each)`; the other pairs are spread evenly over the rest,
    the first `extra` of them taking one more. light = None splits evenly over all blocks."""
    n_light, light_pairs = light or (0, 0)
    rest, blocks = N // 2 - n_light * light_pairs, NBLK - n_light
    assert rest >= 0
    base, extra = rest // blocks, rest % blocks
    return n_light, light_pairs, base, extra, max(light_pairs, base + (extra > 0))


def _block_pairs(N, light):
    """(pairs this block computes, its first pair), on device."""
    n_light, light_pairs, base, extra, _ = _split(N, light)
    bid = fx.Int32(fx.block_idx.x)
    b = bid - n_light
    n_pairs = base + (b < extra).select(1, 0)
    pair0 = n_light * light_pairs + b * base + fx.min(b, extra)
    if n_light == 0:
        return n_pairs, pair0
    return (bid < n_light).select(light_pairs, n_pairs), (bid < n_light).select(bid * light_pairs, pair0)


def layout(N, K, waves, k_split, light=None, tokens=1):
    """(k-steps per part, row slots, rows per wave); asserts the split the op assumes."""
    steps = K // K_STEP
    assert K % K_STEP == 0 and N % 2 == 0 and steps % k_split == 0 and waves % k_split == 0
    slots = waves // k_split
    max_rows = 2 * _split(N, light)[4]
    rows_per_wave = -(-max_rows // slots)
    assert rows_per_wave * slots * k_split * tokens <= RED_WORDS
    return steps // k_split, slots, rows_per_wave


def row_balance(N, K, waves, k_split):
    """Mean rows per row slot over the max: the share of the slowest wave's time that is useful."""
    _, slots, rows = layout(N, K, waves, k_split)
    return N / (NBLK * slots) / rows


def spans(label, mark0, prefetch):
    """Perfetto spans for one emit_gemv; `label` prefixes each name when a kernel has several."""
    first, second = ("issue weights", "poll x") if prefetch else ("poll x", "issue weights")
    names = [first, second, "dot + reduce", "barrier + publish y"]
    return [(f"{label}{n}", mark0 + i, mark0 + i + 1) for i, n in enumerate(names)]


def _geometry(N, K, waves, k_split, depth, light):
    """(plan, depth, issue): this thread's (row, k-step) load order and a function issuing one load."""
    steps_per_part, slots, rows_per_wave = layout(N, K, waves, k_split, light)
    # (row i, k-step s) in issue order: each wave streams whole rows. k-steps outer (one x slice
    # for all rows) measured 4.5x slower at K = 35840: a wave then hops rows on every 1 KB load.
    plan = [(i, s) for i in range(rows_per_wave) for s in range(steps_per_part)]
    tid, bid = fx.Int32(fx.thread_idx.x), fx.Int32(fx.block_idx.x)
    lane, wave = tid % 64, tid // 64
    kpart, slot = wave % k_split, wave // k_split
    n_pairs, pair0 = _block_pairs(N, light)

    def issue(w_rs, i, s, active=None):
        # Rows past this block's range read row N: out of bounds, so the hardware returns 0.
        local = slot + slots * i
        in_range = local < 2 * n_pairs if active is None else (local < 2 * n_pairs) & active
        row = in_range.select(2 * pair0 + local, N)
        k_dword = (kpart * steps_per_part + s) * (K_STEP // 2) + lane * 4
        return buffer_ops.buffer_load(w_rs, row * (K // 2) + k_dword, vec_width=4, cache_modifier=CM_NT)

    return plan, min(depth, len(plan)), issue


def gemv_prefetch(w_rs, N, K, waves, k_split, depth, active=None, light=None):
    """The first `depth` weight loads of emit_gemv, issued now; pass the result as `inflight`.
    Lets a block put a later GEMV's weights in flight before an unrelated op. Where `active` is
    false the loads are out of bounds: they return 0 and move no data, which gives a branch a
    placeholder value to replace."""
    plan, depth, issue = _geometry(N, K, waves, k_split, depth, light)
    return [issue(w_rs, *plan[j], active) for j in range(depth)]


@device
def emit_gemv(x_rs, w_rs, y_rs, partial, trace, x_tag, y_tag, N, K, waves, k_split, depth, prefetch,
              mark0, traced, x_lds=None, inflight=None, light=None, x_lds_ready=False, publish=None, tokens=1,
              y_row_pairs=None):
    """w_rs covers exactly this W ([N, K] bf16, num_records N * K * 2); x_rs and y_rs start at
    their mailboxes. Shapes and switches are Python values, fixed at compile time. `inflight` is
    gemv_prefetch(...) with the same arguments, if the first loads were issued earlier. `light`
    gives the first blocks fewer rows (see _split), for blocks that also run another op.
    With `x_lds_ready`, x_lds already holds x (bf16 pairs, as poll_x leaves it) and x_rs is unused.
    `publish(t, pair, lo, hi, valid)` replaces the y mailbox write: lo, hi are the fp32 sums of the
    pair's two rows for token t. It is called by every thread (valid false past this block's pairs),
    because a Python callable can't be used inside a device if.

    `tokens` = T: x and y are [T, K] and [T, N] (mailboxes and LDS alike), and every weight load
    is used for all T tokens, so the weights are read once for T (MTP verify). y_row_pairs is the
    mailbox's pairs per token (default N / 2), for a GEMV that fills part of a wider row."""
    steps_per_part, slots, rows_per_wave = layout(N, K, waves, k_split, light, tokens)
    assert x_lds is None or x_lds_ready or tokens == 1, "polling x into LDS takes one token"
    rows = range(rows_per_wave)
    plan, depth, issue_w = _geometry(N, K, waves, k_split, depth, light)
    issue = lambda i, s: issue_w(w_rs, i, s)

    tid = fx.Int32(fx.thread_idx.x)
    lane, wave = tid % 64, tid // 64
    kpart, slot = wave % k_split, wave // k_split
    n_pairs, pair0 = _block_pairs(N, light)
    mark(trace, waves, mark0, traced)

    def local_row(i):
        return slot + slots * i

    def x_pair(s):
        """First pair of this lane's 8 x values (pairs [p, p + 4)) at k-step s of its K-part."""
        return ((kpart * steps_per_part + s) * K_STEP + lane * 8) // 2

    def poll_x():
        """x_frag[t][s]: this lane's 8 x values of token t at k-step s (register mode)."""
        if const_expr(x_lds_ready):
            return None
        if const_expr(x_lds is None):
            # One poll per token: batching them holds every chunk's tag words at once, which pushed
            # the 4-token verify kernel past 256 VGPRs into scratch, and spinning on all is heavy.
            frags = []
            for t in range_constexpr(tokens):
                words = poll_pairs(x_rs, [t * (K // 2) + x_pair(s) + h for s in range(steps_per_part)
                                          for h in (0, 2)], x_tag)
                frags.append([fx.Vector.from_elements(words[4 * s: 4 * s + 4], fx.Int32).bitcast(fx.BFloat16)
                              for s in range(steps_per_part)])
            return frags
        # Thread t polls 16 B chunks t, t + threads, ...: two pairs each, value words to LDS.
        threads, chunks = waves * 64, K // 4
        per_thread = -(-chunks // threads)
        for b0 in range_constexpr(0, per_thread, POLL_BATCH):
            # Past the end, re-poll the last chunk: an out-of-bounds load returns tag 0 forever.
            chunk = [fx.min(tid + (b0 + j) * threads, chunks - 1)
                     for j in range(min(POLL_BATCH, per_thread - b0))]
            words = poll_pairs(x_rs, [2 * c for c in chunk], x_tag)
            for j in range_constexpr(len(chunk)):
                fx.ptr_store(words[2 * j], x_lds + 2 * chunk[j])
                fx.ptr_store(words[2 * j + 1], x_lds + 2 * chunk[j] + 1)
        fx.gpu.barrier()
        return None

    def x_at(s, t):
        if const_expr(x_lds is None):
            return x_frag[t][s]
        p = x_lds + t * (K // 2) + x_pair(s)
        return fx.Vector(fx.ptr_load(p, result_type=T.vec(4, T.i32))).bitcast(fx.BFloat16)

    if const_expr(inflight is not None):
        inflight = list(inflight)  # the ring below replaces entries
        mark(trace, waves, mark0 + 1, traced)
        x_frag = poll_x()
    elif const_expr(prefetch):
        inflight = [issue(*plan[j]) for j in range(depth)]
        mark(trace, waves, mark0 + 1, traced)
        x_frag = poll_x()
    else:
        x_frag = poll_x()
        mark(trace, waves, mark0 + 1, traced)
        inflight = [issue(*plan[j]) for j in range(depth)]
    mark(trace, waves, mark0 + 2, traced)

    # acc[i][t]: 4 independent chains per (row, token); only the current row's are live.
    acc = [[[arith.constant(0.0) for _ in range(4)] for _ in range(tokens)] for _ in rows]
    xv = None
    for j in range_constexpr(len(plan)):
        i, s = plan[j]
        nxt = issue(*plan[j + depth]) if const_expr(j + depth < len(plan)) else None
        wv = _as_vec(inflight[j % depth]).bitcast(fx.BFloat16)
        if const_expr(j == 0 or plan[j - 1][1] != s):
            xv = [x_at(s, t) for t in range(tokens)]
        for t in range_constexpr(tokens):
            for q in range_constexpr(4):
                acc[i][t][q] = _dot2_f32_bf16(acc[i][t][q], _pair(xv[t], q), _pair(wv, q))
        if const_expr(nxt is not None):
            inflight[j % depth] = nxt
        if const_expr(s == steps_per_part - 1):
            for t in range_constexpr(tokens):
                a = acc[i][t]
                with arith.fastmath(arith.FastMathFlags.fast):
                    lane_sum = arith.addf(arith.addf(a[0], a[1]), arith.addf(a[2], a[3]))
                    total = _wave_reduce_add_f32(lane_sum)
                if lane == 0:
                    fx.ptr_store(fx.Float32(total), partial + (local_row(i) * k_split + kpart) * tokens + t)
        # Keep the issue order as written: LLVM otherwise reorders this unrolled block freely and
        # decides how many loads are in flight (about one at K = 35840). FlyDSL's GEMMs do this too.
        rocdl.sched_barrier(0)
    mark(trace, waves, mark0 + 3, traced)
    fx.gpu.barrier()

    def pair_sums(p, t):
        lo, hi = fx.Float32(0.0), fx.Float32(0.0)
        for k in range_constexpr(k_split):
            lo = lo + fx.ptr_load(partial + (2 * p * k_split + k) * tokens + t)
            hi = hi + fx.ptr_load(partial + ((2 * p + 1) * k_split + k) * tokens + t)
        return lo, hi

    if const_expr(publish is None):
        if tid < n_pairs:
            for t in range_constexpr(tokens):
                lo, hi = pair_sums(tid, t)
                put_pair(y_rs, t * (y_row_pairs or N // 2) + pair0 + tid, bf16x2(lo, hi), y_tag)
    else:
        # Threads past n_pairs read some block pair's partials (clamped in bounds) and publish nothing.
        p = fx.min(tid, _split(N, light)[4] - 1)
        for t in range_constexpr(tokens):
            lo, hi = pair_sums(p, t)
            publish(t, pair0 + tid, lo, hi, tid < n_pairs)
    fx.gpu.barrier()  # partial is free for the next op, and every pair of y is published
    mark(trace, waves, mark0 + 4, traced)
