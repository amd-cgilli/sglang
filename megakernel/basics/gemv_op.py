"""The GEMV op, y = W x for one token, emitted inline into a kernel (steps 2 and 3).

x and y are tagged-pair mailboxes (common.py). The inner loop is skinny_bf16.py's: 16 B weight
loads, 64 lanes x 8 bf16 = 512 of K per load, v_dot2c_f32_bf16, DPP wave reduction. Work split
over a persistent grid of one block per CU:

    block   a balanced, contiguous range of output rows, in whole pairs
    wave    K-part (wave % k_split) of rows slot, slot + slots, ...  (slot = wave // k_split)
    lane    8 bf16 of K per load

A wave's loads over (row, k-step) are one flat unrolled list kept `depth` loads in flight.
With `prefetch`, the first `depth` loads are issued before x is polled; weights don't depend on
x, so they can be in flight while the producers of x finish.

Marks mark0 .. mark0 + 4 (timeline.py): start, first phase done, second phase done (prefetch:
issue then poll; else poll then issue), dot products done, y published.
"""

import itertools

import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr

import buffer_ops
from common import CM_NT, bf16x2, device, poll_pairs, put_pair
from skinny_bf16 import _as_vec, _dot2_f32_bf16, _pair, _wave_reduce_add_f32
from timeline import mark

NBLK = 256
K_STEP = 512  # 64 lanes x 8 bf16
RED_WORDS = 2048
MARKS = 5


@fx.struct
class Smem:
    partial: fx.Array[fx.Float32, RED_WORDS, 16]  # [row in block, k-part] partial dot products


def layout(N, K, waves, k_split):
    """(k-steps per part, row slots, rows per wave); asserts the split the op assumes."""
    steps = K // K_STEP
    assert K % K_STEP == 0 and N % 2 == 0 and steps % k_split == 0 and waves % k_split == 0
    slots = waves // k_split
    max_rows = 2 * -(-(N // 2) // NBLK)
    rows_per_wave = -(-max_rows // slots)
    assert rows_per_wave * slots * k_split <= RED_WORDS
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


@device
def emit_gemv(x_rs, w_rs, y_rs, partial, trace, x_tag, y_tag, N, K, waves, k_split, depth, prefetch,
              mark0, traced):
    """w_rs covers exactly this W ([N, K] bf16, num_records N * K * 2); x_rs and y_rs start at
    their mailboxes. Shapes and switches are Python values, fixed at compile time."""
    steps_per_part, slots, rows_per_wave = layout(N, K, waves, k_split)
    plan = list(itertools.product(range(rows_per_wave), range(steps_per_part)))  # (row i, k-step s)
    depth = min(depth, len(plan))
    pairs = N // 2

    tid, bid = fx.Int32(fx.thread_idx.x), fx.Int32(fx.block_idx.x)
    lane, wave = tid % 64, tid // 64
    kpart, slot = wave % k_split, wave // k_split
    base, extra = pairs // NBLK, pairs % NBLK
    n_pairs = base + (bid < extra).select(1, 0)
    pair0 = bid * base + fx.min(bid, extra)
    mark(trace, waves, mark0, traced)

    def local_row(i):
        return slot + slots * i

    def issue(i, s):
        # Rows past this block's range read row N: out of bounds, so the hardware returns 0.
        local = local_row(i)
        row = (local < 2 * n_pairs).select(2 * pair0 + local, N)
        k_dword = (kpart * steps_per_part + s) * (K_STEP // 2) + lane * 4
        return buffer_ops.buffer_load(w_rs, row * (K // 2) + k_dword, vec_width=4, cache_modifier=CM_NT)

    def poll_x():
        # This lane's 8 x values per k-step are pairs [p, p + 4): two 16 B loads.
        first = [((kpart * steps_per_part + s) * K_STEP + lane * 8) // 2 for s in range(steps_per_part)]
        words = poll_pairs(x_rs, [p + h for p in first for h in (0, 2)], x_tag)
        return [fx.Vector.from_elements(words[4 * s: 4 * s + 4], fx.Int32).bitcast(fx.BFloat16)
                for s in range(steps_per_part)]

    if const_expr(prefetch):
        inflight = [issue(*plan[j]) for j in range(depth)]
        mark(trace, waves, mark0 + 1, traced)
        x_frag = poll_x()
    else:
        x_frag = poll_x()
        mark(trace, waves, mark0 + 1, traced)
        inflight = [issue(*plan[j]) for j in range(depth)]
    mark(trace, waves, mark0 + 2, traced)

    acc = [arith.constant(0.0) for _ in range(4)]
    for j in range_constexpr(len(plan)):
        i, s = plan[j]
        nxt = issue(*plan[j + depth]) if const_expr(j + depth < len(plan)) else None
        wv = _as_vec(inflight[j % depth]).bitcast(fx.BFloat16)
        for q in range_constexpr(4):
            acc[q] = _dot2_f32_bf16(acc[q], _pair(x_frag[s], q), _pair(wv, q))
        if const_expr(nxt is not None):
            inflight[j % depth] = nxt
        if const_expr(s == steps_per_part - 1):
            with arith.fastmath(arith.FastMathFlags.fast):
                lane_sum = arith.addf(arith.addf(acc[0], acc[1]), arith.addf(acc[2], acc[3]))
                total = _wave_reduce_add_f32(lane_sum)
            if lane == 0:
                fx.ptr_store(fx.Float32(total), partial + local_row(i) * k_split + kpart)
            acc = [arith.constant(0.0) for _ in range(4)]
    mark(trace, waves, mark0 + 3, traced)
    fx.gpu.barrier()

    if tid < n_pairs:
        lo, hi = fx.Float32(0.0), fx.Float32(0.0)
        for k in range_constexpr(k_split):
            lo = lo + fx.ptr_load(partial + 2 * tid * k_split + k)
            hi = hi + fx.ptr_load(partial + (2 * tid + 1) * k_split + k)
        put_pair(y_rs, pair0 + tid, bf16x2(lo, hi), y_tag)
    fx.gpu.barrier()  # partial is free for the next op, and every pair of y is published
    mark(trace, waves, mark0 + 4, traced)
