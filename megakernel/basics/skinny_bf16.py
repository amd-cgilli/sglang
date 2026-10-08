
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import Vector as Vec
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

import buffer_ops


def _ceildiv(x, y):
    return (x + y - 1) // y


_WAVE_THREADS = 64


def _laneid():
    return arith.index_cast(T.i32, fx.thread_idx.x) % _WAVE_THREADS


def _waveid():
    return arith.index_cast(T.i32, fx.thread_idx.x) // _WAVE_THREADS


def _raw(v):
    """Unwrap a DSL Numeric/Vector down to a plain ir.Value."""
    return v.ir_value() if hasattr(v, "ir_value") else v


def _as_vec(loaded):
    """Wrap a buffer_load result as a Vector, whatever its width.

    A `vec_width` of 1 comes back as a bare f32 rather than a <1xf32>, so
    `Vec()` cannot take it directly -- re-pack it into a 1-lane vector first so
    the `.bitcast` to bf16 lanes works uniformly. The kernel only issues
    dwordx4 today, but the narrow case is cheap to keep handled.
    """
    raw = _raw(loaded)
    try:
        ir.VectorType(raw.type)
    except ValueError:
        return Vec.from_elements([raw])
    return Vec(raw)


# --- DPP controls (gfx9 / CDNA) --------------------------------------------
# quad_perm[a,b,c,d] packs as a | b<<2 | c<<4 | d<<6; the mirror controls are
# the 0x14x block. These are reversals, not literal XORs, but once every lane
# in a group holds that group's sum, reversing a group of 2n pairs each half
# with the other -- which is all a butterfly round needs.
_DPP_SWAP_1 = 0xB1  # quad_perm[1,0,3,2]: swap neighbours     (lane ^ 1)
_DPP_SWAP_2 = 0x4E  # quad_perm[2,3,0,1]: swap lane pairs     (lane ^ 2)
_DPP_SWAP_4 = 0x141  # row_half_mirror: reverse within 8       (lane ^ 7)
_DPP_SWAP_8 = 0x140  # row_mirror: reverse within 16           (lane ^ 15)
_DPP_BUTTERFLY = (_DPP_SWAP_1, _DPP_SWAP_2, _DPP_SWAP_4, _DPP_SWAP_8)

_DPP_ALL_ROWS = 0xF
_DPP_ALL_BANKS = 0xF


def _dpp_add_f32(x, dpp_ctrl):
    """`x + dpp(x, dpp_ctrl)`, which the backend folds into one v_add_f32_dpp.

    bound_ctrl=1 makes a read of an inactive lane return 0 -- the identity for
    the add -- rather than leaving the destination untouched.
    """
    x = _raw(x)
    other = rocdl.update_dpp(T.f32, x, x, dpp_ctrl, _DPP_ALL_ROWS, _DPP_ALL_BANKS, True)
    return arith.addf(x, _raw(other))


def _permlane_swap_add_f32(x, swap_op):
    """`x + swap(x)` for the gfx950 v_permlane{16,32}_swap_b32 pair.

    Both ops take (old, src) and hand back the two group-halves separated:
    fed `old == src == x`, ret0 is the even group broadcast over both halves
    and ret1 is the odd group. Their sum is the lane^16 / lane^32 butterfly
    round, landed in every lane.
    """
    xi = arith.bitcast(T.i32, _raw(x))
    pair = swap_op(
        llvm.StructType.get_literal([T.i32, T.i32]), _raw(xi), _raw(xi), False, False
    )
    lo = llvm.extractvalue(T.i32, pair, [0])
    hi = llvm.extractvalue(T.i32, pair, [1])
    return arith.addf(arith.bitcast(T.f32, lo), arith.bitcast(T.f32, hi))


def _dot2_f32_bf16(acc, a2, b2):
    """`acc + a2.lo*b2.lo + a2.hi*b2.hi` as a single v_dot2c_f32_bf16.

    The bf16 pairs go in as raw 32-bit registers, so the whole convert-to-f32
    step disappears: widening bf16 lanes used to be ~62% of this kernel's VALU
    work (a v_lshlrev + v_and per element), and it is all folded into the dot
    here. Accuracy does not suffer -- a bf16 product has at most 16 mantissa
    bits, exact in f32 either way, and dot2 fuses the two products into the
    accumulate rather than rounding a separate pk_add.

    Emitted through call_intrinsic because flydsl's rocdl wrapper has no fdot2;
    `llvm.amdgcn.fdot2.f32.bf16` selects to v_dot2c_f32_bf16 on gfx950. The
    trailing operand is the clamp flag, which we never want.
    """
    acc_raw = _raw(acc)
    return llvm.call_intrinsic(
        acc_raw.type,
        "llvm.amdgcn.fdot2.f32.bf16",
        [_raw(a2), _raw(b2), acc_raw, _raw(arith.constant(False))],
        [],
        [],
    )


def _pair(v, i):
    """Lanes 2i and 2i+1 of `v` as a <2 x bf16> -- one VGPR's worth."""
    return v.shuffle(v, [2 * i, 2 * i + 1])


def _wave_reduce_add_f32(x, width: int = _WAVE_THREADS):
    """Sum `x` across each aligned group of `width` lanes.

    `width` must be a power of two in [1, 64]. Every lane of a group ends up
    holding that group's total; groups never mix, so a wave that owns
    `64 // width` rows reduces all of them in the same log2(width) rounds.

    All butterfly rounds run on the VALU -- no LDS pipe, no lane addressing,
    no lgkmcnt waits, unlike the ds_bpermute crossbar this replaces:

      * strides 1/2/4/8 ride DPP, fused straight into the add.
      * strides 16/32 use v_permlane16_swap_b32 / v_permlane32_swap_b32. DPP
        cannot reach across a 16-lane row, and its row_bcast15/31 controls
        would strand the total in lane 63; the swaps keep it everywhere.

    Every round is a pure lane^stride exchange, so stopping after the rounds
    with stride < width leaves each group holding exactly its own sum.
    """
    assert width & (width - 1) == 0 and 1 <= width <= _WAVE_THREADS

    for i in range_constexpr(len(_DPP_BUTTERFLY)):
        if const_expr((1 << i) < width):
            x = _dpp_add_f32(x, _DPP_BUTTERFLY[i])
    if const_expr(width > 16):
        x = _permlane_swap_add_f32(x, rocdl.permlane16_swap)
    if const_expr(width > 32):
        x = _permlane_swap_add_f32(x, rocdl.permlane32_swap)
    return x


CPOL_NT = 2

# Every lane issues one buffer_load_dwordx4: 4 dwords = 16B = 8 bf16. This is
# fixed rather than tuned -- a narrower load is never the better trade, since
# the lanes it would free up are worth more spent on extra rows (see
# `_threads_per_row`).
ELEMS_PER_LANE = 8
DWORDS_PER_LANE = ELEMS_PER_LANE // 2


def _threads_per_row(K: int) -> int:
    """Widest power-of-two lane group whose dwordx4 tiling still divides K.

    Capped at the wave so a row never spans two waves. Below that cap the
    leftover lanes go to *extra rows* instead of to narrower loads: at K=128 a
    full wave would leave each lane only 2 bf16 (a dword), so instead 16 lanes
    cover the row with wide loads and the wave takes 4 rows.
    """
    tpr = 1
    while tpr * 2 <= _WAVE_THREADS and K % (tpr * 2 * ELEMS_PER_LANE) == 0:
        tpr *= 2
    return tpr


def _k_steps(K: int) -> int:
    """How many K-tiles a lane group sweeps -- one wide load per lane each."""
    return K // (_threads_per_row(K) * ELEMS_PER_LANE)


def build_bf16_skinny_gemm_module(
    *, M: int, N: int, K: int, prefetch_depth: int = 1, num_waves: int = 4
):
    assert M == 1

    # The idea is to issue as wide buffer_load as possible
    # when K is small this is not possible naturally so we can
    # increase the number of rows for each wave
    # Example:
    # - if each wave handles 1 full row
    #   K = 128 -> 2 elements (32bits)/lane
    # - if we assign 4 rows per wave we get only 16 threads per row
    #   K = 128 -> 8 elements/lane

    assert K % ELEMS_PER_LANE == 0

    THREADS_PER_ROW = _threads_per_row(K)
    ROWS_PER_WAVE = _WAVE_THREADS // THREADS_PER_ROW
    ROWS_PER_BLOCK = num_waves * ROWS_PER_WAVE

    # A row group sweeps THREADS_PER_ROW * 16B of one B row per load
    # instruction; at the 64-lane cap that is 1024 contiguous bytes (512 bf16).
    K_STEP = THREADS_PER_ROW * ELEMS_PER_LANE
    DWORDS_PER_STEP = THREADS_PER_ROW * DWORDS_PER_LANE

    K_STEPS = _k_steps(K)
    assert K_STEPS * K_STEP == K

    PREFETCH_DEPTH = min(prefetch_depth, K_STEPS)

    # Assume M=1
    # Each wave owns ROWS_PER_WAVE rows of B, one per group of
    # THREADS_PER_ROW consecutive lanes.
    @flyc.kernel(known_block_size=[num_waves * _WAVE_THREADS, 1, 1])
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        lane_id = _laneid()
        # Lane position inside its row group, and which row of the wave it is on.
        k_lane = lane_id % THREADS_PER_ROW
        row_in_wave = lane_id // THREADS_PER_ROW
        row_idx = (
            fx.block_idx.x * ROWS_PER_BLOCK
            + _waveid() * ROWS_PER_WAVE
            + row_in_wave
        )

        A_gl = buffer_ops.create_buffer_resource(
            A, max_size=False, num_records_bytes=K * 2
        )
        B_gl = buffer_ops.create_buffer_resource(
            B, max_size=False, num_records_bytes=N * (K * 2)
        )
        C_gl = buffer_ops.create_buffer_resource(
            C, max_size=False, num_records_bytes=N * M * 2
        )

        # Load the entire A from global to register, keyed on the lane's slot
        # within its row group so it lines up with that lane's B elements.
        # Row groups past the first re-read the same A -- same addresses, so it
        # is an L1 hit, and it keeps A resident without an LDS round trip.
        # (this can work because A at 7K is ~56VGPRs)
        a_frag = [
            _as_vec(
                buffer_ops.buffer_load(
                    A_gl,
                    offset=k_lane * DWORDS_PER_LANE + s * DWORDS_PER_STEP,
                    vec_width=DWORDS_PER_LANE,
                )
            ).bitcast(fx.BFloat16)
            for s in range_constexpr(K_STEPS)
        ]

        # One row per lane group. Each lane keeps DWORDS_PER_LANE independent
        # f32 accumulators, one per bf16 pair, so the dot2 chains do not
        # serialise on each other; they are summed together at the end.
        acc = [arith.constant(0.0) for _ in range_constexpr(DWORDS_PER_LANE)]

        # buffer_load scales the offset by dwords (our base is bf16 so we have to halve that)
        b_row_base = row_idx * (K // 2)

        def load_b_nt(k):
            # non-temporal load
            return buffer_ops.buffer_load(
                B_gl,
                offset=b_row_base + k_lane * DWORDS_PER_LANE + k * DWORDS_PER_STEP,
                vec_width=DWORDS_PER_LANE,
                cache_modifier=CPOL_NT,
            )

        inflight = [load_b_nt(s) for s in range_constexpr(PREFETCH_DEPTH)]

        for k in range_constexpr(K_STEPS):
            next = load_b_nt(k + PREFETCH_DEPTH) if const_expr(k + PREFETCH_DEPTH < K_STEPS) else None
            b = _as_vec(inflight[k % PREFETCH_DEPTH]).bitcast(fx.BFloat16)
            a = a_frag[k]
            for i in range_constexpr(DWORDS_PER_LANE):
                acc[i] = _dot2_f32_bf16(acc[i], _pair(a, i), _pair(b, i))

            if const_expr(next is not None):
                inflight[k % PREFETCH_DEPTH] = next

        # store
        with arith.fastmath(arith.FastMathFlags.fast):
            ls0 = arith.addf(_raw(acc[0]), _raw(acc[1]))
            ls1 = arith.addf(_raw(acc[2]), _raw(acc[3]))
            lane_sum = arith.addf(_raw(ls0), _raw(ls1))
            # lane_sum = acc[0]
            # for i in range_constexpr(1, DWORDS_PER_LANE):
            #     lane_sum = arith.addf(_raw(lane_sum), _raw(acc[i]))
            total = _wave_reduce_add_f32(lane_sum, THREADS_PER_ROW)

        # One writer per row group. Rows past N fall outside the buffer's
        # num_records, so the store is dropped by the hardware.
        if k_lane == 0:
            buffer_ops.buffer_store(arith.trunc_f(T.bf16, total), C_gl, row_idx)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream,
    ):
        n_threads = num_waves * _WAVE_THREADS
        n_blocks = _ceildiv(N, ROWS_PER_BLOCK)
        kernel(A, B, C).launch(
            grid=(n_blocks, 1, 1),
            block=(n_threads, 1, 1),
            stream=stream,
            value_attrs={
                "rocdl.flat_work_group_size": f"{n_threads},{n_threads}",
                "passthrough": [
                    ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                    ["no-nans-fp-math", "true"],
                    ["unsafe-fp-math", "true"],
                ],
            },
        )

    return launch_gemm