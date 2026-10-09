"""Step 2: one GEMV, y = W x, at full HBM bandwidth, written as a monokernel op (gemv_op.py).

The op reads x from a tagged-pair mailbox and writes y to one, so step 3 can chain them in one
launch. Compared against a pure read of the same bytes (the ceiling) and skinny_bf16.py.

  cd megakernel/basics && bash ../qwen38_flash_next/bench/gpu_run.sh env HIP_VISIBLE_DEVICES=7 \
      FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 step2_gemv.py [--no-trace]
"""

import argparse
import functools
import itertools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

import buffer_ops
from common import CM_NT
from gemv_op import K_STEP, NBLK, alloc_smem, emit_gemv, row_balance, spans
from timeline import mark, new_trace, span_us, write_trace
from skinny_bf16 import build_bf16_skinny_gemm_module

X_TAG = 7  # arbitrary but nonzero: zeroed and out-of-bounds pairs read as tag 0, so 0 would match them
LAUNCHES = 100
ROTATE_BYTES = 3 * 224 << 20  # > 3x the 224 MB Infinity Cache, so every launch reads HBM
SHAPES = {"in_proj": (16480, 2560), "out_proj": (2560, 6144)}  # GDN, Qwen3.8 TP1: (N, K)


@functools.cache
def build_gemv(N, K, waves, k_split, depth, traced, prefetch=True, mark0=0, x_in_lds=False):
    """One launch running one GEMV op; step 3 also uses it as the one-launch-per-op baseline."""
    threads = waves * 64

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def gemv(x_mb: fx.Tensor, w: fx.Tensor, y_mb: fx.Tensor, trace: fx.Tensor, x_tag: fx.Int32,
             y_tag: fx.Int32):
        x_rs = buffer_ops.create_buffer_resource(x_mb, max_size=False, num_records_bytes=K * 4)
        w_rs = buffer_ops.create_buffer_resource(w, max_size=False, num_records_bytes=N * K * 2)
        y_rs = buffer_ops.create_buffer_resource(y_mb, max_size=False, num_records_bytes=N * 4)
        partial, x_lds = alloc_smem(x_in_lds)
        emit_gemv(x_rs, w_rs, y_rs, partial, trace, x_tag, y_tag, N, K, waves, k_split, depth, prefetch,
                  mark0, traced, x_lds)

    @flyc.jit
    def launch(x_mb: fx.Tensor, w: fx.Tensor, y_mb: fx.Tensor, trace: fx.Tensor, x_tag: fx.Int32,
               y_tag: fx.Int32):
        gemv(x_mb, w, y_mb, trace, x_tag, y_tag).launch(grid=(NBLK, 1, 1), block=(threads, 1, 1))

    return launch


@functools.cache
def build_stream_read(nbytes, waves, depth, traced):
    """Read nbytes once with 16 B non-temporal loads, `depth` in flight per lane: the bandwidth ceiling."""
    threads = waves * 64
    loads = -(-nbytes // (16 * NBLK * threads))
    depth = min(depth, loads)

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def stream(src: fx.Tensor, out: fx.Tensor, trace: fx.Tensor):
        bid = fx.Int32(fx.block_idx.x)
        gid = bid * threads + fx.Int32(fx.thread_idx.x)
        rs = buffer_ops.create_buffer_resource(src, max_size=False, num_records_bytes=nbytes)
        mark(trace, waves, 0, traced)

        def issue(i):
            return fx.Vector(buffer_ops.buffer_load(rs, (gid + i * NBLK * threads) * 4, vec_width=4,
                                                    dtype=T.i32, cache_modifier=CM_NT))

        inflight = [issue(i) for i in range_constexpr(depth)]
        fold = fx.Int32(0)
        for i in range_constexpr(loads):
            v = inflight[i % depth]
            if const_expr(i + depth < loads):
                inflight[i % depth] = issue(i + depth)
            fold = fold ^ v[0] ^ v[1] ^ v[2] ^ v[3]
        buffer_ops.buffer_store(fold, buffer_ops.create_buffer_resource(out), gid)
        fx.gpu.barrier()
        mark(trace, waves, 1, traced)

    @flyc.jit
    def launch(src: fx.Tensor, out: fx.Tensor, trace: fx.Tensor):
        stream(src, out, trace).launch(grid=(NBLK, 1, 1), block=(threads, 1, 1))

    return launch


# ===== Host =====

GEMV_SPANS = spans("", 0, prefetch=True)
STREAM_SPANS = [("read", 0, 1)]
TRACE_LAUNCHES = 3  # the last few launches go to the Perfetto trace


def to_mailbox(x, tag):
    """bf16 [n] -> int32 [n] holding n / 2 (bf16x2 word, tag) pairs."""
    words = x.view(torch.int32)
    return torch.stack([words, torch.full_like(words, tag)], -1).flatten()


def from_mailbox(mb):
    pairs = mb.view(-1, 2)
    return pairs[:, 0].contiguous().view(torch.bfloat16), pairs[:, 1]


def weight_copies(N, K):
    n = -(-ROTATE_BYTES // (N * K * 2))
    return [torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * K**-0.5 for _ in range(n)]


def timed(launcher, args_of, n):
    """Mean us per launch over n back-to-back launches, gaps included; args_of(i) = launch i's args.
    flyc.compile gives a ~5 us dispatch; calling the @flyc.jit launcher costs ~60 us of Python,
    which would leave the GPU idle between launches."""
    run = flyc.compile(launcher, *args_of(0))
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(n):
        run(*args_of(i))
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / n


def check(y, w, x, what):
    ref = w.float() @ x.float()
    assert ((y.float() - ref).abs().max() / ref.abs().max()).item() < 1e-2, f"{what}: y is wrong"


def bench_tagged(N, K, ws, x, waves, k_split, depth, traced):
    """Returns (us per launch, median in-kernel us or None, trace or None); asserts y and its tags."""
    f = build_gemv(N, K, waves, k_split, depth, traced)
    x_mb, y_mb = to_mailbox(x, X_TAG), torch.zeros(N, dtype=torch.int32, device="cuda")
    tr = new_trace(LAUNCHES if traced else 1, NBLK, waves)
    per_launch = timed(f, lambda i: (x_mb, ws[i % len(ws)], y_mb, tr[i if traced else 0], X_TAG, i + 1),
                       LAUNCHES)
    y, tags = from_mailbox(y_mb)
    assert (tags == LAUNCHES).all(), "a y pair was not written by the last launch"
    check(y, ws[(LAUNCHES - 1) % len(ws)], x, "tagged gemv")
    return (per_launch, span_us(tr, 0, 4).median().item(), tr) if traced else (per_launch, None, None)


def bench_skinny(N, K, ws, x, depth):
    f = build_bf16_skinny_gemm_module(M=1, N=N, K=K, prefetch_depth=depth)
    y = torch.empty(N, dtype=torch.bfloat16, device="cuda")
    per_launch = timed(f, lambda i: (x[None], ws[i % len(ws)], y, fx.Stream(None)), LAUNCHES)
    check(y, ws[(LAUNCHES - 1) % len(ws)], x, "skinny_bf16")
    return per_launch


def bench_stream(ws, waves, depth, traced):
    """Returns (us per launch, median in-kernel us or None, trace or None)."""
    f = build_stream_read(ws[0].numel() * ws[0].element_size(), waves, depth, traced)
    out = torch.empty(NBLK * waves * 64, dtype=torch.int32, device="cuda")
    tr = new_trace(LAUNCHES if traced else 1, NBLK, waves)
    per_launch = timed(f, lambda i: (ws[i % len(ws)], out, tr[i if traced else 0]), LAUNCHES)
    return (per_launch, span_us(tr, 0, 1).median().item(), tr) if traced else (per_launch, None, None)


def row(label, nbytes, per_launch, kernel=None, note=""):
    gbps = lambda us: nbytes / us / 1e3
    k = f"{gbps(kernel):6.0f} ({kernel:5.1f} us)" if kernel else " " * 17
    print(f"  {label:38} {gbps(per_launch):6.0f}   {k}  {note}")


def main(traced):
    print("GB/s from us per launch (back to back, gaps included) | in-kernel (first to last wave mark)"
          + ("" if traced else "; tracing compiled out, so no in-kernel times or traces"))
    big = [torch.empty(256 << 20, dtype=torch.uint8, device="cuda") for _ in range(3)]
    for depth in (8, 16):
        launch_us, kern, _ = bench_stream(big, 8, depth, traced)
        row(f"stream read 256 MB  waves=8 depth={depth}", 256 << 20, launch_us, kern)
    for name, (N, K) in SHAPES.items():
        ws, x = weight_copies(N, K), torch.randn(K, dtype=torch.bfloat16, device="cuda")
        nbytes = N * K * 2
        print(f"\n{name}: N={N} K={K}, {nbytes / 1e6:.1f} MB, rotating {len(ws)} copies, {LAUNCHES} launches")
        best = {}  # kind -> (us, config, trace); in-kernel us when traced, else us per launch

        def keep(kind, launch_us, kern, cfg, tr):
            cand = (kern if traced else launch_us, cfg, tr)
            best[kind] = min(best.get(kind, cand), cand, key=lambda b: b[0])

        for waves, depth in itertools.product((4, 8), (8, 16)):
            launch_us, kern, tr = bench_stream(ws, waves, depth, traced)
            row(f"stream read  waves={waves} depth={depth}", nbytes, launch_us, kern)
            keep("stream", launch_us, kern, f"waves{waves}_depth{depth}", tr)
        for depth in (1, K // K_STEP):
            row(f"skinny_bf16  depth={depth}", nbytes, bench_skinny(N, K, ws, x, depth))
        splits = [k for k in (1, 2, 4) if (K // K_STEP) % k == 0]
        for waves, k_split, depth in itertools.product((4, 8), splits, (4, 8, 16)):
            if waves % k_split:
                continue
            launch_us, kern, tr = bench_tagged(N, K, ws, x, waves, k_split, depth, traced)
            row(f"tagged gemv  waves={waves} k_split={k_split} depth={depth}", nbytes, launch_us, kern,
                f"row balance {row_balance(N, K, waves, k_split):.0%}")
            keep("tagged", launch_us, kern, f"waves{waves}_ks{k_split}_depth{depth}", tr)
        for kind, (_, cfg, tr) in best.items():
            print(f"  fastest {kind}: {cfg}")
            if traced:
                path = f"traces/step2_{name}_{kind}_{cfg}.json"
                write_trace(path, tr[-TRACE_LAUNCHES:], GEMV_SPANS if kind == "tagged" else STREAM_SPANS)
                print(f"    trace: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-trace", action="store_true", help="compile the kernels without timeline marks")
    main(traced=not parser.parse_args().no_trace)
