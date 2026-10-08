"""Step 3: does the handoff hide? A chain of GEMVs in one launch vs one launch per GEMV.

A chain of L GDN-shaped layers: x -> up (6144 x 2560) -> h -> down (2560 x 6144) -> x' -> ...
Every handoff is a full fan-in: each block of op s + 1 needs all of op s's output.

    per_op    one launch per op (step 2's kernel), back to back
    fused     one launch; each op polls its input, then issues its weight loads
    prefetch  one launch; each op issues `depth` weight loads, then polls its input

The bound is weight bytes / stream-read bandwidth (6.0 TB/s for 256 MB in step 2).

  cd megakernel/basics && bash ../qwen38_flash_next/bench/gpu_run.sh env HIP_VISIBLE_DEVICES=7 \
      FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 step3_chain.py [--no-trace]
"""

import argparse
import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

import buffer_ops
from gemv_op import MARKS, NBLK, Smem, emit_gemv, spans
from step2_gemv import ROTATE_BYTES, X_TAG, build_gemv, from_mailbox, to_mailbox
from timeline import new_trace, span_us, write_trace

WAVES = 4
OPS = {"up": (6144, 2560, 1), "down": (2560, 6144, 4)}  # (N, K, k_split), balanced rows (step 2)
LAUNCHES = 100
STREAM_GBPS = 6000  # step 2, stream read of 256 MB
TRACE_LAUNCHES = 2


def chain(layers):
    """[(op kind, layer)] in execution order, and each op's output offset in the activation buffer."""
    ops = [(kind, l) for l in range(layers) for kind in ("up", "down")]
    offsets, at = [], 0
    for kind, _ in ops:
        offsets.append(at)
        at += OPS[kind][0]  # a mailbox of N bf16 is N int32
    return ops, offsets, at


def rs(t, nbytes, base_bytes=None):
    return buffer_ops.create_buffer_resource(t, max_size=False, num_records_bytes=nbytes,
                                             base_byte_offset=base_bytes)


@functools.cache
def build_chain(layers, depth, prefetch, traced):
    """One launch running the whole chain; launch e polls and publishes with tag e."""
    ops, offsets, _ = chain(layers)
    threads = WAVES * 64

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def fused(x_mb: fx.Tensor, acts: fx.Tensor, w_up: fx.Tensor, w_down: fx.Tensor, trace: fx.Tensor,
              layer0: fx.Int32, epoch: fx.Int32):
        partial = fx.SharedAllocator().allocate(Smem).peek().partial.ptr
        for s in range_constexpr(len(ops)):
            kind, l = ops[s]
            N, K, k_split = OPS[kind]
            x_rs = rs(x_mb, K * 4) if s == 0 else rs(acts, K * 4, offsets[s - 1] * 4)
            y_rs = rs(acts, N * 4, offsets[s] * 4)
            w_rs = rs(w_up if kind == "up" else w_down, N * K * 2, (layer0 + l) * (N * K * 2))
            emit_gemv(x_rs, w_rs, y_rs, partial, trace, X_TAG if s == 0 else epoch, epoch, N, K, WAVES,
                      k_split, depth, prefetch, MARKS * s, traced)

    @flyc.jit
    def launch(x_mb: fx.Tensor, acts: fx.Tensor, w_up: fx.Tensor, w_down: fx.Tensor, trace: fx.Tensor,
               layer0: fx.Int32, epoch: fx.Int32):
        fused(x_mb, acts, w_up, w_down, trace, layer0, epoch).launch(grid=(NBLK, 1, 1), block=(threads, 1, 1))

    return launch


# ===== Host =====


class Chain:
    """Weights rotated over > 3x the Infinity Cache, the input mailbox and the activation mailboxes."""

    def __init__(self, layers):
        self.layers = layers
        self.ops, self.offsets, n_acts = chain(layers)
        self.nbytes = sum(OPS[k][0] * OPS[k][1] * 2 for k, _ in self.ops)
        self.copies = -(-ROTATE_BYTES // self.nbytes)
        rand = lambda N, K: torch.randn(self.copies * layers, N, K, dtype=torch.bfloat16, device="cuda") * K**-0.5
        self.w = {kind: rand(N, K) for kind, (N, K, _) in OPS.items()}
        self.x = torch.randn(OPS["up"][1], dtype=torch.bfloat16, device="cuda")
        self.x_mb = to_mailbox(self.x, X_TAG)
        self.acts = torch.zeros(n_acts, dtype=torch.int32, device="cuda")

    def layer0(self, i):
        return (i % self.copies) * self.layers

    def mailbox(self, s):
        return self.acts[self.offsets[s]: self.offsets[s] + OPS[self.ops[s][0]][0]]

    def check(self, launch):
        """The final output matches a PyTorch chain that rounds every op's output to bf16, and every
        mailbox carries this launch's tag."""
        v = self.x
        for s, (kind, l) in enumerate(self.ops):
            v = (self.w[kind][self.layer0(launch) + l].float() @ v.float()).bfloat16()
            got, tags = from_mailbox(self.mailbox(s))
            assert (tags == launch + 1).all(), f"op {s} ({kind}{l}) has pairs from another launch"
        err = (got.float() - v.float()).abs().max() / v.float().abs().max()
        assert err.item() < 2e-2, f"chain output is wrong: relative error {err.item():.3g}"

    def trace_spans(self, prefetch):
        return [sp for s, (kind, l) in enumerate(self.ops) for sp in spans(f"{kind}{l} ", MARKS * s, prefetch)]


def run_timed(launches_of, n):
    """launches_of(i) -> [(launcher, args)] for chain launch i. Returns mean us per chain."""
    compiled = [flyc.compile(f, *args) for f, args in launches_of(0)]  # compiling also runs launch 0
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(n):
        for run, (_, args) in zip(compiled, launches_of(i)):
            run(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / n


def bench(c, variant, depth, traced):
    """Returns (us per chain, median in-kernel us or None, trace or None)."""
    tr = new_trace(LAUNCHES if traced else 1, NBLK, WAVES)
    t = lambda i: tr[i if traced else 0]
    if variant == "per_op":
        def launches_of(i):
            out = []
            for s, (kind, l) in enumerate(c.ops):
                N, K, k_split = OPS[kind]
                f = build_gemv(N, K, WAVES, k_split, depth, traced, prefetch=True, mark0=MARKS * s)
                x = c.x_mb if s == 0 else c.mailbox(s - 1)
                args = (x, c.w[kind][c.layer0(i) + l], c.mailbox(s), t(i), X_TAG if s == 0 else i + 1, i + 1)
                out.append((f, args))
            return out
    else:
        f = build_chain(c.layers, depth, variant == "prefetch", traced)
        launches_of = lambda i: [(f, (c.x_mb, c.acts, c.w["up"], c.w["down"], t(i), c.layer0(i), i + 1))]
    per_chain = run_timed(launches_of, LAUNCHES)
    c.check(LAUNCHES - 1)
    if not traced:
        return per_chain, None, None
    return per_chain, span_us(tr, 0, MARKS * len(c.ops) - 1).median().item(), tr


def main(traced):
    print("us per chain: back-to-back launches (gaps included) | in-kernel, first to last wave mark"
          + ("" if traced else " (tracing compiled out)"))
    for layers in (1, 4):
        c = Chain(layers)
        bound = c.nbytes / STREAM_GBPS / 1e3
        print(f"\n{layers} layer(s) = {len(c.ops)} GEMVs, {c.nbytes / 1e6:.0f} MB of weights, rotating "
              f"{c.copies} copies; bound at {STREAM_GBPS} GB/s: {bound:.1f} us")
        for variant in ("per_op", "fused", "prefetch"):
            for depth in (8, 16):
                per_chain, kern, tr = bench(c, variant, depth, traced)
                k = f"{kern:6.1f} us ({c.nbytes / kern / 1e3:4.0f} GB/s, {bound / kern:4.0%} of bound)" if kern else ""
                print(f"  {variant:9} depth={depth:2}  {per_chain:6.1f} us ({c.nbytes / per_chain / 1e3:4.0f} GB/s) | {k}")
                if traced and depth == 16:
                    path = f"traces/step3_{layers}layer_{variant}.json"
                    write_trace(path, tr[-TRACE_LAUNCHES:], c.trace_spans(variant != "fused"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-trace", action="store_true", help="compile the kernels without timeline marks")
    main(traced=not parser.parse_args().no_trace)
