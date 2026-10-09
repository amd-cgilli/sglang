"""Step 4b: the GDN token mixer in one launch: in_proj -> gdn_core -> out_proj.

Every block runs the same program, and the mailboxes are the only links between blocks:

    gdn_core_prefetch     state + conv loads for head = block (blocks >= 48 load nothing)
    in_proj  (gemv)       x [2560] -> in_mb [16480] = q | k | v | z | b | a      all blocks, fewer rows on 0..47
    gemv_prefetch         all of this block's out_proj weights, before o exists  blocks 48..255
    then blocks 0..47:    gdn_core in_mb -> o_mb [6144] (one value head each), write conv + SSM state
    and blocks 48..255:   out_proj o_mb -> y_mb [2560]

Checked against qwen38_ref.gdn with real layer-0 weights: both decode STEPS tokens from a state
built by a reference prefill, each from its own state. Timing rotates WEIGHT_COPIES copies of the
projection weights (> 224 MB Infinity Cache) so every launch reads them from HBM.

  cd megakernel/basics && bash ../qwen38_flash_next/bench/gpu_run.sh env HIP_VISIBLE_DEVICES=7 \
      FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 step4b_gdn_mixer.py [--no-trace]
"""

import argparse
import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

import buffer_ops
from gdn_core_op import (CONV_CH, D, IN_DIM, MARKS as CORE_MARKS, STAGE_STRIDE, V_HEADS, ack_pairs, emit_gdn_core, gdn_core_prefetch,
                         gdn_core_write_state)
from gemv_op import MARKS as GEMV_MARKS, NBLK, RED_WORDS, emit_gemv, gemv_prefetch, spans
from step2_gemv import from_mailbox, to_mailbox
from step4_gdn_core import PREFILL, in_proj, load_gdn
from timeline import new_trace, span_us, write_trace

import step4_gdn_core  # noqa: F401  (puts the reference on sys.path)
import qwen38_ref as ref  # noqa: E402

H, HIDDEN_V = ref.H, V_HEADS * D  # 2560, 6144
IN_KSPLIT, OUT_KSPLIT, DEPTH = 1, 4, 16  # balanced rows (step 2), deep prefetch (step 3)
OUT_DEPTH = 64  # all of a wave's out_proj loads (42): they go out while gdn_core runs
# Head blocks also run gdn_core, so they take fewer in_proj rows (25 pairs vs at most 34, which
# keeps the others at 17 rows per wave, as with an even split) and no out_proj rows.
IN_LIGHT, OUT_LIGHT = (48, 25), (48, 0)
CORE_MARK0 = GEMV_MARKS
OUT_MARK0 = GEMV_MARKS + CORE_MARKS
STATE_MARK = OUT_MARK0 + GEMV_MARKS  # end of gdn_core_write_state
STEPS, LAUNCHES, WEIGHT_COPIES = 64, 100, 3
STREAM_GBPS = 6000


@fx.struct
class MixerSmem:
    partial: fx.Array[fx.Float32, RED_WORDS, 16]  # emit_gemv
    xin: fx.Array[fx.Float32, 3 * D, 16]  # emit_gdn_core, as GdnSmem
    qkv: fx.Array[fx.Float32, 3 * D, 16]
    z: fx.Array[fx.Float32, D, 16]
    out: fx.Array[fx.Float32, D, 16]
    scal: fx.Array[fx.Float32, 8, 16]
    stage: fx.Array[fx.Float32, D * STAGE_STRIDE, 16]


@functools.cache
def build(traced):
    @flyc.kernel(known_block_size=[256, 1, 1])
    def mixer(x_mb: fx.Tensor, in_mb: fx.Tensor, o_mb: fx.Tensor, y_mb: fx.Tensor, ack: fx.Tensor,
              w_in: fx.Tensor, w_out: fx.Tensor, conv_w: fx.Tensor, conv: fx.Tensor, ssm: fx.Tensor,
              a_log: fx.Tensor, dt_bias: fx.Tensor, norm: fx.Tensor, trace: fx.Tensor, copy: fx.Int32,
              x_tag: fx.Int32, epoch: fx.Int32):
        rs = lambda t, nbytes, base=None: buffer_ops.create_buffer_resource(
            t, max_size=False, num_records_bytes=nbytes, base_byte_offset=base)
        smem = fx.SharedAllocator().allocate(MixerSmem).peek()
        bid = fx.Int32(fx.block_idx.x)
        x_rs, in_rs, o_rs, y_rs = rs(x_mb, H * 4), rs(in_mb, IN_DIM * 4), rs(o_mb, HIDDEN_V * 4), rs(y_mb, H * 4)
        w_in_rs = rs(w_in, IN_DIM * H * 2, copy * (IN_DIM * H * 2))
        w_out_rs = rs(w_out, H * HIDDEN_V * 2, copy * (H * HIDDEN_V * 2))
        conv_rs, ssm_rs = rs(conv, CONV_CH * 3 * 2), rs(ssm, V_HEADS * D * D * 4)

        ack_rs = rs(ack, ack_pairs() * 8)

        core = gdn_core_prefetch(bid, rs(conv_w, CONV_CH * 4 * 2), conv_rs, ssm_rs, rs(norm, D * 2))
        emit_gemv(x_rs, w_in_rs, in_rs, smem.partial.ptr, trace, x_tag, epoch, IN_DIM, H, 4, IN_KSPLIT, DEPTH,
                  True, 0, traced, light=IN_LIGHT)
        # Every out_proj weight of this block goes in flight now, before o exists; head blocks have
        # no out_proj rows, so theirs are out of bounds and move no data.
        out_w = gemv_prefetch(w_out_rs, H, HIDDEN_V, 4, OUT_KSPLIT, OUT_DEPTH, light=OUT_LIGHT)
        if bid < V_HEADS:
            S = emit_gdn_core(bid, core, in_rs, epoch, o_rs, epoch, ack_rs, rs(a_log, V_HEADS * 4),
                              rs(dt_bias, V_HEADS * 4), smem, trace, CORE_MARK0, traced)
            gdn_core_write_state(bid, S, core, ack_rs, epoch, conv_rs, ssm_rs, smem, trace, STATE_MARK, traced)
        else:
            emit_gemv(o_rs, w_out_rs, y_rs, smem.partial.ptr, trace, epoch, epoch, H, HIDDEN_V, 4, OUT_KSPLIT,
                      OUT_DEPTH, True, OUT_MARK0, traced, inflight=out_w, light=OUT_LIGHT)

    @flyc.jit
    def launch(x_mb: fx.Tensor, in_mb: fx.Tensor, o_mb: fx.Tensor, y_mb: fx.Tensor, ack: fx.Tensor,
               w_in: fx.Tensor, w_out: fx.Tensor, conv_w: fx.Tensor, conv: fx.Tensor, ssm: fx.Tensor,
               a_log: fx.Tensor, dt_bias: fx.Tensor, norm: fx.Tensor, trace: fx.Tensor, copy: fx.Int32,
               x_tag: fx.Int32, epoch: fx.Int32):
        mixer(x_mb, in_mb, o_mb, y_mb, ack, w_in, w_out, conv_w, conv, ssm, a_log, dt_bias, norm, trace, copy,
              x_tag, epoch).launch(grid=(NBLK, 1, 1), block=(256, 1, 1))

    return launch


def trace_spans():
    core = ["poll + conv + norms", "recurrence", "norm + gate + publish o"]
    return (spans("in_proj ", 0, True)
            + [(f"gdn_core {n}", CORE_MARK0 + i, CORE_MARK0 + i + 1) for i, n in enumerate(core)]
            + spans("out_proj ", OUT_MARK0, True)
            + [("gdn_core write state", CORE_MARK0 + CORE_MARKS - 1, STATE_MARK)])


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def main(traced):
    torch.manual_seed(0)
    w = load_gdn(0)
    w_id = {**w, "out_proj.weight": torch.eye(HIDDEN_V, dtype=torch.bfloat16, device="cuda")}
    w_in1 = torch.cat([w[n] for n in ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight",
                                      "in_proj_a.weight")])
    w_in = w_in1[None].repeat(WEIGHT_COPIES, 1, 1)  # copy 0 is the real layer; the rest only rotate HBM reads
    w_out = w["out_proj.weight"][None].repeat(WEIGHT_COPIES, 1, 1)
    st_ref = {"conv": torch.zeros(CONV_CH, 3, dtype=torch.bfloat16, device="cuda"),
              "ssm": torch.zeros(V_HEADS, D, D, device="cuda")}
    ref.gdn(torch.randn(PREFILL, H, dtype=torch.bfloat16, device="cuda"), w, st_ref)
    st_k = {k: v.clone() for k, v in st_ref.items()}

    i32 = lambda n: torch.zeros(n, dtype=torch.int32, device="cuda")
    in_mb, o_mb, y_mb, ack = i32(IN_DIM), i32(HIDDEN_V), i32(H), i32(ack_pairs() * 2)
    tr = new_trace(STEPS if traced else 1, NBLK, 4)
    f = build(traced)

    def args(x_mb, i, copy, x_tag):
        return (x_mb, in_mb, o_mb, y_mb, ack, w_in, w_out, w["conv1d.weight"], st_k["conv"], st_k["ssm"],
                w["A_log"].float(), w["dt_bias"].float(), w["norm.weight"], tr[i if traced and i < STEPS else 0], copy, x_tag, i + 1)

    run, worst = None, {}
    for i in range(STEPS):
        x = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
        x_mb = to_mailbox(x[0], i + 1)
        o_ref = ref.gdn(x, w_id, {k: v.clone() for k, v in st_ref.items()})[0]
        in_ref, y_ref = in_proj(x, w), ref.gdn(x, w, st_ref)[0]
        if run is None:
            run = flyc.compile(f, *args(x_mb, i, 0, i + 1))  # compiling also runs step 0
        else:
            run(*args(x_mb, i, 0, i + 1))
        torch.cuda.synchronize()
        got = {name: from_mailbox(mb) for name, mb in (("in", in_mb), ("o", o_mb), ("y", y_mb))}
        for name, (_, tags) in got.items():
            assert (tags == i + 1).all(), f"step {i}: {name} has pairs from another step"
        m = {"in_proj exact": (got["in"][0] == in_ref).float().mean().item(),
             "o exact": (got["o"][0] == o_ref).float().mean().item(),
             "o err": rel(got["o"][0], o_ref), "y err": rel(got["y"][0], y_ref),
             "ssm err": rel(st_k["ssm"], st_ref["ssm"]), "conv exact": (st_k["conv"] == st_ref["conv"]).float().mean().item()}
        worst = {k: (min if "exact" in k else max)(worst.get(k, v), v) for k, v in m.items()}
        if i in (0, STEPS - 1):
            print(f"  step {i:2}: " + ", ".join(f"{k} {v:.3g}" for k, v in m.items()))
    print("  worst:   " + ", ".join(f"{k} {v:.3g}" for k, v in worst.items()))
    assert worst["y err"] < 2e-2 and worst["ssm err"] < 1e-3, "the mixer drifted from the reference"

    nbytes = (IN_DIM * H + H * HIDDEN_V) * 2 + 2 * V_HEADS * D * D * 4
    bound = nbytes / STREAM_GBPS / 1e3
    print(f"\n{nbytes / 1e6:.1f} MB per launch (weights + state read and write); bound at {STREAM_GBPS} GB/s: {bound:.1f} us")
    if traced:
        kern = span_us(tr, 0, OUT_MARK0 + GEMV_MARKS - 1).median().item()
        done = span_us(tr, 0, STATE_MARK).median().item()
        print(f"in-kernel, median over {STEPS} steps: y published {kern:.1f} us ({bound / kern:.0%} of bound), "
              f"state written {done:.1f} us")
        write_trace("traces/step4b_gdn_mixer.json", tr[-2:], trace_spans())
    x_mb = to_mailbox(torch.randn(H, dtype=torch.bfloat16, device="cuda"), 1)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(STEPS, STEPS + LAUNCHES):
        run(*args(x_mb, i, i % WEIGHT_COPIES, 1))
    end.record()
    torch.cuda.synchronize()
    per = start.elapsed_time(end) * 1e3 / LAUNCHES
    print(f"per launch, back to back: {per:.1f} us ({bound / per:.0%} of bound)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-trace", action="store_true", help="compile the kernel without timeline marks")
    main(traced=not parser.parse_args().no_trace)
