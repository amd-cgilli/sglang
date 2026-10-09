"""Step 4a: gdn_core (gdn_core_op.py) vs qwen38_ref.gdn, real layer-0 weights.

The reference builds a state from a 256-token prefill. Then both decode the same tokens, each
from its own state, for STEPS steps. The kernel reads the in_proj outputs from a mailbox written by
the host (bit-identical to the reference's), and its o is compared with the reference's o
before out_proj, which comes from running the reference with out_proj = identity (exact in bf16).

  cd megakernel/basics && bash ../qwen38_flash_next/bench/gpu_run.sh env HIP_VISIBLE_DEVICES=7 \
      FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 step4_gdn_core.py [--no-trace]
"""

import argparse
import functools
import glob
import json
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

import flydsl.compiler as flyc
import flydsl.expr as fx

import buffer_ops
from gdn_core_op import CONV_CH, D, IN_DIM, V_HEADS, GdnSmem, ack_pairs, emit_gdn_core, gdn_core_prefetch, gdn_core_write_state
from step2_gemv import from_mailbox, to_mailbox
from timeline import mark, new_trace, span_us, write_trace

sys.path.insert(0, __file__.rsplit("/basics/", 1)[0] + "/qwen38_flash_next/reference")
import qwen38_ref as ref  # noqa: E402

CKPT = glob.glob("/sgl-workspace/sglang/hf_cache/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/*")[0]
PREFILL, STEPS, LAUNCHES = 256, 64, 100
SPANS = [("issue loads", 0, 1), ("poll + conv + norms", 1, 2), ("recurrence", 2, 3), ("norm + gate + publish o", 3, 4),
         ("write state", 4, 5)]


@functools.cache
def build(traced):
    @flyc.kernel(known_block_size=[256, 1, 1])
    def gdn_core(in_mb: fx.Tensor, out_mb: fx.Tensor, ack: fx.Tensor, conv_w: fx.Tensor, conv: fx.Tensor,
                 ssm: fx.Tensor, a_log: fx.Tensor, dt_bias: fx.Tensor, norm: fx.Tensor, trace: fx.Tensor,
                 in_tag: fx.Int32, out_tag: fx.Int32):
        rs = lambda t, nbytes: buffer_ops.create_buffer_resource(t, max_size=False, num_records_bytes=nbytes)
        smem = fx.SharedAllocator().allocate(GdnSmem).peek()
        head = fx.Int32(fx.block_idx.x)
        conv_rs, ssm_rs = rs(conv, CONV_CH * 3 * 2), rs(ssm, V_HEADS * D * D * 4)
        mark(trace, 4, 0, traced)
        pre = gdn_core_prefetch(head, rs(conv_w, CONV_CH * 4 * 2), conv_rs, ssm_rs, rs(norm, D * 2))
        ack_rs = rs(ack, ack_pairs() * 8)
        S = emit_gdn_core(head, pre, rs(in_mb, IN_DIM * 4), in_tag, rs(out_mb, V_HEADS * D * 4), out_tag, ack_rs,
                          rs(a_log, V_HEADS * 4), rs(dt_bias, V_HEADS * 4), smem, trace, 1, traced)
        gdn_core_write_state(head, S, pre, ack_rs, out_tag, conv_rs, ssm_rs, smem, trace, 5, traced)

    @flyc.jit
    def launch(in_mb: fx.Tensor, out_mb: fx.Tensor, ack: fx.Tensor, conv_w: fx.Tensor, conv: fx.Tensor,
               ssm: fx.Tensor, a_log: fx.Tensor, dt_bias: fx.Tensor, norm: fx.Tensor, trace: fx.Tensor,
               in_tag: fx.Int32, out_tag: fx.Int32):
        gdn_core(in_mb, out_mb, ack, conv_w, conv, ssm, a_log, dt_bias, norm, trace, in_tag, out_tag).launch(
            grid=(V_HEADS, 1, 1), block=(256, 1, 1))

    return launch


def load_gdn(layer):
    pre = f"model.language_model.layers.{layer}.linear_attn."
    index = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    w = {}
    for key, fname in index.items():
        if key.startswith(pre):
            with safe_open(f"{CKPT}/{fname}", "pt") as f:
                w[key[len(pre):]] = f.get_tensor(key).cuda()
    return w


def in_proj(x, w):
    """The reference's four in_proj GEMVs, concatenated in the mailbox order."""
    names = ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight")
    return torch.cat([ref.linear(x, w[n]) for n in names], -1)[0]


def compare(o_k, o_ref, st_k, st_ref):
    d = (o_k.float() - o_ref.float()).abs()
    return {"o exact": (d == 0).float().mean().item(), "o max err": (d.max() / o_ref.float().abs().max()).item(),
            "ssm max err": ((st_k["ssm"] - st_ref["ssm"]).abs().max() / st_ref["ssm"].abs().max()).item(),
            "conv exact": (st_k["conv"] == st_ref["conv"]).float().mean().item()}


def main(traced):
    torch.manual_seed(0)
    w = load_gdn(0)
    w_id = {**w, "out_proj.weight": torch.eye(V_HEADS * D, dtype=torch.bfloat16, device="cuda")}
    st_ref = {"conv": torch.zeros(CONV_CH, 3, dtype=torch.bfloat16, device="cuda"),
              "ssm": torch.zeros(V_HEADS, D, D, device="cuda")}
    ref.gdn(torch.randn(PREFILL, ref.H, dtype=torch.bfloat16, device="cuda"), w, st_ref)
    st_k = {k: v.clone() for k, v in st_ref.items()}
    print(f"state after a {PREFILL}-token reference prefill: |ssm| max {st_ref['ssm'].abs().max().item():.3g}")

    f = build(traced)
    out_mb = torch.zeros(V_HEADS * D, dtype=torch.int32, device="cuda")
    ack = torch.zeros(ack_pairs() * 2, dtype=torch.int32, device="cuda")
    weights = (w["conv1d.weight"], st_k["conv"], st_k["ssm"], w["A_log"].float(), w["dt_bias"].float(), w["norm.weight"])
    tr = new_trace(STEPS if traced else 1, V_HEADS, 4)
    args = lambda in_mb, i: (in_mb, out_mb, ack, *weights, tr[i if traced else 0], i + 1, i + 1)
    run = None
    worst = {}
    for i in range(STEPS):
        x = torch.randn(1, ref.H, dtype=torch.bfloat16, device="cuda")
        in_mb = to_mailbox(in_proj(x, w), i + 1)
        o_ref = ref.gdn(x, w_id, st_ref)[0]
        if run is None:
            run = flyc.compile(f, *args(in_mb, i))  # compiling also runs step 0
        else:
            run(*args(in_mb, i))
        torch.cuda.synchronize()
        o_k, tags = from_mailbox(out_mb)
        assert (tags == i + 1).all(), f"step {i}: o has pairs from another step"
        m = compare(o_k, o_ref, st_k, st_ref)
        worst = {k: (min if "exact" in k else max)(worst.get(k, v), v) for k, v in m.items()}
        if i in (0, 1, STEPS - 1):
            print(f"  step {i:2}: " + ", ".join(f"{k} {v:.3g}" for k, v in m.items()))
    print("  worst over steps: " + ", ".join(f"{k} {v:.3g}" for k, v in worst.items()))
    assert worst["o max err"] < 2e-2 and worst["ssm max err"] < 1e-4 and worst["conv exact"] == 1.0

    if traced:
        print(f"in-kernel us (first to last wave mark), median over {STEPS} steps: "
              f"o published {span_us(tr, 0, 4).median().item():.2f}, state written {span_us(tr, 0, 5).median().item():.2f}")
        write_trace("traces/step4_gdn_core.json", tr[-2:], SPANS)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(STEPS, STEPS + LAUNCHES):  # the last in_mb carries tag STEPS; o gets a fresh tag
        run(in_mb, out_mb, ack, *weights, tr[0], STEPS, i + 1)
    end.record()
    torch.cuda.synchronize()
    print(f"us per launch, back to back: {start.elapsed_time(end) * 1e3 / LAUNCHES:.2f} "
          f"(moves {2 * V_HEADS * D * D * 4 / 1e6:.1f} MB of state)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-trace", action="store_true", help="compile the kernel without timeline marks")
    main(traced=not parser.parse_args().no_trace)
