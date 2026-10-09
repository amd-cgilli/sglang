"""Step 5: the attention half of a GDN layer in one launch (gdn_block.py) vs qwen38_ref.

    R' = hc_combine(R, n, gdn(x)),  x, n = hc_mix(R)        real layer-0 weights

The state lives in a pool shaped like SGLang's ([layers, slots, ...], one layer view, slot SLOT
read on device), filled from a 256-token reference prefill; R is random each step. Timing replays
the launch back to back (state keeps evolving).

  cd megakernel/basics && bash ../qwen38_flash_next/bench/gpu_run.sh env HIP_VISIBLE_DEVICES=7 \
      FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 step5_gdn_block.py [--no-trace]

Two phases. A: each step starts from the reference's state, and the kernel is also compared with
the reference run on the kernel's own x, which isolates this step's ops from x's rounding flips
(in_proj spreads one flipped x over every output). B: both run free, and the drift must stay small.
A also reruns every step from the same state: the kernel's reductions have a fixed order, so it
must reproduce R' and the state bit for bit, which a race between blocks would break. R' is
compared in bf16 ulps of max(|R|, |R'|), the scale R + a c is rounded at.
"""

import argparse

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

from gdn_block import HC, LAST_MARK, SPANS, build, mailboxes
from gdn_core_op import CONV_CH, D, V_HEADS
from gemv_op import NBLK
from step2_gemv import from_mailbox
from step4_gdn_core import CKPT, PREFILL, load_gdn
from timeline import new_trace, span_us, write_trace

import step4_gdn_core  # noqa: F401  (puts the reference on sys.path)
import qwen38_ref as ref  # noqa: E402

LAYERS, SLOTS, LAYER, SLOT = 2, 6, 1, 3
STEPS, LAUNCHES = 64, 100
STREAM_GBPS = 6000


def load_hc(layer):
    import json
    from safetensors import safe_open

    pre = f"model.language_model.layers.{layer}.attn_hyper_connection."
    index = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    out = {}
    for key, fname in index.items():
        if key.startswith(pre):
            with safe_open(f"{CKPT}/{fname}", "pt") as f:
                out[key[len(pre):]] = f.get_tensor(key).cuda()
    return out


def block_args(w, hc):
    """The weights in gdn_block's argument order, laid out as SGLang holds them."""
    qkvz = torch.cat([w["in_proj_qkv.weight"], w["in_proj_z.weight"]])
    ba = torch.cat([w["in_proj_b.weight"], w["in_proj_a.weight"]])
    return (hc["hc_norm.weight"], hc["input_mix_weight_down.weight"], hc["input_mix_weight_up.weight"],
            hc["block_inject_weight.weight"], qkvz, ba, w["out_proj.weight"], w["conv1d.weight"].view(CONV_CH, 4),
            w["A_log"].float(), w["dt_bias"].float(), w["norm.weight"])


def ulps(got, want, r):
    """Max |got - want| in bf16 ulps of max(|R|, |want|): R' = bf16(R + a c) is rounded at that scale.
    (Measured at |R'| alone, a cancellation R ~ -a c leaves a tiny R' whose spacing means nothing.)"""
    scale = torch.maximum(r.float().abs(), want.float().abs()).clamp_min(1e-30)
    spacing = torch.exp2(torch.floor(torch.log2(scale)) - 7)
    return ((got.float() - want.float()).abs() / spacing).max().item()


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def main(traced):
    torch.manual_seed(0)
    w, hc = load_gdn(0), load_hc(0)
    st_ref = {"conv": torch.zeros(CONV_CH, 3, dtype=torch.bfloat16, device="cuda"),
              "ssm": torch.zeros(V_HEADS, D, D, device="cuda")}
    ref.gdn(torch.randn(PREFILL, ref.H, dtype=torch.bfloat16, device="cuda"), w, st_ref)
    conv_pool = torch.zeros(LAYERS, SLOTS, CONV_CH, 3, dtype=torch.bfloat16, device="cuda")
    ssm_pool = torch.zeros(LAYERS, SLOTS, V_HEADS, D, D, device="cuda")
    conv_pool[LAYER, SLOT], ssm_pool[LAYER, SLOT] = st_ref["conv"], st_ref["ssm"]
    conv, ssm = conv_pool[LAYER], ssm_pool[LAYER]  # SGLang's per-layer views
    cache_idx = torch.tensor([SLOT], dtype=torch.int32, device="cuda")

    weights, mb = block_args(w, hc), mailboxes()
    r = torch.empty(1, HC, dtype=torch.bfloat16, device="cuda")
    r_out = torch.empty_like(r)
    tr = new_trace(2 * STEPS if traced else 1, NBLK, 4)  # reruns overwrite their step's row
    args = lambda i: (r, r_out, *weights, conv, ssm, cache_idx, *mb.values(), tr[i if traced else 0],
                      fx.Stream(torch.cuda.current_stream()))
    run = None

    def step(i, synced):
        """One launch vs the reference; with `synced`, the kernel first gets the reference's state, and
        is also compared with the reference run on the kernel's own x (isolating this step's ops)."""
        nonlocal run
        r.copy_(torch.randn(1, HC, dtype=torch.bfloat16, device="cuda"))
        if synced:
            conv[SLOT], ssm[SLOT] = st_ref["conv"], st_ref["ssm"]
        copy = lambda st: {k: v.clone() for k, v in st.items()}  # ref.gdn replaces the entries it's given
        before = copy(st_ref)
        x, n = ref.hc_mix(r, hc)
        r_ref = ref.hc_combine(r, n, ref.gdn(x, w, st_ref), hc)
        if run is None:
            run = flyc.compile(build(traced), *args(i))  # compiling also runs step 0
        else:
            run(*args(i))
        torch.cuda.synchronize()
        x_k = from_mailbox(mb["x_mb"])[0]
        m = {"x exact": (x_k == x[0]).float().mean().item(), "R' exact": (r_out == r_ref).float().mean().item(),
             "R' max ulps": ulps(r_out, r_ref, r),
             "ssm err": rel(ssm[SLOT], st_ref["ssm"]), "conv exact": (conv[SLOT] == st_ref["conv"]).float().mean().item()}
        if synced:
            m["R' max ulps, ref on kernel x"] = ulps(r_out, ref.hc_combine(r, n, ref.gdn(x_k[None], w, copy(before)), hc), r)
            # Same inputs and state again: a race between blocks would change something.
            got = (r_out.clone(), conv[SLOT].clone(), ssm[SLOT].clone())
            conv[SLOT], ssm[SLOT] = before["conv"], before["ssm"]
            run(*args(i))
            torch.cuda.synchronize()
            m["rerun bit-identical"] = float(all(torch.equal(a, b) for a, b in
                                                 zip(got, (r_out, conv[SLOT], ssm[SLOT]))))
        m["other slots untouched"] = float(ssm_pool[LAYER, :SLOT].abs().max().item() == 0
                                           and ssm_pool[0].abs().max().item() == 0)
        return m

    def summary(name, hist):
        print(f"  {name}:")
        for k in hist[0]:
            v = sorted(h[k] for h in hist)
            worst = v[0] if "exact" in k or "untouched" in k or "identical" in k else v[-1]
            print(f"    {k:30} median {v[len(v) // 2]:.3g}, worst {worst:.3g}")
        low_is_bad = lambda k: "exact" in k or "untouched" in k or "identical" in k
        return {k: (min if low_is_bad(k) else max)(h[k] for h in hist) for k in hist[0]}

    a = summary(f"A: {STEPS} steps, each from the reference's state", [step(i, True) for i in range(STEPS)])
    b = summary(f"B: {STEPS} steps, each side on its own state", [step(STEPS + i, False) for i in range(STEPS)])
    assert mb["sync"][0].item() == 3 * STEPS, "the epoch counter didn't advance once per launch"
    assert a["rerun bit-identical"] == 1.0, "the kernel is not deterministic: a race between blocks"
    assert a["R' max ulps, ref on kernel x"] <= 4 and a["other slots untouched"] == 1.0, "a kernel op is wrong"
    assert b["R' max ulps"] <= 64 and b["ssm err"] < 1e-2, "the free-running state drifted"

    nbytes = sum(t.numel() * t.element_size() for t in weights) + 2 * V_HEADS * D * D * 4
    bound = nbytes / STREAM_GBPS / 1e3
    print(f"\n{nbytes / 1e6:.1f} MB per launch (weights + state read and write); bound at {STREAM_GBPS} GB/s: {bound:.1f} us")
    if traced:
        kern = span_us(tr, 0, LAST_MARK).median().item()
        print(f"in-kernel, median over {STEPS} steps: R' written {kern:.1f} us ({bound / kern:.0%} of bound)")
        write_trace("traces/step5_gdn_block.json", tr[-2:], SPANS)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(LAUNCHES):
        run(*args(0))
    end.record()
    torch.cuda.synchronize()
    per = start.elapsed_time(end) * 1e3 / LAUNCHES
    print(f"per launch, back to back: {per:.1f} us ({bound / per:.0%} of bound; weights hit the 224 MB Infinity "
          f"Cache across launches here, unlike in the model)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-trace", action="store_true", help="compile the kernel without timeline marks")
    main(traced=not parser.parse_args().no_trace)
