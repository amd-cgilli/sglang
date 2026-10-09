"""Step 8: a whole GDN layer for an MTP verify pass (T = 4 tokens) in one launch (gdn_layer_verify.py).

    per token row t:  R'_t = hc_combine1(R_t, gdn_t(...)),  R_out_t = hc_combine2(R'_t, moe(hc_mix2(R'_t)))

The attention half is step 6's (state read, per-token snapshots to SGLang's verify scratch); the
FFN half is step 7's MoE on 4 tokens, experts deduplicated across tokens. Real layer-0 weights
from the MXFP4 checkpoint; each launch starts from the reference's state with a random R.

  cd megakernel/basics && bash ../qwen38_flash_next/bench/gpu_run.sh env HIP_VISIBLE_DEVICES=6 \\
      FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 step8_gdn_layer_verify.py [--no-trace] [--dense-windows]
"""

import argparse

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

from gdn_core_op import CONV_CH, D, V_HEADS
from gdn_layer_verify import LAST_MARK, SPANS, T_VERIFY, build, mailboxes
from gemv_op import NBLK
from hc_op import HC
from moe_op import TOPK
from step2_gemv import from_mailbox
from step4_gdn_core import PREFILL
from step5_gdn_block import block_args, rel, ulps
from step6_gdn_verify import reference as attn_reference
from step7_gdn_layer import ffn_ref, load, moe_ref, moe_stock, moe_weights, route
from timeline import new_trace, span_us, write_trace

import step4_gdn_core  # noqa: F401  (puts the reference on sys.path)
import qwen38_ref as ref  # noqa: E402

T = T_VERIFY
LAYERS, SLOTS, LAYER, SLOT, ROWS, ROW = 2, 6, 1, 3, 3, 1
STEPS, LAUNCHES = 16, 50
STREAM_GBPS = 6000


def main(traced, dedup):
    torch.manual_seed(0)
    w, hc1, hc2 = load(0, "linear_attn"), load(0, "attn_hyper_connection"), load(0, "mlp_hyper_connection")
    mw = moe_weights(0)
    st_ref = {"conv": torch.zeros(CONV_CH, 3, dtype=torch.bfloat16, device="cuda"),
              "ssm": torch.zeros(V_HEADS, D, D, device="cuda")}
    ref.gdn(torch.randn(PREFILL, ref.H, dtype=torch.bfloat16, device="cuda"), w, st_ref)
    conv_pool = torch.zeros(LAYERS, SLOTS, CONV_CH, 3, dtype=torch.bfloat16, device="cuda")
    ssm_pool = torch.zeros(LAYERS, SLOTS, V_HEADS, D, D, device="cuda")
    conv, ssm = conv_pool[LAYER], ssm_pool[LAYER]
    if dedup:  # SGLang's deduplicated window layout (step 6)
        win_phys = torch.zeros(LAYERS, ROWS, CONV_CH, T + 2, dtype=torch.bfloat16, device="cuda")
        win_pool = win_phys.as_strided((LAYERS, ROWS, T, CONV_CH, 3),
                                       (ROWS * CONV_CH * (T + 2), CONV_CH * (T + 2), 1, T + 2, 1))
    else:
        win_pool = torch.zeros(LAYERS, ROWS, T, CONV_CH, 3, dtype=torch.bfloat16, device="cuda")
    snap_pool = torch.zeros(LAYERS, ROWS, T, V_HEADS, D, D, device="cuda")
    win, snap = win_pool[LAYER], snap_pool[LAYER]
    cache_idx = torch.tensor([SLOT], dtype=torch.int32, device="cuda")
    row_idx = torch.tensor([ROW], dtype=torch.int32, device="cuda")

    ffn_w = (hc2["hc_norm.weight"], hc2["input_mix_weight_down.weight"], hc2["input_mix_weight_up.weight"],
             hc2["block_inject_weight.weight"], mw["router"], mw["w_sg"], mw["w13"], mw["s13"], mw["w2"], mw["s2"])
    attn_w, mb = block_args(w, hc1), mailboxes()
    r = torch.empty(T, HC, dtype=torch.bfloat16, device="cuda")
    r_out = torch.empty_like(r)
    tr = new_trace(STEPS if traced else 1, NBLK, 4)
    args = lambda i: (r, r_out, *attn_w, conv, ssm, cache_idx, win, snap, row_idx, *ffn_w, *mb.values(),
                      tr[i if traced else 0], fx.Stream(torch.cuda.current_stream()))
    run = None

    def launch(i):
        nonlocal run
        conv[SLOT], ssm[SLOT] = st_ref["conv"], st_ref["ssm"]
        if run is None:
            run = flyc.compile(build(traced, T, dedup), *args(i))  # compiling also runs this launch
        else:
            run(*args(i))
        torch.cuda.synchronize()

    def step(i):
        r.copy_(torch.randn(T, HC, dtype=torch.bfloat16, device="cuda"))
        r1_ref, snaps, _ = attn_reference(r, w, hc1, st_ref)
        out_ref = torch.cat([ffn_ref(r1_ref[t:t + 1], hc2, mw)[-1] for t in range(T)])
        launch(i)
        r1_k = from_mailbox(mb["r1_mb"])[0].view(T, HC)
        x2_k = from_mailbox(mb["x2_mb"])[0].view(T, -1)
        logits_k = mb["logits_mb"].view(T, -1, 2)[..., 0].contiguous().view(torch.float32)
        m = {"R' max ulps": ulps(r1_k, r1_ref, r),
             "ssm snapshot err": max(rel(snap[ROW, t], snaps[t]["ssm"]) for t in range(T)),
             "conv window exact": min((win[ROW, t] == snaps[t]["conv"]).float().mean().item() for t in range(T)),
             "R_out max ulps": ulps(r_out, out_ref, r1_ref)}
        x2_exact, top_eq, out_kx, y_kx, routes = [], [], [], [], []
        for t in range(T):
            x2_r, logits_r, _, _, _ = ffn_ref(r1_k[t:t + 1], hc2, mw)
            x2_exact.append((x2_k[t] == x2_r[0]).float().mean().item())
            _, ids, wts = route(x2_k[t:t + 1], mw, logits_k[t])  # the kernel's own routing
            top_eq.append(float(set(ids[:TOPK].tolist()) == set(route(x2_k[t:t + 1], mw)[1][:TOPK].tolist())))
            y, _ = moe_ref(x2_k[t:t + 1], ids, wts, mw)
            n2 = ref.hc_mix(r1_k[t:t + 1], hc2)[1]
            out_kx.append(ref.hc_combine(r1_k[t:t + 1], n2, y, hc2))
            y_kx.append(y)
            routes.append((ids, wts))
        y_stock = moe_stock(x2_k, torch.cat([i for i, _ in routes]), torch.cat([w_ for _, w_ in routes]), mw)
        out_stock = torch.cat([ref.hc_combine(r1_k[t:t + 1], ref.hc_mix(r1_k[t:t + 1], hc2)[1], y_stock[t:t + 1], hc2)
                               for t in range(T)])
        m["x2 exact, ref on kernel R'"] = min(x2_exact)
        m["top-10 = ref on kernel x2"] = min(top_eq)
        m["R_out max ulps, ref on kernel x2"] = ulps(r_out, torch.cat(out_kx), r1_k)
        m["R_out max ulps, stock MoE on kernel x2"] = ulps(r_out, out_stock, r1_k)
        m["y stock vs ref on kernel x2"] = rel(y_stock, torch.cat(y_kx))
        m["unique experts"] = float(len(set(torch.cat([i for i, _ in routes]).tolist())))
        got = (r_out.clone(), win[ROW].clone(), snap[ROW].clone())
        launch(i)
        m["rerun bit-identical"] = float(all(torch.equal(a, b) for a, b in zip(got, (r_out, win[ROW], snap[ROW]))))
        return m

    hist = [step(i) for i in range(STEPS)]
    low_is_bad = lambda k: "exact" in k or "identical" in k or "=" in k
    print(f"{STEPS} verify launches ({T} tokens), each from the reference's state:")
    for k in hist[0]:
        v = sorted(h[k] for h in hist)
        print(f"  {k:40} median {v[len(v) // 2]:.3g}, worst {v[0] if low_is_bad(k) else v[-1]:.3g}")
    worst = {k: (min if low_is_bad(k) else max)(h[k] for h in hist) for k in hist[0]}
    assert worst["rerun bit-identical"] == 1.0, "the kernel is not deterministic: a race between blocks"
    assert worst["R_out max ulps, ref on kernel x2"] <= 4, "the MoE stages are wrong"

    experts = sorted(h["unique experts"] for h in hist)[len(hist) // 2]
    expert_bytes = sum(t[0].numel() for t in mw["raw"].values()) * experts
    nbytes = (sum(t.numel() * t.element_size() for t in (*attn_w, *ffn_w[:6])) + expert_bytes
              + (1 + T) * V_HEADS * D * D * 4)
    bound = nbytes / STREAM_GBPS / 1e3
    print(f"\n{nbytes / 1e6:.1f} MB per launch ({experts:.0f} unique experts, {expert_bytes / 1e6:.1f} MB); "
          f"bound at {STREAM_GBPS} GB/s: {bound:.1f} us")
    if traced:
        kern = span_us(tr, 0, LAST_MARK).median().item()
        print(f"in-kernel, median: R_out written {kern:.1f} us ({bound / kern:.0%} of bound)")
        write_trace("traces/step8_gdn_layer_verify.json", tr[-2:], SPANS)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(LAUNCHES):
        run(*args(0))
    end.record()
    torch.cuda.synchronize()
    per = start.elapsed_time(end) * 1e3 / LAUNCHES
    print(f"per launch, back to back: {per:.1f} us ({bound / per:.0%} of bound)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-trace", action="store_true", help="compile the kernel without timeline marks")
    parser.add_argument("--dense-windows", action="store_true", help="dense conv-window scratch, not deduplicated")
    a = parser.parse_args()
    main(traced=not a.no_trace, dedup=not a.dense_windows)
