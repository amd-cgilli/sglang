"""Step 7: a whole GDN layer in one launch (gdn_layer.py): the attention half, then the MXFP4 MoE.

    R' = hc_combine1(R, gdn(hc_mix1(R)));   R_out = hc_combine2(R', moe(hc_mix2(R')))

Real layer-0 weights from the MXFP4 checkpoint, the experts laid out as SGLang holds them (padded,
aiter's shuffle_weight and e8m0_shuffle). Each step starts from the reference's state with a random
R and compares the kernel's handoffs (R', x2, logits, h) and R_out with:

    ref        the torch chain (qwen38_ref + the MXFP4 math in moe_ref), end to end
    on kernel  the same op from the kernel's own inputs, to isolate it from upstream rounding flips
    stock      aiter's fused_moe (what SGLang runs) on the kernel's x2 and routing

  cd megakernel/basics && bash ../qwen38_flash_next/bench/gpu_run.sh env HIP_VISIBLE_DEVICES=7 \
      FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 step7_gdn_layer.py [--no-trace]
"""

import argparse
import glob
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from safetensors import safe_open

import flydsl.compiler as flyc
import flydsl.expr as fx

from gdn_core_op import CONV_CH, D, V_HEADS
from gdn_layer import E, H_VALUES, LAST_MARK, SPANS, build, mailboxes
from gemv_op import NBLK
from hc_op import HC
from moe_op import INTER, INTER_PAD, N_ROUTED, SHARED_ID, TOPK
from step2_gemv import from_mailbox
from step4_gdn_core import PREFILL
from step5_gdn_block import block_args, rel, ulps
from timeline import new_trace, span_us, write_trace

import step4_gdn_core  # noqa: F401  (puts the reference on sys.path)
import qwen38_ref as ref  # noqa: E402

CKPT = glob.glob("/sgl-workspace/sglang/hf_cache/hub/models--amd--Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8/snapshots/*")[0]
LAYERS, SLOTS, LAYER, SLOT = 2, 6, 1, 3
STEPS, LAUNCHES = 32, 100
STREAM_GBPS = 6000
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def load(layer, part):
    pre = f"model.language_model.layers.{layer}.{part}."
    by_file = defaultdict(list)
    for key, fname in json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"].items():
        if key.startswith(pre):
            by_file[fname].append(key)
    out = {}
    for fname, keys in by_file.items():
        with safe_open(f"{CKPT}/{fname}", "pt", device="cuda") as f:
            out.update({k[len(pre):]: f.get_tensor(k) for k in keys})
    return out


def moe_weights(layer):
    """Checkpoint experts stacked [513, ...] (512 = the shared expert), plus SGLang's tensors."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    m = load(layer, "mlp")
    expert = lambda e: f"experts.{e}." if e < N_ROUTED else "shared_expert."
    raw = {f"{p}{s}": torch.stack([m.pop(f"{expert(e)}{p}_proj.{s}") for e in range(E)])
           for p in ("gate", "up", "down") for s in ("weight", "weight_scale")}
    w13 = torch.zeros(E, 2 * INTER_PAD, HC // 8, dtype=torch.uint8, device="cuda")
    s13 = torch.zeros(E, 2 * INTER_PAD, HC // 128, dtype=torch.uint8, device="cuda")
    for i, p in enumerate(("gate", "up")):
        w13[:, i * INTER_PAD: i * INTER_PAD + INTER] = raw[f"{p}weight"]
        s13[:, i * INTER_PAD: i * INTER_PAD + INTER] = raw[f"{p}weight_scale"]
    w2 = torch.zeros(E, ref.H, INTER_PAD // 2, dtype=torch.uint8, device="cuda")
    s2 = torch.zeros(E, ref.H, INTER_PAD // 32, dtype=torch.uint8, device="cuda")
    w2[..., : INTER // 2], s2[..., : INTER // 32] = raw["downweight"], raw["downweight_scale"]
    shuffle_scales = lambda s: e8m0_shuffle(s.view(-1, s.shape[-1])).view(s.shape)
    sgl = {"w13": shuffle_weight(w13, (16, 16)), "s13": shuffle_scales(s13),
           "w2": shuffle_weight(w2, (16, 16)), "s2": shuffle_scales(s2)}
    return {"router": m["gate.weight"], "w_sg": m["shared_expert_gate.weight"], "raw": raw, **sgl}


# ===== MXFP4 reference math =====


def dequant(packed, scale):
    """[rows, B] FP4 bytes (low nibble first), [rows, B / 16] E8M0 -> fp32 [rows, 2 B]."""
    lut = torch.tensor(E2M1 + [-v for v in E2M1], device=packed.device)
    v = torch.stack([lut[(packed & 15).long()], lut[(packed >> 4).long()]], -1).flatten(-2)
    return v * torch.exp2(scale.float() - 127).repeat_interleave(32, -1)


def mx_round(v):
    """aiter's MXFP4 activation quantizer, back in fp32: per 32 values, e = ceil(log2(amax / 6))
    (RoundUp, aiter's default); E2M1 round-to-nearest-even."""
    g = v.float().unflatten(-1, (-1, 32))
    u = (g.abs().amax(-1, keepdim=True) * torch.tensor(1 / 6, dtype=torch.float32)).view(torch.int32)
    e = (((u >> 23) & 0xFF) + ((u & 0x7FFFFF) != 0).int() - 127).float()
    q = g * torch.exp2(-e)
    a = q.abs()
    step = torch.exp2(torch.floor(torch.log2(a.clamp_min(1.0))) - 1)
    r = torch.where(a < 1, torch.round(a * 2) / 2, torch.clamp(torch.round(a / step) * step, max=6.0))
    return (torch.sign(q) * r * torch.exp2(e)).flatten(-2)


def route(x, mw, logits=None):
    """bf16 logits (or the given ones); top 10 of the softmax, renormalized; slot 10 = the shared
    expert, sigmoid gate."""
    logits = ref.linear(x, mw["router"])[0].float() if logits is None else logits
    w, ids = torch.sort(logits.softmax(-1), descending=True, stable=True)  # ties: lowest id first, as the kernel
    w, ids = w[:TOPK], ids[:TOPK]
    shared = torch.sigmoid(ref.linear32(x, mw["w_sg"]))[0]
    return logits, torch.cat([ids, ids.new_tensor([SHARED_ID])]), torch.cat([w / w.sum(), shared])


def moe_ref(x, ids, wts, mw, stock_sum=False):
    """y (bf16 [1, H]) and the stage-1 outputs h ([11, 640], MXFP4-rounded). The 11 terms are
    summed in fp32 and rounded once (the kernel), or with stock_sum as stock does: each term
    rounded to bf16, then bf16 adds (in slot order here; stock's atomics land in any order)."""
    raw, xq = mw["raw"], mx_round(x)
    h, y = [], torch.zeros(1, ref.H, device=x.device)
    for e, w in zip(ids.tolist(), wts):
        g = xq @ dequant(raw["gateweight"][e], raw["gateweight_scale"][e]).T
        u = xq @ dequant(raw["upweight"][e], raw["upweight_scale"][e]).T
        h.append(mx_round(ref.bf16(F.silu(g) * u)))
        term = w * (h[-1] @ dequant(raw["downweight"][e], raw["downweight_scale"][e]).T)
        y = ref.bf16(y + ref.bf16(term)).float() if stock_sum else y + term
    return ref.bf16(y), torch.cat(h)


def moe_stock(x, ids, wts, mw):
    """aiter's fused_moe as SGLang calls it (quark W4A4 MXFP4, separated gate / up)."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    w13, w2 = mw["w13"].view(torch.float4_e2m1fn_x2), mw["w2"].view(torch.float4_e2m1fn_x2)
    w13.is_shuffled = w2.is_shuffled = True
    n = x.shape[0]  # tokens: ids, wts are [n * 11]
    return fused_moe(x, w13, w2, wts.view(n, -1).float(), ids.view(n, -1).int(), activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, w1_scale=mw["s13"], w2_scale=mw["s2"],
                     intermediate_pad=INTER_PAD - INTER, gate_mode=GateMode.SEPARATED.value)


def layer_ref(r, w, hc1, hc2, mw, st):
    """R', x2, logits, ids, h, R_out of the torch chain; st is advanced."""
    x, n = ref.hc_mix(r, hc1)
    r1 = ref.hc_combine(r, n, ref.gdn(x, w, st), hc1)
    return (r1, *ffn_ref(r1, hc2, mw))


def ffn_ref(r1, hc2, mw):
    x2, n2 = ref.hc_mix(r1, hc2)
    logits, ids, wts = route(x2, mw)
    y, h = moe_ref(x2, ids, wts, mw)
    return x2, logits, ids, h, ref.hc_combine(r1, n2, y, hc2)


def main(traced):
    torch.manual_seed(0)
    w, hc1, hc2 = load(0, "linear_attn"), load(0, "attn_hyper_connection"), load(0, "mlp_hyper_connection")
    mw = moe_weights(0)
    st_ref = {"conv": torch.zeros(CONV_CH, 3, dtype=torch.bfloat16, device="cuda"),
              "ssm": torch.zeros(V_HEADS, D, D, device="cuda")}
    ref.gdn(torch.randn(PREFILL, ref.H, dtype=torch.bfloat16, device="cuda"), w, st_ref)
    conv_pool = torch.zeros(LAYERS, SLOTS, CONV_CH, 3, dtype=torch.bfloat16, device="cuda")
    ssm_pool = torch.zeros(LAYERS, SLOTS, V_HEADS, D, D, device="cuda")
    conv, ssm = conv_pool[LAYER], ssm_pool[LAYER]
    cache_idx = torch.tensor([SLOT], dtype=torch.int32, device="cuda")

    ffn_w = (hc2["hc_norm.weight"], hc2["input_mix_weight_down.weight"], hc2["input_mix_weight_up.weight"],
             hc2["block_inject_weight.weight"], mw["router"], mw["w_sg"], mw["w13"], mw["s13"], mw["w2"], mw["s2"])
    attn_w, mb = block_args(w, hc1), mailboxes()
    r = torch.empty(1, HC, dtype=torch.bfloat16, device="cuda")
    r_out = torch.empty_like(r)
    tr = new_trace(STEPS if traced else 1, NBLK, 4)
    args = lambda i: (r, r_out, *attn_w, conv, ssm, cache_idx, *ffn_w, *mb.values(), tr[i if traced else 0],
                      fx.Stream(torch.cuda.current_stream()))
    run = None

    def launch(i):
        nonlocal run
        if run is None:
            run = flyc.compile(build(traced), *args(i))  # compiling also runs this launch
        else:
            run(*args(i))
        torch.cuda.synchronize()

    def step(i):
        conv[SLOT], ssm[SLOT] = st_ref["conv"], st_ref["ssm"]
        r.copy_(torch.randn(1, HC, dtype=torch.bfloat16, device="cuda"))
        st = {k: v.clone() for k, v in st_ref.items()}
        r1_ref, x2_ref, _, _, _, out_ref = layer_ref(r, w, hc1, hc2, mw, st)
        launch(i)
        r1_k = from_mailbox(mb["r1_mb"])[0].view(1, HC)
        x2_k = from_mailbox(mb["x2_mb"])[0].view(1, -1)
        logits_k = mb["logits_mb"].view(-1, 2)[:, 0].view(torch.float32)
        h_k = from_mailbox(mb["h_mb"])[0].view(-1, INTER)
        x2_kr, logits_kx, _, _, out_kr = ffn_ref(r1_k, hc2, mw)  # the FFN half on the kernel's R'
        _, ids_kl, wts_kl = route(x2_k, mw, logits_k)  # the kernel's own routing
        y_kx, h_kx = moe_ref(x2_k, ids_kl, wts_kl, mw)
        n2_k = ref.hc_mix(r1_k, hc2)[1]
        y_stock = moe_stock(x2_k, ids_kl, wts_kl, mw)
        m = {"R' max ulps": ulps(r1_k, r1_ref, r),
             "ssm state err": rel(ssm[SLOT], st["ssm"]),
             "x2 exact, ref on kernel R'": (x2_k == x2_kr).float().mean().item(),
             "logits max err, ref on kernel x2": rel(logits_k, logits_kx),
             "top-10 = ref on kernel x2": float(set(ids_kl[:TOPK].tolist()) == set(route(x2_k, mw)[1][:TOPK].tolist())),
             "h exact, ref on kernel x2": (h_k == h_kx.to(torch.bfloat16).view(-1, INTER)).float().mean().item(),
             "R_out max ulps": ulps(r_out, out_ref, r1_ref),
             "R_out max ulps, ref on kernel R'": ulps(r_out, out_kr, r1_k),
             "R_out max ulps, ref on kernel x2": ulps(r_out, ref.hc_combine(r1_k, n2_k, y_kx, hc2), r1_k),
             "R_out max ulps, stock MoE on kernel x2": ulps(r_out, ref.hc_combine(r1_k, n2_k, y_stock, hc2), r1_k),
             "y stock vs ref on kernel x2": rel(y_stock, y_kx),
             "y stock vs stock-sum ref on kernel x2": rel(y_stock, moe_ref(x2_k, ids_kl, wts_kl, mw, True)[0])}
        got = r_out.clone()
        conv[SLOT], ssm[SLOT] = st_ref["conv"], st_ref["ssm"]
        launch(i)
        m["rerun bit-identical"] = float(torch.equal(got, r_out))
        return m

    hist = [step(i) for i in range(STEPS)]
    low_is_bad = lambda k: "exact" in k or "identical" in k or "=" in k
    print(f"{STEPS} layer launches, each from the reference's state:")
    for k in hist[0]:
        v = sorted(h[k] for h in hist)
        print(f"  {k:40} median {v[len(v) // 2]:.3g}, worst {v[0] if low_is_bad(k) else v[-1]:.3g}")
    worst = {k: (min if low_is_bad(k) else max)(h[k] for h in hist) for k in hist[0]}
    assert worst["rerun bit-identical"] == 1.0, "the kernel is not deterministic: a race between blocks"
    assert worst["R_out max ulps, ref on kernel x2"] <= 8, "the MoE stages are wrong"

    expert_bytes = sum(t[0].numel() for t in mw["raw"].values()) * (TOPK + 1)
    nbytes = (sum(t.numel() * t.element_size() for t in (*attn_w, *ffn_w[:6])) + expert_bytes
              + 2 * V_HEADS * D * D * 4)
    bound = nbytes / STREAM_GBPS / 1e3
    print(f"\n{nbytes / 1e6:.1f} MB per launch ({expert_bytes / 1e6:.1f} MB of experts); "
          f"bound at {STREAM_GBPS} GB/s: {bound:.1f} us")
    if traced:
        kern = span_us(tr, 0, LAST_MARK).median().item()
        print(f"in-kernel, median: R_out written {kern:.1f} us ({bound / kern:.0%} of bound)")
        write_trace("traces/step7_gdn_layer.json", tr[-2:], SPANS)
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
    main(traced=not parser.parse_args().no_trace)
