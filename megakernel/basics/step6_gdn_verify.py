"""Step 6: the attention half of a GDN layer for an MTP verify pass (T = 4 tokens) in one launch.

gdn_block_verify vs qwen38_ref: R' = hc_combine(R, n, gdn(x)) per token row, the 4 GDN steps in
sequence from the slot's state, and the state after each step against SGLang's verify scratch
(conv windows, SSM snapshots). Real layer-0 weights; pools and scratch shaped like SGLang's.

    A: each launch from the reference's state: R', every snapshot, main state untouched, other
       scratch rows untouched, and a rerun from the same inputs is bit-identical
    B: launches chained like SGLang's commit: after each, a random accepted length k is drawn and
       snapshot k - 1 is copied into the main state, on both sides

  cd megakernel/basics && bash ../qwen38_flash_next/bench/gpu_run.sh env HIP_VISIBLE_DEVICES=7 \
      FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 step6_gdn_verify.py [--no-trace]
"""

import argparse
import random

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

from gdn_block_verify import HC, LAST_MARK, SPANS, T_VERIFY, build, mailboxes
from gdn_core_op import CONV_CH, D, V_HEADS
from gemv_op import NBLK
from step2_gemv import from_mailbox
from step4_gdn_core import PREFILL, load_gdn
from step5_gdn_block import block_args, load_hc, rel, ulps
from timeline import new_trace, span_us, write_trace

import step4_gdn_core  # noqa: F401  (puts the reference on sys.path)
import qwen38_ref as ref  # noqa: E402

T = T_VERIFY
LAYERS, SLOTS, LAYER, SLOT, ROWS, ROW = 2, 6, 1, 3, 3, 1
STEPS, LAUNCHES = 32, 100
STREAM_GBPS = 6000


def copy(st):
    return {k: v.clone() for k, v in st.items()}  # ref.gdn replaces the entries it's given


def reference(r, w, hc, st, x=None):
    """R' for T rows and the state after each token, from state st (unchanged). Row by row: a
    4-row matmul sums in another order than a 1-row one, and the kernel's per-token math is the
    1-row (decode) math."""
    mixed = [ref.hc_mix(r[t:t + 1], hc) for t in range(T)]
    xr = torch.cat([xm for xm, _ in mixed])
    x = xr if x is None else x
    st, r_out, snaps = copy(st), [], []
    for t in range(T):
        out = ref.gdn(x[t:t + 1], w, st)
        snaps.append(copy(st))
        r_out.append(ref.hc_combine(r[t:t + 1], mixed[t][1], out, hc))
    return torch.cat(r_out), snaps, xr


def main(traced, dedup):
    torch.manual_seed(0)
    random.seed(0)
    w, hc = load_gdn(0), load_hc(0)
    st_ref = {"conv": torch.zeros(CONV_CH, 3, dtype=torch.bfloat16, device="cuda"),
              "ssm": torch.zeros(V_HEADS, D, D, device="cuda")}
    ref.gdn(torch.randn(PREFILL, ref.H, dtype=torch.bfloat16, device="cuda"), w, st_ref)
    conv_pool = torch.zeros(LAYERS, SLOTS, CONV_CH, 3, dtype=torch.bfloat16, device="cuda")
    ssm_pool = torch.zeros(LAYERS, SLOTS, V_HEADS, D, D, device="cuda")
    conv, ssm = conv_pool[LAYER], ssm_pool[LAYER]
    conv[SLOT], ssm[SLOT] = st_ref["conv"], st_ref["ssm"]
    if dedup:  # SGLang's deduplicated layout: [layers, rows, dim, T + 2] seen as [.., T, dim, 3]
        win_phys = torch.zeros(LAYERS, ROWS, CONV_CH, T + 2, dtype=torch.bfloat16, device="cuda")
        win_pool = win_phys.as_strided((LAYERS, ROWS, T, CONV_CH, 3),
                                       (ROWS * CONV_CH * (T + 2), CONV_CH * (T + 2), 1, T + 2, 1))
    else:
        win_pool = torch.zeros(LAYERS, ROWS, T, CONV_CH, 3, dtype=torch.bfloat16, device="cuda")
    snap_pool = torch.zeros(LAYERS, ROWS, T, V_HEADS, D, D, device="cuda")
    win, snap = win_pool[LAYER], snap_pool[LAYER]
    cache_idx = torch.tensor([SLOT], dtype=torch.int32, device="cuda")
    row_idx = torch.tensor([ROW], dtype=torch.int32, device="cuda")

    weights, mb = block_args(w, hc), mailboxes()
    w_id = {**w, "out_proj.weight": torch.eye(V_HEADS * D, dtype=torch.bfloat16, device="cuda")}
    r = torch.empty(T, HC, dtype=torch.bfloat16, device="cuda")
    r_out = torch.empty_like(r)
    tr = new_trace(2 * STEPS if traced else 1, NBLK, 4)
    args = lambda i: (r, r_out, *weights, conv, ssm, cache_idx, win, snap, row_idx, *mb.values(),
                      tr[i if traced else 0], fx.Stream(torch.cuda.current_stream()))
    run = None

    def launch(i):
        nonlocal run
        if run is None:
            run = flyc.compile(build(traced, T, dedup), *args(i))  # compiling also runs this launch
        else:
            run(*args(i))
        torch.cuda.synchronize()

    def step(i, synced):
        if synced:
            conv[SLOT], ssm[SLOT] = st_ref["conv"], st_ref["ssm"]
        r.copy_(torch.randn(T, HC, dtype=torch.bfloat16, device="cuda"))
        main_before = (conv[SLOT].clone(), ssm[SLOT].clone())
        r_ref, snaps, _ = reference(r, w, hc, st_ref)
        launch(i)
        x_k = from_mailbox(mb["x_mb"])[0].view(T, -1)
        m = {"R' max ulps": ulps(r_out, r_ref, r),
             "ssm snapshot err": max(rel(snap[ROW, t], snaps[t]["ssm"]) for t in range(T)),
             "conv window exact": min((win[ROW, t] == snaps[t]["conv"]).float().mean().item() for t in range(T)),
             "main state untouched": float(torch.equal(conv[SLOT], main_before[0])
                                           and torch.equal(ssm[SLOT], main_before[1])),
             "other rows untouched": float(all(t.abs().max().item() == 0 for t in (
                 snap_pool[0], win_pool[0], snap[:ROW], snap[ROW + 1:], win[:ROW], win[ROW + 1:])))}
        if synced:
            r_kx, snaps_kx, xr = reference(r, w, hc, st_ref, x_k)
            m["x exact"] = (x_k == xr).float().mean().item()
            m["R' max ulps, ref on kernel x"] = ulps(r_out, r_kx, r)
            for t in range(T):
                m[f"snapshot {t} err, ref on kernel x"] = rel(snap[ROW, t], snaps_kx[t]["ssm"])
            o_k = from_mailbox(mb["o_mb"])[0].view(T, -1)
            st = copy(st_ref)
            for t in range(T):
                o_ref = ref.gdn(x_k[t:t + 1], w_id, st)[0]
                m[f"o token {t} exact, ref on kernel x"] = (o_k[t] == o_ref).float().mean().item()
            # out_proj and the combine alone: the reference from the kernel's own o.
            r_ko = torch.cat([ref.hc_combine(r[t:t + 1], ref.hc_mix(r[t:t + 1], hc)[1],
                                             ref.linear(o_k[t:t + 1], w["out_proj.weight"]), hc) for t in range(T)])
            m["R' max ulps, ref on kernel o"] = ulps(r_out, r_ko, r)
            got = (r_out.clone(), win[ROW].clone(), snap[ROW].clone())
            launch(i)
            m["rerun bit-identical"] = float(all(torch.equal(a, b) for a, b in zip(got, (r_out, win[ROW], snap[ROW]))))
        return m, snaps

    def summary(name, hist):
        low_is_bad = lambda k: "exact" in k or "untouched" in k or "identical" in k
        print(f"  {name}:")
        for k in hist[0]:
            v = sorted(h[k] for h in hist)
            print(f"    {k:32} median {v[len(v) // 2]:.3g}, worst {v[0] if low_is_bad(k) else v[-1]:.3g}")
        return {k: (min if low_is_bad(k) else max)(h[k] for h in hist) for k in hist[0]}

    a = summary(f"A: {STEPS} verify launches, each from the reference's state", [step(i, True)[0] for i in range(STEPS)])
    hist = []
    for i in range(STEPS):  # B: commit a random accepted length on both sides, like SGLang after verify
        m, snaps = step(STEPS + i, False)
        k = random.randint(1, T)
        st_ref = copy(snaps[k - 1])
        conv[SLOT], ssm[SLOT] = win[ROW, k - 1], snap[ROW, k - 1]
        hist.append(m)
    b = summary(f"B: {STEPS} verify launches, chained through random accepted lengths", hist)
    assert a["rerun bit-identical"] == 1.0, "the kernel is not deterministic: a race between blocks"
    assert a["main state untouched"] == 1.0 and a["other rows untouched"] == 1.0, "wrote outside its scratch row"
    assert a["R' max ulps, ref on kernel o"] <= 4 and a["ssm snapshot err"] < 1e-3, "a kernel op is wrong"
    assert b["R' max ulps"] <= 64 and b["ssm snapshot err"] < 1e-2, "the chained state drifted"

    nbytes = sum(t.numel() * t.element_size() for t in weights) + (1 + T) * V_HEADS * D * D * 4
    bound = nbytes / STREAM_GBPS / 1e3
    print(f"\n{nbytes / 1e6:.1f} MB per launch (weights once, state read, {T} snapshots written); "
          f"bound at {STREAM_GBPS} GB/s: {bound:.1f} us")
    if traced:
        kern = span_us(tr, 0, LAST_MARK).median().item()
        print(f"in-kernel, median: R' written {kern:.1f} us ({bound / kern:.0%} of bound)")
        write_trace("traces/step6_gdn_verify.json", tr[-2:], SPANS)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(LAUNCHES):
        run(*args(0))
    end.record()
    torch.cuda.synchronize()
    per = start.elapsed_time(end) * 1e3 / LAUNCHES
    print(f"per launch, back to back: {per:.1f} us ({bound / per:.0%} of bound)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-trace", action="store_true", help="compile the kernel without timeline marks")
    parser.add_argument("--dense-windows", action="store_true",
                        help="dense conv-window scratch instead of SGLang's deduplicated one (its default on GPU)")
    a = parser.parse_args()
    main(traced=not a.no_trace, dedup=not a.dense_windows)
