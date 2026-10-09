# First-principles microbenchmarks (FlyDSL, MI350X)

These measure the assumptions behind the megakernel design one at a time, before any model ops exist.

```bash
cd megakernel/basics && bash ../qwen38_flash_next/bench/gpu_run.sh env HIP_VISIBLE_DEVICES=7 \
    FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 step1_sync.py      # or step2_gemv.py ... step5_gdn_block.py
```

| File | |
|---|---|
| `common.py` | device helpers: clock, XCD id, atomics, tagged-pair mailboxes (`put_pair`, `poll_pairs`) |
| `gemv_op.py` | the GEMV op `emit_gemv`, emitted inline into any kernel (steps 2 and 3); x in registers or LDS, schedule always pinned |
| `gdn_core_op.py` | `emit_gdn_core`: GDN conv, gates, delta-rule recurrence and gated norm for one value head |
| `hc_op.py` | the gated residual: hc_norm, hc_up, the inject coefficients, epilogues for hc_down and the combine, the launch epoch |
| `gdn_verify_op.py` | the GDN recurrence for T draft tokens (MTP verify): conv windows and SSM snapshots per token |
| `gdn_block_verify.py` | a GDN layer's attention half for an MTP verify pass (T = 4) in one launch (step 6) |
| `gdn_block.py` | a GDN layer's attention half in one launch (step 5); also what SGLang runs behind the toggle |
| `timeline.py` | per-wave in-kernel timestamps, exported as a Perfetto trace |
| `buffer_ops.py` | copied from the FlyDSL repo `kernels/common/`, as `skinny_bf16.py` imports it |
| `skinny_bf16.py` | reference skinny GEMV, the inner loop of step 2 |

**Monitoring.** A kernel calls `mark(trace, waves, m)` at named points. Lane 0 of each wave
stores the wall clock (10 ns ticks, shared by all CUs) into `int64 [blocks, waves, 8]`, and
column 7 holds the block's XCD. `write_trace` turns spans between marks into a Chrome trace
under `traces/`. Open it at https://ui.perfetto.dev: one process per XCD, one track per wave,
and the last 3 launches with their real gaps.

Tracing is a compile-time switch. `mark(..., enabled)` takes a Python bool checked with
`const_expr`, and the kernel builders take `traced`, so `--no-trace` (steps 2 and 3) compiles
kernels with no clock reads and no trace stores. Its cost is within run-to-run noise: per launch,
tracing on vs off differed by 1.5 to 3.5%, while untraced `skinny_bf16` varied by 1.5 to 2.5%
between the same two runs.

## Step 1: block-to-block handoff (`step1_sync.py`)

A hub block sends "go r" to N spoke blocks. Each spoke replies with its slice of a payload, and
the hub checks every value against the round number. N = 1 is a ping-pong (one round is two
one-way handoffs). N = 255 is a whole-grid fan-in, which is what a stage boundary looks like.

| Protocol | How a value is handed over |
|---|---|
| `counter` (flymk today) | plain stores, a block barrier, then a release atomic add on a counter; the consumer polls relaxed and then issues one acquire fence |
| `counter_nofence` | the same with relaxed atomics and no fences, to show why the fences exist |
| `tagged` (FlyDSL monokernel) | each value is stored as one 8-byte `(value, tag)` agent-scope relaxed store; the consumer polls the data itself until every tag equals the round |

**Results** on GPU 7, 2026-10-08. Each cell is ns per round, as the median of 5 launches of 5000 rounds.
Two separate processes are shown as `a / b`.

| Case | counter | counter_nofence | tagged | tagged / counter |
|---|---|---|---|---|
| ping-pong 4 B, same XCD | 1340 / 1521 | 751 / 916, **all stale** | **810 / 1043** | 0.60 / 0.69 |
| ping-pong 4 B, cross XCD | 1707 / 2157 | 826 / 1279, **all stale** | **991 / 1152** | 0.58 / 0.53 |
| ping-pong 4 KB, cross XCD | 1781 / 2240 | 904 / 1340, **all stale** | **1368 / 1468** | 0.77 / 0.66 |
| fan-in, 255 spokes × 16 B | 8173 / 8541 | 3875 / 4202, **all stale** | **1547 / 1635** | **0.19 / 0.19** |

`counter` and `tagged` read 0 stale values out of 25.6 M. As a negative control, `tagged` with
the tag check removed reads all 5.12 M values stale, so the checker does catch stale data.

**What this says**

- **The fences are required.** Without release and acquire, the consumer reads stale data on
  essentially every value, even within one XCD.
- **Tagged pairs are correct and faster.** A one-way handoff costs about 400 ns within an XCD and
  500 ns across XCDs, versus 700–1000 ns with a counter. The counter pays for an extra round
  trip (the flag first, then the data) plus the cost of the fences.
- **Fan-in is where counters fail.** 255 producers incrementing one counter serialize on that
  address, which costs about 8 µs. That is as much as a grid barrier (11.6 µs measured earlier
  in HIP). With tagged pairs every producer writes its own data, so the fan-in costs about 1.6 µs.
  Over about 400 stage boundaries per decode step, that is roughly 3.3 ms with counters versus
  0.6 ms with tagged pairs, against a 1.4 ms bandwidth floor.
- **Placement:** block `i` runs on XCD `i % 8`, and 256 blocks occupy 256 distinct CUs.
- **Noise:** absolute times move by about 25% between processes, probably because of GPU clock
  state. Ratios within one process are stable. Compare protocols within a single run.

**Decision for the megakernel:** use tagged pairs for every handoff between blocks. The price is
that a value takes 8 bytes instead of 4, and an op that hands data to another block must write
in pair format.

## Step 2: one GEMV at HBM bandwidth (`step2_gemv.py`)

The op is `y = W x` for one token, written as a megakernel op. It polls x from a tagged-pair
mailbox and publishes y as tagged pairs, so step 3 can chain two of them. The inner loop is
`skinny_bf16.py`'s: 16 B non-temporal loads, 64 lanes cover 512 of K, `v_dot2c_f32_bf16`, and
a DPP wave reduction. The grid is persistent, one block per CU:

- **Block:** a balanced, contiguous range of output rows, in whole pairs.
- **Wave:** K-part `wave % k_split` of every `waves / k_split`-th row of the block.
- **Loads:** one flat list over (row, k-step) with `depth` loads in flight. The first `depth`
  are issued before x is polled.

Weights rotate over more than 672 MB of copies (3x the 224 MB Infinity Cache), so every launch
reads HBM. GPU 7, 2026-10-08, 100 launches; GB/s in-kernel is first to last wave mark (median).

| Kernel | in_proj 16480 x 2560 (84 MB) | out_proj 2560 x 6144 (31 MB) |
|---|---|---|
| stream read, ceiling | 5811 (14.5 us) | 5174 (6.1 us) |
| stream read, 256 MB | 6034 | |
| skinny_bf16, per launch with gaps | 5255 | 3558 |
| **tagged gemv**, best | **5205 (16.2 us)**, waves=8 k_split=1 depth=16 | **4475 (7.0 us)**, waves=4 k_split=4 depth=8 |
| tagged gemv, per launch with gaps | 4648 | 3351 |

**What this says**

- **The mailbox interface costs nothing measurable.** The tagged GEMV runs at 90% (in_proj) and
  86% (out_proj) of a pure read of the same bytes. skinny_bf16 is comparable per launch.
- **Row balance matters most.** out_proj has 10 rows per CU. With `k_split=1` a wave gets 2 or 3
  rows (83% balance) and takes 8.0 us; splitting K over 4 waves makes it 100% and 7.0 us.
  in_proj's 5 k-steps can't split, so it stays at 89 to 95% balance.
- **Depth hardly matters once 4 or more loads are in flight.** In the trace, issuing 16 loads per
  lane takes 2.8 us because the memory queue is full, so bandwidth is already saturated.
- **Polling x costs one memory latency (about 1.9 us), even when x is ready.** A wave's loads
  return in issue order, so x arrives after the prefetched weights. That latency would be paid
  anyway, which is why prefetching before the wait matters in step 3.
- **The tail is the block barrier.** "barrier + publish y" waits up to 4.9 us for the block's
  slowest wave, and the whole GPU then waits for the slowest block. In a megakernel the next op's
  prefetch can fill this tail (step 3).
- **Per-launch numbers lose 10 to 25% to launch gaps** for 7 to 16 us kernels. That is the cost a
  megakernel removes.

## Step 3: a chain of GEMVs in one launch (`step3_chain.py`)

The chain is L GDN-shaped layers: x goes through up (6144 x 2560) to h, then down (2560 x 6144) to
x', and so on. Every handoff is a full fan-in, because each block of op s + 1 needs all of op s's
output. Up uses `k_split=1` and down `k_split=4`, both with fully balanced rows (step 2).

| Variant | |
|---|---|
| `per_op` | one launch per op (step 2's kernel), back to back |
| `fused` | one launch; each op polls its input, then issues its weight loads |
| `prefetch` | one launch; each op issues `depth` weight loads per lane, then polls its input |

GPU 7, 2026-10-08, 100 chains, weights rotated past the Infinity Cache. The bound is weight bytes
at 6.0 TB/s (step 2's 256 MB stream read). In-kernel time comes from a traced run; per chain is
from `--no-trace`.

| 4 layers, 8 GEMVs, 252 MB (bound 41.9 us) | per chain | in-kernel | of bound |
|---|---|---|---|
| per_op, depth 16 | 78.2 us | 67.9 us | 62% |
| fused, depth 16 | 63.1 us | 61.0 us | 69% |
| prefetch, depth 8 | 55.0 us | 53.6 us | 78% |
| **prefetch, depth 16** | **50.6 us** | **47.8 us** | **88%** |

With 1 layer (2 GEMVs, 63 MB, bound 10.5 us), prefetch takes 13.9 us in-kernel (76%), against
15.9 us for per_op and 14.8 us for fused. Starting up and draining the grid is a large share of
a chain this short.

**The approach holds.** Tagged-pair handoffs plus prefetching weights before polling bring a
chain of dependent GEMVs to 88% of the bandwidth bound, against 62% with a launch per op.

**Where the time goes**, from `traces/step3_4layer_*.json` (time each op finishes, in us):

| op | up0 | down0 | up1 | down1 | up2 | down2 | up3 | down3 |
|---|---|---|---|---|---|---|---|---|
| per_op | 7.5 | 15.9 | 24.5 | 33.1 | 41.7 | 50.2 | 59.0 | 67.5 |
| fused | 7.5 | 15.2 | 22.8 | 30.6 | 38.2 | 45.8 | 53.2 | 61.0 |
| prefetch | 8.1 | 14.8 | 21.1 | 26.5 | 32.4 | 38.5 | 44.2 | 49.9 |

- **per_op** pays about 1.2 us of launch gap per op, on top of a 7.3 us op.
- **fused** removes the gap: an op's first blocks start while the previous op is finishing. But
  an op only issues its weight loads once its input arrives, so every handoff still costs a full
  memory latency. One op finishes every 7.6 us.
- **prefetch** overlaps that latency with the previous op's tail. In steady state one op finishes
  every 5.7 to 6.7 us, about 6 us on average, against 5.2 us for 31.5 MB at 6 TB/s.
- **Prefetch depth matters here, unlike in step 2.** 16 loads per lane is 64 KB in flight per CU,
  and it beat depth 8 by 11% (47.8 vs 53.6 us). The handoff bubble is roughly one memory latency
  plus the previous op's slowest block, and the bytes in flight must cover it. Prefetching into
  LDS instead of registers would allow more.

**What's left** is about 0.8 us per op: the first op's startup (8.1 us for one op) and the poll
itself. "poll x" still takes about 2.8 us p50, because it ends only when the slowest block of the
previous op has published.

## Side study: one large-K GEMV, 7168 x 35840 (`bench_large_k.py`)

The question was whether this persistent, statically split approach (close in spirit to
stream-K) beats a single ordinary kernel. The weights are 514 MB, with 2 rotated copies (4.6x the
Infinity Cache). GPU 7, 2026-10-09, 50 launches, output checked against PyTorch.

| Kernel | us per launch | GB/s |
|---|---|---|
| stream read, the same bytes (in-kernel 86.0 us) | 88.9 | 5779 |
| **skinny_bf16, depth 4** | **87.2** | **5894** |
| F.linear (hipBLASLt) | 97.1 | 5294 |
| tagged gemv, best (waves=4, k_split=1, depth=8) | 93.9 | 5473 |
| tagged gemv, k_split=1, schedule not pinned | 120.4 | 4268 |

**No: one ordinary grid wins on an isolated GEMV.** skinny_bf16 matches the plain read, beats
hipBLASLt by 10%, and beats our op by 7%.

- **Stream-K's main benefit doesn't apply here.** It balances work that doesn't divide evenly
  across CUs, and 7168 rows divide evenly over 256 CUs (28 rows each, 100% balance).
- **What a big grid does better** is overlap: skinny_bf16 launches 1792 small blocks, so one
  block's prologue (loading x) and epilogue (reduce, store) overlap other blocks' streaming. A
  persistent kernel pays those once per CU, back to back. Here that is staging x into LDS, about
  4 us per block, only partly covered by the weight prefetch.
- **The compiler decides how many loads are in flight.** Unpinned with `k_split=1`, LLVM read each
  x slice from LDS once and kept all 70 in registers (423 VGPRs), and most waits became
  `vmcnt(0)`/`vmcnt(1)`: about one load in flight. A `rocdl.sched_barrier(0)` after every load
  step, as FlyDSL's monokernel GEMMs do, brings it from 120.4 to 93.9 us. `emit_gemv` now always
  does this. Steps 2 and 3 are unchanged by it: 5.2 / 4.3 TB/s, and 48.4 us in-kernel (87% of the
  bound) for the 4-layer prefetch chain, against 47.8 us before.
- **Each wave must stream whole rows.** Making the k-step the outer loop, so one x slice serves all
  of a wave's rows, ran 4.5x slower (about 450 us), pinned or not. A wave then hops rows on every
  1 KB load. Spreading block rows round-robin across CUs changed nothing.

Persistence pays when ops are chained (step 3), not for one big GEMV.

## Step 4a: the GDN recurrence op (`gdn_core_op.py`, `step4_gdn_core.py`)

The op is three calls, so a megakernel can place other work between them: `gdn_core_prefetch`
(phase 1), `emit_gdn_core` (phases 2 to 5) and `gdn_core_write_state` (phase 6). It runs one value
head per block of 256 threads. Its input is the in_proj mailbox
`[q 2048 | k 2048 | v 6144 | z 6144 | b 48 | a 48]`, and its output is `o[h]`, 128 values that
are normalized and gated, ready for out_proj.

| Phase | Threads | Work |
|---|---|---|
| 1. issue loads | all | the state slice `S[h, v, 64 half : +64]`, the conv weights and old conv state of this thread's 4 channels, the norm weight |
| 2. poll + conv | 0..95 | poll 4 channels of q, k or v; conv width 4 + SiLU, rounded to bf16, to LDS |
| | 96..127 | poll 4 channels of z, to LDS |
| | 128 | poll a and b; `decay = exp(-exp(A_log) softplus(a + dt_bias))`, `beta = bf16(sigmoid(b))` |
| 3. norms | waves 0, 1 | `rsqrt(sum q^2 + eps)` and the same for k |
| 4. recurrence | all | thread (v, half): `S *= decay; e = v - S k; S += beta e k; o = S q`; row dots summed over the lane pair with one DPP add |
| 5. publish | all, then 0..63 | per-head RMSNorm (plain weight), `sigmoid(z)` gate, 64 pairs of o |
| 6. write state | all; 0..95 | S back through LDS (one contiguous 1 KB per wave store); conv state `[tap 1, tap 2, new]` |

**Conv state is a write-after-read hazard between blocks.** The q and k channels of key head
`kh` feed value heads `3kh`, `3kh+1` and `3kh+2`, which run on three blocks. All three read the
old conv state, and only `3kh` writes the new one. Each non-owner publishes an ack pair (tag =
epoch) once its conv has consumed the old values. The owner polls both acks before it writes, and
it writes after publishing o, so the wait is off the critical path.

**Test.** Real layer-0 weights. The state comes from a 256-token reference prefill. Then 64 decode
steps run, with the kernel's state evolving on its own. The reference `o` comes from `qwen38_ref.gdn`
with out_proj = identity, which is exact in bf16. Results on GPU 7, 2026-10-09:

| Over 64 steps | Result |
|---|---|
| o bit-exact | 99.5% or more of values; the rest are one bf16 rounding flip (max relative error 3.2e-4) |
| SSM state | max relative error 5.7e-7 |
| conv state | bit-exact |

The flips come from fp32 summation order in the 128-wide dot products, the L2 norm and the
RMSNorm, which differs from torch's einsum and mean.

**Time.** The launch has 48 blocks. Per launch back to back: 9.8 us untraced, 10.6 us traced.

| Phase (traced, p50) | us |
|---|---|
| issue loads | 2.1 |
| poll + conv + norms | 1.7 |
| recurrence | 1.2 |
| norm + gate + publish o | 0.4 |
| write state | 2.0 |

- **Issuing the loads takes the whole state read.** Once a CU has enough loads outstanding, the
  memory pipeline stops accepting new ones, so 64 KB through one CU takes about 2 us. That's latency,
  not bandwidth. In the megakernel these loads go out before in_proj ends (about 16 us), so they
  are hidden.
- **The critical path from input ready to o published is about 3.4 us:** a poll round trip, the conv
  and norms with two block barriers, the recurrence, and the publish.
- **The state write-back is off the critical path.** It overlaps the next op (out_proj).
- **Next levers:** split a head over 2 blocks, which halves both the per-CU state traffic and the
  recurrence time but moves the RMSNorm reduction across blocks. Or shorten the recurrence's
  64-long dependent FMA chains.

## Step 4b: the GDN mixer in one launch (`step4b_gdn_mixer.py`)

The whole GDN token mixer runs in one kernel. Layer-0 weights, 256 blocks of 256 threads:

```
all blocks       gdn_core_prefetch (only blocks 0..47 load anything: head = block)
                 in_proj  x -> in_mb [16480]       blocks 0..47 take 25 pairs, the rest up to 34
                 prefetch all of this block's out_proj weights   (none on blocks 0..47)
blocks 0..47     emit_gdn_core in_mb -> o_mb [6144], then gdn_core_write_state
blocks 48..255   out_proj o_mb -> y_mb [2560]
```

**Correct.** Against `qwen38_ref.gdn` (real out_proj), with 64 decode steps from a 256-token
reference prefill and each side evolving its own state: y is within 0.58%, the SSM state within
4.8e-4, the conv state exact. Most differences start in in_proj: 99.9% of it is bit-exact, and
the GEMV's sum order flips the rest by one bf16 rounding step.

**Fast enough to keep.** The bound is 122 MB (weights, plus the state read and written) at
6.0 TB/s, or 20.4 us. GPU 7, 2026-10-09:

| Version | y published, in-kernel | per launch, untraced |
|---|---|---|
| the three ops alone, sum of in-kernel times (16.1 + 6.7 + 7.0) | 29.8 us | |
| one launch, ops in sequence | 29.4 us | 32.0 us |
| + state write after out_proj, through LDS | 27.5 us | 29.9 us |
| **+ head blocks: fewer in_proj rows, no out_proj; all out_proj weights prefetched** | **24.6 us (83%)** | **26.3 us (77%)** |

**Where the time goes** in the final version (us from launch start):

```
in_proj done 16.2 -> gdn_core: conv 18.8, recurrence 20.5, o published 21.0
                     meanwhile blocks 48..255 stream all 31.5 MB of out_proj weights (16.2 -> ~21)
                  -> out_proj poll done 22.6 -> dot + reduce 24.2 -> y published 24.4
```

- **gdn_core is fully hidden.** Its 4.8 us critical path runs while out_proj's weights stream in,
  which takes about 5.2 us at full bandwidth anyway.
- **What remains is about 3.4 us after o:** the poll round trip, the last dot products and the
  publish. In a multi-layer kernel the next layer's in_proj prefetch can fill this, as in step 3.

**Lessons, each measured on this kernel:**

- **Balance work, not rows.** Head blocks run gdn_core on top of their GEMV share. With an even
  split they finished in_proj 2.7 us late and out_proj 1.8 us late, and set the critical path.
- **Write state last, and contiguously.** On CDNA a wait on a load also waits for every older store
  (`vmcnt` counts both), so 64 KB of state stores delays every later poll of that block. Stored
  straight from the compute layout, each wave store touched 64 rows, and the CU managed about
  20 GB/s (3 us). Staged through LDS it takes 1.1 us.
- **A deep prefetch costs registers.** Putting out_proj's weights in flight on the head blocks
  during gdn_core slowed gdn_core 2x, with about 120 more live VGPRs.
- **Values crossing a device `if` must exist before it.** FlyDSL's rewrite only carries variables
  defined before the branch. `gemv_prefetch(..., active=...)` gives a placeholder whose loads are
  out of bounds and move no data.

## Step 5: a GDN layer's attention half in one launch (`gdn_block.py`, `step5_gdn_block.py`)

`R' = R + c * out_proj(gdn(in_proj(hc_mix(R))))`, the gated read, the GDN mixer and the gated
write of one GDN layer, on SGLang's own weight and state layouts:

```
all blocks        hc_norm: n = gemma_rmsnorm_per_stream(R), into LDS (redundantly: cheaper than a handoff)
blocks 48..207    hc_down: n -> low [320], bf16(silu(acc / 4))      (emit_gemv: x in LDS, custom publish)
all blocks        hc_up: low -> x [2560], 10 outputs each
all blocks        in_proj: x -> [q k v z | b a]  (qkvz and ba are SGLang's two weights; ba on 0..47)
blocks 0..47      gdn_core, then the conv and SSM state at slot cache_idx[0], read on device
blocks 48..255    inject c [4]; out_proj o -> R' with the combine in its epilogue
```

- **Tags without host arguments.** Each launch's tag is `sync[0] + 1`; the last block to finish
  (an atomic count in `sync[1]`) stores it. Every block reads `sync[0]` before it counts itself, so
  none sees the new value, and the launch can be replayed from a graph.
- **Correct.** Real layer-0 weights; the state is in a pool shaped like SGLang's (`[layers, slots,
  ...]`, a layer view, slot 3 of 6), filled by a 256-token reference prefill.

| GPU 7, 2026-10-09 | Result |
|---|---|
| A: 64 steps from the reference's state, R' vs reference | bit-exact at the median step; worst 13 ulps |
| A: R' vs the reference run on the kernel's own x | median 0 ulps, worst 3 |
| A: every step rerun from the same state | bit-identical (no race between blocks) |
| B: 64 free-running steps | R' within 30 ulps; SSM state within 1.6e-3 |
| other slots and layers of the pool | untouched |

  R' is compared in bf16 ulps of `max(|R|, |R'|)`: it is R plus a much smaller update, so one
  rounding flip of the update moves R' by one ulp, while a relative error of R' - R would read
  as several percent. The 13 ulps start in x: hc_down's and hc_up's sums flip a few x values by one
  bf16 step, and in_proj spreads each over all 16480 outputs.
- **Slow so far: 40.7 us in-kernel** against a 22.6 us bound (135 MB). The read (hc_norm, hc_down,
  hc_up) takes ~9.5 us, and in_proj's 84 MB can only stream once x exists: at most ~64 KB per CU
  of it fits in registers ahead of time. Next: warm the 224 MB Infinity Cache with in_proj's
  weights during the read, or move the read into the previous layer's tail.

## SGLang: `SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL=1`

`python/sglang/srt/models/qwen4_exp_gdn_megakernel.py` runs `gdn_block` for each GDN layer of
Qwen4-Exp (Qwen3.8-Flash-Next) when every condition holds:
- decode at batch 1, TP1, not the PLE layer
- an fp32 SSM pool, with no ReplaySSM

The FFN half then runs on the layer's own `mlp_hyper_connection` and MoE, and the residual stream
is left written, as the original path does. After the launch it calls the GDN backend's
prefix-cache state tracking, as `forward_decode` does. Everything else takes the original path.

- **Compilation:** the kernel compiles at the first eligible call. That call also runs the step,
  and it happens in SGLang's eager warmups, before graph capture. If the kernel hasn't been
  compiled when a capture starts, that layer stays on the original path.
- **Arguments:** every pointer is fixed per layer; the slot and the tag are read on device. The
  launch is captured once per GDN layer in the batch-1 decode graph.

**Checked in SGLang** with `qwen38_flash_next/reference/sglang_dump.py`: FP8, TP1, the toggle on,
greedy, top-20 logprobs; GPU 7, 2026-10-09.
- **It runs:** at info level the engine logs the compile during the batch-1 warmup, and graph capture
  completes.
- **Same tokens up to a near-tie:** the short prompt's 20 decode tokens before stock's own near-tie
  are identical to stock's. At the tie (a 0.25-nat gap) the greedy outputs part; stock reproduces its
  own output exactly, so the decode path changed.
- **Within SGLang's noise:** before that point, decode logprobs differ from stock by less than stock
  differs from itself (p-weighted |dlp| 0.025 vs 0.028).
- **Deterministic:** two megakernel runs give the same text.
- **Not covered:** the long prompt ends after its first token, which comes from prefill, so it never
  reaches decode. The dump is `reference/sglang_tp1_fp8_gdn_megakernel.json`.

## FlyDSL notes

- **The JIT cache key covers arguments, module globals and scalar closure values, but not
  closure objects.** A protocol instance captured in a closure was silently ignored, so the first
  compiled variant was reused. Pass such choices as `fx.Constexpr` arguments, and run with
  `FLYDSL_RUNTIME_ENABLE_CACHE=0`.
- **Calling a `@flyc.jit` launcher costs about 60 us of Python per call.** For back-to-back
  timing, use `flyc.compile(launcher, *args)`, which returns a callable that dispatches in about 5 us.
- **Python objects can't be kernel locals that live across an `if`/`for`.** The control-flow
  rewrite tries to carry them as device values. Look them up inline (`PROTOCOLS[name].method(...)`).
- **Helpers that use `if`/`while` on device values need the AST rewrite.** `common.device`
  applies FlyDSL's internal `ASTRewriter.transform`.

**End to end** (`qwen38_flash_next/bench/ab_nomtp_c1.sh`): `amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8`,
TP1, MTP off. Otherwise the server is the AMD team's (`run_server_nomtp.sh`). Random 1024 in / 512
out, 8 requests at concurrency 1, GPU 7, 2026-10-09. One run each.

| | Median TPOT | Output tok/s | Median TTFT |
|---|---|---|---|
| stock | 10.69 ms | 91.6 | 123 ms |
| `SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL=1` | **9.00 ms (-15.8%)** | **108.4 (+18%)** | 120 ms |

That's 1.69 ms per token over 35 GDN layers, about 48 us saved per layer. Stock therefore spends
roughly 90 us on a GDN layer's attention half at batch 1, against the megakernel's ~41 us. Accuracy
on this checkpoint was not checked; the logprob check above used FP8.

## Step 6: MTP verify (`gdn_block_verify.py`, `step6_gdn_verify.py`)

With NEXTN MTP (3 steps, 4 draft tokens, topk 1), every decode step of the target model is a
4-token verify pass. SGLang's contract for a GDN layer (`GDNAttnBackend.forward_extend`, target
verify) is:
- read the conv and SSM state at the request's slot, and leave them unchanged
- after each draft token t, write the state into the verify scratch at `row`: the conv window
  (bf16) and the SSM state (fp32, `[row, t, 48, 128, 128]`)

After verify, SGLang copies the accepted step's scratch over the main state.

`gdn_block_verify` is `gdn_block` with T tokens:
- **GEMVs:** every weight load is used for all T tokens (`tokens=` in `emit_gemv`), so the 135 MB of
  weights are read once per pass.
- **GDN op:** `gdn_verify_op` runs the T steps on the critical path and publishes o. Then
  `gdn_verify_snapshots` recomputes the steps from the saved start state and stores each snapshot
  through LDS. Recomputing is bit-identical and keeps 4 x 64 KB of stores per head off the critical path.
- **Conv windows:** two layouts. Dense is `[row, t, dim, 3]`. SGLang's GPU default for chain drafts is
  deduplicated: one `[dim, T + 2]` row per slot, where step t's window is columns t .. t + 2.

**Checked** (layer-0 weights, 32 launches from the reference's state, then 32 chained through
random accepted lengths), for both window layouts:
- R' from the kernel's own o is within 3 ulps of the reference.
- o is at least 98.4% bit-exact per token against the reference run on the kernel's x.
- Each step's SSM snapshot is within 5e-5, and the conv windows are exact.
- The main state and the other scratch rows are untouched, and reruns are bit-identical.

**Time: 71 to 73 us per pass** (4 tokens) against a 24 us bound. That is half of stock's (see below),
but far from the bound:
- **Register budget:** a wave gets 256 VGPRs. The first version spilled 8 KB a thread to scratch;
  that is fixed by prefetching 24 out_proj loads, not 42, and by polling x one token at a time.
- **Polls:** batching all T tokens into one poll made spinning blocks reload every chunk on every
  retry, which starved the producers (585 us). Small batches only, or spin on the first token.
- **What's left is mostly wave reductions:** one 64-lane DPP reduction per (row, token), 68 per wave
  in in_proj. The fix is MFMA with the tokens as N, as in FlyDSL's monokernel GEMVs.

**End to end with MTP** (`bench/ab_c1.sh` with `SERVER=run_server_mtp.sh`): the AMD team's config,
MXFP4 TP1, NEXTN 3/4/1, simulated accept length 2.32, random 1024 in / 512 out, 8 requests at
concurrency 1, GPU 7, 2026-10-09. One run each.

| | Median TPOT | Output tok/s | Accept length |
|---|---|---|---|
| stock | 8.29 ms | 116.6 | 2.33 |
| `SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL=1` | **7.51 ms (-9.4%)** | **128.1 (+9.8%)** | 2.30 |

Per verify step (TPOT x accept length) that is about 2 ms saved over 35 GDN layers, about 57 us a
layer. Stock's verify GDN half is therefore about 130 us against the kernel's ~73 us.

- **The draft needs a patched config:** this checkpoint stores the MTP experts in bf16 without
  listing them in quark's exclude list, and this branch then builds them as MXFP4 and fails to load
  them. `run_server_mtp.sh` points `--speculative-draft-model-path` at a directory of symlinks whose
  `config.json` lists them.
- **Real MTP:** with real (not simulated) acceptance (`sglang_dump.py`, 64 tokens), the short
  prompt's first 20 tokens match stock. The outputs part at the same near-tie as in the FP8 decode
  check (0.25 nats), where two stock runs also part. Over decode positions 1..20, p-weighted |dlp|
  is 0.096 between two stock runs and 0.049 / 0.090 between the megakernel and each, so it is
  within stock's own noise.

## Step 7: a whole GDN layer in one launch (`moe_op.py`, `gdn_layer.py`, `step7_gdn_layer.py`)

`gdn_layer` is `gdn_block` with R' handed to the FFN half through a mailbox, not written out. The
FFN half runs on SGLang's MXFP4 expert tensors as they are (quark W4A4; aiter's padded, preshuffled
layout; the shared expert fused as expert 512):

| phase | blocks | work |
|---|---|---|
| hc read 2 | all | poll R', n2, low2 (`hc_down`), x2 (`hc_up`); c2 on stage-2 blocks |
| router | all | 2 logits each (`emit_gemv`), bf16-rounded |
| top-k | every block, redundantly | top 10 of 512, softmax over them; shared weight `sigmoid(x . w_sg)`; x2 MXFP4-rounded |
| stage 1 | 36..255 | one (slot, 32-wide group) each: gate and up rows, `h = MXFP4(bf16(silu(g) u))` |
| stage 2 | 0..159 | 16 output rows of all 11 slots, `R_out = bf16(R' + bf16(y) c2)` |

**Matching stock** (aiter's `fused_moe`, found by reading its kernels):
- **Activations:** both quantizations use aiter's default RoundUp scale, `2^ceil(log2(amax * f32(1/6)))`.
  This is not Quark's Even rule, which is 2 to 12% off stock.
- **h:** rounded to bf16 before quantizing, with the hardware `exp2`/`rcp` sigmoid.
- **Stage 2:** stock rounds each expert's term to bf16 and adds the 11 with bf16 atomics in arrival
  order, so stock is nondeterministic. 20 runs on one input gave 20 distinct outputs, spread 0.4 to
  1.2%. The kernel sums in fp32 in a fixed order and rounds once.

**Checked** (`step7_gdn_layer.py`: layer 0, 32 launches from the reference's state):
- Every expert alone is bit-exact with stock's `fused_moe`. The whole MoE differs from stock by
  0.2 to 0.7% (median), less than stock differs from itself.
- On the kernel's own x2, R_out is exact (0 ulps) against the reference. R' is within 3 ulps.
- Reruns are bit-identical.
- End to end, a rounding flip in x2 can flip an expert, as it can in stock: worst case 228 ulps.

**Time: 74.9 us per launch** against a 30 us bound (180 MB: 136 MB for the attention half, 29 MB
for 11 experts, 15 MB for hc 2 and the router). The FFN half takes ~32 us:

| span | median |
|---|---|
| wait for R' (the attention half's tail) | 4.2 us |
| hc read 2 | ~6 us |
| router | ~1.8 us |
| x2 to LDS, top-k | 3.6 us |
| expert loads issued, x2 quantized, stage 1 | 8.2 us |
| stage 2 | 5.6 us |

- **Top-k: one int max reduction a round.** Logits are bf16-rounded, so the key
  `orderable(bf16 bits) << 16 | (511 - index)` orders by value, then by lowest index. This replaced a
  float max plus an index min-reduce a round, and took top-k from 4.1 to 3.0 us.
- **MXFP4 rounding:** uses the hardware convert (`cvt_scalef32_pk_fp4_f32` and back), on 2 threads a
  group, instead of an fp32 emulation with a divide on 80 threads. It is bit-identical.
- **The expert weights (27 MB) go out as soon as the ids are known**, before x2 is quantized. From
  there to the end of stage 1 is load-bound: the 124 blocks with both a stage-1 unit (80 KB) and a
  stage-2 unit (56 KB) wait on 136 KB each. Issuing stage 2's weights after stage 1's didn't help
  (76.3 us).
- **What's left** is mostly serial handoffs: the wait for R', hc_up's 12 wave reductions, and the
  router-to-top-k sync.

**End to end** (`bench/ab_c1.sh` setup, MXFP4 TP1, MTP off, random 1024 / 512, 8 requests at
concurrency 1, GPU 7, 2026-10-09; one run each):

| | Median TPOT | Output tok/s |
|---|---|---|
| stock | 10.77 ms | 91.0 |
| attention half in one launch (step 5) | 8.83 ms (-18%) | 110.2 |
| whole layer in one launch, first version (78 us) | 6.84 ms (-36%) | 141.5 (+56%) |
| **whole layer, top-k and rounding optimized (75 us)** | **6.65 ms (-38%)** | **145.1 (+59%)** |

That is about 112 us saved per GDN layer: stock takes ~190 us for a GDN layer at batch 1, against
75 us here (78 us when measured).

**Logprobs** (`bench/server_logprobs.py`: greedy, top-20, the short prompt's decode positions up to
the first greedy divergence, two runs of each server):
- Megakernel against stock: |dlp| of the argmax is 0.064 to 0.099.
- Stock against stock: 0.084.

So it is within stock's noise. Verify (MTP) still runs the attention half only.

The whole-layer path is taken at batch-1 decode when the MoE passes `_has_mxfp4_moe`. Otherwise
the attention half runs as before. A failed check is logged once, with the reason.

## Step 8: a whole GDN layer for MTP verify in one launch (`moe_verify_op.py`, `gdn_layer_verify.py`, `step8_gdn_layer_verify.py`)

`gdn_layer_verify` is `gdn_block_verify` with R' handed to an FFN half for the 4 draft tokens. The
hc read 2 and the router are the decode ones with `tokens=4`. The MoE is split by **unique
expert**, so each expert's weights are read once for every token that picked it.

| phase | blocks | work |
|---|---|---|
| top-k | every block, wave t = token t | top 10 + shared, packed-key reduce; shared gate from raw x |
| expert table | every block | unique experts of the 4 x 11 entries (<= 41), first-occurrence order, a weight per token |
| stage 1 | all, units (expert, 32-group) in a device loop | 20 FP4 MFMAs a wave (A: 16 weight rows from the loads, B: x of the 4 tokens as columns); h published as FP4 words + scale |
| stage 2 | 0..159 | h of all experts into LDS (FP4, 66 KB); per wave its experts in a device loop, 5 MFMAs each, weights 2 experts ahead as loop-carried state |

**MFMA.** `v_mfma_scale_f32_16x16x128_f8f6f4` with both operands FP4 takes aiter's shuffled weight
layout as is. Our 16 B lane load (row `lane % 16`, K block `lane / 16`) is the A operand, and the
E8M0 scale is the low byte of the scale dword. This was checked bit-exact against torch with a
one-wave probe. The same tokens-as-columns trick, with the 16x16x32 bf16 MFMA, now does `hc_up`
in all four kernels (`emit_hc_up_mfma`): 12 to 4 us at 4 tokens, and the decode layer 74.9 to 67.0 us.

**Register pressure was the hard part.** The first version spilled 4,000 VGPRs and took 1.4 ms.
- Unrolled over 4 tokens and 4 units, LLVM CSE'd and LICM'd loads and decodes across tokens and
  units (x is loop-invariant across units), then interleaved the independent token chains.
- The cure was device loops (`for k in range(...)`, not `range_constexpr`) with nothing hoistable
  in the body, and MFMA, which removes the decodes and dot2s altogether.

**Checked** (layer-0 weights, 32 launches, ~40 unique experts a launch, i.e. close to the worst case):
- R_out is exact (0 ulps) against the reference on the kernel's own x2.
- Against stock's `fused_moe` on the same input it is 6 to 10 ulps (stock is nondeterministic, step 7).
- Reruns are bit-identical.

**Time: 150.9 us per launch** against a 44 us bound (265 MB, 104 MB of it the experts). The attention
half takes ~66 us. Stock's `fused_moe` alone takes 46 us at 4 tokens, before its gate, top-k and
hc kernels.

**End to end with MTP** (`bench/ab_c1.sh`, `SERVER=run_server_mtp.sh`: MXFP4 TP1, NEXTN 3/4/1,
simulated accept 2.32, random 1024 / 512, 8 requests at concurrency 1, GPU 7, 2026-10-09):

| | Median TPOT | Output tok/s | Accept length |
|---|---|---|---|
| stock | 8.36 ms | 116.3 | 2.32 |
| attention half fused (step 6), first version | 7.42 ms | 129.8 | 2.31 |
| whole layer fused, first version (162.6 us) | 6.93 ms | 139.6 | 2.31 |
| **whole layer fused (150.9 us)** | **6.70 ms (-20%)** | **142.7 (+23%)** | 2.34 |

`SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL_FFN=0` keeps the attention half only, for A/Bs.

**Logprobs with real MTP** (`REAL_MTP=1 run_server_mtp.sh`, `bench/server_logprobs.py`, short prompt,
decode positions up to the first greedy divergence):

| pairs | median argmax \|dlp\| | median p-weighted \|dlp\| |
|---|---|---|
| stock vs stock | 0.075 | 0.102 |
| whole layer vs stock (10 pairs) | 0.047 | 0.084 |
| attention half only vs stock (6 pairs) | 0.091 | 0.152 |

At position 2 the first request after a server start sometimes takes another token. Its logprob
varies from -0.04 to -0.55 between runs, and this happens with the attention half alone too.

**Next:** in_proj and out_proj as tokens-as-N MFMA (12 + 6.5 us of dot2 + reductions); the
stage-1 tail (blocks with a 4th unit end ~10 us late); hc_norm 2's read of R' as tagged pairs.
