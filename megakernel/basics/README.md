# First-principles microbenchmarks (FlyDSL, MI350X)

These measure the assumptions behind the megakernel design one at a time, before any model ops exist.

```bash
cd megakernel/basics && bash ../qwen38_flash_next/bench/gpu_run.sh env HIP_VISIBLE_DEVICES=7 \
    FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 step1_sync.py      # or step2_gemv.py, step3_chain.py
```

| File | |
|---|---|
| `common.py` | device helpers: clock, XCD id, atomics, tagged-pair mailboxes (`put_pair`, `poll_pairs`) |
| `gemv_op.py` | the GEMV op `emit_gemv`, emitted inline into any kernel (steps 2 and 3) |
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
