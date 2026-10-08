"""Step 1: what does a producer -> consumer handoff cost, and is it correct?

One hub block (block 0) and N spoke blocks play rounds r = 1, 2, ...:
    hub:    "go r"                          -> every spoke
    spokes: its slice of the payload for r  -> hub, which checks every value
N = 1 is a ping-pong (round = two one-way handoffs); N = 255 is a whole-grid fan-in.

Three protocols run through the same code:
    counter          data with plain stores, then a release atomic on a counter;
                     the consumer polls the counter, then an acquire fence (flymk today)
    counter_nofence  the same with relaxed atomics and no fences (shows why the fences exist)
    tagged           every value is stored as one 8-byte (value, tag) pair; the consumer polls
                     the data itself until all tags equal the round (FlyDSL monokernel)

  cd megakernel/basics && bash ../qwen38_flash_next/bench/gpu_run.sh env HIP_VISIBLE_DEVICES=7 \
      FLYDSL_RUNTIME_CACHE_DIR=$HOME/.flydsl_cache python3 step1_sync.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

from common import ACQUIRE, AGENT, NS_PER_TICK, RELAXED, RELEASE, at, cu_id, device, load, store, wall_clock, xcc_id

NBLK, NTHR = 256, 256
ROUNDS = 5000
REPEATS = 5  # launches per case; run-to-run spread is ~30%, so report the median
MAX_VALUES = 4096


def value_of(r, i):
    """What the spoke sends for element i in round r; unique, so a stale read is detectable."""
    return r * MAX_VALUES + i


def count(cond):
    return cond.select(1, 0)


def my_indices(tid, n):
    """Element indices thread `tid` handles out of n, with a validity flag (n may be < NTHR)."""
    return [(tid + k * NTHR, tid + k * NTHR < n) for k in range(-(-n // NTHR))]


# ===== Protocols: each method emits device code for one side of one handoff =====


class Counter:
    name = "counter"
    pub, sub = RELEASE, ACQUIRE  # orderings on the counter atomics

    @device
    def hub_go(self, buf, r, tid):
        if tid == 0:
            store(at(buf.go, 0), r, self.pub)

    @device
    def spoke_wait_go(self, buf, r, tid):
        if tid == 0:
            self._poll_geq(at(buf.go, 0), r)
        fx.gpu.barrier()

    @device
    def spoke_send(self, buf, r, tid, idx):
        for i, ok in idx:
            if ok:
                store(at(buf.vals, i), value_of(r, i))
        fx.gpu.barrier()  # all of this block's stores are issued before thread 0 publishes
        if tid == 0:
            fx.atomic_add(at(buf.done, 0), 1, syncscope=AGENT, ordering=self.pub)

    @device
    def hub_recv(self, buf, r, tid, idx, n_spokes):
        if tid == 0:
            self._poll_geq(at(buf.done, 0), r * n_spokes)
        fx.gpu.barrier()
        bad = fx.Int32(0)
        for i, ok in idx:
            bad = bad + count(ok & (load(at(buf.vals, i)) != value_of(r, i)))
        fx.gpu.barrier()  # everyone has read round r before thread 0 sends "go r + 1"
        return bad

    @device
    def _poll_geq(self, p, target):
        # Poll relaxed, then fence once: an acquire on every poll would invalidate L2 each time.
        cur = load(p, RELAXED)
        while cur < target:
            cur = load(p, RELAXED)
        if self.sub == ACQUIRE:
            fx.memory_fence(syncscope=AGENT, ordering=ACQUIRE)


class CounterNoFence(Counter):
    name = "counter_nofence"
    pub, sub = RELAXED, RELAXED


class Tagged:
    name = "tagged"

    @staticmethod
    def pack(value, tag):
        return (fx.Int64(tag) << 32) | fx.Int64(fx.Uint32(value))

    @staticmethod
    def tag_of(pair):
        return (pair >> 32).to(fx.Int32)

    @device
    def hub_go(self, buf, r, tid):
        if tid == 0:
            store(at(buf.go_pair, 0), self.pack(r, r), RELAXED)

    @device
    def spoke_wait_go(self, buf, r, tid):
        # Every thread polls the same pair: one load per wave, and no block barrier needed.
        self._poll([(at(buf.go_pair, 0), fx.Boolean(True))], r)

    @device
    def spoke_send(self, buf, r, tid, idx):
        for i, ok in idx:
            if ok:
                store(at(buf.pairs, i), self.pack(value_of(r, i), r), RELAXED)

    @device
    def hub_recv(self, buf, r, tid, idx, n_spokes):
        pairs = self._poll([(at(buf.pairs, i), ok) for i, ok in idx], r)
        bad = fx.Int32(0)
        for k, (i, ok) in enumerate(idx):
            bad = bad + count(ok & (pairs[k].to(fx.Int32) != value_of(r, i)))
        fx.gpu.barrier()  # everyone has read round r before thread 0 sends "go r + 1"
        return bad

    @device
    def _poll(self, ptrs, tag):
        """Load all pairs at once; reload the batch until every valid one carries `tag`."""
        def load_all():
            return fx.Vector.from_elements([load(p, RELAXED) for p, _ in ptrs], fx.Int64)

        def pending(v):
            any_old = fx.Boolean(False)
            for k, (_, ok) in enumerate(ptrs):
                any_old = any_old | (ok & (self.tag_of(v[k]) != tag))
            return any_old

        v = load_all()
        while pending(v):
            v = load_all()
        return v


# ===== Kernels =====

PROTOCOLS = {p.name: p() for p in (Counter, CounterNoFence, Tagged)}


class Buffers:
    """Device views of the kernel's tensor arguments, by name."""

    def __init__(self, go, done, vals, go_pair, pairs):
        self.go, self.done, self.vals, self.go_pair, self.pairs = go, done, vals, go_pair, pairs


# Everything that changes the generated code is a Constexpr argument: FlyDSL's compile cache
# keys on arguments and module globals only, so values captured in a closure would be ignored.
@flyc.kernel(known_block_size=[NTHR, 1, 1])
def exchange(go: fx.Tensor, done: fx.Tensor, vals: fx.Tensor, go_pair: fx.Tensor, pairs: fx.Tensor,
             stale: fx.Tensor, ticks: fx.Tensor, first_spoke: fx.Int32, rounds: fx.Int32,
             proto_name: fx.Constexpr[str], n_spokes: fx.Constexpr[int], per_spoke: fx.Constexpr[int]):
    buf = Buffers(go, done, vals, go_pair, pairs)
    bid, tid = fx.Int32(fx.block_idx.x), fx.Int32(fx.thread_idx.x)
    spoke = bid - first_spoke
    if bid == 0:
        idx = my_indices(tid, n_spokes * per_spoke)
        bad = fx.Int32(0)
        t0 = wall_clock()
        for r in range(1, rounds + 1):
            PROTOCOLS[proto_name].hub_go(buf, r, tid)
            bad = bad + PROTOCOLS[proto_name].hub_recv(buf, r, tid, idx, n_spokes)
        if tid == 0:
            store(at(ticks, 0), wall_clock() - t0)
        fx.atomic_add(at(stale, 0), bad)
    if (spoke >= 0) & (spoke < n_spokes):
        idx = [(spoke * per_spoke + i, ok) for i, ok in my_indices(tid, per_spoke)]
        for r in range(1, rounds + 1):
            PROTOCOLS[proto_name].spoke_wait_go(buf, r, tid)
            PROTOCOLS[proto_name].spoke_send(buf, r, tid, idx)


@flyc.jit
def launch_exchange(go: fx.Tensor, done: fx.Tensor, vals: fx.Tensor, go_pair: fx.Tensor, pairs: fx.Tensor,
                    stale: fx.Tensor, ticks: fx.Tensor, first_spoke: fx.Int32, rounds: fx.Int32,
                    proto_name: fx.Constexpr[str], n_spokes: fx.Constexpr[int], per_spoke: fx.Constexpr[int]):
    exchange(go, done, vals, go_pair, pairs, stale, ticks, first_spoke, rounds, proto_name, n_spokes,
             per_spoke).launch(grid=(NBLK, 1, 1), block=(NTHR, 1, 1))


@flyc.kernel(known_block_size=[NTHR, 1, 1])
def placement(out: fx.Tensor):
    if fx.Int32(fx.thread_idx.x) == 0:
        store(at(out, fx.Int32(fx.block_idx.x)), (xcc_id() << 16) | cu_id())


@flyc.jit
def launch_placement(out: fx.Tensor):
    placement(out).launch(grid=(NBLK, 1, 1), block=(NTHR, 1, 1))


# ===== Host =====


def run_exchange(proto, n_spokes, per_spoke, first_spoke, repeats=REPEATS):
    """Returns (median ns per round over `repeats` launches, total stale values seen by the hub)."""
    i32, i64 = {"dtype": torch.int32, "device": "cuda"}, {"dtype": torch.int64, "device": "cuda"}
    times, stale_total = [], 0
    for _ in range(repeats):
        go, done, vals = torch.zeros(64, **i32), torch.zeros(64, **i32), torch.zeros(MAX_VALUES, **i32)
        go_pair, pairs = torch.zeros(32, **i64), torch.zeros(MAX_VALUES, **i64)
        stale, ticks = torch.zeros(1, **i32), torch.zeros(1, **i64)
        launch_exchange(go, done, vals, go_pair, pairs, stale, ticks, first_spoke, ROUNDS, proto, n_spokes,
                        per_spoke)
        torch.cuda.synchronize()
        times.append(ticks.item() * NS_PER_TICK / ROUNDS)
        stale_total += stale.item()
    return sorted(times)[len(times) // 2], stale_total


def main():
    place = torch.zeros(NBLK, dtype=torch.int32, device="cuda")
    launch_placement(place)
    xcc = (place >> 16).tolist()
    print(f"placement: block -> XCD (first 16) {xcc[:16]}; distinct (XCD, CU) over {NBLK} blocks: "
          f"{len(set(place.tolist()))}")
    same = next(b for b in range(1, NBLK) if xcc[b] == xcc[0])
    other = next(b for b in range(1, NBLK) if xcc[b] != xcc[0])

    cases = [  # (label, spokes, values per spoke, first spoke block)
        (f"ping-pong 4 B, same XCD (0<->{same})", 1, 1, same),
        (f"ping-pong 4 B, cross XCD (0<->{other})", 1, 1, other),
        ("ping-pong 4 KB, cross XCD", 1, 1024, other),
        ("fan-in 255 spokes x 16 B", 255, 4, 1),
    ]
    protos = ("counter", "counter_nofence", "tagged")
    print(f"\nmedian ns per round over {REPEATS} launches of {ROUNDS} rounds (a ping-pong round is two "
          f"one-way handoffs), and stale values over all launches")
    print(f"{'':42}" + "".join(f"{p:>24}" for p in protos))
    for label, n, per, first in cases:
        cells = []
        for p in protos:
            ns, bad = run_exchange(p, n, per, first)
            cells.append(f"{ns:8.0f} ns  {bad:>9} stale")
        print(f"{label:42}" + "".join(f"{c:>24}" for c in cells))


if __name__ == "__main__":
    main()
