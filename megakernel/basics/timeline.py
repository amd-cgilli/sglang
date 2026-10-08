"""In-kernel timestamps per wave, exported as a Chrome trace that ui.perfetto.dev opens directly.

Device: a trace buffer is int64 [blocks, waves, COLS]. `mark(trace, waves, m, enabled)` stores the
wall clock (10 ns ticks, the same on every CU) at column m from lane 0 of each wave. Mark 0
also records the block's XCD in the last column, so the trace can group blocks by chiplet.
`enabled` is a Python bool, resolved at compile time: a kernel built with it off contains no
clock reads or trace stores, so its timing is free of tracing overhead.

Host: `write_trace(path, traces, spans)` writes one slice per (wave, span), where a span is
(name, from mark, to mark). Perfetto shows one process per XCD and one track per wave.
"""

import json
import os

import torch

import flydsl.expr as fx
from flydsl.expr import const_expr

from common import NS_PER_TICK, at, device, store, wall_clock, xcc_id

COLS = 64  # marks 0..62 (5 per GEMV op, so a 12-op chain fits), then the XCD id
XCD_COL = COLS - 1


def new_trace(launches, blocks, waves):
    return torch.zeros(launches, blocks, waves, COLS, dtype=torch.int64, device="cuda")


@device
def mark(trace, waves, m, enabled):
    if const_expr(enabled):
        tid = fx.Int32(fx.thread_idx.x)
        if tid % 64 == 0:
            row = (fx.Int32(fx.block_idx.x) * waves + tid // 64) * COLS
            store(at(trace, row + m), wall_clock())
            if const_expr(m == 0):
                store(at(trace, row + XCD_COL), fx.Int64(xcc_id()))


def span_us(trace, first, last):
    """Per launch: first wave's mark `first` to the last wave's mark `last`, in us."""
    t = trace.flatten(1, 2)
    return (t[:, :, last].max(1).values - t[:, :, first].min(1).values).double() * NS_PER_TICK / 1e3


def write_trace(path, trace, spans):
    """trace: int64 [launches, blocks, waves, COLS]; consecutive launches keep their real gaps."""
    t = trace.cpu()
    t0 = t[..., 0].min().item()
    us = lambda tick: (tick - t0) * NS_PER_TICK / 1e3
    events, named = [], set()
    launches, blocks, waves, _ = t.shape
    for b in range(blocks):
        xcd = int(t[0, b, 0, XCD_COL])
        if xcd not in named:
            named.add(xcd)
            events.append({"ph": "M", "name": "process_name", "pid": xcd, "args": {"name": f"XCD {xcd}"}})
        for w in range(waves):
            tid = b * waves + w
            events.append({"ph": "M", "name": "thread_name", "pid": xcd, "tid": tid,
                           "args": {"name": f"block {b:3} wave {w}"}})
            for launch in range(launches):
                row = t[launch, b, w]
                for name, first, last in spans:
                    events.append({"ph": "X", "name": name, "pid": xcd, "tid": tid, "ts": us(int(row[first])),
                                   "dur": (int(row[last]) - int(row[first])) * NS_PER_TICK / 1e3,
                                   "args": {"launch": launch}})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "ns"}, f)
