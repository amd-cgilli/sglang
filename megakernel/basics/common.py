"""Device helpers shared by the first-principles microbenchmarks (FlyDSL, gfx950)."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr.typing import T

import buffer_ops  # FlyDSL repo kernels/common/buffer_ops.py, as skinny_bf16.py uses it

# Lets a helper called from a kernel use Python if / while on device values, as @flyc.kernel
# bodies can. FlyDSL applies the same rewrite to kernels; it is internal, so pin flydsl 0.3.4.
device = ASTRewriter.transform

AGENT = fx.rocdl.SyncScope.Agent  # coherent across all XCDs of this GPU
RELAXED, ACQUIRE, RELEASE = fx.AtomicOrdering.Monotonic, fx.AtomicOrdering.Acquire, fx.AtomicOrdering.Release
NS_PER_TICK = 10.0  # s_memrealtime runs at 100 MHz


def wall_clock():
    """Device-wide 64-bit realtime counter, the same on every CU."""
    return fx.Int64(llvm.call_intrinsic(ir.IntegerType.get_signless(64), "llvm.amdgcn.s.memrealtime", [], [], []))


def _getreg(name):
    i32 = ir.IntegerType.get_signless(32)
    return fx.Int32(llvm.inline_asm(i32, [], f"s_getreg_b32 $0, hwreg({name})", "=s", has_side_effects=True))


def xcc_id():
    """Which XCD (chiplet: 32 CUs sharing one L2) this wave runs on."""
    return _getreg("HW_REG_XCC_ID") & fx.Int32(0xF)


def cu_id():
    """HW_ID bits [15:8] = cu, sh, se: unique per CU within one XCD."""
    return _getreg("HW_REG_HW_ID") & fx.Int32(0xFF00)


def at(t, i):
    """Pointer to element i of tensor t."""
    return fx.add_offset(t.iter, i)


def load(p, order=None):
    """Plain load, or an agent-scope atomic load with the given ordering."""
    if order is None:
        return fx.generic_load(p)
    return fx.generic_load(p, memory_order=order, syncscope=AGENT)


def store(p, v, order=None):
    if order is None:
        fx.generic_store(p, v)
    else:
        fx.generic_store(p, v, memory_order=order, syncscope=AGENT)


# ===== Tagged-pair mailboxes (FlyDSL kernels/monokernel; measured in step 1) =====
# A mailbox of n bf16 is n / 2 pairs (bf16x2 word, tag), 8 bytes each, written with one store.
# The consumer polls the data itself until every tag equals the one it expects.

CM_NT = 2  # buffer cache policy: non-temporal, for weights read once
CM_DEV = 16  # sc1: coherent at device scope, across XCDs


def bf16x2(lo, hi):
    """Two f32 rounded to bf16 and packed into one int32 word, `lo` in the low half."""
    return fx.Vector.from_elements([lo, hi], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]


def put_pair(rsrc, pair, word, tag):
    buffer_ops.buffer_store(fx.Vector.from_elements([word, tag], fx.Int32), rsrc, pair * 2,
                            cache_modifier=CM_DEV)


def _spin_pause():
    """Opaque to the compiler, so the polling loads are not hoisted out of the retry loop."""
    llvm.InlineAsmOp(None, [], "s_nop 0", "", has_side_effects=True)


@device
def poll_pairs(rsrc, first_pairs, tag):
    """Value words of pairs [p, p + 2) for each p in first_pairs (p even, so one 16 B load each),
    returned once every one of them carries `tag`."""
    def load_all():
        words = []
        for p in first_pairs:
            v = fx.Vector(buffer_ops.buffer_load(rsrc, p * 2, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV))
            words += [v[e] for e in range(4)]
        return fx.Vector.from_elements(words, fx.Int32)

    def pending(v):
        stale = v[1] != tag
        for e in range(3, 4 * len(first_pairs), 2):
            stale = stale | (v[e] != tag)
        return stale

    v = load_all()
    while pending(v):
        _spin_pause()
        v = load_all()
    return [v[e] for e in range(0, 4 * len(first_pairs), 2)]
