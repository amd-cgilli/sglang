"""GDN prefill kernels from AITER (FlyDSL, gfx95).

Used with SGLANG_USE_AITER=1 on gfx95: SGLANG_AITER_GDN_CONV (causal conv) and
SGLANG_AITER_GDN_PREFILL (the whole chunked delta rule as one fused kernel),
both on by default. The lookups return None when a switch is off or the
installed AITER does not ship the kernel, and the callers keep the Triton path.
"""

import functools
import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported

logger = logging.getLogger(__name__)


@functools.cache
def aiter_gdn_prefill_conv():
    """``causal_conv1d_fn``-compatible FlyDSL prefill conv, or None."""
    if not (
        is_gfx95_supported()
        and envs.SGLANG_USE_AITER.get()
        and envs.SGLANG_AITER_GDN_CONV.get()
    ):
        return None
    try:
        from aiter.ops.flydsl.causal_conv1d_flydsl import (
            causal_conv1d_prefill_flydsl_fn,
        )
    except ImportError as exc:
        logger.info("AITER FlyDSL GDN prefill conv unavailable (%s)", exc)
        return None
    return causal_conv1d_prefill_flydsl_fn


@functools.cache
def aiter_gdn_fused_prefill():
    """(gdn_fused_prefill_fwd, gdn_fused_prefill_supported), or None."""
    if not (
        is_gfx95_supported()
        and envs.SGLANG_USE_AITER.get()
        and envs.SGLANG_AITER_GDN_PREFILL.get()
    ):
        return None
    try:
        from aiter.ops.flydsl.gdn_fused_prefill import (
            gdn_fused_prefill_fwd,
            gdn_fused_prefill_supported,
        )
    except ImportError as exc:
        logger.info("AITER FlyDSL fused GDN prefill unavailable (%s)", exc)
        return None
    return gdn_fused_prefill_fwd, gdn_fused_prefill_supported


def fill_track_state_from_h(
    h: torch.Tensor,
    track_state: torch.Tensor,
    track_chunk_idx: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    """Fill the fp32 track snapshot from the per-chunk states ``h`` of the
    Triton fallback, for batches the fused kernel declines.

    ``h`` is [1, total_chunks, H, V, K] with each sequence's 64-token chunks
    contiguous; row n receives the state before chunk ``track_chunk_idx[n]`` of
    sequence n (rows with a negative index are never read). No host sync.
    """
    seq_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_chunks = (seq_lens + 63) // 64
    first_chunk = torch.cumsum(num_chunks, 0) - num_chunks
    src = (first_chunk + track_chunk_idx.clamp(min=0)).clamp(max=h.shape[1] - 1)
    track_state[: src.numel()].copy_(h[0].index_select(0, src.long()))
