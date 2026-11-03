"""Correctness tests for the fused FlyDSL QSA sparse decode (gfx950)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

from sglang.kernels.ops.attention.qsa_flydsl import (
    can_use_flydsl_qsa_sparse_decode,
    flydsl_qsa_sparse_decode,
    recommended_num_splits,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _gfx950_available() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    probe = torch.empty(1, 1, 256, dtype=torch.bfloat16, device="cuda")
    return can_use_flydsl_qsa_sparse_decode(probe, probe, probe)


pytestmark = pytest.mark.skipif(
    not _gfx950_available(), reason="gfx950 FlyDSL required"
)

_DIM = 256
# Qwen3.8-Flash-Next: 512 compressed blocks x ratio 4 + 3.
_TOPK = 2051
_MAX_CTX = 8192


def _make_inputs(rows, num_kv_heads, group, kv_dtype, num_slots=40000, seed=0):
    """Random decode batch with every masking case the kernel must honor.

    Rows keep a sorted valid prefix of varying length, then ``-1`` padding; some
    entries point at or past ``seq_len``; row 0 selects nothing at all. Slot 0 is
    never referenced by a valid token and holds NaN, the way CUDA-graph padding
    leaves it, so any masked lane that reads it must not leak into the output.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(rows, num_kv_heads * group, _DIM, generator=gen, device="cuda")
    k = torch.randn(num_slots, num_kv_heads, _DIM, generator=gen, device="cuda")
    v = torch.randn(num_slots, num_kv_heads, _DIM, generator=gen, device="cuda")
    k[0] = float("nan")
    v[0] = float("nan")
    req_to_token = torch.randint(
        1,
        num_slots,
        (rows + 4, _MAX_CTX),
        generator=gen,
        device="cuda",
        dtype=torch.int32,
    )
    row_req = torch.randperm(rows + 4, generator=gen, device="cuda")[:rows]
    seq_lens = torch.randint(
        _TOPK // 2, _MAX_CTX, (rows,), generator=gen, device="cuda", dtype=torch.int32
    )
    indices = torch.full((rows, _TOPK), -1, dtype=torch.int32, device="cuda")
    for r in range(rows):
        seq_len = int(seq_lens[r])
        count = 0 if (r == 0 and rows > 1) else (r * 977) % _TOPK + 1
        count = min(count, seq_len)
        picked = torch.randperm(seq_len, generator=gen, device="cuda")[:count]
        indices[r, :count] = picked.sort().values.int()
        if count < _TOPK:
            # Positions at or past seq_len are masked even when not -1.
            indices[r, count] = seq_len
    return (
        q.bfloat16(),
        k.to(kv_dtype),
        v.to(kv_dtype),
        req_to_token,
        row_req,
        indices,
        seq_lens,
    )


def _reference(q, k, v, req_to_token, row_req, indices, seq_lens, scale):
    rows, num_q_heads, _ = q.shape
    num_kv_heads = k.shape[1]
    valid = (indices >= 0) & (indices < seq_lens[:, None].long())
    positions = torch.where(valid, indices, 0).long()
    slots = req_to_token[row_req.long()[:, None], positions].long()
    k_sel = k[slots].float().repeat_interleave(num_q_heads // num_kv_heads, dim=2)
    v_sel = v[slots].float().repeat_interleave(num_q_heads // num_kv_heads, dim=2)
    k_sel = torch.where(valid[:, :, None, None], k_sel, 0.0)
    v_sel = torch.where(valid[:, :, None, None], v_sel, 0.0)
    scores = torch.einsum("rhd,rthd->rht", q.float(), k_sel) * scale
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    return torch.einsum("rht,rthd->rhd", probs, v_sel)


@pytest.mark.parametrize(
    "rows, num_kv_heads, group, kv_dtype, num_splits",
    [
        (1, 2, 12, torch.bfloat16, None),  # TP1: Hq=24, Hkv=2
        (7, 1, 3, torch.bfloat16, None),  # TP8: Hq=3, Hkv=1
        (64, 2, 12, torch.bfloat16, None),
        (129, 1, 3, torch.bfloat16, 1),  # single split writes the output directly
        (5, 2, 12, torch.float8_e4m3fn, None),
        (3, 1, 3, torch.float8_e4m3fn, 33),  # one wave round per split
    ],
)
def test_matches_reference(rows, num_kv_heads, group, kv_dtype, num_splits):
    inputs = _make_inputs(rows, num_kv_heads, group, kv_dtype)
    scale = _DIM**-0.5
    if num_splits is None:
        num_splits = recommended_num_splits(rows, num_kv_heads, _TOPK, inputs[0].device)
    out = flydsl_qsa_sparse_decode(*inputs, scale, num_splits=num_splits)
    ref = _reference(*inputs, scale)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
    if rows > 1:
        assert torch.count_nonzero(out[0]) == 0, "an empty row must produce zeros"


def test_slot_offsets_beyond_int32():
    """Slot * Hkv * D overflows int32 once a pool holds > 4M tokens at Hkv=2."""
    rows, num_kv_heads, group = 2, 2, 12
    num_slots = (1 << 31) // (num_kv_heads * _DIM) + 4096
    q, _, _, req_to_token, row_req, indices, seq_lens = _make_inputs(
        rows, num_kv_heads, group, torch.bfloat16, num_slots=64
    )
    shape = (num_slots, num_kv_heads, _DIM)
    k = torch.zeros(shape, dtype=torch.float8_e4m3fn, device="cuda")
    v = torch.zeros_like(k)
    req_to_token = req_to_token + (num_slots - 64)
    gen = torch.Generator(device="cuda").manual_seed(1)
    tail = (64, num_kv_heads, _DIM)
    k[-64:] = torch.randn(tail, generator=gen, device="cuda").to(k.dtype)
    v[-64:] = torch.randn(tail, generator=gen, device="cuda").to(v.dtype)
    inputs = (q, k, v, req_to_token, row_req, indices, seq_lens)
    scale = _DIM**-0.5
    out = flydsl_qsa_sparse_decode(*inputs, scale, num_splits=4)
    ref = _reference(*inputs, scale)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


def test_cuda_graph_replay_tracks_new_indices():
    """A captured launch must re-read indices and lengths on every replay."""
    rows, num_kv_heads, group = 8, 2, 12
    first = _make_inputs(rows, num_kv_heads, group, torch.bfloat16, seed=0)
    second = _make_inputs(rows, num_kv_heads, group, torch.bfloat16, seed=1)
    q, k, v, req_to_token, row_req, indices, seq_lens = (t.clone() for t in first)
    scale = _DIM**-0.5
    num_splits = recommended_num_splits(rows, num_kv_heads, _TOPK, q.device)
    partials = rows * num_kv_heads * group * num_splits
    workspace = (
        torch.empty(partials, device="cuda"),
        torch.empty(partials, device="cuda"),
        torch.empty(partials * _DIM, device="cuda"),
    )

    def run():
        return flydsl_qsa_sparse_decode(
            q,
            k,
            v,
            req_to_token,
            row_req,
            indices,
            seq_lens,
            scale,
            num_splits=num_splits,
            workspace=workspace,
        )

    run()  # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for dst, src in zip((q, k, v, req_to_token, row_req, indices, seq_lens), second):
        dst.copy_(src)
    graph.replay()
    ref = _reference(*second, scale)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


def test_backend_decode_matches_triton_fallback(monkeypatch):
    """The backend must hand the kernel the same rows, lengths and request map
    the Triton fallback uses, including requests resolved per row."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_module

    rows, num_kv_heads, group = 6, 2, 12
    q, k, v, req_to_token, row_req, indices, seq_lens = _make_inputs(
        rows, num_kv_heads, group, torch.bfloat16
    )
    k[0] = 0.0  # the Triton fallback has no NaN guard for masked rows
    v[0] = 0.0
    backend = qsa_module.QwenSparseAttnBackend.__new__(qsa_module.QwenSparseAttnBackend)
    kv_layer = SimpleNamespace(key=lambda: k, value=lambda: v)
    backend.token_to_kv_pool = SimpleNamespace(
        layer_router=SimpleNamespace(read=lambda layer_id: kv_layer)
    )
    backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    backend.qsa_profile = None
    backend._cuda_graph_max_tokens = 0
    backend._fa2_scratch = {}
    backend._flydsl_decode_workspace = None
    metadata = SimpleNamespace(
        is_cuda_graph=False, sequence_lengths=seq_lens, row_req_pool_indices=row_req
    )
    monkeypatch.setattr(backend, "_resolve_metadata", lambda forward_batch: metadata)
    monkeypatch.setattr(qsa_module, "_resolve_trtllm_sparse_decode", lambda: None)
    layer = SimpleNamespace(layer_id=0, scaling=_DIM**-0.5)
    # Per-row requests win over the batch's; poison the batch's to prove it.
    forward_batch = SimpleNamespace(req_pool_indices=torch.zeros_like(row_req))

    outputs = {}
    for enabled in (False, True):
        qsa_module._resolve_flydsl_sparse_decode.cache_clear()
        with envs.SGLANG_ENABLE_QSA_FLYDSL_DECODE.override(enabled):
            outputs[enabled] = backend._forward_paged_attention(
                q, layer, forward_batch, indices
            )
    qsa_module._resolve_flydsl_sparse_decode.cache_clear()
    assert backend._flydsl_decode_workspace is not None, "fused path did not run"
    # Row 0 selects nothing; only the fused kernel defines that output (zeros).
    torch.testing.assert_close(
        outputs[True][1:].float(), outputs[False][1:].float(), atol=1e-2, rtol=1e-2
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
