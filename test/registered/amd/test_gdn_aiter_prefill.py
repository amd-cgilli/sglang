"""AITER FlyDSL GDN prefill (fused delta rule + causal conv) against Triton."""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kernels import gdn_aiter_prefill
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Qwen3.8-Flash-Next TP1: 16 q/k heads, 48 v heads, head dim 128, conv width 4.
HG, H, D, SLOTS = 16, 48, 128, 16
LENS = [3000, 64, 1, 700, 4500]


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _aiter(switch):
    gdn_aiter_prefill.aiter_gdn_fused_prefill.cache_clear()
    gdn_aiter_prefill.aiter_gdn_prefill_conv.cache_clear()
    with envs.SGLANG_USE_AITER.override(True), switch.override(True):
        if switch is envs.SGLANG_AITER_GDN_PREFILL:
            return gdn_aiter_prefill.aiter_gdn_fused_prefill()
        return gdn_aiter_prefill.aiter_gdn_prefill_conv()


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip, "ROCm only")
class TestGDNAiterPrefill(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cu = torch.tensor(
            [0] + torch.tensor(LENS).cumsum(0).tolist(),
            dtype=torch.int32,
            device="cuda",
        )
        self.idx = torch.arange(1, len(LENS) + 1, dtype=torch.int32, device="cuda")

    def test_fused_prefill_matches_triton(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )

        if _aiter(envs.SGLANG_AITER_GDN_PREFILL) is None:
            self.skipTest("AITER without the fused GDN prefill")
        aiter_kernel = TritonGDNKernel()
        gdn_aiter_prefill.aiter_gdn_fused_prefill.cache_clear()
        with envs.SGLANG_AITER_GDN_PREFILL.override(False):
            triton_kernel = TritonGDNKernel()
        self.assertTrue(aiter_kernel.supports_track_state_snapshot)
        self.assertFalse(triton_kernel.supports_track_state_snapshot)

        t = sum(LENS)
        mixed = torch.randn(t, (2 * HG + H) * D, dtype=torch.bfloat16, device="cuda")
        q, k, v = torch.split(mixed, [HG * D, HG * D, H * D], dim=-1)
        q, k, v = q.view(1, t, HG, D), k.view(1, t, HG, D), v.view(1, t, H, D)
        g = -torch.nn.functional.softplus(torch.randn(1, t, H, device="cuda")) * 0.5
        beta = torch.randn(1, t, H, device="cuda").sigmoid()
        pool = 0.05 * torch.randn(SLOTS, H, D, D, dtype=torch.bfloat16, device="cuda")
        chunk_idx = torch.tensor([10, 0, -1, 5, 70], device="cuda")

        def run(kernel):
            p = pool.clone()
            track = torch.zeros(len(LENS), H, D, D, device="cuda")
            o, _, h = kernel.extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=p,
                cache_indices=self.idx,
                query_start_loc=self.cu,
                track_state=track,
                track_chunk_idx=chunk_idx,
            )
            return o, p, track, h

        o_a, p_a, track_a, _ = run(aiter_kernel)
        # Triton reference, plus the fallback fill used when the fused kernel declines.
        o_t, p_t, track_t, h = run(triton_kernel)
        fill = torch.zeros_like(track_t)
        gdn_aiter_prefill.fill_track_state_from_h(h, fill, chunk_idx, self.cu)
        live = chunk_idx >= 0
        self.assertLess(_rel(o_a, o_t), 1e-2)
        self.assertLess(_rel(p_a[self.idx.long()], p_t[self.idx.long()]), 1e-2)
        self.assertLess(_rel(track_a[live], fill[live]), 1e-2)
        # Chunk 0 is the initial state itself.
        torch.testing.assert_close(fill[1], pool[2].float())

    def test_prefill_conv_matches_triton(self):
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

        conv = _aiter(envs.SGLANG_AITER_GDN_CONV)
        if conv is None:
            self.skipTest("AITER without the FlyDSL prefill conv")
        dim, t = (2 * HG + H) * D, sum(LENS)
        x = torch.randn(t, dim, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
        w = 0.3 * torch.randn(dim, 4, dtype=torch.bfloat16, device="cuda")
        b = 0.1 * torch.randn(dim, dtype=torch.bfloat16, device="cuda")
        states = torch.randn(SLOTS, dim, 3, dtype=torch.bfloat16, device="cuda")
        init = torch.tensor([0, 1, 0, 1, 1], dtype=torch.bool, device="cuda")
        outs = []
        for fn in (conv, causal_conv1d_fn):
            s = states.clone()
            o = fn(
                x,
                w,
                b,
                activation="silu",
                conv_states=s,
                has_initial_state=init,
                cache_indices=self.idx,
                query_start_loc=self.cu,
                seq_lens_cpu=LENS,
            )
            outs.append((o, s))
        torch.testing.assert_close(outs[0][0], outs[1][0], atol=4e-2, rtol=2e-2)
        torch.testing.assert_close(outs[0][1], outs[1][1], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
