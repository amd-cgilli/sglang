import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.elementwise.hc_combine import hc_combine, hc_combine_norm
from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import grouped_gemma_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HC_COUNT = 4
HIDDEN_SIZE = 2560


def _reference_hc_combine(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc: int,
    hs: int,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Eager reference mirroring ``GatedResidual._combine_compute``;
    ``compute_dtype=torch.float64`` is the near-exact reference."""
    R = residual.to(compute_dtype).unflatten(-1, (hc, hs))
    gates = 2 * torch.sigmoid(
        F.linear(normed_residual.to(compute_dtype), inject_weight.to(compute_dtype))
        / hc
    )
    injection = block_output.to(compute_dtype).unsqueeze(-2) * gates.unsqueeze(-1)
    return (R + injection).flatten(-2)


def _make_inputs(
    num_tokens: int, dtype: torch.dtype, hc: int = HC_COUNT, hs: int = HIDDEN_SIZE
):
    torch.manual_seed(0)
    block_output = torch.randn(num_tokens, hs, dtype=dtype, device="cuda")
    residual = torch.randn(num_tokens, hc * hs, dtype=dtype, device="cuda")
    normed_residual = torch.randn(num_tokens, hc * hs, dtype=dtype, device="cuda")
    inject_weight = torch.randn(hc, hc * hs, dtype=dtype, device="cuda") * 0.02
    return block_output, residual, normed_residual, inject_weight


# Worst case over M in {1, 7, 128, 8192} against the fp64 reference on B300 (sm103):
# bf16 max rel err 7.8e-3 (1 ulp at a binade edge), fp16 below 1e-3 (1 ulp = 9.8e-4);
# the residual is fp32 reordering flipping the final rounding.
_TOLERANCES = {
    torch.bfloat16: dict(rtol=1e-2, atol=5e-3),
    torch.float16: dict(rtol=1e-3, atol=1e-3),
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 7, 128, 8192])
def test_hc_combine_correctness(dtype, num_tokens):
    block_output, residual, normed_residual, inject_weight = _make_inputs(
        num_tokens, dtype
    )

    out = hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
    )
    expected = _reference_hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
        compute_dtype=torch.float64,
    ).to(dtype)

    torch.testing.assert_close(out, expected, **_TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hc_combine_out_param(dtype):
    block_output, residual, normed_residual, inject_weight = _make_inputs(64, dtype)
    out = torch.empty_like(residual)

    result = hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
        out=out,
    )
    expected = _reference_hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
        compute_dtype=torch.float64,
    ).to(dtype)

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(result, expected, **_TOLERANCES[dtype])


def test_hc_combine_3d_input():
    dtype = torch.bfloat16
    block_output, residual, normed_residual, inject_weight = _make_inputs(32, dtype)
    block_output = block_output.reshape(4, 8, HIDDEN_SIZE)
    residual = residual.reshape(4, 8, HC_COUNT * HIDDEN_SIZE)
    normed_residual = normed_residual.reshape(4, 8, HC_COUNT * HIDDEN_SIZE)

    out = hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
    )
    expected = _reference_hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
        compute_dtype=torch.float64,
    ).to(dtype)

    assert out.shape == residual.shape
    torch.testing.assert_close(out, expected, **_TOLERANCES[dtype])


def test_hc_combine_bad_hidden_size():
    dtype = torch.bfloat16
    block_output, residual, normed_residual, inject_weight = _make_inputs(
        4,
        dtype,
        hc=4,
        hs=1000,  # 4 * 1000 = 4000, not a multiple of 2048
    )
    with pytest.raises(RuntimeError, match="2048"):
        hc_combine(
            block_output,
            residual,
            normed_residual,
            inject_weight,
            4,
            1000,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 7, 128, 8192])
def test_hc_combine_norm_matches_unfused(dtype, num_tokens):
    block_output, residual, normed_residual, inject_weight = _make_inputs(
        num_tokens, dtype
    )
    norm_weight = torch.randn(HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.1
    eps = 1e-6

    out, normed = hc_combine_norm(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        norm_weight,
        HC_COUNT,
        HIDDEN_SIZE,
        eps,
    )
    expected_out = hc_combine(
        block_output, residual, normed_residual, inject_weight, HC_COUNT, HIDDEN_SIZE
    )
    expected_normed = grouped_gemma_rmsnorm(expected_out, norm_weight, HIDDEN_SIZE, eps)

    # The combine half is the same arithmetic; only the norm's fp32 sum order differs.
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(normed, expected_normed, **_TOLERANCES[dtype])


def test_gated_residual_combine_and_normalize():
    from sglang.srt.environ import envs
    from sglang.srt.layers.hyperconnection import (
        GatedResidual,
        HyperConnectionConfig,
    )

    torch.manual_seed(0)
    config = HyperConnectionConfig(
        hc_count=HC_COUNT,
        hidden_size=HIDDEN_SIZE,
        hc_lowrank=320,
        hc_per_branch_norm=True,
    )
    with envs.SGLANG_ENABLE_HC_COMBINE_NORM_FUSION.override(True):
        attn_hc = GatedResidual(config).cuda().to(torch.bfloat16)
        mlp_hc = GatedResidual(config).cuda().to(torch.bfloat16)
    assert attn_hc._fused_combine_norm_ok
    with torch.no_grad():
        for param in [*attn_hc.parameters(), *mlp_hc.parameters()]:
            param.normal_(0, 0.02)
        hyper_input = torch.randn(
            256, HC_COUNT * HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        block_output = torch.randn(
            256, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        _, residuals = attn_hc.mix(hyper_input)

        combined, combined_normed = attn_hc.combine_and_normalize(
            block_output, residuals, mlp_hc.hc_norm
        )
        assert combined_normed is not None
        mixed, (mix_input, mix_input_normed, _) = mlp_hc.mix(
            combined, normalized_input=combined_normed
        )

        expected_combined = attn_hc.combine(block_output, residuals)
        expected_mixed, (_, expected_normed, _) = mlp_hc.mix(expected_combined)

    torch.testing.assert_close(combined, expected_combined, rtol=0, atol=0)
    assert mix_input is combined
    torch.testing.assert_close(
        mix_input_normed, expected_normed, **_TOLERANCES[torch.bfloat16]
    )
    torch.testing.assert_close(mixed, expected_mixed, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
