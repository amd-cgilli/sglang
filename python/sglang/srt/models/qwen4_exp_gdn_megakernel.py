"""Experimental: a Qwen4-Exp GDN layer as one FlyDSL launch.

Behind SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL. The kernels are in megakernel/basics/ of this
repository. The attention half: gated read (hc_mix), GDN mixer (in_proj, conv, gated delta rule,
gated norm, out_proj) and gated write (hc_combine), on the layer's own weights and the GDN
backend's conv and SSM pools. It runs only where it is exact to the original path's math, at batch
1, TP1 and an fp32 SSM state: decode (gdn_block.py) and the MTP target-verify pass over 4 chain
draft tokens (gdn_block_verify.py), which writes the per-token conv windows and SSM snapshots
SGLang commits from after verify. With quark MXFP4 experts (aiter's layout, the shared expert
fused as expert 512) both run the whole layer, the FFN half's gated read, MoE and gated write
included, in one launch (gdn_layer.py, gdn_layer_verify.py).
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import HybridLinearAttnBackend
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.utils import select_verify_intermediate_state_indices
from sglang.srt.layers.layer_boundary.residual.batch import stream_of
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

_KERNEL_DIR = Path(__file__).resolve().parents[4] / "megakernel" / "basics"
VERIFY_TOKENS = 4  # gdn_block_verify is compiled for --speculative-num-draft-tokens 4
# Per kernel ("decode", "verify": the attention half; "layer", "layer_verify": the whole layer):
# compiled launch, mailboxes; the kernel modules once.
_state = {"modules": None, "compiled": {}, "mailboxes": {}, "trace": None, "failed": False, "skips": set()}


def _kernel_modules():
    if _state["modules"] is None:
        if str(_KERNEL_DIR) not in sys.path:
            sys.path.insert(0, str(_KERNEL_DIR))
        import gdn_block
        import gdn_block_verify
        import gdn_layer
        import gdn_layer_verify

        _state["modules"] = {"decode": gdn_block, "verify": gdn_block_verify, "layer": gdn_layer,
                             "layer_verify": gdn_layer_verify}
    return _state["modules"]


def _skip(reason: str) -> bool:
    """Ineligible for a reason worth knowing once (e.g. why verify never takes the kernel)."""
    if reason not in _state["skips"]:
        _state["skips"].add(reason)
        logger.info(f"Qwen4 GDN megakernel not used: {reason}")
    return False


def _gdn_backend() -> Optional[GDNAttnBackend]:
    backend = get_attn_backend()
    if isinstance(backend, HybridLinearAttnBackend):
        backend = backend.linear_attn_backend
    return backend if isinstance(backend, GDNAttnBackend) else None


def _window_layout(win: torch.Tensor) -> Optional[str]:
    """"dense" [rows, T, dim, 3], or "dedup": SGLang's deduplicated view over [rows, dim, T + 2],
    where step t's window is columns t .. t + 2 (memory_pool.py, the GPU default for chain drafts)."""
    rows, steps, dim, width = win.shape
    if win.is_contiguous():
        return "dense"
    if win.stride() == (dim * (steps + width - 1), 1, steps + width - 1, 1):
        return "dedup"
    return None


def _layer_state(layer, forward_batch, verify: bool):
    """(conv [slots, 10240, 3], ssm [slots, 48, 128, 128], cache indices) of this layer, plus for
    verify (window scratch [rows, 4, 10240, 3], snapshot scratch [rows, 4, 48, 128, 128], scratch
    rows), or None when the pool has a layout the kernel doesn't take."""
    backend = _gdn_backend()
    if backend is None or backend.forward_metadata is None:
        return None
    metadata = backend.forward_metadata
    if metadata.replayssm_write_pos is not None:  # ReplaySSM keeps part of the state in a ring
        return None
    cache = backend.req_to_token_pool.mamba2_layer_cache(layer.linear_attn.attn.layer_id)
    conv, ssm, indices = cache.conv[0], cache.temporal, metadata.mamba_cache_indices
    if (conv.dtype != torch.bfloat16 or ssm.dtype != torch.float32 or not conv.is_contiguous()
            or not ssm.is_contiguous() or tuple(conv.shape[1:]) != (10240, 3)
            or tuple(ssm.shape[1:]) != (48, 128, 128) or indices is None or indices.dtype != torch.int32):
        return None
    if not verify:
        return conv, ssm, indices
    if metadata.retrieve_parent_token is not None:
        return _skip("verify with tree drafts (retrieve_parent_token); the kernel takes chains")
    win = cache.intermediate_conv_window[0] if cache.intermediate_conv_window else None
    snap = cache.intermediate_ssm
    if (snap is None or win is None or snap.dtype != torch.float32 or win.dtype != torch.bfloat16
            or not snap.is_contiguous() or tuple(snap.shape[1:]) != (VERIFY_TOKENS, 48, 128, 128)
            or tuple(win.shape[1:]) != (VERIFY_TOKENS, 10240, 3) or _window_layout(win) is None):
        describe = lambda t: None if t is None else (tuple(t.shape), tuple(t.stride()), t.dtype)
        return _skip(f"verify scratch layout not supported: windows {describe(win)}, snapshots {describe(snap)}")
    if snap.shape[0] * snap[0].numel() * 4 >= 2**31:  # the kernel offsets rows in int32 bytes
        return _skip("verify snapshot scratch is over 2 GB")
    rows = select_verify_intermediate_state_indices(
        backend.verify_intermediate_state_indices, forward_batch.req_pool_indices, indices[:1] >= 0,
        backend.req_to_token_pool.size)
    return conv, ssm, indices, win, snap, rows.to(torch.int32)


def _weights(layer):
    """The kernel's weight arguments, from the layer's parameters (cached on the layer: the small
    ones are converted once, the large ones are the parameters themselves)."""
    cached = getattr(layer, "_gdn_megakernel_weights", None)
    if cached is not None:
        return cached
    gdn, hc = layer.linear_attn, layer.attn_hyper_connection
    bf16 = lambda t: t.detach().to(torch.bfloat16).contiguous()
    weights = (
        bf16(hc.hc_norm.weight), bf16(hc.input_mix_weight_down.weight), bf16(hc.input_mix_weight_up.weight),
        bf16(hc.block_inject_weight.weight), bf16(gdn.in_proj_qkvz.weight), bf16(gdn.in_proj_ba.weight),
        bf16(gdn.out_proj.weight), bf16(gdn.conv1d.weight.view(gdn.conv1d.weight.shape[0], -1)),
        gdn.A_log.detach().float().contiguous(), gdn.dt_bias.detach().float().contiguous(), bf16(gdn.norm.weight),
    )
    layer._gdn_megakernel_weights = weights
    return weights


_BF16_WEIGHT_SHAPES = {  # the kernel reads these as plain bf16 matrices
    ("linear_attn", "in_proj_qkvz"): (16384, 2560),
    ("linear_attn", "in_proj_ba"): (96, 2560),
    ("linear_attn", "out_proj"): (2560, 6144),
    ("attn_hyper_connection", "input_mix_weight_down"): (320, 10240),
    ("attn_hyper_connection", "input_mix_weight_up"): (10240, 320),
    ("attn_hyper_connection", "block_inject_weight"): (4, 10240),
}


def _has_bf16_weights(layer) -> bool:
    """A checkpoint that quantizes any of these (e.g. an MXFP4 export that packs the GDN
    projections) keeps the original path: the kernel would read the packed bytes as bf16."""
    cached = layer.__dict__.get("_gdn_megakernel_bf16")
    if cached is None:
        weights = {
            key: getattr(getattr(layer, key[0]), key[1]).weight for key in _BF16_WEIGHT_SHAPES
        }
        cached = all(
            w.dtype == torch.bfloat16 and tuple(w.shape) == _BF16_WEIGHT_SHAPES[key]
            for key, w in weights.items()
        )
        layer._gdn_megakernel_bf16 = cached
    return cached


_MXFP4_MOE_SHAPES = {  # aiter's padded layout: intermediate 640 -> 768, 512 experts + the shared one
    "w13_weight": (513, 1536, 1280), "w13_weight_scale": (513, 1536, 80),
    "w2_weight": (513, 2560, 384), "w2_weight_scale": (513, 2560, 24),
}


def _has_mxfp4_moe(layer) -> bool:
    """The MoE is what gdn_layer.py computes: quark W4A4 MXFP4 experts preshuffled for aiter's
    separated gate / up kernels, softmax top-10 renormalized, the shared expert fused at TP1.
    False with SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL_FFN=0 (the attention half only)."""
    if not envs.SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL_FFN.get():
        return False
    cached = layer.__dict__.get("_gdn_megakernel_mxfp4_moe")
    if cached is None:
        from sglang.srt.layers.quantization.quark.quark import QuarkFusedMoEMethod
        from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import QuarkW4A4MXFp4MoE

        mlp = layer.mlp
        if not isinstance(mlp, Qwen2MoeSparseMoeBlock):
            return False
        experts, topk = mlp.experts, mlp.topk.topk_config
        checks = {
            "one fused shared expert": mlp.num_fused_shared_experts == 1,
            "quark W4A4 MXFP4 experts": (isinstance(experts.quant_method, QuarkFusedMoEMethod)
                                         and isinstance(experts.scheme, QuarkW4A4MXFp4MoE)),
            "silu": experts.moe_runner_config.activation == "silu",
            "aiter's padded uint8 layout": all(
                getattr(experts, n).dtype == torch.uint8 and tuple(getattr(experts, n).shape) == shape
                for n, shape in _MXFP4_MOE_SHAPES.items()),
            "preshuffled": getattr(experts.w13_weight, "is_shuffled", False),
            "softmax top-10 renormalized": (topk.top_k == 10 and topk.renormalize and topk.scoring_func == "softmax"
                                            and topk.correction_bias is None and not topk.use_grouped_topk
                                            and topk.routed_scaling_factor is None),
            "bf16 router and shared gate": (mlp.gate.weight.dtype == torch.bfloat16
                                            and tuple(mlp.gate.weight.shape) == (512, 2560)
                                            and mlp.shared_expert_gate.weight.dtype == torch.bfloat16),
        }
        failed = [name for name, ok in checks.items() if not ok]
        cached = not failed
        if failed:
            _skip(f"whole-layer launch: the MoE is not {', '.join(failed)}")
        layer._gdn_megakernel_mxfp4_moe = cached
    return cached


def _ffn_weights(layer):
    """gdn_layer's FFN-half weight arguments (cached on the layer, as _weights)."""
    cached = getattr(layer, "_gdn_megakernel_ffn_weights", None)
    if cached is None:
        hc, mlp = layer.mlp_hyper_connection, layer.mlp
        bf16 = lambda t: t.detach().to(torch.bfloat16).contiguous()
        experts = mlp.experts
        cached = (
            bf16(hc.hc_norm.weight), bf16(hc.input_mix_weight_down.weight), bf16(hc.input_mix_weight_up.weight),
            bf16(hc.block_inject_weight.weight), bf16(mlp.gate.weight), bf16(mlp.shared_expert_gate.weight),
            experts.w13_weight.data, experts.w13_weight_scale.data, experts.w2_weight.data,
            experts.w2_weight_scale.data,
        )
        layer._gdn_megakernel_ffn_weights = cached
    return cached


def _kind(hidden_states: torch.Tensor, forward_batch) -> Optional[str]:
    """"decode" (one token), "verify" (one request's 4 chain draft tokens), or None."""
    mode = forward_batch.forward_mode
    if mode.is_decode() and hidden_states.shape[0] == 1:
        return "decode"
    if mode.is_target_verify():
        if forward_batch.spec_info.draft_token_num != VERIFY_TOKENS:
            return _skip(f"verify with {forward_batch.spec_info.draft_token_num} draft tokens, not {VERIFY_TOKENS}")
        if hidden_states.shape[0] != VERIFY_TOKENS:
            return None  # a batch of more than one request
        return "verify"
    return None


def eligible(layer, hidden_states: torch.Tensor, forward_batch) -> bool:
    if not envs.SGLANG_OPT_USE_QWEN4_GDN_MEGAKERNEL.get() or _state["failed"]:
        return False
    if layer.ple is not None or get_parallel().tp_size != 1 or not _has_bf16_weights(layer):
        return False
    kind = _kind(hidden_states, forward_batch)
    if not kind:
        return False
    kernel = _whole_layer_kernel(kind) if _has_mxfp4_moe(layer) else kind
    if not any(k[0] == kernel for k in _state["compiled"]) and torch.cuda.is_current_stream_capturing():
        return False  # compiling runs a launch; SGLang's eager warmups compile it before capture
    return bool(_layer_state(layer, forward_batch, kind == "verify"))


def _launch(kind: str, dedup: bool, residual: torch.Tensor, args_between: tuple) -> torch.Tensor:
    """One launch of kernel `kind` on R: (R, R_out, *args_between, *mailboxes, trace, stream)."""
    module = _kernel_modules()[kind]
    if kind not in _state["mailboxes"]:
        _state["mailboxes"][kind] = module.mailboxes(residual.device)
    if _state["trace"] is None:
        _state["trace"] = torch.zeros(1, dtype=torch.int64, device=residual.device)  # unused: untraced build
    r_out = torch.empty_like(residual)
    args = (residual, r_out, *args_between, *_state["mailboxes"][kind].values(), _state["trace"],
            module.fx.Stream(torch.cuda.current_stream()))
    key = (kind, dedup)
    if key not in _state["compiled"]:
        import flydsl.compiler as flyc

        logger.info(f"Compiling the Qwen4 GDN megakernel for {kind} (one launch per GDN layer at batch 1)")
        verify = kind in ("verify", "layer_verify")
        build = module.build(False, module.T_VERIFY, dedup) if verify else module.build(False)
        _state["compiled"][key] = flyc.compile(build, *args)  # also runs this launch
    else:
        _state["compiled"][key](*args)
    return r_out


def run_attention_half(layer, residual: torch.Tensor, forward_batch) -> torch.Tensor:
    """R' = R + c * out_proj(gdn(in_proj(hc_mix(R)))) per token row. Decode updates the layer's GDN
    state and, like GDNAttnBackend.forward_decode, its prefix-cache track copies; verify reads the
    state and writes the per-token snapshots, like forward_extend's target-verify path."""
    kind = _kind(residual, forward_batch)
    state = _layer_state(layer, forward_batch, kind == "verify")
    dedup = kind == "verify" and _window_layout(state[3]) == "dedup"
    r_out = _launch(kind, dedup, residual, (*_weights(layer), *state))
    if kind == "decode":
        _track_decode_state(layer, forward_batch, state)
    return r_out


def _whole_layer_kernel(kind: str) -> str:
    return {"decode": "layer", "verify": "layer_verify"}[kind]


def run_layer(layer, residual: torch.Tensor, forward_batch) -> torch.Tensor:
    """The whole layer: R_out = R' + c2 * moe(hc_mix2(R')) with R' as run_attention_half, per
    token row; decode or verify, with the attention half's state handling."""
    kind = _kind(residual, forward_batch)
    state = _layer_state(layer, forward_batch, kind == "verify")
    dedup = kind == "verify" and _window_layout(state[3]) == "dedup"
    r_out = _launch(_whole_layer_kernel(kind), dedup, residual, (*_weights(layer), *state, *_ffn_weights(layer)))
    if kind == "decode":
        _track_decode_state(layer, forward_batch, state)
    return r_out


def _track_decode_state(layer, forward_batch, state) -> None:
    conv, ssm, indices = state
    _gdn_backend()._track_mamba_state_decode(forward_batch, conv, ssm, indices, layer.linear_attn.attn.layer_id)


def forward_layer(layer, hidden_states: torch.Tensor, forward_batch) -> Optional[torch.Tensor]:
    """The whole decoder layer: in one launch (MXFP4 experts), or the attention half in one and the
    FFN half as the layer's own gated read, MoE and gated write. Leaves the residual stream
    written, as the original path."""
    stream = stream_of(forward_batch)
    if stream.pending is not None:
        return None
    if stream.residual is None:
        residual = layer._widen_streams(hidden_states)  # the stack's first layer
    else:
        stream.check(hidden_states)
        residual = hidden_states
    if _has_mxfp4_moe(layer):
        return stream.write(run_layer(layer, residual, forward_batch))
    residual = run_attention_half(layer, residual, forward_batch)

    hc = layer.mlp_hyper_connection
    ffn_input, normed = hc.mix(residual)
    if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
        ffn_output = layer.mlp(ffn_input, forward_batch, defer_finalize=False)
    else:
        ffn_output = layer.mlp(ffn_input)
    return stream.write(hc.combine(ffn_output, normed))
