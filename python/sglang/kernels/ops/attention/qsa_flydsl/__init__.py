"""SGLang-maintained FlyDSL kernels for Qwen sparse attention (QSA)."""

# AITER owns the FlyDSL toolchain bootstrap and shared tensor/buffer shims.
# Import it before local kernel modules so its vendored FlyDSL path is active.
import aiter as _aiter  # noqa: F401

from .qsa_sparse_decode import (
    can_use_flydsl_qsa_sparse_decode,
    flydsl_qsa_sparse_decode,
    recommended_num_splits,
    workspace_partials,
)

__all__ = [
    "can_use_flydsl_qsa_sparse_decode",
    "flydsl_qsa_sparse_decode",
    "recommended_num_splits",
    "workspace_partials",
]
