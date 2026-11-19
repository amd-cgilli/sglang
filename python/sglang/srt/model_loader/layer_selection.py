"""Load a subset of a checkpoint's decoder layers (``keep_layers``).

``--model-loader-extra-config '{"keep_layers": [0, 1, 2, 3, 44, 45, 46, 47]}'`` (or the
string ``"0-3,44-47"``) builds the model with ``len(keep_layers)`` decoder layers:
checkpoint layer ``keep_layers[i]`` becomes model layer ``i``. The weights of the other
decoder layers are skipped, and safetensors shards that only hold skipped weights are
not read. ``keep_layers = [0, ..., k - 1]`` is the same model as a
``num_hidden_layers = k`` override.

The text config is rewritten for the kept layers:

* per-layer lists (as long as the checkpoint's layer count, e.g. ``layer_types``) are
  subset, so model layer ``i`` keeps the type of checkpoint layer ``keep_layers[i]``;
* layer-index lists (``mlp_only_layers``, ``ple_layer_ids``) are renumbered;
* integer layer patterns (``full_attention_interval``, ``first_k_dense_replace``, ...)
  and per-layer entries of the quantization config cannot be rewritten, so the keep
  list must preserve them (model layer ``i`` must match checkpoint layer
  ``keep_layers[i]``); loading is refused otherwise.

Decoder-layer weights are found by their ``<prefix>layers.<N>.`` names; every other
weight (embeddings, vision towers, MTP/draft layers under another prefix or at an index
>= the checkpoint's layer count) is loaded unchanged. Only the target model is reduced: a
speculative draft model reads the unmodified checkpoint config.

Layers that depend on other layers (e.g. reading the residual of layer ``l - 2``, or
draft taps at fixed layer ids) see the kept neighbours, not the checkpoint's: a
non-contiguous keep list is the operator's responsibility (a warning is logged).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

KEEP_LAYERS_KEY = "keep_layers"

# Integer config fields that define a layer's kind from its index. They cannot be
# subset, so the kept layers must have the kinds of the model layers they become.
_LAYER_PATTERNS = {
    "full_attention_interval": lambda i, n: (i + 1) % n == 0,
    "decoder_sparse_step": lambda i, n: (i + 1) % n == 0,
    "interleave_moe_layer_step": lambda i, n: (i + 1) % n == 0,
    "sliding_window_pattern": lambda i, n: (i + 1) % n == 0,
    "first_k_dense_replace": lambda i, n: i < n,
    "moe_layer_freq": lambda i, n: i % n == 0,
    "moe_layer_start_index": lambda i, n: i >= n,
}

# Config fields listing layer indices: field -> (index base, whether the position of
# an index in the sorted list matters, e.g. it selects a per-layer table).
_LAYER_INDEX_LISTS = {
    "mlp_only_layers": (0, False),
    "ple_layer_ids": (1, True),
}

_logged = set()
_LAYER_RE = re.compile(r"^((?:[^.]+\.)*?)layers\.(\d+)\.")
_NON_DECODER_WORDS = ("vision", "visual", "audio", "image", "speech", "mtp", "draft")


@dataclasses.dataclass
class LayerSelection:
    keep: Tuple[int, ...]
    num_checkpoint_layers: int
    # "<prefix>layers.<N>." of the decoder weights, resolved from the checkpoint.
    prefix: Optional[str] = None

    def __str__(self) -> str:
        return f"{format_layers(self.keep)} of {self.num_checkpoint_layers}"


def parse_keep_layers(value) -> List[int]:
    """``[0, 1, 5]`` or ``"0-3,44-47"`` -> strictly increasing layer indices."""
    if isinstance(value, str):
        keep = []
        for part in value.replace(" ", "").split(","):
            lo, sep, hi = part.partition("-")
            if not lo.isdigit() or (sep and not hi.isdigit()):
                raise ValueError(f"{KEEP_LAYERS_KEY}: bad range {part!r} in {value!r}")
            keep += range(int(lo), int(hi if sep else lo) + 1)
    elif isinstance(value, (list, tuple)) and all(
        isinstance(i, int) and not isinstance(i, bool) for i in value
    ):
        keep = list(value)
    else:
        raise ValueError(
            f"{KEEP_LAYERS_KEY} must be a list of layer indices or a string of "
            f'ranges like "0-3,44-47", not {value!r}'
        )
    if not keep or keep[0] < 0 or any(a >= b for a, b in zip(keep, keep[1:])):
        raise ValueError(
            f"{KEEP_LAYERS_KEY} must be non-empty, non-negative and strictly "
            f"increasing: {value!r}"
        )
    return keep


def format_layers(keep: Sequence[int]) -> str:
    """[0, 1, 2, 3, 44, 45] -> "0-3,44-45"."""
    runs: List[List[int]] = []
    for i in keep:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in runs)


def select_config_layers(
    text_config, keep: Sequence[int], num_checkpoint_layers: int, quant_config=None
) -> LayerSelection:
    """Rewrite ``text_config`` in place for the kept layers (see the module doc)."""
    n_ckpt = num_checkpoint_layers
    if keep[-1] >= n_ckpt:
        raise ValueError(
            f"{KEEP_LAYERS_KEY}: layer {keep[-1]} is out of range, the checkpoint "
            f"has {n_ckpt} decoder layers"
        )
    n_cur = text_config.num_hidden_layers
    if n_cur != n_ckpt and list(keep) != list(range(n_cur)):
        raise ValueError(
            f"{KEEP_LAYERS_KEY} = {format_layers(keep)} conflicts with the "
            f"num_hidden_layers = {n_cur} override; set only {KEEP_LAYERS_KEY}"
        )

    errors = []
    for field, kind in _LAYER_PATTERNS.items():
        n = getattr(text_config, field, None)
        if isinstance(n, int) and not isinstance(n, bool) and n > 0:
            bad = [(i, c) for i, c in enumerate(keep) if kind(i, n) != kind(c, n)]
            if bad:
                errors.append(
                    f"{field} = {n} makes model layer {bad[0][0]} a different kind "
                    f"of layer than checkpoint layer {bad[0][1]}"
                )

    new_ids = {}
    for field, (base, ordinal) in _LAYER_INDEX_LISTS.items():
        ids = getattr(text_config, field, None)
        if not isinstance(ids, (list, tuple)) or not ids:
            continue
        ids = sorted({int(x) - base for x in ids})
        kept = [c for c in keep if c in ids]
        if ordinal and kept != ids[: len(kept)]:
            errors.append(
                f"{field}: layers {kept} would be renumbered among the dropped "
                f"layers {sorted(set(ids) - set(kept))}; keep those too"
            )
        new_ids[field] = [i + base for i, c in enumerate(keep) if c in ids]

    errors += _check_quant_config(quant_config, keep)
    if errors:
        raise ValueError(
            f"{KEEP_LAYERS_KEY} = {format_layers(keep)} cannot be loaded: "
            + "; ".join(errors)
        )

    subset = []
    for field, value in vars(text_config).items():
        if (
            field not in _LAYER_INDEX_LISTS
            and isinstance(value, (list, tuple))
            and len(value) == n_ckpt
        ):
            setattr(text_config, field, type(value)(value[c] for c in keep))
            subset.append(field)
    for field, ids in new_ids.items():
        setattr(text_config, field, ids)
    text_config.num_hidden_layers = len(keep)

    selection = LayerSelection(tuple(keep), n_ckpt)
    if selection.keep in _logged:  # ModelConfig is built several times per process
        return selection
    _logged.add(selection.keep)
    logger.info(
        "Loading decoder layers %s (per-layer config subset: %s; renumbered: %s)",
        selection,
        subset or "none",
        list(new_ids) or "none",
    )
    if keep[-1] - keep[0] + 1 != len(keep):
        logger.warning(
            "%s = %s is not contiguous: layers that depend on other layers (residuals "
            "or caches of earlier layers, draft taps at fixed layer ids) now see the "
            "kept neighbours instead",
            KEEP_LAYERS_KEY,
            format_layers(keep),
        )
    return selection


def _check_quant_config(quant_config, keep) -> List[str]:
    """Per-layer quantization entries (e.g. an exclude list naming ``layers.3.x``) stay
    keyed by the model's layer index, so model layer ``i`` must have the same entries
    as checkpoint layer ``keep[i]``."""
    entries = defaultdict(set)

    def walk(v):
        if isinstance(v, str):
            m = re.search(r"(?:^|\.)layers\.(\d+)\.", v)
            if m:
                entries[int(m.group(1))].add(v[: m.start(1)] + "#" + v[m.end(1) :])
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(k)
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(quant_config)
    for i, c in enumerate(keep):
        if entries[i] != entries[c]:
            return [
                f"the quantization config has different per-layer entries for model "
                f"layer {i} and checkpoint layer {c}"
            ]
    return []


def _weight_map(hf_folder: str, files: Sequence[str], index_file: Optional[str]):
    """weight name -> file, from the safetensors index or the files' headers."""
    path = os.path.join(hf_folder, index_file) if index_file else None
    if path and os.path.isfile(path):
        with open(path) as f:
            weight_map = json.load(f)["weight_map"]
        return {k: os.path.join(hf_folder, v) for k, v in weight_map.items()}
    from safetensors import safe_open

    out = {}
    for file in files:
        if not file.endswith(".safetensors"):
            raise ValueError(f"{KEEP_LAYERS_KEY} needs safetensors weights, not {file}")
        with safe_open(file, framework="pt") as f:
            out.update({k: file for k in f.keys()})
    return out


def _decoder_prefix(names: Iterable[str], num_layers: int) -> str:
    """The ``<prefix>`` of ``<prefix>layers.<N>.`` holding every decoder layer."""
    indices = defaultdict(set)
    for name in names:
        m = _LAYER_RE.match(name)
        if m:
            indices[m.group(1)].add(int(m.group(2)))
    found = [p for p, ids in indices.items() if ids.issuperset(range(num_layers))]
    if len(found) > 1:
        found = [p for p in found if not any(w in p for w in _NON_DECODER_WORDS)]
    if len(found) != 1:
        raise ValueError(
            f"{KEEP_LAYERS_KEY}: cannot tell which '<prefix>layers.<N>.' weights are "
            f"the {num_layers} decoder layers (candidates: {found or sorted(indices)})"
        )
    return found[0]


def select_layer_files(
    selection: LayerSelection,
    hf_folder: str,
    files: List[str],
    index_file: Optional[str],
) -> List[str]:
    """Resolve the decoder prefix and drop the files holding only skipped weights."""
    weight_map = _weight_map(hf_folder, files, index_file)
    selection.prefix = _decoder_prefix(weight_map, selection.num_checkpoint_layers)
    needed = {f for name, f in weight_map.items() if not _skipped(name, selection)}
    indexed = set(weight_map.values())
    kept = [f for f in files if f in needed or f not in indexed]
    if len(kept) < len(files):
        size = sum(os.path.getsize(f) for f in files if f not in kept) / 2**30
        logger.info(
            "%s: reading %d of %d weight files (%.2f GiB of skipped layers not read)",
            KEEP_LAYERS_KEY,
            len(kept),
            len(files),
            size,
        )
    return kept


def _skipped(name: str, selection: LayerSelection) -> bool:
    m = _LAYER_RE.match(name)
    return (
        m is not None
        and m.group(1) == selection.prefix
        and int(m.group(2)) < selection.num_checkpoint_layers
        and int(m.group(2)) not in selection.keep
    )


def select_layer_weights(
    weights: Iterable[Tuple[str, torch.Tensor]], selection: LayerSelection
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Skip the dropped decoder layers' weights and renumber the kept ones."""
    new_index = {c: i for i, c in enumerate(selection.keep)}
    skipped = 0
    for name, tensor in weights:
        m = _LAYER_RE.match(name)
        if m and m.group(1) == selection.prefix:
            n = int(m.group(2))
            if n < selection.num_checkpoint_layers:
                if n not in new_index:
                    skipped += 1
                    continue
                name = f"{selection.prefix}layers.{new_index[n]}.{name[m.end() :]}"
        yield name, tensor
    logger.info(
        "%s: skipped %d weights of the decoder layers not in %s",
        KEEP_LAYERS_KEY,
        skipped,
        selection,
    )


def keep_layers_from_extra_config(extra_config) -> Optional[List[int]]:
    if isinstance(extra_config, str):
        extra_config = json.loads(extra_config or "{}")
    value = (extra_config or {}).get(KEEP_LAYERS_KEY)
    return None if value is None else parse_keep_layers(value)


def _sets_num_hidden_layers(overrides) -> bool:
    if isinstance(overrides, dict):
        return "num_hidden_layers" in overrides or any(
            _sets_num_hidden_layers(v) for v in overrides.values()
        )
    return False


def num_checkpoint_layers(text_config, overrides: Dict, load_raw_text_config) -> int:
    """The checkpoint's decoder layer count (before a num_hidden_layers override)."""
    if _sets_num_hidden_layers(overrides):
        return load_raw_text_config().num_hidden_layers
    return text_config.num_hidden_layers
