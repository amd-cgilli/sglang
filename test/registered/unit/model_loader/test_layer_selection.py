"""Unit tests for loading a subset of a checkpoint's decoder layers (keep_layers)."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import safetensors.torch
import torch

from sglang.srt.model_loader.layer_selection import (
    LayerSelection,
    format_layers,
    parse_keep_layers,
    select_config_layers,
    select_layer_files,
    select_layer_weights,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _hybrid_config(n=8):
    return SimpleNamespace(
        num_hidden_layers=n,
        full_attention_interval=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"] + ["x"] * (n - 4),
        ple_layer_ids=[2],
        rope_scaling={"mrope_section": [1, 2, 3]},
    )


class TestLayerSelection(CustomTestCase):
    def test_parse(self):
        self.assertEqual(parse_keep_layers("0-2,5"), [0, 1, 2, 5])
        self.assertEqual(parse_keep_layers([0, 4]), [0, 4])
        self.assertEqual(format_layers([0, 1, 2, 3, 44, 45, 47]), "0-3,44-45,47")
        for bad in ("3-1", "a", [], [1, 1], [-1], [True]):
            with self.assertRaises(ValueError):
                parse_keep_layers(bad)

    def test_config_subset_and_renumber(self):
        config = _hybrid_config()
        selection = select_config_layers(config, [0, 1, 2, 3], 8)
        self.assertEqual(config.num_hidden_layers, 4)
        self.assertEqual(config.layer_types[3], "full_attention")
        self.assertEqual(config.ple_layer_ids, [2])
        self.assertEqual(selection.keep, (0, 1, 2, 3))

        config = SimpleNamespace(num_hidden_layers=6, mlp_only_layers=[1, 4])
        select_config_layers(config, [0, 4, 5], 6)
        self.assertEqual(config.mlp_only_layers, [1])

    def test_refusals(self):
        # Breaks the full_attention_interval pattern.
        with self.assertRaisesRegex(ValueError, "full_attention_interval"):
            select_config_layers(_hybrid_config(), [0, 1, 3], 8)
        # Renumbers the per-layer table of PLE layer 6 (ple_layer_ids are 1-based).
        config = _hybrid_config()
        config.ple_layer_ids = [2, 7]
        with self.assertRaisesRegex(ValueError, "ple_layer_ids"):
            select_config_layers(config, [0, 2, 4, 5, 6, 7], 8)
        # Per-layer quantization entries differ between model and checkpoint layer.
        quant = {"exclude": ["model.layers.0.mlp.gate", "model.layers.4.attn"]}
        with self.assertRaisesRegex(ValueError, "quantization"):
            select_config_layers(_hybrid_config(), [4, 5, 6, 7], 8, quant)
        with self.assertRaisesRegex(ValueError, "out of range"):
            select_config_layers(_hybrid_config(), [0, 8], 8)
        # A num_hidden_layers override only agrees with a first-k keep list.
        with self.assertRaisesRegex(ValueError, "override"):
            select_config_layers(_hybrid_config(4), [0, 1, 2, 7], 8)
        select_config_layers(_hybrid_config(4), [0, 1, 2, 3], 8)

    def test_files_and_weights(self):
        names = {
            "a.safetensors": ["model.embed_tokens.weight", "model.layers.0.w"],
            "b.safetensors": ["model.layers.1.w", "model.layers.2.w"],
            "c.safetensors": ["model.layers.3.w", "model.layers.4.w", "mtp.layers.0.w"],
            "d.safetensors": ["vision.layers.0.w", "vision.layers.1.w"],
        }
        with tempfile.TemporaryDirectory() as d:
            weight_map = {}
            for file, keys in names.items():
                tensors = {k: torch.tensor([i]) for i, k in enumerate(keys)}
                safetensors.torch.save_file(tensors, os.path.join(d, file))
                weight_map.update({k: file for k in keys})
            with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
                json.dump({"weight_map": weight_map}, f)
            files = sorted(os.path.join(d, f) for f in names)

            selection = LayerSelection((0, 3), 4)
            kept = select_layer_files(
                selection, d, files, "model.safetensors.index.json"
            )
            self.assertEqual(selection.prefix, "model.")
            # b only holds skipped layers. Layer 4 >= the checkpoint's 4 layers
            # (e.g. an MTP layer) is not a decoder layer: it passes unchanged.
            self.assertEqual(
                [os.path.basename(f) for f in kept],
                ["a.safetensors", "c.safetensors", "d.safetensors"],
            )
            weights = [(n, None) for keys in names.values() for n in keys]
            self.assertEqual(
                [n for n, _ in select_layer_weights(weights, selection)],
                [
                    "model.embed_tokens.weight",
                    "model.layers.0.w",
                    "model.layers.1.w",
                    "model.layers.4.w",
                    "mtp.layers.0.w",
                    "vision.layers.0.w",
                    "vision.layers.1.w",
                ],
            )


if __name__ == "__main__":
    unittest.main()
