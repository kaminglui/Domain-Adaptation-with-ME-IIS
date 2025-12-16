import sys
import unittest
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.me_iis_adapter import MaxEntAdapter


def build_toy_joint(num_samples: int, frac_group_a: float, layers):
    """Construct deterministic joint features for (layer, component, class)."""
    num_group_a = int(num_samples * frac_group_a)
    num_group_b = num_samples - num_group_a
    class_probs = torch.zeros(num_samples, 2, dtype=torch.float32)
    class_probs[:num_group_a, 0] = 1.0
    class_probs[num_group_a:, 1] = 1.0
    joint = {}
    for layer in layers:
        gamma = torch.zeros(num_samples, 2, dtype=torch.float32)
        gamma[:num_group_a, 0] = 1.0
        gamma[num_group_a:, 1] = 1.0
        joint[layer] = gamma.unsqueeze(2) * class_probs.unsqueeze(1)
    assert num_group_a + num_group_b == num_samples
    return joint, class_probs


def entropy(weights: torch.Tensor) -> float:
    w = torch.clamp(weights, min=1e-12)
    return float(-(w * torch.log(w)).sum().item())


class TestIISSanity(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        self.device = torch.device("cpu")
        self.layers = ["layer0", "layer1"]
        self.components_per_layer = {layer: 2 for layer in self.layers}
        self.adapter = MaxEntAdapter(
            num_classes=2, layers=self.layers, components_per_layer=self.components_per_layer, device=self.device
        )

    def _initial_error(self, source_flat: torch.Tensor, target_moments: torch.Tensor) -> Tuple[float, torch.Tensor]:
        init_weights = torch.ones(source_flat.size(0), device=self.device) / float(source_flat.size(0))
        init_moments = torch.sum(init_weights.view(-1, 1) * source_flat, dim=0)
        init_error = float((init_moments - target_moments).abs().max().item())
        return init_error, init_weights

    def test_precomputed_joint_matches_moments(self) -> None:
        source_joint, _ = build_toy_joint(200, frac_group_a=0.5, layers=self.layers)
        target_joint, _ = build_toy_joint(200, frac_group_a=0.75, layers=self.layers)

        source_flat = self.adapter.indexer.flatten(source_joint).to(self.device)
        target_moments = self.adapter.indexer.flatten(target_joint).to(self.device).mean(dim=0)
        init_error, init_weights = self._initial_error(source_flat, target_moments)
        init_entropy = entropy(init_weights)

        weights, final_error, history = self.adapter.solve_iis_from_joint(
            source_joint=source_joint,
            target_joint=target_joint,
            max_iter=30,
            iis_tol=0.0,
        )
        self.assertGreater(len(history), 0)
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=6)
        self.assertGreaterEqual(float(weights.min().item()), 0.0)
        self.assertLess(final_error, init_error)
        self.assertTrue(final_error < 0.05 or final_error < 0.1 * init_error)
        final_entropy = entropy(weights)
        self.assertGreater(final_entropy, 0.2 * init_entropy)

    def test_uniform_when_moments_agree(self) -> None:
        source_joint, _ = build_toy_joint(64, frac_group_a=0.6, layers=self.layers)
        target_joint, _ = build_toy_joint(64, frac_group_a=0.6, layers=self.layers)

        source_flat = self.adapter.indexer.flatten(source_joint).to(self.device)
        target_moments = self.adapter.indexer.flatten(target_joint).to(self.device).mean(dim=0)
        init_error, _ = self._initial_error(source_flat, target_moments)

        weights, final_error, history = self.adapter.solve_iis_from_joint(
            source_joint=source_joint,
            target_joint=target_joint,
            max_iter=20,
            iis_tol=0.0,
        )
        self.assertGreater(len(history), 0)
        self.assertLess(final_error, max(1e-4, 0.5 * init_error))
        uniform = torch.ones_like(weights) / float(weights.numel())
        deviation = torch.max((weights - uniform).abs()).item()
        self.assertLess(deviation, 0.05 * uniform[0].item())
        entropy_uniform = entropy(uniform)
        entropy_final = entropy(weights)
        self.assertGreater(entropy_final, 0.9 * entropy_uniform)


if __name__ == "__main__":
    unittest.main()
