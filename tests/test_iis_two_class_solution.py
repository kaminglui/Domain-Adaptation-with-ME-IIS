import unittest

import torch

from models.me_iis_adapter import MaxEntAdapter


class TestIISKnownSolutionTwoClasses(unittest.TestCase):
    def test_exact_solution_two_components_two_classes(self) -> None:
        """
        Synthetic case: 1 layer, 2 components, 2 classes => 4 constraints.
        Each source sample uniquely activates exactly one (component, class) constraint.

        In this setup, the unique max-entropy solution matching target moments is
        q = target_moments, and fractional IIS should recover it (within tolerance).
        """
        device = torch.device("cpu")
        adapter = MaxEntAdapter(num_classes=2, layers=["layer"], components_per_layer={"layer": 2}, device=device)

        # Source: 4 samples, each a unique (j,c) indicator in joint[layer] with shape (N, J, K).
        source = torch.zeros((4, 2, 2), dtype=torch.float32)
        source[0, 0, 0] = 1.0  # (j=0,c=0)
        source[1, 0, 1] = 1.0  # (j=0,c=1)
        source[2, 1, 0] = 1.0  # (j=1,c=0)
        source[3, 1, 1] = 1.0  # (j=1,c=1)
        source_joint = {"layer": source}

        # Target: 10 samples with known proportions [0.4,0.1,0.2,0.3] over the 4 constraints.
        target = torch.zeros((10, 2, 2), dtype=torch.float32)
        target[0:4, 0, 0] = 1.0  # 4/10
        target[4:5, 0, 1] = 1.0  # 1/10
        target[5:7, 1, 0] = 1.0  # 2/10
        target[7:10, 1, 1] = 1.0  # 3/10
        target_joint = {"layer": target}

        weights, final_error, history = adapter.solve_iis_from_joint(
            source_joint=source_joint, target_joint=target_joint, max_iter=3, iis_tol=1e-10
        )

        self.assertGreaterEqual(len(history), 1)
        self.assertAlmostEqual(float(weights.sum().item()), 1.0, places=8)
        self.assertTrue(bool(torch.all(weights >= 0)))
        self.assertLess(final_error, 1e-8)

        expected = torch.tensor([0.4, 0.1, 0.2, 0.3], dtype=weights.dtype)
        self.assertTrue(torch.allclose(weights.cpu(), expected, atol=1e-8))


if __name__ == "__main__":
    unittest.main()

