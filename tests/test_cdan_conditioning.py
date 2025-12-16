import unittest

import torch

from src.experiments.losses import entropy
from src.experiments.methods.cdan import _conditional_representation


class TestCdanConditioning(unittest.TestCase):
    def test_conditional_representation_shape(self) -> None:
        torch.manual_seed(0)
        b, d, c = 5, 7, 3
        feats = torch.randn(b, d)
        probs = torch.softmax(torch.randn(b, c), dim=1)
        cond = _conditional_representation(feats, probs)
        self.assertEqual(tuple(cond.shape), (b, d * c))

    def test_entropy_conditioning_weights_decrease_with_entropy(self) -> None:
        probs_low_entropy = torch.tensor([[0.99, 0.005, 0.005], [0.9, 0.05, 0.05]], dtype=torch.float32)
        probs_high_entropy = torch.tensor([[1 / 3, 1 / 3, 1 / 3], [0.34, 0.33, 0.33]], dtype=torch.float32)
        w_low = 1.0 + torch.exp(-entropy(probs_low_entropy))
        w_high = 1.0 + torch.exp(-entropy(probs_high_entropy))
        self.assertTrue(bool((w_low > w_high).all().item()))


if __name__ == "__main__":
    unittest.main()

