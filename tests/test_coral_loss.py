import unittest

import torch

from src.experiments.losses import coral_loss


class TestCoralLoss(unittest.TestCase):
    def test_coral_loss_scalar_and_grad(self) -> None:
        src = torch.randn(4, 8, requires_grad=True)
        tgt = torch.randn(3, 8, requires_grad=True)
        loss = coral_loss(src, tgt)
        self.assertEqual(loss.ndim, 0)
        loss.backward()
        self.assertIsNotNone(src.grad)
        self.assertIsNotNone(tgt.grad)

    def test_coral_loss_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            coral_loss(torch.randn(4, 8), torch.randn(4, 7))


if __name__ == "__main__":
    unittest.main()

