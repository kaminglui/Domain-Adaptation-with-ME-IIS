import unittest

import torch

from src.experiments.losses import joint_mmd_loss, mmd_loss


class TestMmdLosses(unittest.TestCase):
    def test_mmd_nonzero_for_shifted_distributions(self) -> None:
        torch.manual_seed(0)
        src = torch.randn(16, 8)
        tgt = torch.randn(16, 8) + 2.0
        loss = mmd_loss(src, tgt)
        self.assertGreater(float(loss.item()), 1e-4)

    def test_joint_mmd_nonzero_for_shifted_joint(self) -> None:
        torch.manual_seed(0)
        feats_s = torch.randn(16, 8)
        feats_t = torch.randn(16, 8) + 1.0
        probs_s = torch.softmax(torch.randn(16, 3), dim=1)
        probs_t = torch.softmax(torch.randn(16, 3) + 0.5, dim=1)
        loss = joint_mmd_loss([feats_s, probs_s], [feats_t, probs_t])
        self.assertGreater(float(loss.item()), 1e-4)


if __name__ == "__main__":
    unittest.main()

