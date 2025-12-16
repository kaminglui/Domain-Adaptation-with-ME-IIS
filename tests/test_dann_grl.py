import unittest

import torch

from models.dann import GradientReversal


class TestGradientReversal(unittest.TestCase):
    def test_grl_flips_gradient_sign(self) -> None:
        x = torch.randn(4, 3, requires_grad=True)
        grl = GradientReversal()
        y = grl(x, lambda_=1.5).sum()
        y.backward()
        expected = -1.5 * torch.ones_like(x)
        self.assertTrue(torch.allclose(x.grad, expected))


if __name__ == "__main__":
    unittest.main()

