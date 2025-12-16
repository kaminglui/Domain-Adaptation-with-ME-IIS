import unittest

import torch

from src.experiments.loss_checks import assert_valid_ce_labels, safe_cross_entropy
from src.experiments.pseudo_labeling import build_pseudo_labels
from utils.test_utils import build_tiny_model


class TestPseudoLabeling(unittest.TestCase):
    def test_build_pseudo_labels_produces_valid_class_ids(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cpu")
        num_classes = 2
        model = build_tiny_model(num_classes=num_classes, feature_dim=8, bottleneck_dim=8).to(device)
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.data = torch.tensor([5.0, 0.0], device=device)

        images = torch.randn(10, 3, 8, 8)
        dummy_labels = torch.zeros(10, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(images, dummy_labels)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)

        indices, labels = build_pseudo_labels(
            model=model,
            target_loader=loader,
            device=device,
            conf_thresh=0.9,
            max_ratio=1.0,
            num_source_samples=10,
        )
        self.assertGreater(len(indices), 0)
        self.assertEqual(len(indices), len(labels))
        for lab in labels:
            self.assertGreaterEqual(int(lab), 0)
            self.assertLess(int(lab), num_classes)

    def test_safe_cross_entropy_accepts_ignore_index(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, 2, -100, 1], dtype=torch.long)
        assert_valid_ce_labels(labels, num_classes=3, ignore_index=-100)
        loss = safe_cross_entropy(logits, labels, num_classes=3, ignore_index=-100)
        self.assertTrue(torch.isfinite(loss).all().item())


if __name__ == "__main__":
    unittest.main()

