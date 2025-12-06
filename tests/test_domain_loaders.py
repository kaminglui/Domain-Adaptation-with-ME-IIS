import tempfile
import unittest
from pathlib import Path

import torch

from datasets.domain_loaders import get_domain_loaders
from utils.test_utils import create_office31_like, create_office_home_like


class TestDomainLoaders(unittest.TestCase):
    def test_office_home_toy_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_office_home_like(root, domains=("Ar", "Cl"), images_per_class=5, image_size=256)
            source_loader, target_loader, target_eval_loader = get_domain_loaders(
                dataset_name="office_home",
                source_domain="Ar",
                target_domain="Cl",
                batch_size=2,
                root=str(root),
                num_workers=0,
            )
            self.assertEqual(len(source_loader.dataset.classes), len(target_eval_loader.dataset.classes))
            self.assertEqual(source_loader.dataset.class_to_idx, target_eval_loader.dataset.class_to_idx)

            batch = next(iter(source_loader))
            images, labels = batch
            self.assertEqual(images.shape[1:], torch.Size([3, 224, 224]))
            self.assertTrue(torch.is_floating_point(images))
            self.assertTrue(torch.is_tensor(labels))
            self.assertGreaterEqual(labels.min().item(), 0)
            self.assertLess(labels.max().item(), len(source_loader.dataset.classes))

            capped_source, capped_target, capped_eval = get_domain_loaders(
                dataset_name="office_home",
                source_domain="Ar",
                target_domain="Cl",
                batch_size=2,
                root=str(root),
                num_workers=0,
                max_samples_per_domain=3,
            )
            self.assertEqual(len(capped_source.dataset), 3)
            self.assertEqual(len(capped_target.dataset), 3)
            self.assertEqual(len(capped_eval.dataset), 3)

    def test_office31_class_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_office31_like(root, domains=("A", "W"), images_per_class=4, image_size=256)
            source_loader, target_loader, target_eval_loader = get_domain_loaders(
                dataset_name="office31",
                source_domain="A",
                target_domain="W",
                batch_size=2,
                root=str(root),
                num_workers=0,
            )
            self.assertEqual(source_loader.dataset.class_to_idx, target_loader.dataset.class_to_idx)
            self.assertEqual(source_loader.dataset.class_to_idx, target_eval_loader.dataset.class_to_idx)
            batch = next(iter(target_eval_loader))
            images, labels = batch
            self.assertEqual(images.shape[1:], torch.Size([3, 224, 224]))
            self.assertTrue(torch.is_tensor(labels))
            self.assertGreaterEqual(labels.min().item(), 0)
            self.assertLess(labels.max().item(), len(source_loader.dataset.classes))


if __name__ == "__main__":
    unittest.main()
