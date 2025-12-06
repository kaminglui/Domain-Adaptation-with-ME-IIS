import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import scripts.adapt_me_iis as adapt_me_iis
import scripts.train_source as train_source
from datasets.domain_loaders import get_domain_loaders
from models.me_iis_adapter import MaxEntAdapter
from utils.test_utils import build_tiny_model, create_office_home_like, temporary_workdir


class TestIISKnownSolution(unittest.TestCase):
    def test_two_point_known_solution(self) -> None:
        """
        When each sample uniquely activates one constraint, IIS should recover
        the target component proportions in a single step.
        """
        device = torch.device("cpu")
        adapter = MaxEntAdapter(
            num_classes=1, layers=["layer"], components_per_layer={"layer": 2}, device=device
        )
        # Source: sample 0 -> component 0, sample 1 -> component 1
        source_joint = {"layer": torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]], dtype=torch.float32)}
        # Target: 4 samples in comp0, 1 sample in comp1  => target moments [0.8, 0.2]
        target_joint = {
            "layer": torch.tensor(
                [
                    [[1.0], [0.0]],
                    [[1.0], [0.0]],
                    [[1.0], [0.0]],
                    [[1.0], [0.0]],
                    [[0.0], [1.0]],
                ],
                dtype=torch.float32,
            )
        }

        weights, final_error, history = adapter.solve_iis_from_joint(
            source_joint=source_joint, target_joint=target_joint, max_iter=5, iis_tol=1e-6
        )

        self.assertGreater(len(history), 0)
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=6)
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(final_error < 1e-4)
        expected = torch.tensor([0.8, 0.2])
        self.assertTrue(torch.allclose(weights.cpu(), expected, atol=1e-3))


class TestClassMappingAlignment(unittest.TestCase):
    def test_office_home_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Source has two classes; target intentionally misses one.
            create_office_home_like(root, domains=("Ar",), classes=("c0", "c1"), images_per_class=2, image_size=32)
            create_office_home_like(root, domains=("Cl",), classes=("c0",), images_per_class=2, image_size=32)
            with self.assertRaises(ValueError):
                get_domain_loaders(
                    dataset_name="office_home",
                    source_domain="Ar",
                    target_domain="Cl",
                    batch_size=2,
                    root=str(root),
                    num_workers=0,
                )


class TestFeatureMassAggregation(unittest.TestCase):
    def test_multi_layer_feature_mass(self) -> None:
        layers = ["l0", "l1", "l2"]
        comps = {layer: 2 for layer in layers}
        adapter = MaxEntAdapter(num_classes=2, layers=layers, components_per_layer=comps, device=torch.device("cpu"))
        num_samples = 4
        class_probs = torch.full((num_samples, 2), 0.5, dtype=torch.float32)
        joint = {}
        for layer in layers:
            gamma = torch.full((num_samples, 2), 0.5, dtype=torch.float32)
            joint[layer] = gamma.unsqueeze(2) * class_probs.unsqueeze(1)

        flat, feature_mass = adapter._validate_joint_features(joint, name="joint", rel_mass_tol=1e-6)
        expected_mass = float(len(layers))
        self.assertTrue(torch.allclose(feature_mass, torch.full((num_samples,), expected_mass), atol=1e-6))
        self.assertEqual(flat.shape[1], adapter.indexer.total_constraints)


class TestEndToEndDryRun(unittest.TestCase):
    def test_train_and_adapt_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            data_root = workdir / "data"
            create_office_home_like(
                data_root, domains=("Ar", "Cl"), classes=("c0", "c1"), images_per_class=3, image_size=32
            )

            source_args = argparse.Namespace(
                dataset_name="office_home",
                data_root=str(data_root),
                source_domain="Ar",
                target_domain="Cl",
                num_epochs=1,
                resume_from=None,
                save_every=0,
                batch_size=2,
                lr_backbone=1e-3,
                lr_classifier=1e-2,
                weight_decay=1e-3,
                num_workers=0,
                deterministic=True,
                seed=0,
                dry_run_max_batches=2,
                dry_run_max_samples=4,
            )
            with temporary_workdir(workdir):
                with mock.patch("scripts.train_source.build_model", build_tiny_model), mock.patch(
                    "scripts.train_source.tqdm", lambda iterable, **_: iterable
                ), mock.patch("scripts.train_source.sys.stdout.flush", lambda: None):
                    train_source.train_source(source_args)

            final_ckpt = workdir / "checkpoints" / "source_only_Ar_to_Cl_seed0.pth"
            self.assertTrue(final_ckpt.exists())

            adapt_args = argparse.Namespace(
                dataset_name="office_home",
                data_root=str(data_root),
                source_domain="Ar",
                target_domain="Cl",
                checkpoint=str(final_ckpt),
                batch_size=2,
                num_workers=0,
                num_latent_styles=2,
                components_per_layer=None,
                feature_layers="layer3,layer4",
                source_prob_mode="onehot",
                iis_iters=3,
                iis_tol=0.1,
                adapt_epochs=1,
                resume_adapt_from=None,
                save_adapt_every=0,
                finetune_backbone=False,
                backbone_lr_scale=0.1,
                classifier_lr=1e-2,
                weight_decay=1e-3,
                dry_run_max_samples=4,
                dry_run_max_batches=1,
                deterministic=True,
                seed=0,
            )
            with temporary_workdir(workdir):
                patchers = [
                    mock.patch("scripts.adapt_me_iis.build_model", build_tiny_model),
                    mock.patch("scripts.adapt_me_iis.tqdm", lambda iterable, **_: iterable),
                    mock.patch("scripts.adapt_me_iis._append_csv_safe", lambda *_, **__: None),
                    mock.patch("scripts.adapt_me_iis._save_npz_safe", lambda *_, **__: None),
                    mock.patch("scripts.adapt_me_iis.sys.stdout.flush", lambda: None),
                ]
                with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4]:
                    adapt_me_iis.adapt_me_iis(adapt_args)

            adapt_ckpt = workdir / "checkpoints" / "me_iis_Ar_to_Cl_layer3-layer4_seed0.pth"
            self.assertTrue(adapt_ckpt.exists())
            ckpt_data = torch.load(adapt_ckpt, map_location="cpu")
            weights = ckpt_data.get("weights")
            self.assertIsNotNone(weights)
            self.assertAlmostEqual(float(weights.sum().item()), 1.0, places=4)
            batch_cap = adapt_args.dry_run_max_batches * adapt_args.batch_size
            sample_cap = adapt_args.dry_run_max_samples
            expected_len = min(sample_cap, batch_cap) if adapt_args.dry_run_max_batches > 0 else sample_cap
            self.assertEqual(weights.numel(), expected_len)


if __name__ == "__main__":
    unittest.main()
