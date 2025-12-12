import argparse
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

import scripts.adapt_me_iis as adapt_me_iis
import scripts.train_source as train_source
from datasets.domain_loaders import get_domain_loaders
from models.me_iis_adapter import ConstraintIndexer, MaxEntAdapter
from utils.test_utils import build_tiny_model, create_office_home_like, temporary_workdir


def _one_layer_joint_from_components(components: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Helper to wrap a (N, J) component matrix into a single-layer joint dict with K=1.
    """
    if components.dim() != 2:
        raise ValueError(f"components must be 2D (N, J), got shape {tuple(components.shape)}")
    return {"layer": components.unsqueeze(2).clone()}


class TestIISKnownSolution(unittest.TestCase):
    def test_exact_solution_known_joint(self) -> None:
        """
        When each sample uniquely activates one constraint, IIS should recover
        the target proportions (e.g., 4:1 => 0.8/0.2).
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
        expected = torch.tensor([0.8, 0.2], dtype=weights.dtype)
        self.assertTrue(torch.allclose(weights.cpu(), expected, atol=1e-3))


class TestIISConvergenceProperties(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.layer = "layer"
        self.adapter = MaxEntAdapter(
            num_classes=1, layers=[self.layer], components_per_layer={self.layer: 2}, device=self.device
        )

    def test_monotonic_moment_error(self) -> None:
        source_components = torch.tensor(
            [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]], dtype=torch.float32, device=self.device
        )
        target_components = torch.tensor(
            [[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]], dtype=torch.float32, device=self.device
        )
        weights, final_error, history = self.adapter.solve_iis_from_joint(
            source_joint=_one_layer_joint_from_components(source_components.cpu()),
            target_joint=_one_layer_joint_from_components(target_components.cpu()),
            max_iter=6,
            iis_tol=0.0,
        )
        self.assertGreater(len(history), 1)
        errors = [h.max_moment_error for h in history]
        for prev, curr in zip(errors, errors[1:]):
            self.assertLessEqual(curr, prev + 1e-6)
        self.assertGreaterEqual(final_error, 0.0)

    def test_group_weight_shift_ratio(self) -> None:
        source_components = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
        )
        target_components = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        weights, final_error, history = self.adapter.solve_iis_from_joint(
            source_joint=_one_layer_joint_from_components(source_components),
            target_joint=_one_layer_joint_from_components(target_components),
            max_iter=10,
            iis_tol=0.0,
        )
        group_a_mass = float(weights[:3].sum().item())
        group_b_mass = float(weights[3:].sum().item())
        self.assertAlmostEqual(group_a_mass + group_b_mass, 1.0, places=6)
        ratio = group_a_mass / group_b_mass
        self.assertAlmostEqual(ratio, 1.0, places=3)
        self.assertLess(history[-1].max_moment_error, 1e-3)

    def test_unreachable_constraint(self) -> None:
        source_components = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        target_components = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        buf = io.StringIO()
        with redirect_stdout(buf):
            weights, final_error, history = self.adapter.solve_iis_from_joint(
                source_joint=_one_layer_joint_from_components(source_components),
                target_joint=_one_layer_joint_from_components(target_components),
                max_iter=4,
                iis_tol=0.0,
            )
        output = buf.getvalue()
        self.assertIn("Unachievable constraint", output)
        self.assertIn("component=1, class=0", output)
        self.assertTrue(self.adapter.unachievable_constraints)
        layer, comp_idx, class_idx = self.adapter.decode_constraint(self.adapter.unachievable_constraints[0])
        self.assertEqual(layer, self.layer)
        self.assertEqual(comp_idx, 1)
        self.assertEqual(class_idx, 0)
        self.assertGreater(history[-1].max_moment_error, 1e-3)
        self.assertAlmostEqual(float(weights.sum().item()), 1.0, places=6)

    def test_converges_to_uniform_when_source_equals_target(self) -> None:
        source_components = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )
        target_components = source_components.clone()
        weights, final_error, history = self.adapter.solve_iis_from_joint(
            source_joint=_one_layer_joint_from_components(source_components),
            target_joint=_one_layer_joint_from_components(target_components),
            max_iter=5,
            iis_tol=0.0,
        )
        uniform = torch.full_like(weights, 1.0 / weights.numel())
        self.assertTrue(torch.allclose(weights, uniform, atol=5e-3))
        self.assertLess(final_error, 1e-6)
        entropy_uniform = float(-(uniform * torch.log(uniform)).sum().item())
        entropy_final = float(-(weights * torch.log(weights + 1e-12)).sum().item())
        self.assertGreater(entropy_final, 0.95 * entropy_uniform)


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
        expected_tensor = torch.full((num_samples,), expected_mass, dtype=feature_mass.dtype, device=feature_mass.device)
        self.assertTrue(torch.allclose(feature_mass, expected_tensor, atol=1e-6))
        self.assertEqual(flat.shape[1], adapter.indexer.total_constraints)


class TestConstraintIndexer(unittest.TestCase):
    def test_flatten_and_decode(self) -> None:
        layers = ["layer3", "layer4"]
        comp_map = {"layer3": 2, "layer4": 1}
        num_classes = 3
        indexer = ConstraintIndexer(layers, comp_map, num_classes)
        num_samples = 2
        joint = {
            "layer3": torch.zeros(num_samples, 2, num_classes),
            "layer4": torch.zeros(num_samples, 1, num_classes),
        }
        joint["layer3"][0, 0, 0] = 1.0
        joint["layer3"][1, 1, 2] = 2.0
        joint["layer4"][0, 0, 2] = 3.0

        flat = indexer.flatten(joint)
        self.assertEqual(flat.shape, (num_samples, 9))
        self.assertEqual(flat[0, 0].item(), 1.0)
        self.assertEqual(flat[1, 5].item(), 2.0)
        self.assertEqual(flat[0, 8].item(), 3.0)

        layer, comp_idx, class_idx = indexer.decode(0)
        self.assertEqual((layer, comp_idx, class_idx), ("layer3", 0, 0))
        layer, comp_idx, class_idx = indexer.decode(5)
        self.assertEqual((layer, comp_idx, class_idx), ("layer3", 1, 2))
        layer, comp_idx, class_idx = indexer.decode(8)
        self.assertEqual((layer, comp_idx, class_idx), ("layer4", 0, 2))
        with self.assertRaises(ValueError):
            indexer.decode(9)


class TestAdapterConfiguration(unittest.TestCase):
    def test_get_components_per_layer_str(self) -> None:
        comp_map = {"layer3": 2, "layer4": 3, "avgpool": 4}
        adapter = MaxEntAdapter(
            num_classes=3,
            layers=["layer3", "layer4", "avgpool"],
            components_per_layer=comp_map,
            device=torch.device("cpu"),
        )
        comp_str = adapter.get_components_per_layer_str(["layer3", "layer4", "avgpool"])
        self.assertEqual(comp_str, "2,3,4")

    def test_invalid_gmm_selection_mode_raises(self) -> None:
        adapter = MaxEntAdapter(
            num_classes=2,
            layers=["layer3"],
            components_per_layer={"layer3": 1},
            device=torch.device("cpu"),
            gmm_selection_mode="badmode",
        )
        feats = torch.zeros(4, 2)
        with self.assertRaises(ValueError):
            adapter.fit_target_structure({"layer3": feats})


class TestGMMBIC(unittest.TestCase):
    def test_bic_prefers_multiple_components(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        layer = "layer_bic"
        cluster_a = torch.randn(30, 2) * 0.05 + torch.tensor([0.0, 0.0])
        cluster_b = torch.randn(30, 2) * 0.05 + torch.tensor([4.0, 4.0])
        feats = torch.cat([cluster_a, cluster_b], dim=0)

        adapter = MaxEntAdapter(
            num_classes=2,
            layers=[layer],
            components_per_layer={layer: 1},
            device=torch.device("cpu"),
            seed=0,
            gmm_selection_mode="bic",
            gmm_bic_min_components=1,
            gmm_bic_max_components=3,
        )
        adapter.fit_target_structure({layer: feats})
        self.assertGreaterEqual(adapter.components_per_layer[layer], 2)
        gamma = adapter._predict_gamma(layer, feats)
        assignments = torch.argmax(gamma, dim=1)
        split = cluster_a.size(0)
        self.assertNotEqual(torch.mode(assignments[:split]).values.item(), torch.mode(assignments[split:]).values.item())

    def test_bic_not_at_upper_bound_for_clear_clusters(self) -> None:
        torch.manual_seed(1)
        np.random.seed(1)
        layer = "layer_bic_upper"
        centers = [torch.tensor([-4.0, 0.0]), torch.tensor([0.0, 4.0]), torch.tensor([4.0, -4.0])]
        clusters = [c + 0.05 * torch.randn(40, 2) for c in centers]
        feats = torch.cat(clusters, dim=0)

        adapter = MaxEntAdapter(
            num_classes=3,
            layers=[layer],
            components_per_layer={layer: 1},
            device=torch.device("cpu"),
            seed=1,
            gmm_selection_mode="bic",
            gmm_bic_min_components=1,
            gmm_bic_max_components=5,
        )
        adapter.fit_target_structure({layer: feats})
        chosen = adapter.get_components_per_layer()[layer]
        self.assertGreater(chosen, 1)
        self.assertLess(chosen, adapter.gmm_bic_max_components)
        self.assertEqual(adapter.get_components_per_layer_str([layer]), str(chosen))


class TestPseudoLabelAdaptation(unittest.TestCase):
    def test_pseudo_label_smoke(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cpu")
        num_classes = 2
        model = build_tiny_model(num_classes=num_classes, feature_dim=8).to(device)
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.data = torch.tensor([2.2, 0.0], device=device)

        # Source data (labels align with pseudo labels to keep gradients consistent).
        source_images = torch.randn(8, 3, 8, 8)
        source_labels = torch.zeros(8, dtype=torch.long)
        source_ds = torch.utils.data.TensorDataset(source_images, source_labels)
        source_loader = torch.utils.data.DataLoader(
            adapt_me_iis.IndexedDataset(source_ds), batch_size=2, shuffle=True
        )
        weights_vec = torch.ones(len(source_ds)) / float(len(source_ds))

        # Target data with confident predictions for class 0.
        target_images = torch.randn(6, 3, 8, 8)
        target_labels_dummy = torch.zeros(len(target_images), dtype=torch.long)
        target_ds = torch.utils.data.TensorDataset(target_images, target_labels_dummy)
        target_loader = torch.utils.data.DataLoader(target_ds, batch_size=3, shuffle=False)
        pseudo_indices, pseudo_labels = adapt_me_iis.build_pseudo_label_dataset(
            model=model,
            target_loader=target_loader,
            device=device,
            conf_thresh=0.9,
            max_ratio=1.0,
            num_source_samples=len(source_ds),
        )
        self.assertGreater(len(pseudo_indices), 0)
        pseudo_ds = adapt_me_iis.PseudoLabeledDataset(target_ds, pseudo_indices, pseudo_labels)
        pseudo_loader = torch.utils.data.DataLoader(pseudo_ds, batch_size=2, shuffle=True)

        def _compute_pseudo_loss() -> float:
            total, count = 0.0, 0
            eval_loader = torch.utils.data.DataLoader(pseudo_ds, batch_size=2, shuffle=False)
            with torch.no_grad():
                for imgs, labels in eval_loader:
                    logits, _ = model(imgs.to(device), return_features=False)
                    loss = F.cross_entropy(logits, labels.to(device), reduction="sum")
                    total += float(loss.item())
                    count += labels.size(0)
            return total / max(1, count)

        initial_pseudo_loss = _compute_pseudo_loss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss, acc, batches, pseudo_used, pseudo_total = adapt_me_iis.adapt_epoch(
            model=model,
            optimizer=optimizer,
            source_loader=source_loader,
            source_weights_vec=weights_vec,
            device=device,
            max_batches=2,
            pseudo_loader=pseudo_loader,
            pseudo_loss_weight=1.0,
        )
        final_pseudo_loss = _compute_pseudo_loss()
        self.assertGreater(pseudo_total, 0)
        self.assertGreaterEqual(pseudo_used, 0)
        self.assertLess(final_pseudo_loss, initial_pseudo_loss + 1e-4)


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
                gmm_selection_mode="fixed",
                gmm_bic_min_components=2,
                gmm_bic_max_components=8,
                cluster_backend="gmm",
                vmf_kappa=20.0,
                cluster_clean_ratio=1.0,
                kmeans_n_init=10,
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
                use_pseudo_labels=False,
                pseudo_conf_thresh=0.9,
                pseudo_max_ratio=1.0,
                pseudo_loss_weight=1.0,
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
