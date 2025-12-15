import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from models.dann import DomainDiscriminator, DomainDiscriminatorConfig, GradientReversal
from src.experiments.methods import dann, source_only
from src.experiments.run_config import RunConfig
from utils.test_utils import build_tiny_model, create_office_home_like


class TestDANNSanity(unittest.TestCase):
    def test_grl_reverses_gradient(self) -> None:
        x = torch.randn(4, 3, requires_grad=True)
        grl = GradientReversal()
        y = grl(x, lambda_=1.0)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.allclose(x.grad, -torch.ones_like(x)))

    def test_domain_discriminator_shape(self) -> None:
        disc = DomainDiscriminator(in_dim=10, config=DomainDiscriminatorConfig(hidden_dim=8, dropout=0.0))
        out = disc(torch.randn(5, 10))
        self.assertEqual(tuple(out.shape), (5, 2))

    def test_dann_training_step_saves_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            data_root = workdir / "data"
            create_office_home_like(data_root, domains=("Ar", "Cl"), images_per_class=6, image_size=256)
            runs_root = workdir / "outputs" / "runs"

            cfg_src = RunConfig(
                dataset_name="office_home",
                data_root=str(data_root),
                source_domain="Ar",
                target_domain="Cl",
                method="source_only",
                epochs_source=1,
                epochs_adapt=0,
                batch_size=2,
                num_workers=0,
                seed=0,
                deterministic=True,
                dry_run_max_samples=8,
                dry_run_max_batches=2,
            )
            with mock.patch("src.experiments.methods.source_only.build_model", build_tiny_model):
                src_res = source_only.run(cfg_src, runs_root=runs_root, force_rerun=False, save_every_epochs=1)
            src_ckpt = Path(src_res["checkpoint"])
            self.assertTrue(src_ckpt.exists())

            cfg_dann = RunConfig(
                dataset_name="office_home",
                data_root=str(data_root),
                source_domain="Ar",
                target_domain="Cl",
                method="dann",
                epochs_source=1,
                epochs_adapt=1,
                batch_size=2,
                num_workers=0,
                finetune_backbone=False,
                backbone_lr_scale=0.1,
                classifier_lr=1e-2,
                seed=0,
                deterministic=True,
                dry_run_max_samples=8,
                dry_run_max_batches=2,
                method_params={
                    "disc_hidden_dim": 8,
                    "disc_dropout": 0.0,
                    "dann_lambda_schedule": "constant",
                    "domain_loss_weight": 1.0,
                },
            )

            with mock.patch("src.experiments.methods.dann.build_model", build_tiny_model):
                res = dann.run(cfg_dann, source_checkpoint=src_ckpt, runs_root=runs_root, force_rerun=False)
            self.assertIn(res["status"], ("trained", "resumed", "skipped"))
            ckpt_path = Path(res["checkpoint"])
            self.assertTrue(ckpt_path.exists())
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.assertIn("discriminator", ckpt)


if __name__ == "__main__":
    unittest.main()

