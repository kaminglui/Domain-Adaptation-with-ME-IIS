import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import scripts.adapt_me_iis as adapt_me_iis
import scripts.train_source as train_source
from utils.test_utils import build_tiny_model, create_office_home_like, temporary_workdir


class TestCheckpointResume(unittest.TestCase):
    def _source_args(self, data_root: Path, resume_from=None, num_epochs: int = 2):
        return argparse.Namespace(
            dataset_name="office_home",
            data_root=str(data_root),
            source_domain="Ar",
            target_domain="Cl",
            num_epochs=num_epochs,
            resume_from=str(resume_from) if resume_from else None,
            save_every=1,
            batch_size=2,
            lr_backbone=1e-3,
            lr_classifier=1e-2,
            weight_decay=1e-3,
            num_workers=0,
            deterministic=True,
            seed=0,
            dry_run_max_batches=4,
            dry_run_max_samples=8,
        )

    def _adapt_args(self, data_root: Path, checkpoint: Path, resume_from=None, adapt_epochs: int = 2):
        return argparse.Namespace(
            dataset_name="office_home",
            data_root=str(data_root),
            source_domain="Ar",
            target_domain="Cl",
            checkpoint=str(checkpoint),
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
            iis_iters=5,
            iis_tol=0.05,
            adapt_epochs=adapt_epochs,
            resume_adapt_from=str(resume_from) if resume_from else None,
            save_adapt_every=1,
            finetune_backbone=False,
            backbone_lr_scale=0.1,
            classifier_lr=1e-2,
            weight_decay=1e-3,
            use_pseudo_labels=False,
            pseudo_conf_thresh=0.9,
            pseudo_max_ratio=1.0,
            pseudo_loss_weight=1.0,
            dry_run_max_samples=8,
            dry_run_max_batches=2,
            deterministic=True,
            seed=0,
        )

    def test_train_and_adapt_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            data_root = workdir / "data"
            create_office_home_like(data_root, domains=("Ar", "Cl"), images_per_class=6, image_size=256)

            source_args = self._source_args(data_root, resume_from=None, num_epochs=2)
            with temporary_workdir(workdir):
                with mock.patch("scripts.train_source.build_model", build_tiny_model), mock.patch(
                    "scripts.train_source.sys.stdout.flush", lambda: None
                ):
                    train_source.train_source(source_args)

            final_ckpt = workdir / "checkpoints" / "source_only_Ar_to_Cl_seed0.pth"
            epoch_ckpt = workdir / "checkpoints" / "source_only_Ar_to_Cl_seed0_epoch1.pth"
            self.assertTrue(final_ckpt.exists())
            self.assertTrue(epoch_ckpt.exists())

            resume_args = self._source_args(data_root, resume_from=epoch_ckpt, num_epochs=4)
            with temporary_workdir(workdir):
                with mock.patch("scripts.train_source.build_model", build_tiny_model), mock.patch(
                    "scripts.train_source.sys.stdout.flush", lambda: None
                ):
                    train_source.train_source(resume_args)

            resumed_state = torch.load(final_ckpt, map_location="cpu")
            self.assertGreaterEqual(resumed_state.get("epoch", -1), 1)

            adapt_args = self._adapt_args(data_root, checkpoint=final_ckpt, resume_from=None, adapt_epochs=2)
            with temporary_workdir(workdir):
                patchers = [
                    mock.patch("scripts.adapt_me_iis.build_model", build_tiny_model),
                    mock.patch("scripts.adapt_me_iis.tqdm", lambda iterable, **_: iterable),
                    mock.patch("scripts.adapt_me_iis._save_checkpoint_safe", lambda checkpoint, path: torch.save(checkpoint, path)),
                    mock.patch("scripts.adapt_me_iis._append_csv_safe", lambda *_, **__: None),
                    mock.patch("scripts.adapt_me_iis._save_npz_safe", lambda *_, **__: None),
                    mock.patch("scripts.adapt_me_iis.sys.stdout.flush", lambda: None),
                ]
                with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], patchers[5]:
                    adapt_me_iis.adapt_me_iis(adapt_args)

            adapt_epoch_ckpt = workdir / "checkpoints" / "me_iis_Ar_to_Cl_layer3-layer4_seed0_epoch1.pth"
            adapt_final_ckpt = workdir / "checkpoints" / "me_iis_Ar_to_Cl_layer3-layer4_seed0.pth"
            self.assertTrue(adapt_epoch_ckpt.exists())
            self.assertTrue(adapt_final_ckpt.exists())

            resume_adapt_args = self._adapt_args(
                data_root, checkpoint=final_ckpt, resume_from=adapt_epoch_ckpt, adapt_epochs=3
            )
            resume_adapt_args.dry_run_max_batches = 4
            with temporary_workdir(workdir):
                patchers = [
                    mock.patch("scripts.adapt_me_iis.build_model", build_tiny_model),
                    mock.patch("scripts.adapt_me_iis.tqdm", lambda iterable, **_: iterable),
                    mock.patch("scripts.adapt_me_iis._save_checkpoint_safe", lambda checkpoint, path: torch.save(checkpoint, path)),
                    mock.patch("scripts.adapt_me_iis._append_csv_safe", lambda *_, **__: None),
                    mock.patch("scripts.adapt_me_iis._save_npz_safe", lambda *_, **__: None),
                    mock.patch("scripts.adapt_me_iis.sys.stdout.flush", lambda: None),
                ]
                with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], patchers[5]:
                    adapt_me_iis.adapt_me_iis(resume_adapt_args)

            resumed_adapt_state = torch.load(adapt_final_ckpt, map_location="cpu")
            self.assertGreaterEqual(resumed_adapt_state.get("epoch", -1), 1)


if __name__ == "__main__":
    unittest.main()
