import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.experiments.methods import source_only
from src.experiments.run_config import RunConfig, compute_run_id, get_run_dir
from utils.test_utils import build_tiny_model, create_office_home_like


class TestRunConfigAndCheckpointing(unittest.TestCase):
    def test_run_id_deterministic(self) -> None:
        cfg = RunConfig(
            dataset_name="office_home",
            data_root="datasets/Office-Home",
            source_domain="Ar",
            target_domain="Cl",
            method="source_only",
            epochs_source=2,
            epochs_adapt=0,
            batch_size=2,
            num_workers=0,
            seed=0,
            deterministic=True,
        )
        self.assertEqual(cfg.run_id, compute_run_id(cfg))
        self.assertEqual(cfg.run_id, RunConfig(**{**cfg.__dict__}).run_id)

    def test_resume_then_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            data_root = workdir / "data"
            create_office_home_like(data_root, domains=("Ar", "Cl"), images_per_class=6, image_size=256)

            runs_root = workdir / "outputs" / "runs"
            cfg = RunConfig(
                dataset_name="office_home",
                data_root=str(data_root),
                source_domain="Ar",
                target_domain="Cl",
                method="source_only",
                epochs_source=2,
                epochs_adapt=0,
                batch_size=2,
                num_workers=0,
                lr_backbone=1e-3,
                lr_classifier=1e-2,
                weight_decay=1e-3,
                seed=0,
                deterministic=True,
                dry_run_max_samples=8,
                dry_run_max_batches=0,
            )

            original_train_one_epoch = source_only._train_one_epoch
            call_count = {"n": 0}

            def fail_on_second_epoch(*args, **kwargs):
                call_count["n"] += 1
                if call_count["n"] >= 2:
                    raise RuntimeError("simulated interruption")
                return original_train_one_epoch(*args, **kwargs)

            with mock.patch("src.experiments.methods.source_only.build_model", build_tiny_model):
                with mock.patch("src.experiments.methods.source_only._train_one_epoch", side_effect=fail_on_second_epoch):
                    with self.assertRaises(RuntimeError):
                        source_only.run(cfg, runs_root=runs_root, force_rerun=False, save_every_epochs=1)

                run_dir = get_run_dir(cfg, runs_root=runs_root)
                state_path = run_dir / "state.json"
                self.assertTrue(state_path.exists())
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertFalse(state.get("completed", True))
                self.assertEqual(int(state.get("last_completed_epoch", -1)), 0)

                res = source_only.run(cfg, runs_root=runs_root, force_rerun=False, save_every_epochs=1)
                self.assertEqual(res["status"], "trained")
                ckpt = torch.load(Path(res["checkpoint"]), map_location="cpu")
                self.assertTrue(bool(ckpt.get("completed", False)))
                self.assertGreaterEqual(int(ckpt.get("epoch", -1)), 1)

                res2 = source_only.run(cfg, runs_root=runs_root, force_rerun=False, save_every_epochs=1)
                self.assertEqual(res2["status"], "skipped")


if __name__ == "__main__":
    unittest.main()

