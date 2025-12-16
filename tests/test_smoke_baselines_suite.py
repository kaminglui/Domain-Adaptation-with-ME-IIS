import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.experiments.run_config import RunConfig
from src.experiments.runner import run_one
from utils.test_utils import build_tiny_model, create_office_home_like


class TestBaselineSmokeSuite(unittest.TestCase):
    def test_methods_run_and_signatures_differ(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            data_root = workdir / "data"
            create_office_home_like(data_root, domains=("Ar", "Cl"), images_per_class=3, image_size=64)

            runs_root = workdir / "outputs" / "runs"
            base = dict(
                dataset_name="office_home",
                data_root=str(data_root),
                source_domain="Ar",
                target_domain="Cl",
                epochs_source=1,
                epochs_adapt=1,
                batch_size=2,
                num_workers=0,
                seed=0,
                deterministic=True,
                dry_run_max_samples=12,
                dry_run_max_batches=2,
                bottleneck_dim=8,
            )

            methods = ["source_only", "dann", "dan", "jan", "cdan", "me_iis"]
            fingerprints = {}

            patchers = [
                mock.patch("src.experiments.methods.source_only.build_model", build_tiny_model),
                mock.patch("src.experiments.methods.dann.build_model", build_tiny_model),
                mock.patch("src.experiments.methods.dan.build_model", build_tiny_model),
                mock.patch("src.experiments.methods.jan.build_model", build_tiny_model),
                mock.patch("src.experiments.methods.cdan.build_model", build_tiny_model),
                mock.patch("src.experiments.methods.me_iis.build_model", build_tiny_model),
                mock.patch("src.experiments.eval_harness.build_model", build_tiny_model),
            ]
            with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], patchers[5], patchers[6]:
                for method in methods:
                    method_params = {}
                    if method == "me_iis":
                        method_params = {
                            "iis_iters": 1,
                            "gmm_selection_mode": "fixed",
                            "num_latent_styles": 2,
                            "cluster_clean_ratio": 1.0,
                        }
                    cfg = RunConfig(method=method, method_params=method_params, **base)
                    res = run_one(cfg, force_rerun=True, runs_root=runs_root, write_metrics=True)
                    run_dir = Path(res["run_dir"])
                    self.assertTrue((run_dir / "signature.json").exists())
                    self.assertTrue((run_dir / "metrics.csv").exists())

                    sig = json.loads((run_dir / "signature.json").read_text(encoding="utf-8"))
                    fp = sig.get("comparison_fingerprint")
                    self.assertIsInstance(fp, str)
                    fingerprints[method] = fp

                    if method == "me_iis":
                        ckpt = torch.load(Path(res["checkpoint"]), map_location="cpu")
                        weights = ckpt.get("weights")
                        self.assertIsNotNone(weights)
                        self.assertAlmostEqual(float(weights.sum().item()), 1.0, places=4)

                    stdout_path = run_dir / "logs" / "stdout.txt"
                    self.assertTrue(stdout_path.exists())
                    stdout = stdout_path.read_text(encoding="utf-8", errors="ignore")
                    self.assertIn(f"[METHOD] {method}", stdout)

            # Ensure method routing is not degenerate: each baseline should have a distinct fingerprint.
            self.assertEqual(len(fingerprints), len(methods))
            self.assertEqual(len(set(fingerprints.values())), len(methods))


if __name__ == "__main__":
    unittest.main()
