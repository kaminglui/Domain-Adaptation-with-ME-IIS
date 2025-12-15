import csv
import tempfile
import unittest
from pathlib import Path

from src.experiments.notebook_summary import collect_expected_runs
from src.experiments.run_config import RunConfig, get_run_dir
from utils.logging_utils import OFFICE_HOME_ME_IIS_FIELDS, upsert_csv_row


class TestUpsertCsvRow(unittest.TestCase):
    def test_upsert_idempotent_by_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            row = {
                "dataset": "office-home",
                "source": "Ar",
                "target": "Cl",
                "seed": 0,
                "method": "source_only",
                "run_id": "deadbeef00",
                "status": "trained",
                "error": "",
                "target_acc": 12.34,
                "source_acc": 56.78,
            }
            upsert_csv_row(str(path), OFFICE_HOME_ME_IIS_FIELDS, row, unique_key="run_id")
            # Same run_id, different target_acc should overwrite (still 1 row).
            row2 = dict(row)
            row2["target_acc"] = 99.0
            upsert_csv_row(str(path), OFFICE_HOME_ME_IIS_FIELDS, row2, unique_key="run_id")

            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].get("run_id"), "deadbeef00")
            self.assertEqual(rows[0].get("target_acc"), "99.0")

    def test_upsert_migrates_and_dedupes_legacy_csv_without_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.csv"
            legacy_fields = ["dataset", "source", "target", "seed", "method", "target_acc", "source_acc"]
            legacy_row = {
                "dataset": "office-home",
                "source": "Ar",
                "target": "Cl",
                "seed": "0",
                "method": "source_only",
                "target_acc": "1.0",
                "source_acc": "2.0",
            }
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=legacy_fields)
                writer.writeheader()
                writer.writerow(legacy_row)
                writer.writerow(legacy_row)  # duplicate

            new_row = {
                "dataset": "office-home",
                "source": "Ar",
                "target": "Cl",
                "seed": 0,
                "method": "me_iis",
                "run_id": "feedface99",
                "status": "trained",
                "error": "",
                "target_acc": 3.0,
                "source_acc": 4.0,
            }
            upsert_csv_row(str(path), OFFICE_HOME_ME_IIS_FIELDS, new_row, unique_key="run_id")

            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Legacy duplicate should be deduped to 1, plus the new row => 2 total.
            self.assertEqual(len(rows), 2)
            run_ids = [r.get("run_id", "").strip() for r in rows]
            self.assertIn("feedface99", run_ids)
            self.assertTrue(all(run_ids))


class TestNotebookSummaryCollection(unittest.TestCase):
    def test_collect_expected_runs_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir) / "outputs" / "runs"

            ok_cfg = RunConfig(
                dataset_name="office_home",
                data_root="datasets/Office-Home",
                source_domain="Ar",
                target_domain="Cl",
                method="source_only",
                epochs_source=1,
                epochs_adapt=0,
                batch_size=2,
                num_workers=0,
                seed=0,
                deterministic=True,
            )
            failed_cfg = RunConfig(
                dataset_name="office_home",
                data_root="datasets/Office-Home",
                source_domain="Ar",
                target_domain="Cl",
                method="me_iis",
                epochs_source=1,
                epochs_adapt=1,
                batch_size=2,
                num_workers=0,
                seed=0,
                deterministic=True,
                method_params={"feature_layers": ["layer4"]},
            )

            ok_dir = get_run_dir(ok_cfg, runs_root=runs_root)
            ok_dir.mkdir(parents=True, exist_ok=True)
            (ok_dir / "metrics.csv").write_text(
                "method,seed,target_acc,source_acc,run_id,timestamp\n"
                f"{ok_cfg.method},{ok_cfg.seed},12.0,34.0,{ok_cfg.run_id},t0\n",
                encoding="utf-8",
            )

            failed_dir = get_run_dir(failed_cfg, runs_root=runs_root)
            (failed_dir / "logs").mkdir(parents=True, exist_ok=True)
            (failed_dir / "logs" / "stderr.txt").write_text("Traceback: boom\n", encoding="utf-8")

            rows = collect_expected_runs([ok_cfg, failed_cfg], runs_root=runs_root)
            by_method = {r["method"]: r for r in rows}
            self.assertEqual(by_method["source_only"]["status"], "OK")
            self.assertEqual(by_method["me_iis"]["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()

