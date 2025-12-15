import os
import unittest

import torch
import torch.nn as nn

from utils.data_utils import build_loader
from utils.resource_utils import (
    DataLoaderTuning,
    ResourceSnapshot,
    auto_tune_dataloader,
    format_bytes,
    tune_checkpoint_saving,
)


class _TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return torch.zeros(3, 4, 4), torch.tensor(0)


class TestResourceUtils(unittest.TestCase):
    def test_format_bytes(self) -> None:
        self.assertEqual(format_bytes(None), "unknown")
        self.assertEqual(format_bytes(0), "0.00B")
        self.assertEqual(format_bytes(1024), "1.00KB")
        self.assertEqual(format_bytes(1024**2), "1.00MB")

    def test_build_loader_accepts_resource_kwargs(self) -> None:
        ds = _TinyDataset()
        _ = build_loader(
            ds,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            seed=0,
            pin_memory=True,
            persistent_workers=True,  # should be ignored when num_workers=0
            prefetch_factor=4,  # should be ignored when num_workers=0
        )
        _ = build_loader(
            ds,
            batch_size=2,
            shuffle=False,
            num_workers=1,
            seed=0,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def test_auto_tune_dataloader_env_gated(self) -> None:
        snapshot = ResourceSnapshot(
            cpu_total_bytes=32 * 1024**3,
            cpu_available_bytes=32 * 1024**3,
            cpu_method="test",
            cpu_count=16,
            cuda_available=False,
            cuda_device=None,
            cuda_name=None,
            cuda_total_bytes=None,
            cuda_free_bytes=None,
            disk_path=".",
            disk_total_bytes=100 * 1024**3,
            disk_free_bytes=100 * 1024**3,
            data_path=None,
            data_disk_total_bytes=None,
            data_disk_free_bytes=None,
        )

        os.environ.pop("ME_IIS_AUTO_RESOURCES", None)
        t0 = auto_tune_dataloader(
            base_batch_size=32,
            base_num_workers=4,
            device=torch.device("cpu"),
            resources=snapshot,
            model=None,
            input_size=224,
            num_classes=10,
        )
        self.assertIsInstance(t0, DataLoaderTuning)
        self.assertEqual(t0.batch_size, 32)
        self.assertEqual(t0.num_workers, 4)
        self.assertFalse(t0.pin_memory)
        self.assertFalse(t0.persistent_workers)
        self.assertIsNone(t0.prefetch_factor)

        os.environ["ME_IIS_AUTO_RESOURCES"] = "1"
        t1 = auto_tune_dataloader(
            base_batch_size=32,
            base_num_workers=4,
            device=torch.device("cpu"),
            resources=snapshot,
            model=None,
            input_size=224,
            num_classes=10,
        )
        # CPU-only: batch_size unchanged, but workers/prefetch should increase.
        self.assertEqual(t1.batch_size, 32)
        self.assertGreaterEqual(t1.num_workers, 4)
        self.assertTrue(t1.persistent_workers if t1.num_workers > 0 else True)
        os.environ.pop("ME_IIS_AUTO_RESOURCES", None)

    def test_tune_checkpoint_saving_can_reduce_frequency(self) -> None:
        model = nn.Linear(10, 10)
        tuned = tune_checkpoint_saving(
            disk_free_bytes=10_000,
            total_epochs=10,
            save_every_epochs_requested=1,
            model=model,
            reserve_bytes=0,
        )
        # For this tiny model + small disk budget, saving every epoch should be reduced.
        self.assertGreaterEqual(int(tuned.save_every_epochs), 1)
        self.assertNotEqual(int(tuned.save_every_epochs), 1)


if __name__ == "__main__":
    unittest.main()

