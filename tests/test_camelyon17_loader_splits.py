from __future__ import annotations

from typing import Dict

import pytest
import torch

from src.datasets.wilds_camelyon17 import build_camelyon17_loaders


class _StubSubset(torch.utils.data.Dataset):
    def __init__(self, split: str, n: int = 4):
        self.split = split
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        x = torch.zeros(3, 8, 8) + float(idx)
        y = torch.tensor(0, dtype=torch.long)
        metadata = torch.zeros(1, dtype=torch.long)
        return x, y, metadata


class _StubWILDSDataset:
    def __init__(self):
        self.subsets: Dict[str, _StubSubset] = {}

    def get_subset(self, split: str, transform=None):  # noqa: ANN001
        ds = _StubSubset(split)
        self.subsets[split] = ds
        return ds


def test_camelyon17_loader_has_required_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("wilds")
    import src.datasets.wilds_camelyon17 as mod

    stub = _StubWILDSDataset()

    def _fake_get_dataset(*, root_dir: str, download: bool, unlabeled: bool):  # noqa: ANN001
        return stub

    def _fake_get_loaders():  # noqa: ANN001
        def _train_loader(_typ: str, dataset, batch_size: int, **kwargs):  # noqa: ANN001
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        def _eval_loader(_typ: str, dataset, batch_size: int, **kwargs):  # noqa: ANN001
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return _train_loader, _eval_loader

    monkeypatch.setattr(mod, "get_camelyon17_dataset", _fake_get_dataset)
    monkeypatch.setattr(mod, "_get_wilds_loaders", _fake_get_loaders)

    out = build_camelyon17_loaders(
        {
            "data_root": "unused",
            "download": False,
            "unlabeled": True,
            "split_mode": "uda_target",
            "eval_split": "test",
            "adapt_split": "test_unlabeled",
            "train_transform": None,
            "eval_transform": None,
            "batch_size": 2,
            "include_indices_in_train": True,
            "num_workers": 0,
        }
    )

    splits = out["splits"]
    assert splits.labeled_train is not None
    assert splits.labeled_val is not None
    assert splits.labeled_test is not None
    assert splits.unlabeled_val is not None
    assert splits.unlabeled_test is not None
    # train_unlabeled and id_val are optional in WILDS; ensure the attribute exists.
    assert hasattr(splits, "unlabeled_train")
    assert hasattr(splits, "labeled_id_val")

