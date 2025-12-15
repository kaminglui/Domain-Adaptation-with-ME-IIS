from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class DropLabelsDataset(Dataset):
    """Wrap a dataset and replace labels with a fixed sentinel value."""

    def __init__(self, dataset: Dataset, label_value: int = -1):
        self.dataset = dataset
        self.label_value = int(label_value)
        for attr in ("class_to_idx", "classes", "root", "transform"):
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, _ = self.dataset[idx]
        return image, torch.tensor(self.label_value, dtype=torch.long)


def assert_labels_dropped(labels: Any, label_value: int = -1) -> None:
    if torch.is_tensor(labels):
        if labels.numel() == 0:
            return
        if not bool((labels.to(dtype=torch.long) == int(label_value)).all().item()):
            raise AssertionError("Target labels were not dropped (unexpected label values found).")

