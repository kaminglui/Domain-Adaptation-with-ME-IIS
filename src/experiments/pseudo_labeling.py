from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


class PseudoLabeledDataset(Subset):
    """Subset wrapper that replaces labels with provided pseudo labels."""

    def __init__(self, base_dataset: Dataset, indices: List[int], pseudo_labels: List[int]):
        super().__init__(base_dataset, indices)
        if len(indices) != len(pseudo_labels):
            raise ValueError("Length of indices and pseudo_labels must match.")
        self.pseudo_labels = [int(l) for l in pseudo_labels]

    def __getitem__(self, idx: int):
        image, _ = super().__getitem__(idx)
        label = self.pseudo_labels[idx]
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.to(dtype=torch.long)
        return image, label


@torch.no_grad()
def build_pseudo_labels(
    model: nn.Module,
    target_loader: DataLoader,
    device: torch.device,
    conf_thresh: float,
    max_ratio: float,
    num_source_samples: int,
) -> Tuple[List[int], List[int]]:
    """
    Generate pseudo labels for target samples with confidence above conf_thresh.

    Returns indices relative to target_loader.dataset and the corresponding predicted labels.
    max_ratio limits the number of pseudo samples kept to max_ratio * num_source_samples.
    """
    was_training = model.training
    model.eval()
    candidates: List[Tuple[float, int, int]] = []

    max_keep = None
    if max_ratio >= 0:
        max_keep = int(max_ratio * float(num_source_samples))
        if max_keep <= 0:
            if was_training:
                model.train()
            return [], []

    running_idx = 0
    for batch in target_loader:
        if len(batch) == 2:
            images, _ = batch
        else:
            images = batch[0]
        images = images.to(device)
        logits, _ = model(images, return_features=False)
        probs = F.softmax(logits, dim=1)
        max_prob, pred = probs.max(dim=1)
        bs = images.size(0)
        for i in range(bs):
            dataset_idx = running_idx + i
            if float(max_prob[i].item()) >= conf_thresh:
                candidates.append((float(max_prob[i].item()), dataset_idx, int(pred[i].item())))
        running_idx += bs

    if was_training:
        model.train()

    if not candidates:
        return [], []

    candidates.sort(key=lambda x: x[0], reverse=True)
    if max_keep is not None and max_keep > 0:
        candidates = candidates[:max_keep]

    pseudo_indices = [c[1] for c in candidates]
    pseudo_labels = [c[2] for c in candidates]
    return pseudo_indices, pseudo_labels

