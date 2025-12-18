from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Algorithm, unpack_wilds_batch


class ERM(Algorithm):
    def __init__(self, *, featurizer: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = nn.Linear(int(feature_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.featurizer(x)
        return self.classifier(feats)

    def update(self, labeled_batch: Any, unlabeled_batch: Any | None = None) -> Dict[str, Any]:
        batch = unpack_wilds_batch(labeled_batch)
        if batch.y is None:
            raise ValueError("ERM.update expected labels but batch.y is None.")
        logits = self.forward(batch.x)
        loss = F.cross_entropy(logits, batch.y)
        acc = (logits.argmax(dim=1) == batch.y).float().mean()
        return {
            "loss": loss,
            "loss_cls": loss.detach(),
            "acc": acc.detach(),
            "batch_size": int(batch.x.shape[0]),
        }

    def parameter_groups(self) -> Iterable[Dict[str, Any]]:
        return [
            {"name": "featurizer", "params": self.featurizer.parameters()},
            {"name": "classifier", "params": self.classifier.parameters()},
        ]

