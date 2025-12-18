from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AlgorithmBatch:
    x: torch.Tensor
    y: Optional[torch.Tensor]
    metadata: Optional[torch.Tensor] = None
    idx: Optional[torch.Tensor] = None


def unpack_wilds_batch(batch: Any) -> AlgorithmBatch:
    """
    WILDS loaders typically yield (x, y, metadata). We also support an optional
    trailing idx field for per-sample weighting.
    """
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Expected batch tuple/list, got {type(batch)}.")
    if len(batch) == 3:
        x, y, metadata = batch
        return AlgorithmBatch(x=x, y=y, metadata=metadata, idx=None)
    if len(batch) == 4:
        x, y, metadata, idx = batch
        return AlgorithmBatch(x=x, y=y, metadata=metadata, idx=idx)
    raise ValueError(f"Unsupported batch format with {len(batch)} elements.")


class Algorithm(nn.Module):
    """
    Common interface for ERM/UDA algorithms.

    `update(...)` computes the per-step loss/metrics; the training loop owns
    iteration/epochs, logging, checkpointing, and optimizer stepping.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def update(self, labeled_batch: Any, unlabeled_batch: Any | None = None) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def parameter_groups(self) -> Iterable[Dict[str, Any]]:
        return [{"params": self.parameters()}]

    def on_epoch_start(self, epoch: int, *, device: torch.device) -> Dict[str, Any]:
        return {}

    def on_epoch_end(self, epoch: int, *, device: torch.device) -> Dict[str, Any]:
        return {}

