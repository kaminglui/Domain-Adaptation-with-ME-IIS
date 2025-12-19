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
        self._feature_info_logged: bool = False
        self._backbone_name: Optional[str] = None
        self._backbone_pretrained: Optional[bool] = None
        self._feature_dim: Optional[int] = None
        self._feature_layer: Optional[str] = None

    def set_backbone_info(
        self,
        *,
        backbone_name: str,
        pretrained: bool,
        feature_dim: int,
        feature_layer: Optional[str] = None,
    ) -> None:
        self._backbone_name = str(backbone_name)
        self._backbone_pretrained = bool(pretrained)
        self._feature_dim = int(feature_dim)
        self._feature_layer = None if feature_layer is None else str(feature_layer)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        featurizer = getattr(self, "featurizer", None)
        if featurizer is None:
            raise AttributeError(
                f"{type(self).__name__} has no attribute 'featurizer'; cannot compute f(x)."
            )
        feats = featurizer(x)
        if not self._feature_info_logged:
            bb = self._backbone_name or "UNKNOWN"
            pt = self._backbone_pretrained
            feat_dim = self._feature_dim
            layer = self._feature_layer or "UNKNOWN"
            mode = "train" if self.training else "eval"
            shape = tuple(feats.shape)
            print(
                f"[features] backbone={bb} pretrained={pt} feat_layer={layer} feature_dim={feat_dim} "
                f"f(x)_shape={shape} model_mode={mode}"
            )
            self._feature_info_logged = True
        return feats

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
