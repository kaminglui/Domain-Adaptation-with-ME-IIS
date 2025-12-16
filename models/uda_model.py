from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        use_bn: bool = True,
        use_relu: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(int(in_dim), int(out_dim))
        self.bn = nn.BatchNorm1d(int(out_dim)) if use_bn else None
        self.relu = nn.ReLU(inplace=True) if use_relu else None
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else None
        self.out_features = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class UdaModel(nn.Module):
    """Backbone + bottleneck + classifier wrapper for UDA baselines."""

    def __init__(self, backbone: nn.Module, bottleneck: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.bottleneck = bottleneck
        self.classifier = classifier

        out_features = getattr(bottleneck, "out_features", None)
        self.feature_dim = int(out_features) if out_features is not None else None

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        penultimate = self.backbone(x)
        feats = self.bottleneck(penultimate)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits, None

    def forward_with_intermediates(
        self, x: torch.Tensor, feature_layers: Tuple[str, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        penultimate, intermediates = self.backbone.forward_intermediates(x, layers=feature_layers)
        feats = self.bottleneck(penultimate)
        logits = self.classifier(feats)
        return logits, feats, intermediates

