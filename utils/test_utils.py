import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn


@contextmanager
def temporary_workdir(path: Path):
    """Temporarily change the working directory for isolation in tests."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _write_image(path: Path, image_size: int = 256) -> None:
    arr = np.random.randint(0, 255, size=(image_size, image_size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def create_office_home_like(
    root: Path,
    domains: Sequence[str] = ("Ar", "Cl"),
    classes: Sequence[str] = ("class_a", "class_b"),
    images_per_class: int = 4,
    image_size: int = 256,
) -> Path:
    """Create a minimal Office-Home style directory tree with random images."""
    domain_map = {"Ar": "Art", "Cl": "Clipart", "Pr": "Product", "Rw": "RealWorld"}
    for code in domains:
        subdir = domain_map.get(code, code)
        for cls in classes:
            for idx in range(images_per_class):
                _write_image(root / subdir / cls / f"img_{idx}.jpg", image_size=image_size)
    return root


def create_office31_like(
    root: Path,
    domains: Sequence[str] = ("A", "W"),
    classes: Sequence[str] = ("class_a", "class_b"),
    images_per_class: int = 4,
    image_size: int = 256,
) -> Path:
    """Create a minimal Office-31 style directory tree with random images."""
    domain_map = {"A": "amazon", "D": "dslr", "W": "webcam"}
    for code in domains:
        subdir = domain_map.get(code, code)
        for cls in classes:
            for idx in range(images_per_class):
                _write_image(root / subdir / cls / f"img_{idx}.jpg", image_size=image_size)
    return root


class TinyBackbone(nn.Module):
    """Lightweight backbone used to keep synthetic tests fast."""

    def __init__(self, feature_dim: int = 32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(3 * 4 * 4, feature_dim)
        self.out_features = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.proj(x))
        return x

    def forward_intermediates(self, x: torch.Tensor, layers: Iterable[str]):
        feats = self.forward(x)
        intermediates = {layer: feats for layer in layers}
        return feats, intermediates


class TinyModel(nn.Module):
    """Tiny classifier that mimics the ME-IIS model interface."""

    def __init__(self, num_classes: int, feature_dim: int = 32, bottleneck_dim: int = 32):
        super().__init__()
        self.backbone = TinyBackbone(feature_dim=feature_dim)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        self.feature_dim = int(bottleneck_dim)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.backbone(x)
        feats = torch.relu(self.bottleneck(feats))
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits, None

    def forward_with_intermediates(self, x: torch.Tensor, feature_layers):
        feats, intermediates = self.backbone.forward_intermediates(x, feature_layers)
        feats = torch.relu(self.bottleneck(feats))
        logits = self.classifier(feats)
        return logits, feats, intermediates


def build_tiny_model(
    num_classes: int,
    feature_dim: int = 32,
    pretrained: bool = True,
    *,
    bottleneck_dim: int = 32,
    bottleneck_bn: bool = True,
    bottleneck_relu: bool = True,
    bottleneck_dropout: float = 0.0,
    **_: object,
) -> nn.Module:
    """Factory to mirror models.classifier.build_model for tests."""
    # Keep the tiny model simple: ignore BN/dropout toggles.
    _ = (pretrained, bottleneck_bn, bottleneck_relu, bottleneck_dropout)
    return TinyModel(num_classes=num_classes, feature_dim=feature_dim, bottleneck_dim=int(bottleneck_dim))
