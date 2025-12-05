from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


SUPPORTED_LAYERS = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
# Names follow torchvision ResNet-50 modules (e.g., resnet.layer3, resnet.avgpool).


class ResNet50Backbone(nn.Module):
    """ResNet-50 feature extractor that can expose intermediate pooled features."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.out_features = 2048

    def _pool_flatten(self, x: torch.Tensor) -> torch.Tensor:
        # Use global average pooling to project spatial activations to a vector.
        return torch.flatten(F.adaptive_avg_pool2d(x, output_size=1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward returning the penultimate (avgpooled) features."""
        out = self.resnet.conv1(x)
        out = self.resnet.bn1(out)
        out = self.resnet.relu(out)
        out = self.resnet.maxpool(out)

        out = self.resnet.layer1(out)
        out = self.resnet.layer2(out)
        out = self.resnet.layer3(out)
        out = self.resnet.layer4(out)
        out = self.resnet.avgpool(out)
        feats = torch.flatten(out, 1)
        return feats

    def forward_intermediates(self, x: torch.Tensor, layers: Tuple[str, ...]):
        """
        Forward pass that additionally returns pooled activations from specified layers.

        Args:
            x: input tensor.
            layers: tuple of layer names (subset of SUPPORTED_LAYERS).

        Returns:
            penultimate features (Tensor) and a dict layer_name -> pooled features (Tensor).
        """
        invalid = set(layers) - set(SUPPORTED_LAYERS)
        if invalid:
            raise ValueError(f"Unsupported layers requested: {invalid}. Supported: {SUPPORTED_LAYERS}")

        feats_dict: Dict[str, torch.Tensor] = OrderedDict()
        out = self.resnet.conv1(x)
        out = self.resnet.bn1(out)
        out = self.resnet.relu(out)
        out = self.resnet.maxpool(out)

        out = self.resnet.layer1(out)
        if "layer1" in layers:
            feats_dict["layer1"] = self._pool_flatten(out)

        out = self.resnet.layer2(out)
        if "layer2" in layers:
            feats_dict["layer2"] = self._pool_flatten(out)

        out = self.resnet.layer3(out)
        if "layer3" in layers:
            feats_dict["layer3"] = self._pool_flatten(out)

        out = self.resnet.layer4(out)
        if "layer4" in layers:
            feats_dict["layer4"] = self._pool_flatten(out)

        out = self.resnet.avgpool(out)
        penultimate = torch.flatten(out, 1)
        if "avgpool" in layers:
            feats_dict["avgpool"] = penultimate
        return penultimate, feats_dict


class FullModel(nn.Module):
    """Wrapper returning logits and optionally features."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits, None

    def forward_with_intermediates(
        self, x: torch.Tensor, feature_layers: Tuple[str, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward that returns logits, penultimate feats, and selected layer activations."""
        feats, intermediates = self.backbone.forward_intermediates(x, layers=feature_layers)
        logits = self.classifier(feats)
        return logits, feats, intermediates


def _self_test_forward_with_intermediates() -> None:
    """Quick shape sanity check for the backbone hooks."""
    device = torch.device("cpu")
    model = ResNet50Backbone(pretrained=False).to(device)
    model.eval()
    dummy = torch.randn(2, 3, 224, 224, device=device)
    layers = ("layer3", "layer4", "avgpool")
    _, feats = model.forward_intermediates(dummy, layers)
    expected_shapes = {"layer3": (2, 1024), "layer4": (2, 2048), "avgpool": (2, 2048)}
    for name in layers:
        if name not in feats:
            raise AssertionError(f"Missing requested layer '{name}' in intermediates.")
        if feats[name].shape != torch.Size(expected_shapes[name]):
            raise AssertionError(f"Unexpected shape for {name}: {feats[name].shape}, expected {expected_shapes[name]}.")
    print("[Backbone self-test] forward_with_intermediates outputs have expected shapes.")


if __name__ == "__main__":
    _self_test_forward_with_intermediates()
