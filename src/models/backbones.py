from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


BackboneName = Literal["densenet121", "resnet50"]


@dataclass(frozen=True)
class BackboneOutput:
    model: nn.Module
    feature_dim: int
    name: str


def _imagenet_weights(name: BackboneName, pretrained: bool):
    if not pretrained:
        return None
    if name == "densenet121":
        return models.DenseNet121_Weights.IMAGENET1K_V1
    if name == "resnet50":
        return models.ResNet50_Weights.IMAGENET1K_V1
    raise ValueError(f"Unknown backbone '{name}'.")


def replace_batchnorm_with_instancenorm(module: nn.Module) -> nn.Module:
    """
    Recursively replace BatchNorm{1,2,3}d with InstanceNorm{1,2,3}d (affine=True).
    """

    def _swap(child: nn.Module) -> nn.Module:
        if isinstance(child, nn.BatchNorm1d):
            inst = nn.InstanceNorm1d(child.num_features, affine=True, track_running_stats=True)
            inst.weight.data.copy_(child.weight.data)
            inst.bias.data.copy_(child.bias.data)
            inst.running_mean.data.copy_(child.running_mean.data)
            inst.running_var.data.copy_(child.running_var.data)
            return inst
        if isinstance(child, nn.BatchNorm2d):
            inst = nn.InstanceNorm2d(child.num_features, affine=True, track_running_stats=True)
            inst.weight.data.copy_(child.weight.data)
            inst.bias.data.copy_(child.bias.data)
            inst.running_mean.data.copy_(child.running_mean.data)
            inst.running_var.data.copy_(child.running_var.data)
            return inst
        if isinstance(child, nn.BatchNorm3d):
            inst = nn.InstanceNorm3d(child.num_features, affine=True, track_running_stats=True)
            inst.weight.data.copy_(child.weight.data)
            inst.bias.data.copy_(child.bias.data)
            inst.running_mean.data.copy_(child.running_mean.data)
            inst.running_var.data.copy_(child.running_var.data)
            return inst
        return child

    for name, child in list(module.named_children()):
        new_child = _swap(child)
        if new_child is not child:
            setattr(module, name, new_child)
        else:
            replace_batchnorm_with_instancenorm(child)
    return module


class DenseNet121Backbone(nn.Module):
    def __init__(self, *, pretrained: bool = False):
        super().__init__()
        weights = _imagenet_weights("densenet121", pretrained)
        net = models.densenet121(weights=weights)
        self.features = net.features
        self.out_features = int(net.classifier.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        return torch.flatten(x, 1)


class ResNet50Backbone(nn.Module):
    def __init__(self, *, pretrained: bool = False, replace_bn_with_in: bool = False):
        super().__init__()
        weights = _imagenet_weights("resnet50", pretrained)
        net = models.resnet50(weights=weights)
        net.fc = nn.Identity()
        if replace_bn_with_in:
            net = replace_batchnorm_with_instancenorm(net)
        self.net = net
        self.out_features = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_backbone(
    name: BackboneName,
    *,
    pretrained: bool = False,
    replace_batchnorm_with_instancenorm: bool = False,
) -> BackboneOutput:
    name_l = name.lower()
    if name_l == "densenet121":
        model = DenseNet121Backbone(pretrained=pretrained)
        if replace_batchnorm_with_instancenorm:
            model = replace_batchnorm_with_instancenorm(model)
        return BackboneOutput(model=model, feature_dim=model.out_features, name="densenet121")
    if name_l == "resnet50":
        model = ResNet50Backbone(
            pretrained=pretrained, replace_bn_with_in=replace_batchnorm_with_instancenorm
        )
        return BackboneOutput(model=model, feature_dim=model.out_features, name="resnet50")
    raise ValueError(f"Unknown backbone '{name}'. Supported: densenet121, resnet50.")

