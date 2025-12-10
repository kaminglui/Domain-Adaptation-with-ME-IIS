# models – Network Components

## Purpose
Defines the ResNet-50 backbone, classifier head, and ME–IIS adapter used across training and adaptation.

## Contents
- `backbone.py` – ResNet-50 feature extractor with intermediate hooks and pooled outputs.
- `classifier.py` – Linear classifier head and `build_model` helper returning `FullModel`.
- `me_iis_adapter.py` – Core ME–IIS constraint construction, IIS solver, and adaptation helpers.
- `__init__.py` – Package marker.

## Main APIs
- `ResNet50Backbone(pretrained=True)` – Outputs 2048-d features; supports intermediate layers `layer1`–`layer4`, `avgpool`.
- `ClassifierHead(in_dim, num_classes)` – Linear head with He init.
- `build_model(num_classes, pretrained=True)` – Builds `FullModel(backbone, head)`.
- `FullModel.forward(x, return_features=False)` – Returns `(logits, features_or_None)`.
- `FullModel.forward_with_intermediates(x, feature_layers)` – Returns logits, penultimate features, and requested intermediate activations.
- `MaxEntAdapter` and helpers (in `me_iis_adapter.py`) – Implements ME–IIS math (not altered by scripts).

## Notes
- Default supported feature layers: `["layer1","layer2","layer3","layer4","avgpool"]`.
- Adaptation scripts select layers via CLI (`--feature_layers`).
