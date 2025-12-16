from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def assert_valid_ce_labels(
    labels: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: Optional[int] = None,
    name: str = "labels",
) -> None:
    if not torch.is_tensor(labels):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(labels)}")
    if labels.dtype != torch.long:
        raise AssertionError(f"{name} dtype must be torch.long, got {labels.dtype}")
    if labels.numel() == 0:
        return
    if int(num_classes) <= 0:
        raise ValueError("num_classes must be > 0 for label validation.")

    if ignore_index is None:
        min_v = int(labels.min().item())
        max_v = int(labels.max().item())
        if min_v < 0 or max_v >= int(num_classes):
            raise AssertionError(f"{name} values out of range: [{min_v}, {max_v}] vs num_classes={int(num_classes)}")
        return

    ignore = int(ignore_index)
    mask = labels != ignore
    if not bool(mask.any().item()):
        return
    valid = labels[mask]
    min_v = int(valid.min().item())
    max_v = int(valid.max().item())
    if min_v < 0 or max_v >= int(num_classes):
        raise AssertionError(
            f"{name} values out of range (excluding ignore_index={ignore}): "
            f"[{min_v}, {max_v}] vs num_classes={int(num_classes)}"
        )


def safe_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    assert_valid_ce_labels(labels, num_classes=int(num_classes), ignore_index=ignore_index)
    if ignore_index is None:
        return F.cross_entropy(logits, labels, reduction=reduction)
    return F.cross_entropy(logits, labels, ignore_index=int(ignore_index), reduction=reduction)

