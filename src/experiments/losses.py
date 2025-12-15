from __future__ import annotations

import torch


def coral_loss(source: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Deep CORAL loss: (1 / (4 d^2)) * ||C_s - C_t||_F^2

    Args:
        source: (N_s, d) feature tensor.
        target: (N_t, d) feature tensor.
    """
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be 2D tensors (N, d).")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target must have the same feature dimension.")

    ns = int(source.shape[0])
    nt = int(target.shape[0])
    d = int(source.shape[1])

    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)

    if ns <= 1:
        cov_s = torch.zeros((d, d), device=source.device, dtype=source.dtype)
    else:
        cov_s = (source.t() @ source) / max(ns - 1, 1)

    if nt <= 1:
        cov_t = torch.zeros((d, d), device=target.device, dtype=target.dtype)
    else:
        cov_t = (target.t() @ target) / max(nt - 1, 1)

    loss = torch.mean((cov_s - cov_t) ** 2)
    return loss / (4.0 * float(d * d) + eps)

