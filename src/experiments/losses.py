from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch


def gaussian_kernel_matrix(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: Optional[float] = None,
) -> torch.Tensor:
    """
    Multi-kernel Gaussian (RBF) matrix over concatenated [source; target].

    Returns:
        kernels: (N+M, N+M) tensor
    """
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be 2D tensors (N, d).")
    if int(kernel_num) <= 0:
        raise ValueError("kernel_num must be > 0.")
    if float(kernel_mul) <= 0:
        raise ValueError("kernel_mul must be > 0.")

    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)
    l2_distance = ((total0 - total1) ** 2).sum(2)

    n_samples = int(total.shape[0])
    if fix_sigma is None:
        denom = max(1, n_samples * n_samples - n_samples)
        bandwidth = l2_distance.detach().sum() / float(denom)
    else:
        bandwidth = torch.tensor(float(fix_sigma), device=total.device, dtype=total.dtype)

    bandwidth = bandwidth / (float(kernel_mul) ** (int(kernel_num) // 2))
    bandwidth_list = [bandwidth * (float(kernel_mul) ** i) for i in range(int(kernel_num))]
    kernels = sum(torch.exp(-l2_distance / (bw + 1e-12)) for bw in bandwidth_list)
    return kernels


def mmd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: Optional[float] = None,
) -> torch.Tensor:
    """
    Maximum Mean Discrepancy with a Gaussian kernel mixture.
    """
    kernels = gaussian_kernel_matrix(
        source,
        target,
        kernel_mul=float(kernel_mul),
        kernel_num=int(kernel_num),
        fix_sigma=fix_sigma,
    )
    n = int(source.shape[0])
    if n <= 0:
        return torch.tensor(0.0, device=source.device)
    k_xx = kernels[:n, :n]
    k_yy = kernels[n:, n:]
    k_xy = kernels[:n, n:]
    k_yx = kernels[n:, :n]
    return k_xx.mean() + k_yy.mean() - k_xy.mean() - k_yx.mean()


def joint_mmd_loss(
    source_reprs: Sequence[torch.Tensor],
    target_reprs: Sequence[torch.Tensor],
    *,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: Optional[float] = None,
) -> torch.Tensor:
    """
    Joint MMD (JAN-style): product of Gaussian kernels across representations.
    """
    if len(source_reprs) != len(target_reprs):
        raise ValueError("source_reprs and target_reprs must have the same length.")
    if not source_reprs:
        raise ValueError("At least one representation is required for joint MMD.")

    joint_kernel = None
    for src, tgt in zip(source_reprs, target_reprs):
        k = gaussian_kernel_matrix(src, tgt, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        joint_kernel = k if joint_kernel is None else joint_kernel * k

    assert joint_kernel is not None
    n = int(source_reprs[0].shape[0])
    if n <= 0:
        return torch.tensor(0.0, device=source_reprs[0].device)
    k_xx = joint_kernel[:n, :n]
    k_yy = joint_kernel[n:, n:]
    k_xy = joint_kernel[:n, n:]
    k_yx = joint_kernel[n:, :n]
    return k_xx.mean() + k_yy.mean() - k_xy.mean() - k_yx.mean()


def entropy(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-sample entropy for probability vectors.
    """
    p = probs.clamp(min=float(eps), max=1.0)
    return -(p * p.log()).sum(dim=1)

