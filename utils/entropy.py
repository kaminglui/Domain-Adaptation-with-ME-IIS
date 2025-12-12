"""
Entropy and filtering helpers for ME-IIS.
"""

import math
import numpy as np


def prediction_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute entropy H(p) = -sum_k p_k log(p_k + eps) for each row of probs.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"prediction_entropy expects 2D array, got shape {arr.shape}.")
    clipped = np.clip(arr, 0.0, 1.0)
    return -np.sum(clipped * np.log(clipped + eps), axis=1)


def select_low_entropy_indices(entropy: np.ndarray, keep_ratio: float) -> np.ndarray:
    """
    Return indices of the lowest-entropy samples given a keep ratio in (0, 1].
    """
    if keep_ratio <= 0 or keep_ratio > 1:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}.")
    ent = np.asarray(entropy, dtype=np.float64).reshape(-1)
    n = ent.shape[0]
    n_keep = max(1, int(math.ceil(float(n) * float(keep_ratio))))
    order = np.argsort(ent)
    return order[:n_keep]
