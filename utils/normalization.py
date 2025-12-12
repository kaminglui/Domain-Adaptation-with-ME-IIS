"""
Feature normalization utilities.
"""

import numpy as np


def l2_normalize(X: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalization with zero-vector safety.
    Zero rows remain zero; non-zero rows are scaled to unit norm within tolerance.
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("Input must have at least one dimension for normalization.")
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    safe_norm = np.where(norm > eps, norm, 1.0)
    normalized = arr / safe_norm
    normalized = np.where(norm > eps, normalized, 0.0)
    return normalized.astype(np.float64, copy=False)
