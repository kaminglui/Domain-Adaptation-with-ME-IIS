"""
Shared clustering backend interface for ME-IIS.
"""

from abc import ABC, abstractmethod

import numpy as np


class ClusteringBackend(ABC):
    """Abstract interface for per-layer clustering backends."""

    @property
    @abstractmethod
    def n_components(self) -> int:
        """Return the number of mixture components / prototypes."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Fit the backend to a 2D feature array."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample responsibilities with shape (N, n_components)."""
