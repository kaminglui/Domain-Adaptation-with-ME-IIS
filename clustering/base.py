"""
Shared latent-probability backend interface for ME-IIS.

The backend is responsible only for producing P[M_i=j | a_i] for a given
layer's activations a_i. IIS math (Eq. 14–18) consumes these probabilities
without caring about the underlying model (GMM, vMF-softmax, etc.).
"""

from abc import ABC, abstractmethod

import numpy as np


class LatentBackend(ABC):
    """Abstract interface for per-layer latent probability backends."""

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


# Backwards compatibility: previous code referred to this interface as ClusteringBackend.
ClusteringBackend = LatentBackend
