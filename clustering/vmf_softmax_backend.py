"""
vMF-like prototype softmax backend using unit-normalized features and KMeans centroids.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from clustering.base import ClusteringBackend
from utils.normalization import l2_normalize


@dataclass
class VMFSoftmaxConfig:
    kappa: float = 20.0
    random_state: Optional[int] = None
    kmeans_n_init: int = 10
    eps: float = 1e-12


class VMFSoftmaxBackend(ClusteringBackend):
    """Softmax over cosine similarities to KMeans prototypes (spherical vMF analogue)."""

    def __init__(self, n_components: int, config: Optional[VMFSoftmaxConfig] = None):
        self._n_components = int(n_components)
        self.config = config or VMFSoftmaxConfig()
        self.centroids_: Optional[np.ndarray] = None

    @property
    def n_components(self) -> int:
        return int(self._n_components)

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"VMFSoftmaxBackend.fit expects 2D array, got shape {X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("VMFSoftmaxBackend.fit received zero samples.")
        X_norm = l2_normalize(X, axis=1, eps=self.config.eps)
        kmeans = KMeans(
            n_clusters=self._n_components,
            random_state=self.config.random_state,
            n_init=self.config.kmeans_n_init,
        )
        kmeans.fit(X_norm)
        centroids = l2_normalize(kmeans.cluster_centers_, axis=1, eps=self.config.eps)
        self.centroids_ = centroids.astype(np.float64, copy=False)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("VMFSoftmaxBackend.predict_proba called before fit().")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"VMFSoftmaxBackend.predict_proba expects 2D array, got shape {X.shape}.")
        X_norm = l2_normalize(X, axis=1, eps=self.config.eps)
        sims = X_norm @ self.centroids_.T  # cosine similarities
        logits = self.config.kappa * sims
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        denom = exp_logits.sum(axis=1, keepdims=True)
        probs = exp_logits / denom
        return probs.astype(np.float64, copy=False)
