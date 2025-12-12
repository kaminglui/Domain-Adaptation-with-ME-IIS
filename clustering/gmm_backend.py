"""
GaussianMixture backend implementing the ClusteringBackend interface.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture

from clustering.base import LatentBackend


@dataclass
class GMMBackendConfig:
    selection_mode: str = "fixed"
    bic_min_components: int = 2
    bic_max_components: int = 8
    max_subsample: int = 20000
    covariance_type: str = "diag"
    reg_covar: float = 1e-6


class GMMBackend(LatentBackend):
    """Wraps sklearn GaussianMixture with optional BIC-based component selection."""

    def __init__(
        self,
        n_components: int,
        random_state: Optional[int] = None,
        config: Optional[GMMBackendConfig] = None,
        layer_name: Optional[str] = None,
    ):
        self._n_components = int(n_components)
        self.random_state = random_state
        self.config = config or GMMBackendConfig()
        self.layer_name = layer_name or ""
        self.model: Optional[GaussianMixture] = None

    @property
    def n_components(self) -> int:
        if self.model is not None:
            return int(self.model.n_components)
        return int(self._n_components)

    def _fit_model(self, n_components: int, X: np.ndarray) -> None:
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=self.config.covariance_type,
            reg_covar=self.config.reg_covar,
            random_state=self.random_state,
        )
        self.model.fit(X)

    def _select_components_via_bic(self, X: np.ndarray) -> int:
        if X.ndim != 2:
            raise ValueError(f"Expected 2D features for BIC selection, got shape {X.shape}.")
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("Cannot run BIC selection with zero target samples.")

        feats_np_sub = X
        if n_samples > self.config.max_subsample:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n_samples, size=self.config.max_subsample, replace=False)
            feats_np_sub = X[idx]

        min_comp = max(1, int(self.config.bic_min_components))
        max_comp = max(min_comp, int(self.config.bic_max_components))
        max_comp = min(max_comp, feats_np_sub.shape[0])
        min_comp = min(min_comp, max_comp)

        best_bic = np.inf
        best_m: Optional[int] = None
        for m in range(min_comp, max_comp + 1):
            if m > feats_np_sub.shape[0]:
                continue
            gmm = GaussianMixture(
                n_components=m,
                covariance_type=self.config.covariance_type,
                reg_covar=self.config.reg_covar,
                random_state=self.random_state,
            )
            gmm.fit(feats_np_sub)
            bic = gmm.bic(feats_np_sub)
            if bic < best_bic:
                best_bic = bic
                best_m = m

        if best_m is None:
            raise RuntimeError(
                f"BIC component selection failed for layer '{self.layer_name}' with min={min_comp}, max={max_comp}."
            )

        self._fit_model(best_m, X)
        print(f"[GMM] layer={self.layer_name or '<unnamed>'} selection=bic M*={best_m} BIC={best_bic:.3e}")
        return best_m

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"GMMBackend.fit expects 2D array, got shape {X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("GMMBackend.fit received zero samples.")

        mode = self.config.selection_mode
        if mode not in {"fixed", "bic"}:
            raise ValueError(f"Unknown gmm selection_mode '{mode}'.")

        if mode == "fixed":
            self._fit_model(self._n_components, X)
        else:
            selected = self._select_components_via_bic(X)
            self._n_components = int(selected)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GMMBackend.predict_proba called before fit().")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"GMMBackend.predict_proba expects 2D array, got shape {X.shape}.")
        gamma = self.model.predict_proba(X)
        return gamma.astype(np.float64, copy=False)
