"""
Factory helpers to build clustering backends from configuration.
"""

from typing import Optional

from clustering.base import LatentBackend
from clustering.gmm_backend import GMMBackend, GMMBackendConfig
from clustering.vmf_softmax_backend import VMFSoftmaxBackend, VMFSoftmaxConfig


def create_backend(
    backend_name: str,
    n_components: int,
    seed: Optional[int],
    layer_name: Optional[str] = None,
    gmm_selection_mode: str = "fixed",
    gmm_bic_min_components: int = 2,
    gmm_bic_max_components: int = 8,
    gmm_covariance_type: str = "diag",
    gmm_reg_covar: float = 1e-6,
    kmeans_n_init: int = 10,
    vmf_kappa: float = 20.0,
) -> LatentBackend:
    """
    Build a clustering backend instance based on a string identifier.

    Args:
        backend_name: which backend to create ('gmm' or 'vmf_softmax').
        n_components: number of clusters/components requested.
        seed: random seed forwarded to deterministic components.
        layer_name: optional layer name for logging.
        gmm_selection_mode: only used for GMM backend ('fixed' or 'bic').
        gmm_bic_min_components: min components for BIC search (GMM only).
        gmm_bic_max_components: max components for BIC search (GMM only).
        kmeans_n_init: n_init for KMeans (vMF-softmax only).
        vmf_kappa: concentration parameter for vMF-softmax.
    """
    name = backend_name.lower()
    if name == "gmm":
        cfg = GMMBackendConfig(
            selection_mode=gmm_selection_mode,
            bic_min_components=gmm_bic_min_components,
            bic_max_components=gmm_bic_max_components,
            covariance_type=str(gmm_covariance_type),
            reg_covar=float(gmm_reg_covar),
        )
        return GMMBackend(n_components=n_components, random_state=seed, config=cfg, layer_name=layer_name)
    if name == "vmf_softmax":
        cfg = VMFSoftmaxConfig(kappa=vmf_kappa, random_state=seed, kmeans_n_init=kmeans_n_init)
        return VMFSoftmaxBackend(n_components=n_components, config=cfg)
    raise ValueError(f"Unknown clustering backend '{backend_name}'. Supported: gmm, vmf_softmax.")
