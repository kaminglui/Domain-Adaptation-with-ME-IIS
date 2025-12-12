"""
Clustering backends for ME-IIS.

Exports the shared backend interface along with concrete implementations.
"""

from .base import ClusteringBackend, LatentBackend
from .factory import create_backend
from .gmm_backend import GMMBackend, GMMBackendConfig
from .vmf_softmax_backend import VMFSoftmaxBackend, VMFSoftmaxConfig

__all__ = [
    "LatentBackend",
    "ClusteringBackend",
    "create_backend",
    "GMMBackend",
    "GMMBackendConfig",
    "VMFSoftmaxBackend",
    "VMFSoftmaxConfig",
]
