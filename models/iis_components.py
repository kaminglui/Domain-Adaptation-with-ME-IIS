"""
Modular IIS components used by MaxEntAdapter.

The implementations follow Pal & Miller Eq. (14)–(18) exactly for latent-variable
constraints:
    - Eq. (14): P_g[C=c, M_i=j] = (1/T) * sum_t I(c(t)=c) * P[M_i=j | a_i(t)]
    - Eq. (15): P_m[C=c, M_i=j] = (1/T) * sum_t P[C=c | f(t), a(t)] * P[M_i=j | a_i(t)]
    - Eq. (18): Δλ = (1 / (N_d + N_c)) * log(P_g / P_m)
This module only handles how these probabilities are built and consumed; the IIS
updates themselves are unchanged.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from clustering.base import LatentBackend
from utils.entropy import prediction_entropy, select_low_entropy_indices


class TargetEntropyFilter:
    """Select low-entropy target indices for optional clean clustering."""

    def __init__(self, keep_ratio: float = 1.0):
        if keep_ratio <= 0 or keep_ratio > 1:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}.")
        self.keep_ratio = float(keep_ratio)

    def select(self, class_probs: torch.Tensor) -> Optional[np.ndarray]:
        """
        Return indices of the lowest-entropy rows in class_probs.
        Uses utils.entropy.prediction_entropy for deterministic filtering.
        """
        if self.keep_ratio >= 1.0:
            return None
        probs_np = class_probs.detach().cpu().numpy()
        ent = prediction_entropy(probs_np)
        return select_low_entropy_indices(ent, keep_ratio=self.keep_ratio)


class ConstraintIndexer:
    """Utility to flatten (layer, component, class) constraint tensors."""

    def __init__(self, layers: List[str], components_per_layer: Dict[str, int], num_classes: int):
        self.layers = list(layers)
        self.components_per_layer = dict(components_per_layer)
        self.num_classes = int(num_classes)
        self.offsets: Dict[str, int] = {}
        offset = 0
        for layer in self.layers:
            self.offsets[layer] = offset
            offset += self.components_per_layer[layer] * self.num_classes
        self.total_constraints = offset

    def flatten(self, joint: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten dict of joint features into shape (N, C_total)."""
        flats = []
        for layer in self.layers:
            flats.append(joint[layer].reshape(joint[layer].shape[0], -1))
        return torch.cat(flats, dim=1)

    def decode(self, flat_idx: int) -> Tuple[str, int, int]:
        """
        Map a flattened constraint index back to (layer, component_idx, class_idx).
        """
        if flat_idx < 0 or flat_idx >= self.total_constraints:
            raise ValueError(f"Flat index {flat_idx} out of range [0, {self.total_constraints}).")
        for layer in self.layers:
            start = self.offsets[layer]
            width = self.components_per_layer[layer] * self.num_classes
            end = start + width
            if start <= flat_idx < end:
                local = flat_idx - start
                comp_idx = local // self.num_classes
                class_idx = local % self.num_classes
                return layer, int(comp_idx), int(class_idx)
        raise RuntimeError(f"Failed to decode flat index {flat_idx}.")


class JointConstraintBuilder:
    """
    Build joint latent/class constraint tensors joint_{i,j,c}(t) = P[M_i=j|a_i(t)] * class_weight_c(t),
    ensuring the feature-mass property required by the IIS derivation.
    """

    def __init__(
        self,
        layers: List[str],
        components_per_layer: Dict[str, int],
        num_classes: int,
        device: torch.device,
        backends: Optional[Dict[str, LatentBackend]] = None,
    ):
        self.layers = list(layers)
        self.components_per_layer = dict(components_per_layer)
        self.num_classes = int(num_classes)
        self.device = device
        self.backends: Dict[str, LatentBackend] = backends if backends is not None else {}
        self.indexer = ConstraintIndexer(self.layers, self.components_per_layer, self.num_classes)
        self.expected_feature_mass = float(len(self.layers))  # N_c in Eq. (18) when N_d=0

    def set_backends(self, backends: Dict[str, LatentBackend]) -> None:
        self.backends = backends

    def update_components(self, components_per_layer: Dict[str, int]) -> None:
        self.components_per_layer = dict(components_per_layer)
        self.indexer = ConstraintIndexer(self.layers, self.components_per_layer, self.num_classes)

    def _prepare_class_probs(self, class_probs: torch.Tensor) -> torch.Tensor:
        probs = torch.as_tensor(class_probs, dtype=torch.float64, device=self.device)
        if probs.ndim != 2 or probs.shape[1] != self.num_classes:
            raise ValueError(
                f"class_probs must have shape (N, K) with K={self.num_classes}, got {tuple(probs.shape)}."
            )
        if torch.any(probs < -1e-12):
            raise ValueError("class_probs must be non-negative.")
        row_sums = probs.sum(dim=1, keepdim=True)
        if torch.any(row_sums <= 0):
            raise ValueError("class_probs rows must have positive mass.")
        return probs / row_sums

    def _validate_joint(self, joint: Dict[str, torch.Tensor], rel_mass_tol: float) -> Tuple[torch.Tensor, torch.Tensor]:
        expected_total = self.expected_feature_mass
        per_layer_masses: List[torch.Tensor] = []
        for layer in self.layers:
            if layer not in joint:
                raise ValueError(f"joint missing layer '{layer}'.")
            tensor = torch.as_tensor(joint[layer], dtype=torch.float64, device=self.device)
            if tensor.dim() != 3 or tensor.shape[2] != self.num_classes:
                raise ValueError(
                    f"joint[{layer}] must have shape (N, J_l, K) with K={self.num_classes}. Got {tuple(tensor.shape)}."
                )
            if tensor.shape[1] != self.components_per_layer[layer]:
                raise ValueError(
                    f"joint[{layer}] second dim must equal components_per_layer[{layer}]={self.components_per_layer[layer]}."
                )
            if torch.any(tensor < -1e-8):
                raise ValueError(f"joint[{layer}] contains negative entries.")
            layer_mass = tensor.sum(dim=(1, 2))
            per_layer_masses.append(layer_mass)
        masses = torch.stack(per_layer_masses, dim=1)  # (N, L)
        eps = 1e-8
        layer_dev = float((masses - 1.0).abs().max().item())
        rel_layer_dev = layer_dev / (1.0 + eps)
        if rel_layer_dev > rel_mass_tol:
            raise ValueError(
                f"Per-layer feature mass deviates from 1 (max_abs_dev={layer_dev:.4e}, rel_dev={rel_layer_dev:.3e}, tol={rel_mass_tol:.3e})."
            )
        total_mass = masses.sum(dim=1)
        max_total_dev = float((total_mass - expected_total).abs().max().item())
        rel_total_dev = max_total_dev / (abs(expected_total) + eps)
        if rel_total_dev > rel_mass_tol:
            raise ValueError(
                f"Total feature mass deviates from expected ~{expected_total:.6f} "
                f"(max_abs_dev={max_total_dev:.4e}, rel_dev={rel_total_dev:.3e}, tol={rel_mass_tol:.3e})."
            )
        flat = self.indexer.flatten({layer: torch.as_tensor(joint[layer], dtype=torch.float64, device=self.device) for layer in self.layers})
        return flat, total_mass

    def build_joint(self, layer_features: Dict[str, torch.Tensor], class_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Build joint constraints using the configured latent backends.
        Implements joint_{i,j,c}(t) = P[M_i=j | a_i(t)] * class_weight_c(t).
        """
        probs = self._prepare_class_probs(class_probs)
        joint: Dict[str, torch.Tensor] = {}
        for layer in self.layers:
            backend = self.backends.get(layer)
            if backend is None:
                raise RuntimeError(f"Latent backend for layer '{layer}' has not been fit.")
            feats = torch.as_tensor(layer_features[layer], dtype=torch.float64, device=self.device)
            gamma = backend.predict_proba(feats.detach().cpu().numpy())
            gamma_t = torch.from_numpy(np.asarray(gamma, dtype=np.float64)).to(self.device)
            joint[layer] = gamma_t.unsqueeze(2) * probs.unsqueeze(1)
        return joint

    def build_joint_from_responsibilities(
        self, responsibilities: Dict[str, torch.Tensor], class_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Build joint constraints directly from provided responsibilities.
        Useful for deterministic tests without invoking a backend.
        """
        probs = self._prepare_class_probs(class_probs)
        joint: Dict[str, torch.Tensor] = {}
        for layer in self.layers:
            gamma = torch.as_tensor(responsibilities[layer], dtype=torch.float64, device=self.device)
            joint[layer] = gamma.unsqueeze(2) * probs.unsqueeze(1)
        return joint

    def validate_and_flatten(self, joint: Dict[str, torch.Tensor], rel_mass_tol: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Validate mass constraints and return (flat_joint, feature_mass_per_sample).
        """
        return self._validate_joint(joint, rel_mass_tol=rel_mass_tol)


@dataclass
class IISDelta:
    delta: torch.Tensor
    pm: torch.Tensor
    updated_weights: torch.Tensor


class IISUpdater:
    """
    Implements the IIS updates for fractional features following Eq. (14)–(18).
    """

    def __init__(self, num_latent: int, num_discrete: int = 0, eps: float = 1e-12):
        if num_latent <= 0:
            raise ValueError("num_latent must be positive.")
        if num_discrete < 0:
            raise ValueError("num_discrete cannot be negative.")
        self.num_latent = int(num_latent)
        self.num_discrete = int(num_discrete)
        self.eps = float(eps)
        self._denominator = float(self.num_latent + self.num_discrete)

    @property
    def mass_constant(self) -> float:
        """Return N_d + N_c used in Eq. (18)'s denominator."""
        return self._denominator

    def compute_pg(self, flat_joint: torch.Tensor) -> torch.Tensor:
        """Eq. (14): average ground-truth/pseudo-label constraints over target samples."""
        if flat_joint.ndim != 2:
            raise ValueError(f"flat_joint must be 2D (T, C), got shape {tuple(flat_joint.shape)}.")
        return flat_joint.mean(dim=0)

    def compute_pm(self, weights: torch.Tensor, flat_joint: torch.Tensor) -> torch.Tensor:
        """Eq. (15): model estimate using current weights over source samples."""
        if weights.ndim != 1:
            raise ValueError("weights must be 1D.")
        if flat_joint.shape[0] != weights.shape[0]:
            raise ValueError("weights length must match number of source samples.")
        return torch.sum(weights.view(-1, 1) * flat_joint, dim=0)

    def delta_lambda(self, pg: torch.Tensor, pm: torch.Tensor) -> torch.Tensor:
        """Eq. (18): Δλ = (1/(N_d + N_c)) * log(P_g / P_m) with epsilon safety."""
        pg_safe = torch.clamp(pg, min=self.eps)
        pm_safe = torch.clamp(pm, min=self.eps)
        return torch.log(pg_safe / pm_safe) / self._denominator

    def update_weights(self, weights: torch.Tensor, flat_joint: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Weight update q_new ∝ q * exp(sum_k f_k(t) * Δλ_k) followed by renormalization."""
        exponent = torch.sum(flat_joint * delta.unsqueeze(0), dim=1)
        new_weights = weights * torch.exp(exponent)
        new_weights = torch.clamp(new_weights, min=0.0)
        total = new_weights.sum()
        if total.item() <= self.eps:
            raise RuntimeError("Weights collapsed to zero mass during IIS update.")
        return new_weights / total

    def step(self, weights: torch.Tensor, flat_joint: torch.Tensor, pg: torch.Tensor) -> IISDelta:
        """
        Perform a single IIS delta computation and weight update.
        Returns the delta, current P_m, and the updated weights.
        """
        pm = self.compute_pm(weights, flat_joint)
        delta = self.delta_lambda(pg, pm)
        updated_weights = self.update_weights(weights, flat_joint, delta)
        return IISDelta(delta=delta, pm=pm, updated_weights=updated_weights)
