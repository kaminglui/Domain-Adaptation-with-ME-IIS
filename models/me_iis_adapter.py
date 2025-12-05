from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from sklearn.mixture import GaussianMixture


@dataclass
class IISIterationStats:
    delta_norm: float
    kl_moments: float
    max_moment_error: float
    mean_moment_error: float
    l2_moment_error: float
    weight_min: float
    weight_max: float
    weight_entropy: float
    feature_mass_min: float
    feature_mass_max: float
    feature_mass_mean: float
    feature_mass_std: float


class ConstraintIndexer:
    """Utility to flatten (layer, component, class) constraint tensors."""

    def __init__(self, layers: List[str], components_per_layer: Dict[str, int], num_classes: int):
        self.layers = list(layers)
        self.components_per_layer = components_per_layer
        self.num_classes = num_classes
        self.offsets: Dict[str, int] = {}
        offset = 0
        for layer in self.layers:
            self.offsets[layer] = offset
            offset += self.components_per_layer[layer] * num_classes
        self.total_constraints = offset

    def flatten(self, joint: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten dict of joint features into shape (N, C_total)."""
        flats = []
        for layer in self.layers:
            flats.append(joint[layer].reshape(joint[layer].shape[0], -1))
        return torch.cat(flats, dim=1)


class MaxEntAdapter:
    """Maximum Entropy IIS adapter with multi-layer fractional features (Pal & Miller style)."""

    def __init__(
        self,
        num_classes: int,
        layers: List[str],
        components_per_layer: Dict[str, int],
        device: torch.device = torch.device("cpu"),
    ):
        self.num_classes = num_classes
        self.layers = layers
        self.components_per_layer = components_per_layer
        self.device = device
        self.indexer = ConstraintIndexer(layers, components_per_layer, num_classes)
        self.gmms: Dict[str, GaussianMixture] = {
            layer: GaussianMixture(
                n_components=components_per_layer[layer], covariance_type="diag", reg_covar=1e-6
            )
            for layer in layers
        }

    def fit_target_structure(self, target_layer_features: Dict[str, torch.Tensor]) -> None:
        """Fit one GMM per layer on target activations."""
        for layer, gmm in self.gmms.items():
            feats_np = target_layer_features[layer].detach().cpu().numpy()
            gmm.fit(feats_np)

    def _predict_gamma(self, layer: str, features: torch.Tensor) -> torch.Tensor:
        feats_np = features.detach().cpu().numpy()
        gamma = self.gmms[layer].predict_proba(feats_np)
        return torch.from_numpy(gamma).to(self.device)

    def get_joint_features(
        self, layer_features: Dict[str, torch.Tensor], class_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Return joint features dict: layer -> (N, J_l, K)."""
        joint: Dict[str, torch.Tensor] = {}
        for layer in self.layers:
            gamma = self._predict_gamma(layer, layer_features[layer])  # (N, J_l)
            joint[layer] = gamma.unsqueeze(2) * class_probs.unsqueeze(1)
        return joint

    def _kl_div(self, p: torch.Tensor, q: torch.Tensor) -> float:
        p_safe = p + 1e-8
        q_safe = q + 1e-8
        kl = torch.sum(p_safe * torch.log(p_safe / q_safe))
        return float(kl.detach().cpu().item())

    def solve_iis(
        self,
        source_layer_feats: Dict[str, torch.Tensor],
        source_class_probs: torch.Tensor,
        target_layer_feats: Dict[str, torch.Tensor],
        target_class_probs: torch.Tensor,
        max_iter: int = 15,
        iis_tol: float = 0.0,
        f_mass_rel_tol: float = 1e-2,
        precomputed_source_joint: Optional[Dict[str, torch.Tensor]] = None,
        precomputed_target_joint: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[IISIterationStats]]:
        """
        Solve for Q(x) over source samples using Pal & Miller fractional IIS.

        Args:
            source_layer_feats: dict layer -> (N_s, D_l) source activations.
            source_class_probs: (N_s, K) P(Ĉ|x_s) from source model.
            target_layer_feats: dict layer -> (N_t, D_l) target activations.
            target_class_probs: (N_t, K) P(Ĉ|x_t) from source model.
            max_iter: number of IIS iterations.
            iis_tol: convergence tolerance on max_abs_error for early stopping.
            f_mass_rel_tol: relative std threshold to warn on feature-mass non-constancy.
            precomputed_source_joint: optional joint feature tensor dict for synthetic tests.
            precomputed_target_joint: optional joint feature tensor dict for synthetic tests.
        """
        device = self.device
        # Compute target constraint moments
        if precomputed_target_joint is not None:
            target_joint = precomputed_target_joint
        else:
            target_joint = self.get_joint_features(target_layer_feats, target_class_probs)
        target_flat = self.indexer.flatten(target_joint)  # (N_t, C)
        target_moments = target_flat.to(device).mean(dim=0)  # (C,)

        # Build source joint features (fractional features summed across layers)
        if precomputed_source_joint is not None:
            source_joint = precomputed_source_joint
        else:
            source_joint = self.get_joint_features(source_layer_feats, source_class_probs)
        source_flat = self.indexer.flatten(source_joint).to(device)  # (N_s, C)

        N_s = source_flat.size(0)
        weights = torch.ones(N_s, device=device) / float(N_s)
        lambdas = torch.zeros(self.indexer.total_constraints, device=device)
        feature_mass = source_flat.sum(dim=1)  # should be constant (#layers) for our construction
        f_mass_mean = float(feature_mass.mean().item())
        f_mass_std = float(feature_mass.std().item())
        mass_constant = f_mass_mean
        eps = 1e-8
        if abs(f_mass_mean) < eps:
            raise RuntimeError("Feature mass mean is zero; cannot apply fractional IIS update.")
        rel_mass_std = f_mass_std / (f_mass_mean + eps)
        if rel_mass_std > f_mass_rel_tol:
            print(
                f"[IIS][WARNING] Feature mass not approximately constant: mean={f_mass_mean:.6f}, "
                f"std={f_mass_std:.6f}, rel_std={rel_mass_std:.3e}. Pal & Miller fractional IIS assumption "
                "may be violated. Inspect feature construction."
            )
        else:
            print(
                f"[IIS] Feature mass check ok: mean={f_mass_mean:.6f}, std={f_mass_std:.6f}, "
                f"rel_std={rel_mass_std:.3e}"
            )

        history: List[IISIterationStats] = []
        for it in range(max_iter):
            current_moments = torch.sum(weights.view(-1, 1) * source_flat, dim=0)  # (C,)
            ratio = torch.ones_like(current_moments)
            active_mask = (target_moments > eps) | (current_moments > eps)
            ratio[active_mask] = (target_moments[active_mask] + eps) / (current_moments[active_mask] + eps)
            delta_update = torch.zeros_like(current_moments)
            delta_update[active_mask] = torch.log(ratio[active_mask]) / (mass_constant + eps)
            lambdas = lambdas + delta_update

            # Update weights using new lambdas
            exponent = torch.sum(source_flat * delta_update.unsqueeze(0), dim=1)
            weights = weights * torch.exp(exponent)
            if torch.any(weights < -1e-9):
                print("[IIS][WARNING] Negative weights encountered; clamping to 0 before renormalization.")
            weights = torch.clamp(weights, min=0.0)
            weight_sum = weights.sum()
            if weight_sum.item() <= eps:
                raise RuntimeError("Weights collapsed to zero mass during IIS update.")
            weights = weights / weight_sum
            if torch.any(weights < 0) or abs(weights.sum().item() - 1.0) > 1e-6:
                raise RuntimeError("Weights after IIS update are not a valid probability distribution.")

            updated_moments = torch.sum(weights.view(-1, 1) * source_flat, dim=0)
            moment_error = updated_moments - target_moments
            max_abs_error = float(moment_error.abs().max().detach().cpu().item())
            l2_error = float(moment_error.pow(2).sum().sqrt().detach().cpu().item())
            delta_norm = float(delta_update.norm().detach().cpu().item())
            kl = self._kl_div(target_moments, updated_moments)
            w_cpu = weights.detach().cpu()
            entropy = float(-(weights * torch.log(weights + 1e-12)).detach().cpu().sum().item())
            hist = IISIterationStats(
                delta_norm=delta_norm,
                kl_moments=kl,
                max_moment_error=max_abs_error,
                mean_moment_error=float(moment_error.abs().mean().detach().cpu().item()),
                l2_moment_error=l2_error,
                weight_min=float(w_cpu.min().item()),
                weight_max=float(w_cpu.max().item()),
                weight_entropy=entropy,
                feature_mass_min=float(feature_mass.min().detach().cpu().item()),
                feature_mass_max=float(feature_mass.max().detach().cpu().item()),
                feature_mass_mean=f_mass_mean,
                feature_mass_std=f_mass_std,
            )
            history.append(hist)
            print(
                f"[IIS] iter {it+1}/{max_iter} | ||delta|| {delta_norm:.4f} | "
                f"KL {kl:.4e} | max mom err {hist.max_moment_error:.4e} | "
                f"L2 err {hist.l2_moment_error:.4e} | H(Q) {hist.weight_entropy:.4f}"
            )
            if iis_tol > 0 and max_abs_error < iis_tol:
                print(f"[IIS] Converged at iter {it+1} with max_abs_error {max_abs_error:.4e} < tol {iis_tol:.4e}")
                break
        if history:
            last = history[-1]
            print(
                f"IIS finished: iters={len(history)}, max_abs_error={last.max_moment_error:.4e}, "
                f"l2_error={last.l2_moment_error:.4e}, entropy={last.weight_entropy:.4f}"
            )
        return weights.detach(), history
