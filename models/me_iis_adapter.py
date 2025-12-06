from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
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
        seed: Optional[int] = None,
        gmm_selection_mode: str = "fixed",
        gmm_bic_min_components: int = 2,
        gmm_bic_max_components: int = 8,
    ):
        self.num_classes = num_classes
        self.layers = layers
        self.components_per_layer = dict(components_per_layer)
        self.device = device
        self.indexer = ConstraintIndexer(layers, self.components_per_layer, num_classes)
        self.expected_feature_mass = float(len(layers))
        self.seed = seed
        self.gmm_selection_mode = gmm_selection_mode
        self.gmm_bic_min_components = gmm_bic_min_components
        self.gmm_bic_max_components = gmm_bic_max_components
        self.gmms: Dict[str, GaussianMixture] = self._build_layer_gmms()

    def _build_layer_gmms(self) -> Dict[str, GaussianMixture]:
        return {
            layer: GaussianMixture(
                n_components=self.components_per_layer[layer],
                covariance_type="diag",
                reg_covar=1e-6,
                random_state=self.seed,
            )
            for layer in self.layers
        }

    def _refresh_indexer(self) -> None:
        """Rebuild the constraint indexer after component count changes."""
        self.indexer = ConstraintIndexer(self.layers, self.components_per_layer, self.num_classes)

    def _select_components_via_bic(
        self,
        layer_name: str,
        feats_np: np.ndarray,
        min_components: int,
        max_components: int,
    ) -> GaussianMixture:
        """Fit several GMMs and pick the component count with the lowest BIC."""
        if feats_np.ndim != 2:
            raise ValueError(f"Expected 2D features for BIC selection, got shape {feats_np.shape}.")
        n_samples = feats_np.shape[0]
        if n_samples == 0:
            raise ValueError("Cannot run BIC selection with zero target samples.")
        max_subsample = 20000
        if n_samples > max_subsample:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n_samples, size=max_subsample, replace=False)
            feats_np_sub = feats_np[idx]
        else:
            feats_np_sub = feats_np

        min_comp = max(1, int(min_components))
        max_comp = max(min_comp, int(max_components))
        max_comp = min(max_comp, feats_np_sub.shape[0])
        min_comp = min(min_comp, max_comp)

        best_bic = np.inf
        best_m = None
        for m in range(min_comp, max_comp + 1):
            if m > feats_np_sub.shape[0]:
                continue
            gmm = GaussianMixture(
                n_components=m,
                covariance_type="diag",
                reg_covar=1e-6,
                random_state=self.seed,
            )
            gmm.fit(feats_np_sub)
            bic = gmm.bic(feats_np_sub)
            if bic < best_bic:
                best_bic = bic
                best_m = m

        if best_m is None:
            raise RuntimeError(
                f"BIC component selection failed for layer '{layer_name}' with min={min_components}, max={max_components}."
            )

        final_gmm = GaussianMixture(
            n_components=best_m,
            covariance_type="diag",
            reg_covar=1e-6,
            random_state=self.seed,
        )
        final_gmm.fit(feats_np)
        print(f"[GMM] layer={layer_name} selection=bic M*={best_m} BIC={best_bic:.3e}")
        return final_gmm

    def _assert_non_negative(self, tensor: torch.Tensor, name: str, atol: float = 1e-8) -> None:
        min_val = float(tensor.min().item())
        if min_val < -atol:
            raise ValueError(f"{name} contains negative entries (min={min_val:.3e}). Constraints must be non-negative.")

    def _assert_valid_distribution(self, weights: torch.Tensor, name: str = "weights", atol: float = 1e-6) -> None:
        if torch.isnan(weights).any():
            raise RuntimeError(f"{name} contains NaNs.")
        min_val = float(weights.min().detach().cpu().item())
        sum_val = float(weights.sum().detach().cpu().item())
        if min_val < -atol:
            raise RuntimeError(f"{name} has negative entries (min={min_val:.3e}).")
        if abs(sum_val - 1.0) > atol:
            raise RuntimeError(f"{name} must sum to 1 within atol={atol}. Got sum={sum_val:.6f}.")

    def _validate_joint_features(
        self,
        joint: Dict[str, torch.Tensor],
        name: str,
        rel_mass_tol: float,
        expected_mass: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        expected = self.expected_feature_mass if expected_mass is None else float(expected_mass)
        for layer in self.layers:
            if layer not in joint:
                raise ValueError(f"{name} missing layer '{layer}'.")
            tensor = joint[layer]
            if tensor.dim() != 3 or tensor.shape[2] != self.num_classes:
                raise ValueError(
                    f"{name}[{layer}] must have shape (N, J_l, K) with K={self.num_classes}. Got {tuple(tensor.shape)}."
                )
            self._assert_non_negative(tensor, f"{name}[{layer}]")
        flat = self.indexer.flatten(joint).to(self.device)
        feature_mass = flat.sum(dim=1)
        eps = 1e-8
        max_dev = float((feature_mass - expected).abs().max().item())
        rel_dev = max_dev / (abs(expected) + eps)
        if rel_dev > rel_mass_tol:
            raise ValueError(
                f"{name} feature mass deviates from expected ~{expected:.6f} "
                f"(max_abs_dev={max_dev:.4e}, rel_dev={rel_dev:.3e}, tol={rel_mass_tol:.3e})."
            )
        return flat, feature_mass

    def fit_target_structure(self, target_layer_features: Dict[str, torch.Tensor]) -> None:
        """Fit one GMM per layer on target activations."""
        if self.gmm_selection_mode not in {"fixed", "bic"}:
            raise ValueError(f"Unknown gmm_selection_mode '{self.gmm_selection_mode}'.")

        for layer in self.layers:
            feats_np = target_layer_features[layer].detach().cpu().numpy()
            if self.gmm_selection_mode == "fixed":
                gmm = self.gmms.get(layer)
                if gmm is None or gmm.n_components != self.components_per_layer[layer]:
                    gmm = GaussianMixture(
                        n_components=self.components_per_layer[layer],
                        covariance_type="diag",
                        reg_covar=1e-6,
                        random_state=self.seed,
                    )
                gmm.fit(feats_np)
            elif self.gmm_selection_mode == "bic":
                gmm = self._select_components_via_bic(
                    layer_name=layer,
                    feats_np=feats_np,
                    min_components=self.gmm_bic_min_components,
                    max_components=self.gmm_bic_max_components,
                )
            else:
                raise ValueError(f"Unknown gmm_selection_mode '{self.gmm_selection_mode}'.")

            self.gmms[layer] = gmm
            self.components_per_layer[layer] = int(gmm.n_components)

        self._refresh_indexer()

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
        source_layer_feats: Optional[Dict[str, torch.Tensor]],
        source_class_probs: Optional[torch.Tensor],
        target_layer_feats: Optional[Dict[str, torch.Tensor]],
        target_class_probs: Optional[torch.Tensor],
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
            if target_layer_feats is None or target_class_probs is None:
                raise ValueError("target_layer_feats and target_class_probs are required when no precomputed target joint is provided.")
            target_joint = self.get_joint_features(target_layer_feats, target_class_probs)
        target_flat, _ = self._validate_joint_features(
            target_joint, name="target_joint", rel_mass_tol=f_mass_rel_tol
        )
        target_moments = target_flat.to(device).mean(dim=0)  # (C,)

        # Build source joint features (fractional features summed across layers)
        if precomputed_source_joint is not None:
            source_joint = precomputed_source_joint
        else:
            if source_layer_feats is None or source_class_probs is None:
                raise ValueError("source_layer_feats and source_class_probs are required when no precomputed source joint is provided.")
            source_joint = self.get_joint_features(source_layer_feats, source_class_probs)
        source_flat, feature_mass = self._validate_joint_features(
            source_joint, name="source_joint", rel_mass_tol=f_mass_rel_tol
        )
        source_flat = source_flat.to(device)  # (N_s, C)

        N_s = source_flat.size(0)
        weights = torch.ones(N_s, device=device) / float(N_s)
        lambdas = torch.zeros(self.indexer.total_constraints, device=device)
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
        converged = False
        for it in range(max_iter):
            current_moments = torch.sum(weights.view(-1, 1) * source_flat, dim=0)  # (C,)
            ratio = torch.ones_like(current_moments)
            active_mask = (target_moments > eps) | (current_moments > eps)
            ratio[active_mask] = (target_moments[active_mask] + eps) / (current_moments[active_mask] + eps)
            ratio = torch.clamp(ratio, min=1e-6, max=1e6)
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
            self._assert_valid_distribution(weights, name="IIS weights")

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
                converged = True
                break
        if history:
            last = history[-1]
            print(
                f"IIS finished: iters={len(history)}, max_abs_error={last.max_moment_error:.4e}, "
                f"l2_error={last.l2_moment_error:.4e}, entropy={last.weight_entropy:.4f}"
            )
        if iis_tol > 0 and converged and history and history[-1].max_moment_error > iis_tol:
            raise RuntimeError(
                f"IIS termination did not satisfy tolerance iis_tol={iis_tol:.3e} "
                f"(final max_abs_error={history[-1].max_moment_error:.4e})."
            )
        if iis_tol > 0 and not converged:
            print(
                f"[IIS][WARN] Did not meet tolerance iis_tol={iis_tol:.3e} within {max_iter} iterations. "
                f"Final max_abs_error={history[-1].max_moment_error if history else float('nan'):.4e}."
            )
        self._assert_valid_distribution(weights, name="IIS final weights")
        self._assert_non_negative(target_moments, "target moments")
        self._assert_non_negative(source_flat, "source joint features")
        return weights.detach(), history

    def solve_iis_from_joint(
        self,
        source_joint: Dict[str, torch.Tensor],
        target_joint: Dict[str, torch.Tensor],
        max_iter: int = 15,
        iis_tol: float = 0.0,
        f_mass_rel_tol: float = 1e-2,
    ) -> Tuple[torch.Tensor, float, List[IISIterationStats]]:
        """Run IIS directly on precomputed joint constraint features."""
        weights, history = self.solve_iis(
            source_layer_feats=None,
            source_class_probs=None,
            target_layer_feats=None,
            target_class_probs=None,
            max_iter=max_iter,
            iis_tol=iis_tol,
            f_mass_rel_tol=f_mass_rel_tol,
            precomputed_source_joint=source_joint,
            precomputed_target_joint=target_joint,
        )
        final_error = history[-1].max_moment_error if history else float("nan")
        return weights, final_error, history
