from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.me_iis_adapter import IISIterationStats, MaxEntAdapter

from .base import Algorithm, unpack_wilds_batch


NormalizeWeightsMode = Literal["global", "batch"]
UpdateFrequency = Literal["epoch"]


@dataclass(frozen=True)
class MEIISConfig:
    K: int = 8
    covariance_type: str = "diag"
    regularization: float = 1e-6
    max_iis_iters: int = 15
    iis_step_size: float = 1.0
    iis_damping: float = 0.0
    constraint_weight: float = 1.0
    confidence_threshold: float = 0.90
    min_confident_samples: int = 512
    max_entropy: Optional[float] = None
    ema_constraints: float = 0.0
    update_frequency: UpdateFrequency = "epoch"
    clip_weights_max: Optional[float] = None
    normalize_weights: NormalizeWeightsMode = "batch"
    topk_debug: int = 10


class MEIIS(Algorithm):
    """
    ME-IIS (Maximum Entropy Importance-weighted IIS) as a reweighting algorithm.

    This implementation uses the existing Pal-Miller fractional IIS adapter to
    solve for a max-entropy distribution over source samples matching target
    (pseudo-labeled) joint constraints.
    """

    def __init__(
        self,
        *,
        featurizer: nn.Module,
        feature_dim: int,
        num_classes: int,
        seed: int,
        config: MEIISConfig,
    ):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = nn.Linear(int(feature_dim), int(num_classes))
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.cfg = config

        self.layer_name = "feat"
        self.adapter = MaxEntAdapter(
            num_classes=int(num_classes),
            layers=[self.layer_name],
            components_per_layer={self.layer_name: int(config.K)},
            device=torch.device("cpu"),
            seed=int(seed),
            gmm_selection_mode="fixed",
            gmm_covariance_type=str(config.covariance_type),
            gmm_reg_covar=float(config.regularization),
        )

        self.register_buffer("source_weights", torch.empty(0, dtype=torch.float64), persistent=True)
        self.register_buffer("target_moments_ema", torch.empty(0, dtype=torch.float64), persistent=True)

        self.last_iis_history: list[IISIterationStats] = []
        self.ess_history: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.featurizer(x)
        return self.classifier(feats)

    def parameter_groups(self) -> Iterable[Dict[str, Any]]:
        return [
            {"name": "featurizer", "params": self.featurizer.parameters()},
            {"name": "classifier", "params": self.classifier.parameters()},
        ]

    def _ensure_source_weights(self, n_source: int) -> None:
        if self.source_weights.numel() == int(n_source):
            return
        if int(n_source) <= 0:
            raise ValueError(f"n_source must be positive, got {n_source}.")
        w = torch.full((int(n_source),), 1.0 / float(n_source), dtype=torch.float64)
        self.source_weights = w

    def update(self, labeled_batch: Any, unlabeled_batch: Any | None = None) -> Dict[str, Any]:
        batch = unpack_wilds_batch(labeled_batch)
        if batch.y is None:
            raise ValueError("MEIIS.update expected labels but batch.y is None.")

        logits = self.forward(batch.x)
        per_sample = F.cross_entropy(logits, batch.y, reduction="none")

        loss: torch.Tensor
        if batch.idx is None or self.source_weights.numel() == 0:
            loss = per_sample.mean()
        else:
            idx_cpu = batch.idx.detach().cpu().to(torch.long)
            w = self.source_weights[idx_cpu].to(device=per_sample.device, dtype=per_sample.dtype)
            denom = w.sum().clamp_min(1e-12)
            if self.cfg.normalize_weights == "batch":
                loss = (w * per_sample).sum() / denom
            else:
                # global weights already sum to 1; normalize for stable batch scaling.
                loss = (w * per_sample).sum() / denom

        acc = (logits.argmax(dim=1) == batch.y).float().mean()
        return {
            "loss": loss,
            "loss_cls": loss.detach(),
            "acc": acc.detach(),
            "batch_size": int(batch.x.shape[0]),
        }

    @torch.no_grad()
    def update_importance_weights(
        self,
        *,
        source_loader: Any,
        target_loader: Any,
        device: torch.device,
        max_target_fit_samples: int = 20000,
        max_target_joint_samples: int = 4096,
    ) -> Dict[str, Any]:
        """
        Compute ME-IIS source weights from the current model using:
          - source joint constraints: responsibilities * onehot(source_label)
          - target constraints: responsibilities * softmax(model(x_t)), filtered by confidence/entropy
        """
        self.eval()

        # --- Pass 1: scan target for confidence stats + sample features for clustering fit
        tau = float(self.cfg.confidence_threshold)
        min_keep = int(self.cfg.min_confident_samples)
        max_ent = self.cfg.max_entropy

        fit_feats: list[torch.Tensor] = []
        fit_probs: list[torch.Tensor] = []
        conf_all: list[float] = []
        kept = 0
        total = 0

        for batch_raw in target_loader:
            batch = unpack_wilds_batch(batch_raw)
            x = batch.x.to(device, non_blocking=True)
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            conf = probs.max(dim=1).values
            ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=1)

            conf_all.extend([float(v) for v in conf.detach().cpu().tolist()])
            total += int(x.shape[0])

            mask = conf >= tau
            if max_ent is not None:
                mask = mask & (ent <= float(max_ent))

            if int(mask.sum().item()) == 0:
                continue

            x_keep = x[mask]
            probs_keep = probs[mask]
            feats_keep = self.featurizer(x_keep)

            kept += int(x_keep.shape[0])
            if sum(t.numel() for t in fit_feats) < max_target_fit_samples * int(feats_keep.shape[1]):
                fit_feats.append(feats_keep.detach().cpu())
                fit_probs.append(probs_keep.detach().cpu())
            if kept >= max_target_fit_samples:
                break

        if total == 0:
            raise RuntimeError("Target loader yielded zero samples; cannot update ME-IIS weights.")

        if kept < min_keep and conf_all:
            # Fallback: lower tau to hit min_keep based on empirical confidence distribution.
            conf_sorted = np.sort(np.asarray(conf_all, dtype=np.float64))[::-1]
            if len(conf_sorted) >= min_keep:
                tau = min(tau, float(conf_sorted[min_keep - 1]))
            else:
                tau = min(tau, float(conf_sorted[-1]))

        # Need at least some features to fit.
        if not fit_feats:
            return {"status": "skipped_no_confident_target", "tau_used": tau, "kept": kept, "total": total}

        feats_fit = torch.cat(fit_feats, dim=0)
        probs_fit = torch.cat(fit_probs, dim=0)

        # Fit clustering structure on (filtered) target activations.
        self.adapter.device = torch.device("cpu")
        self.adapter.fit_target_structure({self.layer_name: feats_fit}, target_class_probs=probs_fit)

        # --- Pass 2: compute target moments (optionally also keep a small target_joint for validation)
        target_moments_sum: Optional[torch.Tensor] = None
        target_count = 0
        target_joint_keep: list[torch.Tensor] = []

        for batch_raw in target_loader:
            batch = unpack_wilds_batch(batch_raw)
            x = batch.x.to(device, non_blocking=True)
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            conf = probs.max(dim=1).values
            ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=1)

            mask = conf >= tau
            if max_ent is not None:
                mask = mask & (ent <= float(max_ent))
            if int(mask.sum().item()) == 0:
                continue

            x_keep = x[mask]
            probs_keep = probs[mask]
            feats_keep = self.featurizer(x_keep).detach().cpu()
            probs_keep_cpu = probs_keep.detach().cpu()
            joint = self.adapter.get_joint_features({self.layer_name: feats_keep}, probs_keep_cpu)[self.layer_name]
            flat = joint.reshape(joint.shape[0], -1).to(dtype=torch.float64)

            if target_moments_sum is None:
                target_moments_sum = flat.sum(dim=0)
            else:
                target_moments_sum = target_moments_sum + flat.sum(dim=0)
            target_count += int(flat.shape[0])

            if sum(t.shape[0] for t in target_joint_keep) < max_target_joint_samples:
                target_joint_keep.append(joint.to(dtype=torch.float32))

        if target_moments_sum is None or target_count <= 0:
            return {"status": "skipped_no_confident_target", "tau_used": tau, "kept": kept, "total": total}

        target_moments = target_moments_sum / float(target_count)
        if float(self.cfg.ema_constraints) > 0:
            if self.target_moments_ema.numel() == 0:
                self.target_moments_ema = target_moments.clone()
            else:
                alpha = float(self.cfg.ema_constraints)
                self.target_moments_ema = alpha * self.target_moments_ema + (1.0 - alpha) * target_moments
            target_moments_used = self.target_moments_ema.clone()
        else:
            target_moments_used = target_moments

        # --- Pass 3: build full source_joint (N_s, J, K)
        n_source = len(source_loader.dataset)
        self._ensure_source_weights(n_source)
        J = int(self.adapter.components_per_layer[self.layer_name])
        Kc = int(self.num_classes)

        source_joint = torch.zeros((int(n_source), J, Kc), dtype=torch.float32)
        for batch_raw in source_loader:
            batch = unpack_wilds_batch(batch_raw)
            if batch.idx is None or batch.y is None:
                raise ValueError("ME-IIS weight update requires source batches with (y, idx).")

            x = batch.x.to(device, non_blocking=True)
            feats = self.featurizer(x).detach().cpu()
            y = batch.y.detach().cpu().to(torch.long)
            idx = batch.idx.detach().cpu().to(torch.long)
            onehot = F.one_hot(y, num_classes=Kc).to(dtype=torch.float32)
            joint = self.adapter.get_joint_features({self.layer_name: feats}, onehot)[self.layer_name].to(dtype=torch.float32)
            source_joint[idx] = joint

        if not target_joint_keep:
            # Minimal joint batch for mass validation inside solve_iis.
            target_joint_keep = [
                self.adapter.get_joint_features({self.layer_name: feats_fit[: min(256, feats_fit.shape[0])]}, probs_fit[: min(256, probs_fit.shape[0])])[self.layer_name].to(dtype=torch.float32)
            ]
        target_joint_small = torch.cat(target_joint_keep, dim=0)

        # Solve IIS for max-entropy source weights matching target moments.
        weights, _, history = self.adapter.solve_iis_from_joint(
            source_joint={self.layer_name: source_joint},
            target_joint={self.layer_name: target_joint_small},
            max_iter=int(self.cfg.max_iis_iters),
            iis_tol=0.0,
            iis_step_size=float(self.cfg.iis_step_size) * float(self.cfg.constraint_weight),
            iis_damping=float(self.cfg.iis_damping),
            target_moments_override=target_moments_used,
            verbose=False,
        )

        if self.cfg.clip_weights_max is not None:
            weights = torch.clamp(weights, min=0.0, max=float(self.cfg.clip_weights_max))
            weights = weights / weights.sum().clamp_min(1e-12)

        self.source_weights = weights.detach().cpu().to(dtype=torch.float64)
        self.last_iis_history = list(history)

        ess = float(1.0 / torch.sum(self.source_weights**2).clamp_min(1e-12).item())
        self.ess_history.append(ess)

        topk = int(self.cfg.topk_debug)
        top_idx = torch.argsort(self.source_weights, descending=True)[:topk].cpu().tolist()
        top_w = self.source_weights[torch.as_tensor(top_idx, dtype=torch.long)].cpu().tolist()

        obj_hist = [float(h.objective) for h in history]
        return {
            "status": "updated",
            "tau_used": float(tau),
            "target_confident": int(target_count),
            "ess": ess,
            "w_min": float(self.source_weights.min().item()),
            "w_max": float(self.source_weights.max().item()),
            "topk_idx": top_idx,
            "topk_w": top_w,
            "iis_objective": obj_hist,
        }

