from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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
    use_confidence_filtered_constraints: bool = True
    target_conf_thresh: float = 0.90
    target_conf_mode: Literal["maxprob", "entropy"] = "maxprob"
    target_conf_min_count: int = 256
    max_entropy: Optional[float] = None
    ema_constraints: float = 0.0
    update_frequency: UpdateFrequency = "epoch"
    weight_clip_max: float = 10.0
    weight_mix_alpha: float = 0.8
    normalize_weights: NormalizeWeightsMode = "batch"
    topk_debug: int = 10
    debug: bool = False
    debug_strict_monotonicity: bool = False


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
        feats = self.extract_features(x)
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
        run_dir: str | Path | None = None,
        epoch: int | None = None,
    ) -> Dict[str, Any]:
        """
        Compute ME-IIS source weights from the current model using:
          - source joint constraints: responsibilities * onehot(source_label)
          - target constraints: responsibilities * softmax(model(x_t)), filtered by confidence/entropy
        """
        was_training = self.training
        self.eval()
        try:
            if epoch is not None:
                print(f"[MEIIS] update_importance_weights epoch={int(epoch)} model_mode=eval was_training={was_training}")
            use_conf = bool(self.cfg.use_confidence_filtered_constraints)
            conf_mode = str(self.cfg.target_conf_mode)
            if conf_mode not in {"maxprob", "entropy"}:
                raise ValueError(f"Unknown target_conf_mode='{conf_mode}'. Expected 'maxprob' or 'entropy'.")
            tau_initial = float(self.cfg.target_conf_thresh)
            tau_used = float(tau_initial)
            min_keep = int(self.cfg.target_conf_min_count)
            max_ent = self.cfg.max_entropy

            fallback_used = False
            fallback_policy = "none"

            def _compute_conf_ent(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                conf = probs.max(dim=1).values
                ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=1)
                return conf, ent

            def _mask_from_scores(conf: torch.Tensor, ent: torch.Tensor, *, tau: float, force_all: bool) -> torch.Tensor:
                if force_all or not use_conf:
                    mask = torch.ones(conf.shape[0], dtype=torch.bool, device=conf.device)
                else:
                    if conf_mode == "maxprob":
                        mask = conf >= float(tau)
                    else:
                        mask = ent <= float(tau)
                if max_ent is not None:
                    mask = mask & (ent <= float(max_ent))
                return mask

            def _tau_for_min_count(scores: list[float], *, min_count: int) -> float:
                arr = np.asarray(scores, dtype=np.float64)
                if arr.size == 0:
                    return float(tau_used)
                sorted_vals = np.sort(arr)[::-1] if conf_mode == "maxprob" else np.sort(arr)
                idx = int(min(max(0, int(min_count) - 1), int(sorted_vals.size) - 1))
                return float(sorted_vals[idx])

            def _collect_fit(*, tau: float, force_all: bool) -> tuple[torch.Tensor, torch.Tensor, list[float], int, int]:
                feats_list: list[torch.Tensor] = []
                probs_list: list[torch.Tensor] = []
                scores_all: list[float] = []
                kept = 0
                total = 0

                for batch_raw in target_loader:
                    batch = unpack_wilds_batch(batch_raw)
                    x = batch.x.to(device, non_blocking=True)
                    logits = self.forward(x)
                    probs = F.softmax(logits, dim=1)
                    conf, ent = _compute_conf_ent(probs)
                    score = conf if conf_mode == "maxprob" else ent
                    scores_all.extend([float(v) for v in score.detach().cpu().tolist()])
                    total += int(x.shape[0])

                    mask = _mask_from_scores(conf, ent, tau=tau, force_all=force_all)
                    if int(mask.sum().item()) == 0:
                        continue

                    x_keep = x[mask]
                    probs_keep = probs[mask]
                    feats_keep = self.extract_features(x_keep)

                    finite_rows = torch.isfinite(feats_keep).all(dim=1) & torch.isfinite(probs_keep).all(dim=1)
                    if int((~finite_rows).sum().item()) > 0:
                        feats_keep = feats_keep[finite_rows]
                        probs_keep = probs_keep[finite_rows]
                    if feats_keep.numel() == 0:
                        continue

                    feats_list.append(feats_keep.detach().cpu())
                    probs_list.append(probs_keep.detach().cpu())
                    kept += int(feats_keep.shape[0])
                    if kept >= int(max_target_fit_samples):
                        break

                feats_fit = torch.cat(feats_list, dim=0) if feats_list else torch.empty(0)
                probs_fit = torch.cat(probs_list, dim=0) if probs_list else torch.empty(0)
                return feats_fit, probs_fit, scores_all, kept, total

            # Pass 1: fit target clustering.
            feats_fit, probs_fit, scores_all, kept_fit, total_fit = _collect_fit(tau=tau_used, force_all=not use_conf)
            if total_fit == 0:
                raise RuntimeError("Target loader yielded zero samples; cannot update ME-IIS weights.")

            if use_conf and kept_fit < min_keep:
                fallback_used = True
                fallback_policy = "lower_threshold"
                tau_used = _tau_for_min_count(scores_all, min_count=min_keep)
                feats_fit, probs_fit, _scores2, kept_fit, _total2 = _collect_fit(tau=tau_used, force_all=False)

            if kept_fit < min_keep:
                fallback_used = True
                fallback_policy = "use_all"
                feats_fit, probs_fit, _scores3, kept_fit, _total3 = _collect_fit(tau=tau_used, force_all=True)

            if feats_fit.numel() == 0:
                return {
                    "status": "skipped_no_target_for_fit",
                    "tau_initial": float(tau_initial),
                    "tau_used": float(tau_used),
                    "target_total": int(total_fit),
                    "target_selected_fit": int(kept_fit),
                    "fallback_used": bool(fallback_used),
                    "fallback_policy": str(fallback_policy),
                }

            self.adapter.device = torch.device("cpu")
            self.adapter.fit_target_structure({self.layer_name: feats_fit}, target_class_probs=probs_fit)

            # Pass 2: compute target moments (Pg).
            def _collect_target_moments(*, tau: float, force_all: bool) -> tuple[Optional[torch.Tensor], int, int, int, list[torch.Tensor]]:
                moments_sum: Optional[torch.Tensor] = None
                count = 0
                total = 0
                dropped_nonfinite = 0
                joint_keep: list[torch.Tensor] = []

                for batch_raw in target_loader:
                    batch = unpack_wilds_batch(batch_raw)
                    x = batch.x.to(device, non_blocking=True)
                    logits = self.forward(x)
                    probs = F.softmax(logits, dim=1)
                    conf, ent = _compute_conf_ent(probs)
                    total += int(x.shape[0])

                    mask = _mask_from_scores(conf, ent, tau=tau, force_all=force_all)
                    if int(mask.sum().item()) == 0:
                        continue

                    x_keep = x[mask]
                    probs_keep = probs[mask]
                    feats_keep = self.extract_features(x_keep).detach().cpu()
                    probs_keep_cpu = probs_keep.detach().cpu()
                    joint = self.adapter.get_joint_features({self.layer_name: feats_keep}, probs_keep_cpu)[self.layer_name]
                    flat = joint.reshape(joint.shape[0], -1).to(dtype=torch.float64)

                    finite_rows = torch.isfinite(flat).all(dim=1)
                    if int((~finite_rows).sum().item()) > 0:
                        dropped_nonfinite += int((~finite_rows).sum().item())
                        flat = flat[finite_rows]
                        joint = joint[finite_rows]
                    if flat.numel() == 0:
                        continue

                    moments_sum = flat.sum(dim=0) if moments_sum is None else moments_sum + flat.sum(dim=0)
                    count += int(flat.shape[0])

                    if sum(t.shape[0] for t in joint_keep) < max_target_joint_samples:
                        joint_keep.append(joint.to(dtype=torch.float32))

                return moments_sum, count, total, dropped_nonfinite, joint_keep

            target_moments_sum, target_count, target_total, dropped_nonfinite, target_joint_keep = _collect_target_moments(
                tau=tau_used, force_all=not use_conf
            )

            if use_conf and target_count < min_keep:
                fallback_used = True
                fallback_policy = "use_all"
                feats_fit, probs_fit, _scores4, kept_fit, _total4 = _collect_fit(tau=tau_used, force_all=True)
                if feats_fit.numel() == 0:
                    return {
                        "status": "skipped_no_target_for_fit",
                        "tau_initial": float(tau_initial),
                        "tau_used": float(tau_used),
                        "target_total": int(target_total),
                        "target_selected_fit": int(kept_fit),
                        "fallback_used": True,
                        "fallback_policy": str(fallback_policy),
                    }
                self.adapter.fit_target_structure({self.layer_name: feats_fit}, target_class_probs=probs_fit)
                target_moments_sum, target_count, target_total, dropped_nonfinite, target_joint_keep = _collect_target_moments(
                    tau=tau_used, force_all=True
                )

            if target_moments_sum is None or target_count <= 0:
                return {
                    "status": "skipped_no_target_moments",
                    "tau_initial": float(tau_initial),
                    "tau_used": float(tau_used),
                    "target_total": int(target_total),
                    "target_selected": int(target_count),
                    "fallback_used": bool(fallback_used),
                    "fallback_policy": str(fallback_policy),
                    "target_dropped_nonfinite": int(dropped_nonfinite),
                }

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

            # Pass 3: build full source_joint (N_s, J, K).
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
                feats = self.extract_features(x).detach().cpu()
                y = batch.y.detach().cpu().to(torch.long)
                idx = batch.idx.detach().cpu().to(torch.long)
                onehot = F.one_hot(y, num_classes=Kc).to(dtype=torch.float32)
                joint = self.adapter.get_joint_features({self.layer_name: feats}, onehot)[self.layer_name].to(dtype=torch.float32)
                source_joint[idx] = joint

            if not target_joint_keep:
                target_joint_keep = [
                    self.adapter.get_joint_features(
                        {self.layer_name: feats_fit[: min(256, feats_fit.shape[0])]},
                        probs_fit[: min(256, probs_fit.shape[0])],
                    )[self.layer_name].to(dtype=torch.float32)
                ]
            target_joint_small = torch.cat(target_joint_keep, dim=0)

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

            weights = weights.detach().cpu().to(dtype=torch.float64)
            weights = torch.clamp(weights, min=0.0)
            weights = weights / weights.sum().clamp_min(1e-12)

            clip_max = float(self.cfg.weight_clip_max)
            if clip_max > 0:
                weights = torch.clamp(weights, min=0.0, max=clip_max)
                weights = weights / weights.sum().clamp_min(1e-12)
            mix_alpha = float(self.cfg.weight_mix_alpha)
            if not (0.0 <= mix_alpha <= 1.0):
                raise ValueError(f"weight_mix_alpha must be in [0,1], got {mix_alpha}.")
            if mix_alpha < 1.0:
                uniform = torch.full_like(weights, 1.0 / float(weights.numel()))
                weights = mix_alpha * weights + (1.0 - mix_alpha) * uniform
                weights = weights / weights.sum().clamp_min(1e-12)

            self.source_weights = weights.to(dtype=torch.float64)
            self.last_iis_history = list(history)

            ess = float(1.0 / torch.sum(self.source_weights**2).clamp_min(1e-12).item())
            self.ess_history.append(ess)
            w_entropy = float(-(self.source_weights * torch.log(self.source_weights + 1e-12)).sum().item())

            topk = int(self.cfg.topk_debug)
            top_idx = torch.argsort(self.source_weights, descending=True)[:topk].cpu().tolist()
            top_w = self.source_weights[torch.as_tensor(top_idx, dtype=torch.long)].cpu().tolist()

            obj_hist = [float(h.objective) for h in history]
            obj_decreased = any(curr + 1e-10 < prev for prev, curr in zip(obj_hist, obj_hist[1:]))
            last = history[-1] if history else None

            ratio = float(target_count) / float(max(1, target_total))
            debug_dump_path: Optional[str] = None
            if obj_decreased and bool(self.cfg.debug) and run_dir is not None:
                try:
                    run_dir_p = Path(run_dir)
                    run_dir_p.mkdir(parents=True, exist_ok=True)
                    stamp = f"epoch{int(epoch)}" if epoch is not None else "epochNA"
                    dump_path = run_dir_p / f"meiis_debug_{stamp}_objective_decreased.json"
                    dump_payload = {
                        "status": "objective_decreased",
                        "epoch": None if epoch is None else int(epoch),
                        "tau_initial": float(tau_initial),
                        "tau_used": float(tau_used),
                        "target_total": int(target_total),
                        "target_selected": int(target_count),
                        "target_selected_ratio": ratio,
                        "fallback_used": bool(fallback_used),
                        "fallback_policy": str(fallback_policy),
                        "iis_objective": obj_hist,
                        "iis_final_max_moment_error": None if last is None else float(last.max_moment_error),
                        "iis_final_mean_moment_error": None if last is None else float(last.mean_moment_error),
                        "iis_final_l2_moment_error": None if last is None else float(last.l2_moment_error),
                        "iis_num_unachievable": None if last is None else int(last.num_unachievable_constraints),
                        "ess": ess,
                        "weight_entropy": w_entropy,
                        "w_min": float(self.source_weights.min().item()),
                        "w_max": float(self.source_weights.max().item()),
                        "topk_idx": top_idx,
                        "topk_w": top_w,
                    }
                    dump_path.write_text(json.dumps(dump_payload, indent=2, sort_keys=True), encoding="utf-8")
                    debug_dump_path = str(dump_path)
                    print(f"[MEIIS][WARN] IIS objective decreased; wrote debug dump to {debug_dump_path}")
                except Exception as exc:
                    print(f"[MEIIS][WARN] IIS objective decreased but failed to write debug dump: {exc}")
            if obj_decreased and bool(self.cfg.debug_strict_monotonicity):
                raise RuntimeError("IIS objective decreased (strict mode enabled).")
            print(
                "[MEIIS] "
                f"tau_used={float(tau_used):.4f} target_selected={int(target_count)}/{int(target_total)} "
                f"(ratio={ratio:.3f}) fallback={fallback_policy if fallback_used else 'none'} "
                f"ESS={ess:.1f} H(w)={w_entropy:.4f} w_min={float(self.source_weights.min().item()):.3e} "
                f"w_max={float(self.source_weights.max().item()):.3e} "
                f"iis_max_err={float(last.max_moment_error) if last is not None else float('nan'):.3e}"
            )

            return {
                "status": "updated",
                "tau_initial": float(tau_initial),
                "tau_used": float(tau_used),
                "target_total": int(target_total),
                "target_selected": int(target_count),
                "target_selected_ratio": ratio,
                "target_dropped_nonfinite": int(dropped_nonfinite),
                "fallback_used": bool(fallback_used),
                "fallback_policy": str(fallback_policy),
                "weight_clip_max": float(self.cfg.weight_clip_max),
                "weight_mix_alpha": float(self.cfg.weight_mix_alpha),
                "ess": ess,
                "weight_entropy": w_entropy,
                "w_min": float(self.source_weights.min().item()),
                "w_max": float(self.source_weights.max().item()),
                "topk_idx": top_idx,
                "topk_w": top_w,
                "iis_objective": obj_hist,
                "iis_objective_decreased": bool(obj_decreased),
                "debug_dump_path": debug_dump_path,
                "iis_final_max_moment_error": None if last is None else float(last.max_moment_error),
                "iis_final_mean_moment_error": None if last is None else float(last.mean_moment_error),
                "iis_final_l2_moment_error": None if last is None else float(last.l2_moment_error),
                "iis_num_unachievable": None if last is None else int(last.num_unachievable_constraints),
            }
        finally:
            self.train(was_training)
