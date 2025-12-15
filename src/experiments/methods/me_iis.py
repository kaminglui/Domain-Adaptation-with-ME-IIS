from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tv_datasets

from datasets.domain_loaders import get_domain_loaders
from eval import evaluate
from models.classifier import build_model
from models.me_iis_adapter import IISIterationStats, MaxEntAdapter
from src.experiments.checkpointing import (
    RunState,
    ensure_run_dirs,
    find_existing_completed_checkpoint,
    find_resume_checkpoint,
    load_checkpoint,
    load_state,
    save_checkpoint,
    save_state,
)
from src.experiments.data import DropLabelsDataset
from src.experiments.run_artifacts import RunArtifacts
from src.experiments.run_config import RunConfig, get_run_dir, save_config
from src.experiments.pseudo_labeling import PseudoLabeledDataset, build_pseudo_labels
from utils.data_utils import build_loader, make_generator, make_worker_init_fn
from utils.experiment_utils import build_components_map, parse_feature_layers
from utils.feature_utils import extract_features
from utils.seed_utils import get_device, set_seed


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        return image, label, idx


def _num_classes_from_dataset(dataset: Dataset) -> int:
    base = dataset
    while isinstance(base, Subset):
        base = base.dataset  # type: ignore[assignment]
    if hasattr(base, "classes"):
        return len(base.classes)  # type: ignore[attr-defined]
    if hasattr(base, "dataset") and hasattr(base.dataset, "classes"):
        return len(base.dataset.classes)  # type: ignore[attr-defined]
    raise ValueError("Unable to infer number of classes from dataset.")


def _class_probs_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, mode: str, num_classes: int
) -> torch.Tensor:
    if mode == "softmax":
        return F.softmax(logits, dim=1)
    if mode == "onehot":
        return F.one_hot(labels.to(torch.int64), num_classes=num_classes).float().to(logits.device)
    raise ValueError(f"Unknown source_prob_mode '{mode}'")


def adapt_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    source_loader: DataLoader,
    source_weights_vec: torch.Tensor,
    device: torch.device,
    max_batches: int = 0,
    pseudo_loader: Optional[DataLoader] = None,
    pseudo_loss_weight: float = 1.0,
) -> Tuple[float, float, int, int, int]:
    model.train()
    total_loss, total_acc, total_src = 0.0, 0.0, 0
    batches_seen = 0
    pseudo_used = 0
    pseudo_total = len(pseudo_loader.dataset) if pseudo_loader is not None else 0
    pseudo_iter = iter(pseudo_loader) if pseudo_loader is not None else None

    for images, labels, idxs in source_loader:
        if max_batches > 0 and batches_seen >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        batch_weights = source_weights_vec[idxs].to(device)

        pseudo_batch = None
        if pseudo_iter is not None:
            try:
                pseudo_batch = next(pseudo_iter)
            except StopIteration:
                pseudo_iter = None
                pseudo_batch = None

        optimizer.zero_grad()
        logits, _ = model(images, return_features=False)
        ce = F.cross_entropy(logits, labels, reduction="none")
        loss_src = (batch_weights * ce).sum() / (batch_weights.sum() + 1e-8)

        pseudo_loss = torch.tensor(0.0, device=device)
        pseudo_bs = 0
        if pseudo_batch is not None:
            images_tgt, labels_tgt = pseudo_batch
            images_tgt = images_tgt.to(device)
            labels_tgt = labels_tgt.to(device)
            logits_tgt, _ = model(images_tgt, return_features=False)
            pseudo_loss = F.cross_entropy(logits_tgt, labels_tgt)
            pseudo_bs = labels_tgt.size(0)
            pseudo_used += pseudo_bs

        loss = loss_src + pseudo_loss_weight * pseudo_loss
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        acc = float((torch.argmax(logits, dim=1) == labels).float().mean().item() * 100.0)
        total_loss += float(loss.item()) * (bs + pseudo_bs)
        total_acc += acc * bs
        total_src += bs
        batches_seen += 1

    denom = total_src + pseudo_used if (pseudo_loader is not None and (total_src + pseudo_used) > 0) else total_src
    avg_loss = total_loss / denom if denom > 0 else 0.0
    src_acc = total_acc / total_src if total_src > 0 else 0.0
    return avg_loss, src_acc, batches_seen, pseudo_used, pseudo_total


def _save_iis_npz(path: Path, weights: torch.Tensor, history: List[IISIterationStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        weights=weights.detach().cpu().numpy(),
        delta_norm=[h.delta_norm for h in history],
        kl=[h.kl_moments for h in history],
        moment_max=[h.max_moment_error for h in history],
        moment_mean=[h.mean_moment_error for h in history],
        moment_l2=[h.l2_moment_error for h in history],
        num_unachievable_constraints=[h.num_unachievable_constraints for h in history],
        w_min=[h.weight_min for h in history],
        w_max=[h.weight_max for h in history],
        w_entropy=[h.weight_entropy for h in history],
        feature_mass_min=[h.feature_mass_min for h in history],
        feature_mass_max=[h.feature_mass_max for h in history],
        feature_mass_mean=[h.feature_mass_mean for h in history],
        feature_mass_std=[h.feature_mass_std for h in history],
    )


def run(
    config: RunConfig,
    source_checkpoint: Path,
    force_rerun: bool = False,
    runs_root: Optional[Path] = None,
    save_every_epochs: int = 1,
) -> Dict[str, Any]:
    run_dir = get_run_dir(config, runs_root=runs_root)
    artifacts = RunArtifacts(run_dir=run_dir, run_id=config.run_id, stage="adapt", method=config.method)
    ensure_run_dirs(artifacts)
    save_config(run_dir, config)

    if not force_rerun:
        completed = find_existing_completed_checkpoint(artifacts)
        if completed is not None:
            return {"status": "skipped", "run_dir": str(run_dir), "checkpoint": str(completed)}

    resume_path = None if force_rerun else find_resume_checkpoint(artifacts)

    if config.data_root is None:
        raise ValueError("data_root must be set for adaptation.")
    data_root = Path(config.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not Path(source_checkpoint).exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint}")

    set_seed(config.seed, deterministic=config.deterministic)
    device = get_device(deterministic=config.deterministic)
    data_generator = make_generator(config.seed)
    worker_init = make_worker_init_fn(config.seed)

    source_loader, _, target_eval_loader = get_domain_loaders(
        dataset_name=config.dataset_name,
        source_domain=config.source_domain,
        target_domain=config.target_domain,
        batch_size=config.batch_size,
        root=str(data_root),
        num_workers=config.num_workers,
        debug_classes=False,
        max_samples_per_domain=config.dry_run_max_samples if config.dry_run_max_samples > 0 else None,
        generator=data_generator,
        worker_init_fn=worker_init,
    )
    source_train_raw = source_loader.dataset
    target_eval_raw = target_eval_loader.dataset

    if not hasattr(source_train_raw, "class_to_idx") or not hasattr(target_eval_raw, "class_to_idx"):
        raise ValueError("Domain datasets must expose class_to_idx for ME-IIS.")
    if source_train_raw.class_to_idx != target_eval_raw.class_to_idx:  # type: ignore[attr-defined]
        raise ValueError("Source and target class mappings differ from canonical mapping.")
    eval_transform = getattr(target_eval_raw, "transform", None)
    if eval_transform is None:
        raise ValueError("Target evaluation dataset is missing a transform.")

    source_feat_raw = tv_datasets.ImageFolder(source_train_raw.root, transform=eval_transform)  # type: ignore[attr-defined]
    if source_feat_raw.class_to_idx != source_train_raw.class_to_idx:  # type: ignore[attr-defined]
        raise ValueError("Source eval dataset mapping does not match canonical mapping.")

    target_feat_raw = DropLabelsDataset(target_eval_raw)

    sample_cap: Optional[int] = config.dry_run_max_samples if config.dry_run_max_samples > 0 else None
    if config.dry_run_max_batches > 0:
        batch_cap = config.dry_run_max_batches * config.batch_size
        sample_cap = batch_cap if sample_cap is None else min(sample_cap, batch_cap)
    if sample_cap is not None:
        sample_cap = max(1, int(sample_cap))
        src_idx = list(range(min(sample_cap, len(source_train_raw))))
        tgt_idx = list(range(min(sample_cap, len(target_eval_raw))))
        if not src_idx or not tgt_idx:
            raise ValueError("Dry-run sample cap produced an empty dataset; increase limits.")
        source_train_ds: Dataset = Subset(source_train_raw, src_idx)
        source_feat_ds: Dataset = Subset(source_feat_raw, src_idx)
        target_eval_ds: Dataset = Subset(target_eval_raw, tgt_idx)
        target_feat_ds: Dataset = Subset(target_feat_raw, tgt_idx)
    else:
        source_train_ds = source_train_raw
        source_feat_ds = source_feat_raw
        target_eval_ds = target_eval_raw
        target_feat_ds = target_feat_raw

    num_classes = _num_classes_from_dataset(source_train_ds)
    model = build_model(num_classes=num_classes, pretrained=config.backbone_pretrained).to(device)
    source_ckpt = torch.load(source_checkpoint, map_location=device)
    model.backbone.load_state_dict(source_ckpt["backbone"])
    model.classifier.load_state_dict(source_ckpt["classifier"])

    target_eval_loader = build_loader(
        target_eval_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
        generator=data_generator,
        drop_last=False,
    )
    baseline_acc, _ = evaluate(model, target_eval_loader, device)

    # IIS configuration (method_params)
    feature_layers_param = config.method_params.get("feature_layers")
    if feature_layers_param is None:
        feature_layers = list(config.feature_layers)
    elif isinstance(feature_layers_param, (list, tuple)):
        feature_layers = [str(x).strip() for x in feature_layers_param if str(x).strip()]
        if not feature_layers:
            raise ValueError("feature_layers must contain at least one layer name.")
    else:
        feature_layers = parse_feature_layers(str(feature_layers_param))
    components_map = build_components_map(
        feature_layers=feature_layers,
        default_components=int(config.method_params.get("num_latent_styles", 5)),
        override_str=config.method_params.get("components_per_layer"),
    )
    layer_tag = "-".join(feature_layers)
    components_str = ",".join(str(components_map[layer]) for layer in feature_layers)

    source_prob_mode = str(config.method_params.get("source_prob_mode", "softmax"))
    iis_iters = int(config.method_params.get("iis_iters", 15))
    iis_tol = float(config.method_params.get("iis_tol", 1e-3))
    cluster_backend = str(config.method_params.get("cluster_backend", "gmm"))
    gmm_selection_mode = str(config.method_params.get("gmm_selection_mode", "fixed"))
    gmm_bic_min = int(config.method_params.get("gmm_bic_min_components", 2))
    gmm_bic_max = int(config.method_params.get("gmm_bic_max_components", 8))
    vmf_kappa = float(config.method_params.get("vmf_kappa", 20.0))
    cluster_clean_ratio = float(config.method_params.get("cluster_clean_ratio", 1.0))
    kmeans_n_init = int(config.method_params.get("kmeans_n_init", 10))

    use_pseudo_labels = bool(config.method_params.get("use_pseudo_labels", False))
    pseudo_conf_thresh = float(config.method_params.get("pseudo_conf_thresh", 0.9))
    pseudo_max_ratio = float(config.method_params.get("pseudo_max_ratio", 1.0))
    pseudo_loss_weight = float(config.method_params.get("pseudo_loss_weight", 1.0))

    weights: Optional[torch.Tensor] = None
    history: List[IISIterationStats] = []
    start_epoch = 0
    last_completed_epoch = -1
    adapt_batches_seen = 0
    scheduler_loaded = False
    resume_success = False

    def _make_optimizer_and_scheduler(current_model: nn.Module):
        if not config.finetune_backbone:
            for p in current_model.backbone.parameters():
                p.requires_grad = False
        backbone_lr = config.classifier_lr * config.backbone_lr_scale
        groups = [
            {"params": [p for p in current_model.backbone.parameters() if p.requires_grad], "lr": backbone_lr},
            {"params": list(current_model.classifier.parameters()), "lr": config.classifier_lr},
        ]
        groups = [g for g in groups if len(g["params"]) > 0]
        opt = optim.SGD(groups, momentum=config.momentum, weight_decay=config.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(config.epochs_adapt)))
        return opt, sched

    optimizer, scheduler = _make_optimizer_and_scheduler(model)
    src_acc = 0.0

    if resume_path is not None:
        resume_ckpt = load_checkpoint(resume_path, map_location=device)
        backbone_state = resume_ckpt.get("backbone")
        classifier_state = resume_ckpt.get("classifier")
        if backbone_state is None or classifier_state is None:
            raise RuntimeError("Adaptation checkpoint missing backbone/classifier state.")
        model.backbone.load_state_dict(backbone_state, strict=False)
        model.classifier.load_state_dict(classifier_state, strict=False)
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "scheduler" in resume_ckpt:
            try:
                scheduler.load_state_dict(resume_ckpt["scheduler"])
                scheduler_loaded = True
            except Exception:
                scheduler_loaded = False
        adapt_batches_seen = int(resume_ckpt.get("adapt_batches_seen", adapt_batches_seen))
        last_completed_epoch = int(resume_ckpt.get("epoch", last_completed_epoch))
        start_epoch = max(last_completed_epoch + 1, 0)
        src_acc = float(resume_ckpt.get("source_acc", src_acc))
        weights = resume_ckpt.get("weights")
        history_payload = resume_ckpt.get("history", [])
        if isinstance(history_payload, list):
            try:
                history = [IISIterationStats(**h) for h in history_payload]
            except Exception:
                history = []
        resume_success = True

    if not scheduler_loaded and start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    if weights is None:
        source_feat_loader = build_loader(
            source_feat_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            seed=config.seed,
            generator=data_generator,
            drop_last=False,
        )
        target_feat_loader = build_loader(
            target_feat_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            seed=config.seed,
            generator=data_generator,
            drop_last=False,
        )
        with torch.no_grad():
            source_feats, source_logits, source_labels = extract_features(
                model,
                source_feat_loader,
                device,
                feature_layers,
                max_batches=config.dry_run_max_batches,
            )
            target_feats, target_logits, _target_labels = extract_features(
                model,
                target_feat_loader,
                device,
                feature_layers,
                max_batches=config.dry_run_max_batches,
            )
            source_probs = _class_probs_from_logits(
                logits=source_logits.to(device),
                labels=source_labels.to(device),
                mode=source_prob_mode,
                num_classes=num_classes,
            )
            target_probs = F.softmax(target_logits.to(device), dim=1)

        adapter = MaxEntAdapter(
            num_classes=num_classes,
            layers=list(feature_layers),
            components_per_layer=components_map,
            device=device,
            seed=config.seed,
            gmm_selection_mode=gmm_selection_mode,
            gmm_bic_min_components=gmm_bic_min,
            gmm_bic_max_components=gmm_bic_max,
            cluster_backend=cluster_backend,
            vmf_kappa=vmf_kappa,
            cluster_clean_ratio=cluster_clean_ratio,
            kmeans_n_init=kmeans_n_init,
        )
        adapter.fit_target_structure({k: v.to(device) for k, v in target_feats.items()}, target_class_probs=target_probs)
        weights, history = adapter.solve_iis(
            source_layer_feats={k: v.to(device) for k, v in source_feats.items()},
            source_class_probs=source_probs,
            target_layer_feats={k: v.to(device) for k, v in target_feats.items()},
            target_class_probs=target_probs,
            max_iter=iis_iters,
            iis_tol=iis_tol,
        )
        weights = weights.detach().cpu()

        iis_npz = artifacts.run_dir / "checkpoints" / f"me_iis_history_{config.run_id}.npz"
        _save_iis_npz(iis_npz, weights, history)

    if weights is None:
        raise RuntimeError("Failed to obtain IIS weights.")
    if len(weights) != len(source_train_ds):
        raise ValueError("IIS weight vector length does not match source training set.")

    weighted_loader = build_loader(
        IndexedDataset(source_train_ds),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        generator=data_generator,
        drop_last=False,
    )

    pseudo_loader: Optional[DataLoader] = None
    if use_pseudo_labels:
        target_feat_loader_pl = build_loader(
            target_feat_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            seed=config.seed,
            generator=data_generator,
            drop_last=False,
        )
        pseudo_indices, pseudo_labels = build_pseudo_labels(
            model=model,
            target_loader=target_feat_loader_pl,
            device=device,
            conf_thresh=pseudo_conf_thresh,
            max_ratio=pseudo_max_ratio,
            num_source_samples=len(source_train_ds),
        )
        if len(pseudo_indices) > 0:
            pseudo_ds = PseudoLabeledDataset(target_feat_loader_pl.dataset, pseudo_indices, pseudo_labels)
            pseudo_loader = build_loader(
                pseudo_ds,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                seed=config.seed,
                generator=data_generator,
                drop_last=False,
            )

    total_epochs = int(config.epochs_adapt)
    for epoch in range(start_epoch, total_epochs):
        remaining_batches = 0
        if config.dry_run_max_batches > 0:
            remaining_batches = max(config.dry_run_max_batches - adapt_batches_seen, 0)
            if remaining_batches == 0:
                break

        loss, src_acc, batches_used, pseudo_used, pseudo_total = adapt_epoch(
            model=model,
            optimizer=optimizer,
            source_loader=weighted_loader,
            source_weights_vec=weights,
            device=device,
            max_batches=remaining_batches,
            pseudo_loader=pseudo_loader,
            pseudo_loss_weight=pseudo_loss_weight,
        )
        adapt_batches_seen += batches_used
        scheduler.step()
        tgt_acc, _ = evaluate(model, target_eval_loader, device)
        last_completed_epoch = epoch

        ckpt_payload: Dict[str, Any] = {
            "backbone": model.backbone.state_dict(),
            "classifier": model.classifier.state_dict(),
            "weights": weights,
            "history": [h.__dict__ for h in history],
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": last_completed_epoch,
            "num_epochs": total_epochs,
            "adapt_batches_seen": adapt_batches_seen,
            "source_acc": float(src_acc),
            "baseline_target_acc": float(baseline_acc),
            "target_acc": float(tgt_acc),
            "completed": False,
            "config": asdict(config),
            "me_iis": {
                "feature_layers": feature_layers,
                "layer_tag": layer_tag,
                "components_per_layer": components_str,
                "cluster_backend": cluster_backend,
                "gmm_selection_mode": gmm_selection_mode,
            },
            "loss": float(loss),
            "pseudo_used": int(pseudo_used),
            "pseudo_total": int(pseudo_total),
        }
        save_checkpoint(artifacts.last_checkpoint_path, ckpt_payload)
        if save_every_epochs > 0 and (epoch + 1) % save_every_epochs == 0:
            save_checkpoint(artifacts.epoch_checkpoint_path(epoch + 1), ckpt_payload)
        save_state(
            artifacts.state_path,
            RunState(
                completed=False,
                stage=artifacts.stage,
                method=artifacts.method,
                run_id=artifacts.run_id,
                last_completed_epoch=last_completed_epoch,
                total_epochs=total_epochs,
                final_checkpoint=str(artifacts.final_checkpoint_path.relative_to(artifacts.run_dir)),
                last_checkpoint=str(artifacts.last_checkpoint_path.relative_to(artifacts.run_dir)),
            ),
        )

        if config.dry_run_max_batches > 0 and adapt_batches_seen >= config.dry_run_max_batches:
            break

    ckpt_final: Dict[str, Any] = {
        "backbone": model.backbone.state_dict(),
        "classifier": model.classifier.state_dict(),
        "weights": weights,
        "history": [h.__dict__ for h in history],
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": last_completed_epoch,
        "num_epochs": total_epochs,
        "adapt_batches_seen": adapt_batches_seen,
        "source_acc": float(src_acc),
        "baseline_target_acc": float(baseline_acc),
        "completed": True,
        "config": asdict(config),
    }
    save_checkpoint(artifacts.final_checkpoint_path, ckpt_final)
    save_state(
        artifacts.state_path,
        RunState(
            completed=True,
            stage=artifacts.stage,
            method=artifacts.method,
            run_id=artifacts.run_id,
            last_completed_epoch=last_completed_epoch,
            total_epochs=total_epochs,
            final_checkpoint=str(artifacts.final_checkpoint_path.relative_to(artifacts.run_dir)),
            last_checkpoint=str(artifacts.last_checkpoint_path.relative_to(artifacts.run_dir)),
        ),
    )

    state = load_state(artifacts.state_path)
    return {
        "status": "trained" if not resume_success else "resumed",
        "run_dir": str(run_dir),
        "checkpoint": str(artifacts.final_checkpoint_path),
        "completed": bool(state.completed) if state else True,
        "epoch": int(last_completed_epoch),
        "baseline_target_acc": float(baseline_acc),
    }
