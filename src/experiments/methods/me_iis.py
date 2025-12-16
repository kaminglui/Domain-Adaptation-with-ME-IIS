from __future__ import annotations

import json
from contextlib import nullcontext
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
from src.experiments.budget import compute_step_budget
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
from src.experiments.loss_checks import safe_cross_entropy
from src.experiments.run_artifacts import RunArtifacts
from src.experiments.run_config import RunConfig, get_run_dir, save_config
from src.experiments.pseudo_labeling import PseudoLabeledDataset, build_pseudo_labels
from src.experiments.signature import hash_model_state, write_signature
from utils.data_utils import build_loader, make_generator, make_worker_init_fn
from utils.experiment_utils import build_components_map, parse_feature_layers
from utils.feature_utils import extract_features
from utils.resource_utils import (
    auto_resources_enabled,
    auto_tune_dataloader,
    detect_resources,
    format_bytes,
    tune_checkpoint_saving,
    write_resource_snapshot,
)
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


def _grad_norm(module: nn.Module) -> float:
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total_sq += float(g.norm(2).item()) ** 2
    return float(total_sq**0.5)


def adapt_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    source_loader: DataLoader,
    source_weights_vec: torch.Tensor,
    device: torch.device,
    *,
    amp_enabled: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
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

        optimizer.zero_grad(set_to_none=True)
        pseudo_loss = torch.tensor(0.0, device=device)
        pseudo_bs = 0
        with (torch.autocast("cuda", dtype=torch.float16) if amp_enabled else nullcontext()):
            logits, _ = model(images, return_features=False)
            ce = safe_cross_entropy(logits, labels, num_classes=int(logits.shape[1]), reduction="none")
            loss_src = (batch_weights * ce).sum() / (batch_weights.sum() + 1e-8)

            if pseudo_batch is not None:
                images_tgt, labels_tgt = pseudo_batch
                images_tgt = images_tgt.to(device)
                labels_tgt = labels_tgt.to(device)
                logits_tgt, _ = model(images_tgt, return_features=False)
                pseudo_loss = safe_cross_entropy(logits_tgt, labels_tgt, num_classes=int(logits_tgt.shape[1]))
                pseudo_bs = labels_tgt.size(0)
                pseudo_used += pseudo_bs

            loss = loss_src + float(pseudo_loss_weight) * pseudo_loss

        if amp_enabled:
            if scaler is None:
                raise RuntimeError("AMP enabled but GradScaler is None.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
    save_every_epochs: int = 0,
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
    model = build_model(
        num_classes=num_classes,
        pretrained=config.backbone_pretrained,
        bottleneck_dim=int(config.bottleneck_dim),
        bottleneck_bn=bool(config.bottleneck_bn),
        bottleneck_relu=bool(config.bottleneck_relu),
        bottleneck_dropout=float(config.bottleneck_dropout),
    ).to(device)
    source_ckpt = torch.load(source_checkpoint, map_location=device)
    model.backbone.load_state_dict(source_ckpt["backbone"])
    model.bottleneck.load_state_dict(source_ckpt["bottleneck"])
    model.classifier.load_state_dict(source_ckpt["classifier"])

    if not config.finetune_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    amp_enabled = bool(config.method_params.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    resources = detect_resources(disk_path=run_dir, data_path=data_root, cuda_device=int(device.index or 0))
    if auto_resources_enabled(default=False):
        dl_tuning = auto_tune_dataloader(
            base_batch_size=config.batch_size,
            base_num_workers=config.num_workers,
            device=device,
            resources=resources,
            model=model,
            input_size=int(config.input_size),
            num_classes=int(num_classes),
        )
        ckpt_tuning = tune_checkpoint_saving(
            disk_free_bytes=resources.disk_free_bytes,
            total_epochs=int(config.epochs_adapt),
            save_every_epochs_requested=int(save_every_epochs),
            model=model,
        )
        write_resource_snapshot(
            run_dir,
            resources,
            tuning={"dataloader": asdict(dl_tuning), "checkpoint": asdict(ckpt_tuning)},
        )
        if dl_tuning.batch_size != config.batch_size or dl_tuning.num_workers != config.num_workers:
            print(
                f"[AUTO_RESOURCES] me_iis: batch_size {config.batch_size}->{dl_tuning.batch_size}, "
                f"num_workers {config.num_workers}->{dl_tuning.num_workers} "
                f"(GPU free={format_bytes(resources.cuda_free_bytes)} disk_free={format_bytes(resources.disk_free_bytes)})"
            )
        save_every_epochs = int(ckpt_tuning.save_every_epochs)
    else:
        dl_tuning = auto_tune_dataloader(
            base_batch_size=config.batch_size,
            base_num_workers=config.num_workers,
            device=device,
            resources=resources,
            model=None,
            input_size=int(config.input_size),
            num_classes=int(num_classes),
        )

    loader_kwargs = dl_tuning.as_loader_kwargs()
    batch_size = int(dl_tuning.batch_size)
    num_workers = int(dl_tuning.num_workers)

    target_eval_loader = build_loader(
        target_eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=config.seed,
        generator=data_generator,
        drop_last=False,
        **loader_kwargs,
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
    gmm_reg_covar = float(config.method_params.get("gmm_reg_covar", 1e-6))
    vmf_kappa = float(config.method_params.get("vmf_kappa", 20.0))
    cluster_clean_ratio = float(config.method_params.get("cluster_clean_ratio", 1.0))
    kmeans_n_init = int(config.method_params.get("kmeans_n_init", 10))

    use_pseudo_labels = bool(config.method_params.get("use_pseudo_labels", False))
    pseudo_conf_thresh = float(config.method_params.get("pseudo_conf_thresh", 0.9))
    pseudo_max_ratio = float(config.method_params.get("pseudo_max_ratio", 1.0))
    pseudo_loss_weight = float(config.method_params.get("pseudo_loss_weight", 1.0))
    weight_clip_max_raw = config.method_params.get("weight_clip_max")
    weight_clip_max = None if weight_clip_max_raw is None else float(weight_clip_max_raw)
    weight_mix_alpha = float(config.method_params.get("weight_mix_alpha", 0.0))

    source_batches_per_epoch = int((len(source_train_ds) + batch_size - 1) // batch_size)
    target_batches_per_epoch = int((len(target_feat_ds) + batch_size - 1) // batch_size)
    budget = compute_step_budget(
        method=str(config.method),
        epochs_source=int(config.epochs_source),
        epochs_adapt=int(config.epochs_adapt),
        source_batches_per_epoch=int(source_batches_per_epoch),
        target_batches_per_epoch=int(target_batches_per_epoch),
    )
    epoch_step_cap = int(budget.adapt_steps_per_epoch)
    print(
        "[BUDGET] "
        f"source_batches/epoch={int(budget.source_batches_per_epoch)} "
        f"target_batches/epoch={int(budget.target_batches_per_epoch)} "
        f"adapt_steps/epoch={int(budget.adapt_steps_per_epoch)} "
        f"steps_total={int(budget.steps_total)}"
    )

    loss_terms_enabled = {
        "cls": True,
        "domain_adv": False,
        "mmd": False,
        "jmmd": False,
        "cdan": False,
        "iis": True,
        "pseudo": bool(use_pseudo_labels),
    }
    signature = {
        "method_name": config.method,
        "source_checkpoint": str(Path(source_checkpoint)),
        "method_params_used": {
            "amp": bool(amp_enabled),
            "feature_layers": list(feature_layers),
            "num_latent_styles": int(config.method_params.get("num_latent_styles", 5)),
            "components_per_layer": config.method_params.get("components_per_layer"),
            "source_prob_mode": str(source_prob_mode),
            "iis_iters": int(iis_iters),
            "iis_tol": float(iis_tol),
            "cluster_backend": str(cluster_backend),
            "gmm_selection_mode": str(gmm_selection_mode),
            "gmm_bic_min_components": int(gmm_bic_min),
            "gmm_bic_max_components": int(gmm_bic_max),
            "gmm_reg_covar": float(gmm_reg_covar),
            "vmf_kappa": float(vmf_kappa),
            "cluster_clean_ratio": float(cluster_clean_ratio),
            "kmeans_n_init": int(kmeans_n_init),
            "use_pseudo_labels": bool(use_pseudo_labels),
            "pseudo_conf_thresh": float(pseudo_conf_thresh),
            "pseudo_max_ratio": float(pseudo_max_ratio),
            "pseudo_loss_weight": float(pseudo_loss_weight),
            "weight_clip_max": weight_clip_max,
            "weight_mix_alpha": float(weight_mix_alpha),
            **({"one_batch_debug": True} if bool(config.method_params.get("one_batch_debug", False)) else {}),
        },
        "loss_terms_enabled": loss_terms_enabled,
        "model_components": {
            "backbone": type(model.backbone).__name__,
            "bottleneck": type(model.bottleneck).__name__,
            "bottleneck_dim": int(config.bottleneck_dim),
            "classifier": type(model.classifier).__name__,
            "domain_discriminator_present": False,
        },
        "dataloader": {
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            **{k: v for k, v in loader_kwargs.items() if v is not None},
        },
        "step_budget": {
            "source_batches_per_epoch": int(budget.source_batches_per_epoch),
            "target_batches_per_epoch": int(budget.target_batches_per_epoch),
            "adapt_steps_per_epoch": int(budget.adapt_steps_per_epoch),
            "steps_source": int(budget.steps_source),
            "steps_adapt": int(budget.steps_adapt),
            "steps_total": int(budget.steps_total),
        },
        "model_state_hash": hash_model_state(model),
    }
    write_signature(run_dir, signature)
    print(f"[METHOD] {config.method} | enabled_losses={loss_terms_enabled}")
    print(f"[AMP] enabled={bool(amp_enabled)}")
    print(
        "[MODEL] "
        f"backbone={signature['model_components']['backbone']} "
        f"bottleneck_dim={int(config.bottleneck_dim)} "
        "domain_discriminator_present=False"
    )

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
            {"params": list(current_model.bottleneck.parameters()), "lr": config.classifier_lr},
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
        bottleneck_state = resume_ckpt.get("bottleneck")
        classifier_state = resume_ckpt.get("classifier")
        if backbone_state is None or bottleneck_state is None or classifier_state is None:
            raise RuntimeError("Adaptation checkpoint missing backbone/bottleneck/classifier state.")
        model.backbone.load_state_dict(backbone_state, strict=False)
        model.bottleneck.load_state_dict(bottleneck_state, strict=False)
        model.classifier.load_state_dict(classifier_state, strict=False)
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "scheduler" in resume_ckpt:
            try:
                scheduler.load_state_dict(resume_ckpt["scheduler"])
                scheduler_loaded = True
            except Exception:
                scheduler_loaded = False
        if amp_enabled and "scaler" in resume_ckpt:
            try:
                scaler.load_state_dict(resume_ckpt["scaler"])
            except Exception:
                pass
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
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=config.seed,
            generator=data_generator,
            drop_last=False,
            **loader_kwargs,
        )
        target_feat_loader = build_loader(
            target_feat_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=config.seed,
            generator=data_generator,
            drop_last=False,
            **loader_kwargs,
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
            gmm_reg_covar=gmm_reg_covar,
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

        if weight_clip_max is not None:
            if weight_clip_max <= 0:
                raise ValueError("weight_clip_max must be > 0 when provided.")
            weights = torch.clamp(weights, max=float(weight_clip_max))
            weights = weights / (weights.sum() + 1e-12)

        if weight_mix_alpha > 0:
            alpha = float(max(0.0, min(1.0, float(weight_mix_alpha))))
            uniform = torch.full_like(weights, 1.0 / float(max(1, int(weights.numel()))))
            weights = (1.0 - alpha) * weights + alpha * uniform
            weights = weights / (weights.sum() + 1e-12)

        iis_npz = artifacts.run_dir / "checkpoints" / f"me_iis_history_{config.run_id}.npz"
        _save_iis_npz(iis_npz, weights, history)

        diagnostics_dir = artifacts.run_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        iis_history_path = diagnostics_dir / "iis_history.json"
        weights_summary_path = diagnostics_dir / "weights_summary.json"

        weights_safe = weights.detach().cpu().clamp(min=1e-12)
        q = torch.tensor([0.01, 0.50, 0.99], dtype=weights_safe.dtype, device=weights_safe.device)
        q01, q50, q99 = torch.quantile(weights_safe, q).tolist()
        weights_entropy = float((-(weights_safe * weights_safe.log()).sum()).item())
        weights_summary = {
            "num_samples": int(weights_safe.numel()),
            "min": float(weights_safe.min().item()),
            "p1": float(q01),
            "p50": float(q50),
            "p99": float(q99),
            "max": float(weights_safe.max().item()),
            "entropy": float(weights_entropy),
            "weight_clip_max": weight_clip_max,
            "weight_mix_alpha": float(weight_mix_alpha),
        }
        weights_summary_path.write_text(json.dumps(weights_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        decoded_unachievable = []
        for flat_idx in getattr(adapter, "unachievable_constraints", []) or []:
            try:
                layer, comp_idx, class_idx = adapter.decode_constraint(int(flat_idx))
                decoded_unachievable.append(
                    {"flat_idx": int(flat_idx), "layer": str(layer), "component": int(comp_idx), "class": int(class_idx)}
                )
            except Exception:
                decoded_unachievable.append({"flat_idx": int(flat_idx)})

        iis_history = {
            "layers": list(feature_layers),
            "selected_components_per_layer": adapter.get_components_per_layer(),
            "cluster_backend": str(cluster_backend),
            "gmm_selection_mode": str(gmm_selection_mode),
            "gmm_reg_covar": float(gmm_reg_covar),
            "iis_iters_requested": int(iis_iters),
            "iis_tol": float(iis_tol),
            "num_unachievable_constraints": int(len(decoded_unachievable)),
            "unachievable_constraints": decoded_unachievable,
            "history": [h.__dict__ for h in history],
        }
        iis_history_path.write_text(json.dumps(iis_history, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if weights is None:
        raise RuntimeError("Failed to obtain IIS weights.")
    if len(weights) != len(source_train_ds):
        raise ValueError("IIS weight vector length does not match source training set.")

    weighted_loader = build_loader(
        IndexedDataset(source_train_ds),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=config.seed,
        generator=data_generator,
        drop_last=False,
        **loader_kwargs,
    )

    one_batch_debug = bool(config.method_params.get("one_batch_debug", False))

    pseudo_loader: Optional[DataLoader] = None
    if use_pseudo_labels:
        target_feat_loader_pl = build_loader(
            target_feat_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=config.seed,
            generator=data_generator,
            drop_last=False,
            **loader_kwargs,
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
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                seed=config.seed,
                generator=data_generator,
                drop_last=False,
                **loader_kwargs,
            )

    if one_batch_debug:
        # IIS residual summary (method-specific diagnostics).
        if history:
            h = history[-1]
            print(
                "[ONE_BATCH_DEBUG] me_iis "
                f"iis_iters={len(history)} max_moment_error={float(h.max_moment_error):.6f} "
                f"mean_moment_error={float(h.mean_moment_error):.6f} "
                f"num_unachievable={int(h.num_unachievable_constraints)} "
                f"w_min={float(h.weight_min):.6f} w_p99={float(h.weight_p99):.6f} w_max={float(h.weight_max):.6f} "
                f"w_entropy={float(h.weight_entropy):.6f}"
            )
        else:
            print("[ONE_BATCH_DEBUG] me_iis iis_iters=0 (no IIS history recorded)")

        model.train()
        images, labels, idxs = next(iter(weighted_loader))
        images = images.to(device)
        labels = labels.to(device)
        batch_weights = weights[idxs].to(device)

        pseudo_loss = torch.tensor(0.0, device=device)
        pseudo_keep = 0
        pseudo_total = 0
        if pseudo_loader is not None:
            try:
                images_tgt, labels_tgt = next(iter(pseudo_loader))
                images_tgt = images_tgt.to(device)
                labels_tgt = labels_tgt.to(device)
                with (torch.autocast("cuda", dtype=torch.float16) if amp_enabled else nullcontext()):
                    logits_tgt, _ = model(images_tgt, return_features=False)
                    pseudo_loss = safe_cross_entropy(logits_tgt, labels_tgt, num_classes=int(logits_tgt.shape[1]))
                pseudo_total = int(labels_tgt.numel())
                pseudo_keep = pseudo_total
            except StopIteration:
                pseudo_loss = torch.tensor(0.0, device=device)

        optimizer.zero_grad(set_to_none=True)
        with (torch.autocast("cuda", dtype=torch.float16) if amp_enabled else nullcontext()):
            logits, _ = model(images, return_features=False)
            ce = safe_cross_entropy(logits, labels, num_classes=int(logits.shape[1]), reduction="none")
            loss_src = (batch_weights * ce).sum() / (batch_weights.sum() + 1e-8)
            loss = loss_src + float(pseudo_loss_weight) * pseudo_loss
        if not torch.isfinite(loss).all().item():
            raise RuntimeError("Non-finite loss encountered in one-batch debug.")
        if amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        gn_backbone = _grad_norm(model.backbone)
        gn_bottleneck = _grad_norm(model.bottleneck)
        gn_head = _grad_norm(model.classifier)
        if amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        src_acc = float((torch.argmax(logits, dim=1) == labels).float().mean().item() * 100.0)

        print(
            "[ONE_BATCH_DEBUG] me_iis "
            f"weighted_cls_loss={float(loss_src.item()):.6f} pseudo_loss={float(pseudo_loss.item()):.6f} "
            f"pseudo_keep={pseudo_keep}/{pseudo_total} src_acc={src_acc:.3f} "
            f"grad_norm_backbone={gn_backbone:.6f} grad_norm_bottleneck={gn_bottleneck:.6f} grad_norm_head={gn_head:.6f}"
        )
        return {
            "status": "one_batch_debug",
            "run_dir": str(run_dir),
            "weighted_cls_loss": float(loss_src.item()),
            "pseudo_loss": float(pseudo_loss.item()),
            "pseudo_keep": int(pseudo_keep),
            "pseudo_total": int(pseudo_total),
            "source_acc_batch": float(src_acc),
            "grad_norm_backbone": float(gn_backbone),
            "grad_norm_bottleneck": float(gn_bottleneck),
            "grad_norm_head": float(gn_head),
            "iis_iters": int(len(history)),
            "iis_max_moment_error": float(history[-1].max_moment_error) if history else None,
            "iis_num_unachievable": int(history[-1].num_unachievable_constraints) if history else None,
        }

    total_epochs = int(config.epochs_adapt)
    for epoch in range(start_epoch, total_epochs):
        remaining_batches = 0
        if config.dry_run_max_batches > 0:
            remaining_batches = max(config.dry_run_max_batches - adapt_batches_seen, 0)
            if remaining_batches == 0:
                break

        max_batches_epoch = int(epoch_step_cap)
        if remaining_batches > 0:
            max_batches_epoch = int(min(max_batches_epoch, remaining_batches))

        loss, src_acc, batches_used, pseudo_used, pseudo_total = adapt_epoch(
            model=model,
            optimizer=optimizer,
            source_loader=weighted_loader,
            source_weights_vec=weights,
            device=device,
            amp_enabled=bool(amp_enabled),
            scaler=scaler,
            max_batches=max_batches_epoch,
            pseudo_loader=pseudo_loader,
            pseudo_loss_weight=pseudo_loss_weight,
        )
        adapt_batches_seen += batches_used
        scheduler.step()
        tgt_acc, _ = evaluate(model, target_eval_loader, device)
        last_completed_epoch = epoch

        ckpt_payload: Dict[str, Any] = {
            "backbone": model.backbone.state_dict(),
            "bottleneck": model.bottleneck.state_dict(),
            "classifier": model.classifier.state_dict(),
            "weights": weights,
            "history": [h.__dict__ for h in history],
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            **({"scaler": scaler.state_dict()} if amp_enabled else {}),
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
        "bottleneck": model.bottleneck.state_dict(),
        "classifier": model.classifier.state_dict(),
        "weights": weights,
        "history": [h.__dict__ for h in history],
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        **({"scaler": scaler.state_dict()} if amp_enabled else {}),
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
