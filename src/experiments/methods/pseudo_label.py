from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from datasets.domain_loaders import get_domain_loaders
from eval import evaluate
from models.classifier import build_model
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
from src.experiments.pseudo_labeling import PseudoLabeledDataset, build_pseudo_labels
from src.experiments.run_artifacts import RunArtifacts
from src.experiments.run_config import RunConfig, get_run_dir, save_config
from utils.data_utils import build_loader, make_generator, make_worker_init_fn
from utils.seed_utils import get_device, set_seed


def _infer_num_classes(dataset: Dataset) -> int:
    base = dataset
    while hasattr(base, "dataset"):
        base = base.dataset  # type: ignore[attr-defined]
    if hasattr(base, "classes"):
        return len(base.classes)  # type: ignore[attr-defined]
    raise ValueError("Unable to infer number of classes from dataset.")


def _adapt_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    source_loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
    pseudo_loader: Optional[DataLoader] = None,
    pseudo_loss_weight: float = 1.0,
) -> Dict[str, float]:
    model.train()
    pseudo_iter = iter(pseudo_loader) if pseudo_loader is not None else None

    total_loss_sum = 0.0
    total_src = 0
    total_acc = 0.0
    batches_seen = 0
    pseudo_used = 0
    pseudo_total = len(pseudo_loader.dataset) if pseudo_loader is not None else 0

    for images, labels in source_loader:
        if max_batches > 0 and batches_seen >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        pseudo_batch = None
        if pseudo_iter is not None:
            try:
                pseudo_batch = next(pseudo_iter)
            except StopIteration:
                pseudo_iter = None
                pseudo_batch = None

        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(images, return_features=False)
        loss_src = F.cross_entropy(logits, labels)
        loss_src.backward()

        bs = labels.size(0)
        loss_src_value = float(loss_src.detach().item())
        total_loss_sum += loss_src_value * bs

        pseudo_loss_value = 0.0
        pseudo_bs = 0
        if pseudo_batch is not None:
            images_tgt, labels_tgt = pseudo_batch
            images_tgt = images_tgt.to(device)
            labels_tgt = labels_tgt.to(device)
            logits_tgt, _ = model(images_tgt, return_features=False)
            pseudo_loss = F.cross_entropy(logits_tgt, labels_tgt)
            (pseudo_loss_weight * pseudo_loss).backward()

            pseudo_bs = labels_tgt.size(0)
            pseudo_used += pseudo_bs
            pseudo_loss_value = float(pseudo_loss.detach().item())
            total_loss_sum += float(pseudo_loss_weight) * pseudo_loss_value * pseudo_bs

        optimizer.step()

        acc = float((torch.argmax(logits, dim=1) == labels).float().mean().item() * 100.0)
        total_acc += acc * bs
        total_src += bs
        batches_seen += 1

    denom = total_src + pseudo_used if (pseudo_loader is not None and (total_src + pseudo_used) > 0) else total_src
    avg_loss = total_loss_sum / denom if denom > 0 else 0.0
    avg_src_acc = total_acc / total_src if total_src > 0 else 0.0
    return {
        "loss": float(avg_loss),
        "source_acc": float(avg_src_acc),
        "batches_seen": float(batches_seen),
        "pseudo_used": float(pseudo_used),
        "pseudo_total": float(pseudo_total),
    }


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
    if device.type == "cuda":
        torch.cuda.empty_cache()
    data_generator = make_generator(config.seed)
    worker_init = make_worker_init_fn(config.seed)

    source_loader, _target_loader_train, target_eval_loader = get_domain_loaders(
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
    source_train_ds: Dataset = source_loader.dataset
    target_eval_ds: Dataset = target_eval_loader.dataset

    source_loader = build_loader(
        source_train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        generator=data_generator,
        drop_last=False,
    )
    target_eval_loader = build_loader(
        target_eval_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
        generator=data_generator,
        drop_last=False,
    )

    target_feat_ds: Dataset = DropLabelsDataset(target_eval_ds)
    target_feat_loader = build_loader(
        target_feat_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
        generator=data_generator,
        drop_last=False,
    )

    num_classes = _infer_num_classes(source_train_ds)
    model = build_model(num_classes=int(num_classes), pretrained=config.backbone_pretrained).to(device)
    source_ckpt = torch.load(source_checkpoint, map_location=device)
    model.backbone.load_state_dict(source_ckpt["backbone"])
    model.classifier.load_state_dict(source_ckpt["classifier"])

    if not config.finetune_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    backbone_lr = config.classifier_lr * config.backbone_lr_scale
    groups = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": backbone_lr},
        {"params": list(model.classifier.parameters()), "lr": config.classifier_lr},
    ]
    groups = [g for g in groups if len(g["params"]) > 0]
    optimizer = optim.SGD(groups, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config.epochs_adapt)))

    conf_thresh = float(config.method_params.get("pseudo_conf_thresh", 0.9))
    max_ratio = float(config.method_params.get("pseudo_max_ratio", 1.0))
    pseudo_loss_weight = float(config.method_params.get("pseudo_loss_weight", 1.0))

    pseudo_indices, pseudo_labels = build_pseudo_labels(
        model=model,
        target_loader=target_feat_loader,
        device=device,
        conf_thresh=conf_thresh,
        max_ratio=max_ratio,
        num_source_samples=len(source_train_ds),
    )
    if len(pseudo_indices) != len(pseudo_labels):
        raise RuntimeError("Pseudo-labeling returned mismatched indices/labels lengths.")
    if pseudo_indices:
        min_idx = min(int(i) for i in pseudo_indices)
        max_idx = max(int(i) for i in pseudo_indices)
        if min_idx < 0 or max_idx >= len(target_feat_ds):
            raise ValueError(
                "Pseudo-label indices out of range for target dataset: "
                f"[{min_idx}, {max_idx}] vs dataset size {len(target_feat_ds)}"
            )
    if pseudo_labels:
        min_label = min(int(l) for l in pseudo_labels)
        max_label = max(int(l) for l in pseudo_labels)
        if min_label < 0 or max_label >= int(num_classes):
            raise ValueError(
                "Pseudo-label class ids out of range: "
                f"[{min_label}, {max_label}] vs num_classes={int(num_classes)}"
            )
    print(
        f"[pseudo_label] pseudo_count={len(pseudo_indices)} conf_thresh={conf_thresh} "
        f"max_ratio={max_ratio} pseudo_loss_weight={pseudo_loss_weight}"
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pseudo_loader: Optional[DataLoader] = None
    if len(pseudo_indices) > 0:
        pseudo_ds = PseudoLabeledDataset(target_feat_loader.dataset, pseudo_indices, pseudo_labels)
        pseudo_loader = build_loader(
            pseudo_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            seed=config.seed,
            generator=data_generator,
            drop_last=False,
        )

    start_epoch = 0
    last_completed_epoch = -1
    scheduler_loaded = False
    resume_success = False
    adapt_batches_seen = 0
    src_acc = 0.0

    if resume_path is not None:
        resume_ckpt = load_checkpoint(resume_path, map_location=device)
        model.backbone.load_state_dict(resume_ckpt["backbone"], strict=False)
        model.classifier.load_state_dict(resume_ckpt["classifier"], strict=False)
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "scheduler" in resume_ckpt:
            try:
                scheduler.load_state_dict(resume_ckpt["scheduler"])
                scheduler_loaded = True
            except Exception:
                scheduler_loaded = False
        last_completed_epoch = int(resume_ckpt.get("epoch", last_completed_epoch))
        start_epoch = max(last_completed_epoch + 1, 0)
        adapt_batches_seen = int(resume_ckpt.get("adapt_batches_seen", adapt_batches_seen))
        src_acc = float(resume_ckpt.get("source_acc", src_acc))
        resume_success = True

    if not scheduler_loaded and start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    total_epochs = int(config.epochs_adapt)
    for epoch in range(start_epoch, total_epochs):
        remaining_batches = 0
        if config.dry_run_max_batches > 0:
            remaining_batches = max(config.dry_run_max_batches - adapt_batches_seen, 0)
            if remaining_batches == 0:
                break

        stats = _adapt_epoch(
            model=model,
            optimizer=optimizer,
            source_loader=source_loader,
            device=device,
            max_batches=remaining_batches,
            pseudo_loader=pseudo_loader,
            pseudo_loss_weight=pseudo_loss_weight,
        )
        adapt_batches_seen += int(stats["batches_seen"])
        scheduler.step()
        tgt_acc, _ = evaluate(model, target_eval_loader, device)
        last_completed_epoch = epoch
        src_acc = float(stats["source_acc"])

        ckpt_payload: Dict[str, Any] = {
            "backbone": model.backbone.state_dict(),
            "classifier": model.classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": last_completed_epoch,
            "num_epochs": total_epochs,
            "adapt_batches_seen": adapt_batches_seen,
            "source_acc": float(src_acc),
            "target_acc": float(tgt_acc),
            "completed": False,
            "config": asdict(config),
            "pseudo_conf_thresh": float(conf_thresh),
            "pseudo_max_ratio": float(max_ratio),
            "pseudo_loss_weight": float(pseudo_loss_weight),
            "pseudo_count": int(len(pseudo_indices)),
            "loss": float(stats["loss"]),
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
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": last_completed_epoch,
        "num_epochs": total_epochs,
        "adapt_batches_seen": adapt_batches_seen,
        "source_acc": float(src_acc),
        "completed": True,
        "config": asdict(config),
        "pseudo_count": int(len(pseudo_indices)),
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
        "pseudo_count": int(len(pseudo_indices)),
    }
