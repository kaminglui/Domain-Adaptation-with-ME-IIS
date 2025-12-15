from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

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
from src.experiments.run_artifacts import RunArtifacts
from src.experiments.run_config import RunConfig, get_run_dir, save_config
from utils.data_utils import build_loader, make_generator, make_worker_init_fn
from utils.resource_utils import (
    auto_tune_dataloader,
    auto_resources_enabled,
    detect_resources,
    format_bytes,
    tune_checkpoint_saving,
    write_resource_snapshot,
)
from utils.seed_utils import get_device, set_seed


def _infer_num_classes(loader: DataLoader) -> int:
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset  # type: ignore[attr-defined]
    if hasattr(dataset, "classes"):
        return len(dataset.classes)  # type: ignore[attr-defined]
    raise ValueError("Unable to infer number of classes from dataset.")


def _compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == labels).float().mean().item() * 100.0)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, float, int]:
    model.train()
    running_loss, running_acc, total = 0.0, 0.0, 0
    batches_seen = 0
    for images, labels in loader:
        if max_batches > 0 and batches_seen >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(images, return_features=False)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        running_loss += float(loss.item()) * bs
        running_acc += _compute_accuracy(logits.detach(), labels) * bs
        total += bs
        batches_seen += 1

    if total == 0:
        return 0.0, 0.0, batches_seen
    return running_loss / total, running_acc / total, batches_seen


def run(
    config: RunConfig,
    force_rerun: bool = False,
    runs_root: Optional[Path] = None,
    save_every_epochs: int = 1,
) -> Dict[str, Any]:
    run_dir = get_run_dir(config, runs_root=runs_root)
    artifacts = RunArtifacts(run_dir=run_dir, run_id=config.run_id, stage="source", method=config.method)
    ensure_run_dirs(artifacts)
    save_config(run_dir, config)

    if not force_rerun:
        completed = find_existing_completed_checkpoint(artifacts)
        if completed is not None:
            return {"status": "skipped", "run_dir": str(run_dir), "checkpoint": str(completed)}

    resume_path = None if force_rerun else find_resume_checkpoint(artifacts)

    if config.data_root is None:
        raise ValueError("data_root must be set for training.")
    data_root = Path(config.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

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

    source_ds = source_loader.dataset
    target_eval_ds = target_eval_loader.dataset

    num_classes = _infer_num_classes(source_loader)
    model = build_model(num_classes=num_classes, pretrained=config.backbone_pretrained).to(device)

    if auto_resources_enabled(default=False):
        resources = detect_resources(disk_path=run_dir, data_path=data_root, cuda_device=int(device.index or 0))
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
            total_epochs=int(config.epochs_source),
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
                f"[AUTO_RESOURCES] source_only: batch_size {config.batch_size}->{dl_tuning.batch_size}, "
                f"num_workers {config.num_workers}->{dl_tuning.num_workers} "
                f"(GPU free={format_bytes(resources.cuda_free_bytes)} disk_free={format_bytes(resources.disk_free_bytes)})"
            )
        save_every_epochs = int(ckpt_tuning.save_every_epochs)
    else:
        dl_tuning = auto_tune_dataloader(
            base_batch_size=config.batch_size,
            base_num_workers=config.num_workers,
            device=device,
            resources=detect_resources(disk_path=run_dir, data_path=data_root, cuda_device=int(device.index or 0)),
            model=None,
            input_size=int(config.input_size),
            num_classes=int(num_classes),
        )

    loader_kwargs = dl_tuning.as_loader_kwargs()
    source_loader = build_loader(
        source_ds,
        batch_size=dl_tuning.batch_size,
        shuffle=True,
        num_workers=dl_tuning.num_workers,
        seed=config.seed,
        generator=data_generator,
        drop_last=False,
        **loader_kwargs,
    )
    target_eval_loader = build_loader(
        target_eval_ds,
        batch_size=dl_tuning.batch_size,
        shuffle=False,
        num_workers=dl_tuning.num_workers,
        seed=config.seed,
        generator=data_generator,
        drop_last=False,
        **loader_kwargs,
    )

    def _make_optimizer_and_scheduler(current_model: nn.Module):
        params = [
            {"params": current_model.backbone.parameters(), "lr": config.lr_backbone},
            {"params": current_model.classifier.parameters(), "lr": config.lr_classifier},
        ]
        opt = optim.SGD(params, momentum=config.momentum, weight_decay=config.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, config.epochs_source))
        return opt, sched

    optimizer, scheduler = _make_optimizer_and_scheduler(model)
    start_epoch = 0
    last_completed_epoch = -1
    scheduler_loaded = False
    best_target_acc = 0.0
    final_source_acc = 0.0
    batches_seen_total = 0

    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, map_location=device)
        backbone_state = ckpt.get("backbone")
        classifier_state = ckpt.get("classifier")
        if backbone_state is None or classifier_state is None:
            raise RuntimeError("Resume checkpoint missing backbone/classifier state.")
        model.backbone.load_state_dict(backbone_state, strict=False)
        model.classifier.load_state_dict(classifier_state, strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
                scheduler_loaded = True
            except Exception:
                scheduler_loaded = False
        best_target_acc = float(ckpt.get("best_target_acc", best_target_acc))
        final_source_acc = float(ckpt.get("source_acc", final_source_acc))
        last_completed_epoch = int(ckpt.get("epoch", last_completed_epoch))
        batches_seen_total = int(ckpt.get("batches_seen_total", batches_seen_total))
        start_epoch = max(last_completed_epoch + 1, 0)

    if not scheduler_loaded and start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    total_epochs = int(config.epochs_source)
    for epoch in range(start_epoch, total_epochs):
        remaining_batches = 0
        if config.dry_run_max_batches > 0:
            remaining_batches = max(config.dry_run_max_batches - batches_seen_total, 0)
            if remaining_batches == 0:
                break

        loss, src_acc, batches_used = _train_one_epoch(
            model=model,
            loader=source_loader,
            optimizer=optimizer,
            device=device,
            max_batches=remaining_batches,
        )
        batches_seen_total += batches_used
        scheduler.step()
        tgt_acc, _ = evaluate(model, target_eval_loader, device)
        best_target_acc = max(best_target_acc, tgt_acc)
        final_source_acc = src_acc
        last_completed_epoch = epoch

        ckpt_payload: Dict[str, Any] = {
            "backbone": model.backbone.state_dict(),
            "classifier": model.classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": last_completed_epoch,
            "num_epochs": total_epochs,
            "best_target_acc": best_target_acc,
            "source_acc": final_source_acc,
            "batches_seen_total": batches_seen_total,
            "completed": False,
            "config": asdict(config),
            "loss": float(loss),
            "target_acc": float(tgt_acc),
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

        if config.dry_run_max_batches > 0 and batches_seen_total >= config.dry_run_max_batches:
            break

    ckpt_final: Dict[str, Any] = {
        "backbone": model.backbone.state_dict(),
        "classifier": model.classifier.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": last_completed_epoch,
        "num_epochs": total_epochs,
        "best_target_acc": best_target_acc,
        "source_acc": final_source_acc,
        "batches_seen_total": batches_seen_total,
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
        "status": "trained",
        "run_dir": str(run_dir),
        "checkpoint": str(artifacts.final_checkpoint_path),
        "completed": bool(state.completed) if state else True,
        "epoch": int(last_completed_epoch),
        "best_target_acc": float(best_target_acc),
        "source_acc": float(final_source_acc),
    }
