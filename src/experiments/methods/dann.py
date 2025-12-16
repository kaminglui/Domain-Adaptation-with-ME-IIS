from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets

from datasets.domain_loaders import get_domain_loaders
from eval import evaluate
from models.classifier import build_model
from models.dann import DomainDiscriminator, DomainDiscriminatorConfig, GradientReversal
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
from src.experiments.data import DropLabelsDataset, assert_labels_dropped
from src.experiments.loss_checks import safe_cross_entropy
from src.experiments.run_artifacts import RunArtifacts
from src.experiments.run_config import RunConfig, get_run_dir, save_config
from src.experiments.signature import hash_model_state, write_signature
from utils.data_utils import build_loader, make_generator, make_worker_init_fn
from utils.resource_utils import (
    auto_resources_enabled,
    auto_tune_dataloader,
    detect_resources,
    format_bytes,
    tune_checkpoint_saving,
    write_resource_snapshot,
)
from utils.seed_utils import get_device, set_seed


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def _dataset_root(ds: Dataset) -> str:
    root = getattr(ds, "root", None)
    if root is not None:
        return str(root)
    base = getattr(ds, "dataset", None)
    root = getattr(base, "root", None) if base is not None else None
    if root is not None:
        return str(root)
    raise ValueError("Dataset does not expose a 'root' attribute.")


def _dataset_transform(ds: Dataset):
    t = getattr(ds, "transform", None)
    if t is not None:
        return t
    base = getattr(ds, "dataset", None)
    t = getattr(base, "transform", None) if base is not None else None
    return t


def _maybe_subset_like(reference: Dataset, other: Dataset) -> Dataset:
    if hasattr(reference, "indices"):
        try:
            from torch.utils.data import Subset

            return Subset(other, list(reference.indices))  # type: ignore[attr-defined]
        except Exception:
            return other
    return other


def _infer_feature_dim(model: nn.Module, device: torch.device) -> int:
    dummy = torch.zeros(2, 3, 224, 224, device=device)
    _, feats = model(dummy, return_features=True)
    if feats is None:
        raise RuntimeError("Model did not return features (return_features=True).")
    return int(feats.shape[1])


def _dann_lambda(step: int, total_steps: int, schedule: str) -> float:
    if total_steps <= 0:
        return 1.0
    if schedule == "constant":
        return 1.0
    if schedule == "linear":
        return float(step) / float(total_steps)
    if schedule == "grl":
        # Common DANN schedule: 2/(1+exp(-10p)) - 1
        p = float(step) / float(total_steps)
        return float(2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0)
    raise ValueError(f"Unknown dann_lambda_schedule '{schedule}'")


def _grad_norm(module: nn.Module) -> float:
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total_sq += float(g.norm(2).item()) ** 2
    return float(total_sq**0.5)


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

    source_loader, target_loader, target_eval_loader = get_domain_loaders(
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
    target_train_ds: Dataset = DropLabelsDataset(target_loader.dataset)
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
    target_loader = build_loader(
        target_train_ds,
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

    # Load source checkpoint
    num_classes = None
    if hasattr(source_train_ds, "classes"):
        num_classes = len(source_train_ds.classes)  # type: ignore[attr-defined]
    if num_classes is None:
        raise ValueError("Unable to infer num_classes for DANN from source dataset.")
    model = build_model(
        num_classes=int(num_classes),
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

    feat_dim = getattr(model, "feature_dim", None)
    if feat_dim is None:
        feat_dim = _infer_feature_dim(model, device=device)

    disc_cfg = DomainDiscriminatorConfig(
        hidden_dim=int(config.method_params.get("disc_hidden_dim", 1024)),
        dropout=float(config.method_params.get("disc_dropout", 0.0)),
    )
    discriminator = DomainDiscriminator(in_dim=int(feat_dim), config=disc_cfg).to(device)
    grl = GradientReversal()
    disc_lr = float(config.method_params.get("disc_lr", config.classifier_lr))

    amp_enabled = bool(config.method_params.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    loss_terms_enabled = {
        "cls": True,
        "domain_adv": True,
        "mmd": False,
        "jmmd": False,
        "cdan": False,
        "iis": False,
        "pseudo": False,
    }
    signature = {
        "method_name": config.method,
        "source_checkpoint": str(Path(source_checkpoint)),
        "method_params_used": {
            "amp": bool(amp_enabled),
            "disc_hidden_dim": int(disc_cfg.hidden_dim),
            "disc_dropout": float(disc_cfg.dropout),
            "disc_lr": float(disc_lr),
            "grl_lambda_schedule": str(
                config.method_params.get("grl_lambda_schedule", config.method_params.get("dann_lambda_schedule", "grl"))
            ),
            "domain_loss_weight": float(config.method_params.get("domain_loss_weight", 1.0)),
            **({"one_batch_debug": True} if bool(config.method_params.get("one_batch_debug", False)) else {}),
        },
        "loss_terms_enabled": loss_terms_enabled,
        "model_components": {
            "backbone": type(model.backbone).__name__,
            "bottleneck": type(model.bottleneck).__name__,
            "bottleneck_dim": int(config.bottleneck_dim),
            "classifier": type(model.classifier).__name__,
            "domain_discriminator_present": True,
            "domain_discriminator": type(discriminator).__name__,
            "domain_discriminator_in_dim": int(feat_dim),
            "domain_discriminator_hidden_dim": int(disc_cfg.hidden_dim),
        },
        "model_state_hash": hash_model_state(model),
    }

    if not config.finetune_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

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
                f"[AUTO_RESOURCES] dann: batch_size {config.batch_size}->{dl_tuning.batch_size}, "
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

    source_loader = build_loader(
        source_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=config.seed,
        generator=data_generator,
        drop_last=False,
        **loader_kwargs,
    )
    target_loader = build_loader(
        target_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=config.seed,
        generator=data_generator,
        drop_last=False,
        **loader_kwargs,
    )
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

    budget = compute_step_budget(
        method=str(config.method),
        epochs_source=int(config.epochs_source),
        epochs_adapt=int(config.epochs_adapt),
        source_batches_per_epoch=int(len(source_loader)),
        target_batches_per_epoch=int(len(target_loader)),
    )
    print(
        "[BUDGET] "
        f"source_batches/epoch={int(budget.source_batches_per_epoch)} "
        f"target_batches/epoch={int(budget.target_batches_per_epoch)} "
        f"adapt_steps/epoch={int(budget.adapt_steps_per_epoch)} "
        f"steps_total={int(budget.steps_total)}"
    )

    signature["dataloader"] = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        **{k: v for k, v in loader_kwargs.items() if v is not None},
    }
    signature["step_budget"] = {
        "source_batches_per_epoch": int(budget.source_batches_per_epoch),
        "target_batches_per_epoch": int(budget.target_batches_per_epoch),
        "adapt_steps_per_epoch": int(budget.adapt_steps_per_epoch),
        "steps_source": int(budget.steps_source),
        "steps_adapt": int(budget.steps_adapt),
        "steps_total": int(budget.steps_total),
    }
    write_signature(run_dir, signature)
    print(f"[METHOD] {config.method} | enabled_losses={loss_terms_enabled}")
    print(f"[AMP] enabled={bool(amp_enabled)}")
    print(
        "[MODEL] "
        f"backbone={signature['model_components']['backbone']} "
        f"bottleneck_dim={int(config.bottleneck_dim)} "
        "domain_discriminator_present=True"
    )

    backbone_lr = config.classifier_lr * config.backbone_lr_scale
    groups = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": backbone_lr},
        {"params": list(model.bottleneck.parameters()), "lr": config.classifier_lr},
        {"params": list(model.classifier.parameters()), "lr": config.classifier_lr},
        {"params": list(discriminator.parameters()), "lr": float(disc_lr)},
    ]
    groups = [g for g in groups if len(g["params"]) > 0]
    optimizer = optim.SGD(groups, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config.epochs_adapt)))

    start_epoch = 0
    last_completed_epoch = -1
    scheduler_loaded = False
    resume_success = False
    adapt_batches_seen = 0

    if resume_path is not None:
        resume_ckpt = load_checkpoint(resume_path, map_location=device)
        model.backbone.load_state_dict(resume_ckpt["backbone"], strict=False)
        model.bottleneck.load_state_dict(resume_ckpt["bottleneck"], strict=False)
        model.classifier.load_state_dict(resume_ckpt["classifier"], strict=False)
        discriminator.load_state_dict(resume_ckpt["discriminator"], strict=False)
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
        last_completed_epoch = int(resume_ckpt.get("epoch", last_completed_epoch))
        start_epoch = max(last_completed_epoch + 1, 0)
        adapt_batches_seen = int(resume_ckpt.get("adapt_batches_seen", adapt_batches_seen))
        resume_success = True

    if not scheduler_loaded and start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    domain_loss_weight = float(config.method_params.get("domain_loss_weight", 1.0))
    lambda_schedule = str(
        config.method_params.get("grl_lambda_schedule", config.method_params.get("dann_lambda_schedule", "grl"))
    )

    steps_per_epoch = min(len(source_loader), len(target_loader))
    total_steps = max(1, steps_per_epoch * max(1, int(config.epochs_adapt)))
    global_step = start_epoch * steps_per_epoch

    total_epochs = int(config.epochs_adapt)
    src_acc = 0.0

    one_batch_debug = bool(config.method_params.get("one_batch_debug", False))
    if one_batch_debug:
        model.train()
        discriminator.train()
        x_s, y_s = next(iter(source_loader))
        x_t, y_t = next(iter(target_loader))
        assert_labels_dropped(y_t, label_value=-1)

        x_s = x_s.to(device)
        y_s = y_s.to(device)
        x_t = x_t.to(device)

        optimizer.zero_grad(set_to_none=True)

        with (torch.autocast("cuda", dtype=torch.float16) if amp_enabled else nullcontext()):
            logits_s, feats_s = model(x_s, return_features=True)
            _logits_t, feats_t = model(x_t, return_features=True)
            if feats_s is None or feats_t is None:
                raise RuntimeError("Model did not return features for DANN (one-batch debug).")

            cls_loss = safe_cross_entropy(logits_s, y_s, num_classes=int(logits_s.shape[1]))
            lam = _dann_lambda(global_step, total_steps, schedule=lambda_schedule)
            dom_logits_s = discriminator(grl(feats_s, lam))
            dom_logits_t = discriminator(grl(feats_t, lam))

            dom_labels_s = torch.zeros(dom_logits_s.size(0), dtype=torch.long, device=device)
            dom_labels_t = torch.ones(dom_logits_t.size(0), dtype=torch.long, device=device)
            dom_loss_s = safe_cross_entropy(dom_logits_s, dom_labels_s, num_classes=int(dom_logits_s.shape[1]))
            dom_loss_t = safe_cross_entropy(dom_logits_t, dom_labels_t, num_classes=int(dom_logits_t.shape[1]))
            dom_loss = 0.5 * (dom_loss_s + dom_loss_t)

            loss = cls_loss + domain_loss_weight * dom_loss
        if not torch.isfinite(loss).all().item():
            raise RuntimeError("Non-finite loss encountered in one-batch debug.")
        if amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        with torch.no_grad():
            dom_preds_s = torch.argmax(dom_logits_s, dim=1)
            dom_preds_t = torch.argmax(dom_logits_t, dim=1)
            dom_acc = float(
                torch.cat([dom_preds_s == dom_labels_s, dom_preds_t == dom_labels_t]).float().mean().item() * 100.0
            )
            src_acc = float((torch.argmax(logits_s, dim=1) == y_s).float().mean().item() * 100.0)

        gn_backbone = _grad_norm(model.backbone)
        gn_bottleneck = _grad_norm(model.bottleneck)
        gn_head = _grad_norm(model.classifier)
        gn_disc = _grad_norm(discriminator)
        if amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        print(
            "[ONE_BATCH_DEBUG] dann "
            f"cls_loss={float(cls_loss.item()):.6f} dom_loss={float(dom_loss.item()):.6f} "
            f"lambda={float(lam):.6f} dom_acc={dom_acc:.3f} src_acc={src_acc:.3f} "
            f"grad_norm_backbone={gn_backbone:.6f} grad_norm_bottleneck={gn_bottleneck:.6f} "
            f"grad_norm_head={gn_head:.6f} grad_norm_disc={gn_disc:.6f}"
        )
        return {
            "status": "one_batch_debug",
            "run_dir": str(run_dir),
            "cls_loss": float(cls_loss.item()),
            "domain_loss": float(dom_loss.item()),
            "grl_lambda": float(lam),
            "domain_acc": float(dom_acc),
            "source_acc_batch": float(src_acc),
            "grad_norm_backbone": float(gn_backbone),
            "grad_norm_bottleneck": float(gn_bottleneck),
            "grad_norm_head": float(gn_head),
            "grad_norm_disc": float(gn_disc),
        }

    for epoch in range(start_epoch, total_epochs):
        remaining_batches = 0
        if config.dry_run_max_batches > 0:
            remaining_batches = max(config.dry_run_max_batches - adapt_batches_seen, 0)
            if remaining_batches == 0:
                break

        model.train()
        discriminator.train()

        epoch_loss = 0.0
        epoch_dom_loss = 0.0
        epoch_dom_acc = 0.0
        epoch_src_acc = 0.0
        epoch_batches = 0

        for batch_idx, ((x_s, y_s), (x_t, y_t)) in enumerate(zip(source_loader, target_loader)):
            if remaining_batches > 0 and epoch_batches >= remaining_batches:
                break

            x_s = x_s.to(device)
            y_s = y_s.to(device)
            x_t = x_t.to(device)
            assert_labels_dropped(y_t, label_value=-1)

            optimizer.zero_grad(set_to_none=True)

            with (torch.autocast("cuda", dtype=torch.float16) if amp_enabled else nullcontext()):
                logits_s, feats_s = model(x_s, return_features=True)
                _logits_t, feats_t = model(x_t, return_features=True)
                if feats_s is None or feats_t is None:
                    raise RuntimeError("Model did not return features for DANN.")

                cls_loss = safe_cross_entropy(logits_s, y_s, num_classes=int(logits_s.shape[1]))

                lam = _dann_lambda(global_step, total_steps, schedule=lambda_schedule)
                dom_logits_s = discriminator(grl(feats_s, lam))
                dom_logits_t = discriminator(grl(feats_t, lam))

                dom_labels_s = torch.zeros(dom_logits_s.size(0), dtype=torch.long, device=device)
                dom_labels_t = torch.ones(dom_logits_t.size(0), dtype=torch.long, device=device)
                dom_loss = 0.5 * (
                    safe_cross_entropy(dom_logits_s, dom_labels_s, num_classes=int(dom_logits_s.shape[1]))
                    + safe_cross_entropy(dom_logits_t, dom_labels_t, num_classes=int(dom_logits_t.shape[1]))
                )

                loss = cls_loss + domain_loss_weight * dom_loss

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                src_acc = float((torch.argmax(logits_s, dim=1) == y_s).float().mean().item() * 100.0)
                dom_preds_s = torch.argmax(dom_logits_s, dim=1)
                dom_preds_t = torch.argmax(dom_logits_t, dim=1)
                dom_acc = float(
                    torch.cat([dom_preds_s == dom_labels_s, dom_preds_t == dom_labels_t]).float().mean().item() * 100.0
                )
            epoch_src_acc += src_acc
            epoch_loss += float(loss.item())
            epoch_dom_loss += float(dom_loss.item())
            epoch_dom_acc += float(dom_acc)
            epoch_batches += 1
            adapt_batches_seen += 1
            global_step += 1

            if config.dry_run_max_batches > 0 and adapt_batches_seen >= config.dry_run_max_batches:
                break

        scheduler.step()
        tgt_acc, _ = evaluate(model, target_eval_loader, device)

        last_completed_epoch = epoch
        avg_loss = epoch_loss / max(1, epoch_batches)
        avg_dom_loss = epoch_dom_loss / max(1, epoch_batches)
        avg_dom_acc = epoch_dom_acc / max(1, epoch_batches)
        avg_src_acc = epoch_src_acc / max(1, epoch_batches)
        print(
            f"[DANN] epoch={epoch + 1}/{total_epochs} "
            f"loss={avg_loss:.6f} dom_loss={avg_dom_loss:.6f} dom_acc={avg_dom_acc:.3f} "
            f"lambda_schedule={lambda_schedule}"
        )

        ckpt_payload: Dict[str, Any] = {
            "backbone": model.backbone.state_dict(),
            "bottleneck": model.bottleneck.state_dict(),
            "classifier": model.classifier.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            **({"scaler": scaler.state_dict()} if amp_enabled else {}),
            "epoch": last_completed_epoch,
            "num_epochs": total_epochs,
            "adapt_batches_seen": adapt_batches_seen,
            "source_acc": float(avg_src_acc),
            "target_acc": float(tgt_acc),
            "completed": False,
            "config": asdict(config),
            "loss": float(avg_loss),
            "domain_loss_weight": float(domain_loss_weight),
            "lambda_schedule": lambda_schedule,
            "domain_loss": float(avg_dom_loss),
            "domain_acc": float(avg_dom_acc),
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
        "discriminator": discriminator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        **({"scaler": scaler.state_dict()} if amp_enabled else {}),
        "epoch": last_completed_epoch,
        "num_epochs": total_epochs,
        "adapt_batches_seen": adapt_batches_seen,
        "source_acc": float(src_acc),
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
    }
