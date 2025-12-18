from __future__ import annotations

import json
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from src.algorithms.base import unpack_wilds_batch


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Keep performance knobs (e.g. cudnn.benchmark) under caller control,
        # but enable deterministic convolution algorithms when available.
        torch.backends.cudnn.deterministic = True


def _extract_acc(metrics: Mapping[str, Any]) -> Optional[float]:
    for key in ("acc_avg", "accuracy", "acc"):
        if key in metrics:
            try:
                return float(metrics[key])
            except Exception:
                continue
    return None


@torch.no_grad()
def eval_wilds_split(
    *,
    algorithm: torch.nn.Module,
    loader: Any,
    dataset: Any,
    device: torch.device,
) -> Dict[str, Any]:
    algorithm.eval()
    y_pred = []
    y_true = []
    metadata = []

    for batch_raw in loader:
        batch = unpack_wilds_batch(batch_raw)
        x = batch.x.to(device, non_blocking=True)
        logits = algorithm.predict(x)  # type: ignore[attr-defined]
        y_pred.append(logits.detach().cpu())
        if batch.y is not None:
            y_true.append(batch.y.detach().cpu())
        if batch.metadata is not None:
            metadata.append(batch.metadata.detach().cpu())

    y_pred_t = torch.cat(y_pred, dim=0)
    y_true_t = torch.cat(y_true, dim=0) if y_true else None
    metadata_t = torch.cat(metadata, dim=0) if metadata else None

    if hasattr(dataset, "eval"):
        try:
            out = dataset.eval(y_pred_t.numpy(), None if y_true_t is None else y_true_t.numpy(), None if metadata_t is None else metadata_t.numpy())
        except Exception:
            out = dataset.eval(y_pred_t, y_true_t, metadata_t)
        return dict(out)

    if y_true_t is None:
        raise RuntimeError("Dataset has no eval(...) and labels are missing; cannot compute metrics.")
    acc = float((y_pred_t.argmax(dim=1) == y_true_t).float().mean().item())
    return {"acc": acc}


def _build_optimizer(algorithm: Any, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    opt_name = str(cfg.get("optimizer", "adamw")).lower()
    base_lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.0))

    lr_overrides = cfg.get("lr_overrides", None) or {}

    param_groups = []
    for group in algorithm.parameter_groups():  # type: ignore[attr-defined]
        g = dict(group)
        name = g.get("name")
        if name is not None and name in lr_overrides:
            g["lr"] = float(lr_overrides[name])
        else:
            g["lr"] = float(g.get("lr", base_lr))
        g["weight_decay"] = float(g.get("weight_decay", weight_decay))
        param_groups.append(g)

    if opt_name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        nesterov = bool(cfg.get("nesterov", True))
        return torch.optim.SGD(param_groups, lr=base_lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    if opt_name == "adam":
        return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)
    if opt_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{opt_name}'. Supported: adamw, adam, sgd.")


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _save_ckpt(
    path: Path,
    *,
    algorithm: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_val: float,
    cfg: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "cfg": _to_jsonable(cfg),
        "algorithm": algorithm.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": None if not torch.cuda.is_available() else torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(payload, path)


def _load_ckpt(
    path: Path,
    *,
    algorithm: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> Tuple[int, float]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        ckpt = torch.load(path, map_location="cpu")
    algorithm.load_state_dict(ckpt["algorithm"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = int(ckpt.get("epoch", 0))
    best_val = float(ckpt.get("best_val", float("-inf")))

    rng = ckpt.get("rng", {})
    try:
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    except Exception:
        pass
    return epoch, best_val


def train(
    *,
    cfg: Mapping[str, Any],
    run_dir: Path,
    algorithm: Any,
    wilds_dataset: Any,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    unlabeled_loader: Any | None = None,
    id_val_loader: Any | None = None,
) -> Dict[str, Any]:
    """
    Unified training loop for Camelyon17 experiments.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    results_path = run_dir / "results.json"
    cfg_path = run_dir / "config.json"

    _save_json(cfg_path, cfg)

    if results_path.exists() and best_path.exists() and not bool(cfg.get("force_rerun", False)):
        return json.loads(results_path.read_text(encoding="utf-8"))

    seed = int(cfg.get("seed", 0))
    deterministic = bool(cfg.get("deterministic", True))
    set_seed(seed, deterministic=deterministic)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    algorithm.to(device)

    amp_enabled = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler: Optional[torch.cuda.amp.GradScaler] = torch.cuda.amp.GradScaler(enabled=amp_enabled) if device.type == "cuda" else None
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()

    optimizer = _build_optimizer(algorithm, cfg)
    scheduler_name = str(cfg.get("scheduler", "none")).lower()
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.get("epochs", 1)))
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.get("step_size", 10)), gamma=float(cfg.get("gamma", 0.1)))

    start_epoch = 0
    best_val = float("-inf")
    if last_path.exists() and bool(cfg.get("resume", True)):
        start_epoch, best_val = _load_ckpt(last_path, algorithm=algorithm, optimizer=optimizer, scaler=scaler)
        start_epoch += 1

    epochs = int(cfg.get("epochs", 1))
    grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
    log_every = max(1, int(cfg.get("log_every", 100)))
    patience = int(cfg.get("early_stop_patience", 10))
    best_epoch = start_epoch - 1
    epochs_since_improve = 0

    meiis_weight_updates: list[dict[str, Any]] = []

    def _iter_unlabeled():
        if unlabeled_loader is None:
            while True:
                yield None
        while True:
            for b in unlabeled_loader:
                yield b

    unlabeled_iter = _iter_unlabeled()

    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        algorithm.train()

        if hasattr(algorithm, "update_importance_weights") and unlabeled_loader is not None:
            try:
                upd = algorithm.update_importance_weights(  # type: ignore[attr-defined]
                    source_loader=train_loader,
                    target_loader=unlabeled_loader,
                    device=device,
                )
                meiis_weight_updates.append({"epoch": epoch, **_to_jsonable(upd)})
            except Exception as exc:
                meiis_weight_updates.append({"epoch": epoch, "status": f"failed:{exc}"})

        optimizer.zero_grad(set_to_none=True)
        running = {"loss": 0.0, "acc": 0.0, "n": 0}
        step = 0

        for batch_raw in train_loader:
            unlabeled_batch = next(unlabeled_iter)
            with autocast_ctx:
                out = algorithm.update(batch_raw, unlabeled_batch)  # type: ignore[attr-defined]
                loss = out["loss"] / float(grad_accum_steps)

            if amp_enabled and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                if amp_enabled and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            bs = int(out.get("batch_size", 0) or out.get("batch_size_src", 0) or 0)
            running["loss"] += float(out["loss"].detach().item()) * max(1, bs)
            running["acc"] += float(out.get("acc", 0.0)) * max(1, bs)
            running["n"] += max(1, bs)

            if step % log_every == 0 and step > 0:
                avg_loss = running["loss"] / max(1, running["n"])
                avg_acc = running["acc"] / max(1, running["n"])
                print(f"[train] epoch={epoch} step={step} loss={avg_loss:.4f} acc={avg_acc:.4f}")
            step += 1

        # Flush any partial accumulation.
        if step % grad_accum_steps != 0:
            if amp_enabled and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

        val_metrics = eval_wilds_split(algorithm=algorithm, loader=val_loader, dataset=wilds_dataset, device=device)
        val_acc = _extract_acc(val_metrics)
        if val_acc is None:
            raise RuntimeError(f"Unable to extract val accuracy from metrics keys: {list(val_metrics.keys())}")

        print(f"[val] epoch={epoch} acc={val_acc:.4f}")

        improved = val_acc > best_val
        if improved:
            best_val = val_acc
            best_epoch = epoch
            _save_ckpt(best_path, algorithm=algorithm, optimizer=optimizer, scaler=scaler, epoch=epoch, best_val=best_val, cfg=cfg)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        _save_ckpt(last_path, algorithm=algorithm, optimizer=optimizer, scaler=scaler, epoch=epoch, best_val=best_val, cfg=cfg)

        if patience > 0 and epochs_since_improve >= patience:
            print(f"[early_stop] patience={patience} hit at epoch={epoch}; best_epoch={best_epoch} best_val={best_val:.4f}")
            break

    # Evaluate using best checkpoint (preferred) or last.
    if best_path.exists():
        _load_ckpt(best_path, algorithm=algorithm, optimizer=optimizer, scaler=scaler)
    elif last_path.exists():
        _load_ckpt(last_path, algorithm=algorithm, optimizer=optimizer, scaler=scaler)

    eval_id_val = None
    if id_val_loader is not None:
        eval_id_val = eval_wilds_split(algorithm=algorithm, loader=id_val_loader, dataset=wilds_dataset, device=device)
    eval_val = eval_wilds_split(algorithm=algorithm, loader=val_loader, dataset=wilds_dataset, device=device)
    eval_test = eval_wilds_split(algorithm=algorithm, loader=test_loader, dataset=wilds_dataset, device=device)

    results = {
        "run_id": str(cfg.get("run_id", "")),
        "status": "done",
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "metrics": {
            "id_val": eval_id_val,
            "val": eval_val,
            "test": eval_test,
        },
        "meiis": {"weight_updates": meiis_weight_updates},
        "wall_time_sec": float(time.time() - t0),
    }
    _save_json(results_path, results)
    return results
