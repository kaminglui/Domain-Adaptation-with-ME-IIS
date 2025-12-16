from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from datasets.domain_loaders import get_domain_loaders
from eval import evaluate
from models.classifier import build_model
from utils.data_utils import build_loader, make_generator, make_worker_init_fn
from utils.seed_utils import get_device, set_seed


def _infer_num_classes(loader: DataLoader) -> int:
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset  # type: ignore[attr-defined]
    if hasattr(dataset, "classes"):
        return len(dataset.classes)  # type: ignore[attr-defined]
    raise ValueError("Unable to infer number of classes from dataset.")


def _build_eval_loader(
    dataset_name: str,
    data_root: Path,
    domain: str,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    gen = make_generator(seed)
    worker_init = make_worker_init_fn(seed)
    _src_loader, _tgt_loader, eval_loader = get_domain_loaders(
        dataset_name=dataset_name,
        source_domain=domain,
        target_domain=domain,
        batch_size=batch_size,
        root=str(data_root),
        num_workers=num_workers,
        debug_classes=False,
        max_samples_per_domain=None,
        generator=gen,
        worker_init_fn=worker_init,
    )
    return build_loader(
        eval_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
        generator=gen,
        drop_last=False,
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_name: str,
    data_root: Path,
    domain: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    deterministic: bool,
) -> float:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    set_seed(seed, deterministic=deterministic)
    device = get_device(deterministic=deterministic)

    loader = _build_eval_loader(
        dataset_name=dataset_name,
        data_root=data_root,
        domain=domain,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    num_classes = _infer_num_classes(loader)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "backbone" not in ckpt or "bottleneck" not in ckpt or "classifier" not in ckpt:
        raise RuntimeError("Checkpoint missing 'backbone', 'bottleneck', and/or 'classifier' keys.")

    bottleneck_state = ckpt["bottleneck"]
    bottleneck_dim = None
    if isinstance(bottleneck_state, dict):
        w = bottleneck_state.get("fc.weight")
        if w is None:
            w = bottleneck_state.get("weight")
        if torch.is_tensor(w) and w.ndim == 2:
            bottleneck_dim = int(w.shape[0])
    if bottleneck_dim is None:
        raise RuntimeError("Unable to infer bottleneck_dim from checkpoint.")

    model = build_model(num_classes=num_classes, pretrained=False, bottleneck_dim=bottleneck_dim).to(device)
    model.backbone.load_state_dict(ckpt["backbone"], strict=False)
    model.bottleneck.load_state_dict(ckpt["bottleneck"], strict=False)
    model.classifier.load_state_dict(ckpt["classifier"], strict=False)
    acc, _ = evaluate(model, loader, device)
    return float(acc)


def evaluate_source_and_target(
    checkpoint_path: Path,
    dataset_name: str,
    data_root: Path,
    source_domain: str,
    target_domain: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    deterministic: bool,
) -> Tuple[float, float]:
    source_acc = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        dataset_name=dataset_name,
        data_root=data_root,
        domain=source_domain,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        deterministic=deterministic,
    )
    target_acc = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        dataset_name=dataset_name,
        data_root=data_root,
        domain=target_domain,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        deterministic=deterministic,
    )
    return source_acc, target_acc
