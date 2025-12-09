import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm  # auto selects notebook or CLI progress bars without ipywidgets

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.domain_loaders import DEFAULT_OFFICE31_ROOT, DEFAULT_OFFICE_HOME_ROOT, get_domain_loaders
from eval import evaluate
from models.classifier import build_model
from models.me_iis_adapter import IISIterationStats, MaxEntAdapter
from utils.data_utils import build_loader, make_generator, make_worker_init_fn
from utils.logging_utils import OFFICE_HOME_ME_IIS_FIELDS, append_csv
from utils.feature_utils import extract_features
from utils.env_utils import is_colab
from utils.seed_utils import get_device, set_seed
from torchvision import datasets


def _save_npz_safe(data: Dict[str, Any], path: Path) -> None:
    """
    Save numpy artifacts (e.g., IIS weights/history) with clear logging
    and basic error handling.
    """
    path = Path(os.fspath(path))
    out_dir = path.parent
    try:
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[IIS] Saving npz artifact to: {path}")
        sys.stdout.flush()
        np.savez(path, **data)
        print(f"[IIS] Done saving npz artifact: {path}")
    except Exception as e:
        print(f"[IIS][ERROR] Failed to save npz artifact to {path}: {e}")
        traceback.print_exc()


def _append_csv_safe(row: Dict[str, Any], path: Path, fieldnames: Iterable[str]) -> None:
    """
    Append a row to a CSV result file with logging and error handling.
    """
    path = Path(os.fspath(path))
    out_dir = path.parent
    try:
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        print(f"[RESULTS] Appending row to: {path}")
        sys.stdout.flush()
        append_csv(path=str(path), fieldnames=fieldnames, row=row)
        if is_new:
            # append_csv writes the header for new files.
            pass
        print(f"[RESULTS] Done appending row to: {path}")
    except Exception as e:
        print(f"[RESULTS][ERROR] Failed to append row to {path}: {e}")
        traceback.print_exc()


def _save_checkpoint_safe(checkpoint: Dict[str, Any], path: Path) -> None:
    """
    Save a checkpoint with logging and error handling for filesystem issues.
    """
    path = Path(os.fspath(path))
    ckpt_dir = path.parent
    try:
        if not ckpt_dir.exists():
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CKPT] Saving checkpoint to: {path}")
        sys.stdout.flush()
        torch.save(checkpoint, path)
        print(f"[CKPT] Done saving checkpoint: {path}")
    except Exception as e:
        print(f"[CKPT][ERROR] Failed to save checkpoint to {path}: {e}")
        traceback.print_exc()


def _build_adapt_ckpt_path(args: argparse.Namespace, layer_tag: str, epoch: Optional[int] = None) -> Path:
    # Use Windows-safe filenames: replace "src->tgt" with "src_to_tgt"
    base = f"me_iis_{args.source_domain}_to_{args.target_domain}_{layer_tag}_seed{args.seed}"
    if epoch is not None:
        base += f"_epoch{epoch}"
    return Path("checkpoints") / f"{base}.pth"


class IndexedDataset(Dataset):
    """Wrap dataset to also return index for weight lookup."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label, idx


class PseudoLabeledDataset(Subset):
    """Subset wrapper that replaces labels with provided pseudo labels."""

    def __init__(self, base_dataset: Dataset, indices: List[int], pseudo_labels: List[int]):
        super().__init__(base_dataset, indices)
        if len(indices) != len(pseudo_labels):
            raise ValueError("Length of indices and pseudo_labels must match.")
        self.pseudo_labels = [int(l) for l in pseudo_labels]

    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        label = self.pseudo_labels[idx]
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.to(dtype=torch.long)
        return image, label


def _parse_feature_layers(layer_str: str) -> Tuple[str, ...]:
    layers = tuple([l.strip() for l in layer_str.split(",") if l.strip()])
    if not layers:
        raise ValueError("At least one feature layer must be specified (e.g., 'layer3,layer4,avgpool').")
    return layers


def _build_component_map(
    feature_layers: Iterable[str], override: str, default_components: int
) -> Dict[str, int]:
    comp_map = {layer: default_components for layer in feature_layers}
    if override:
        for item in override.split(","):
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Invalid components_per_layer entry '{item}'. Use 'layer:count'.")
            name, count = item.split(":")
            name = name.strip()
            if name not in comp_map:
                raise ValueError(f"Got components override for unknown layer '{name}'.")
            comp_map[name] = int(count)
    for layer, count in comp_map.items():
        if count <= 0:
            raise ValueError(f"Number of components for {layer} must be positive, got {count}.")
    return comp_map


def _num_classes_from_dataset(dataset: Dataset) -> int:
    base = dataset
    while isinstance(base, Subset):
        base = base.dataset  # type: ignore
    if hasattr(base, "classes"):
        return len(base.classes)  # type: ignore
    if hasattr(base, "dataset") and hasattr(base.dataset, "classes"):
        return len(base.dataset.classes)  # type: ignore
    raise ValueError("Unable to infer number of classes from dataset.")


def _dataset_tag(name: str) -> str:
    if name == "office_home":
        return "office-home"
    if name == "office31":
        return "office-31"
    return name


def _normalize_dataset_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def _resolve_officehome_root_from(base: Path) -> Path:
    """Best-effort to find the Office-Home root inside KaggleHub download folder."""
    if not base.exists() or not base.is_dir():
        return base
    realworld_candidates = ["RealWorld", "Real World", "Real_World", "Real"]
    candidates = [base] + [p for p in base.iterdir() if p.is_dir()]
    for cand in candidates:
        if all((cand / sub).exists() for sub in ["Art", "Clipart", "Product"]) and any(
            (cand / rw).exists() for rw in realworld_candidates
        ):
            return cand
    return base


def _resolve_office31_root_from(base: Path) -> Path:
    """Best-effort to find the Office-31 root inside KaggleHub download folder."""
    if not base.exists() or not base.is_dir():
        return base
    candidates = [base] + [p for p in base.iterdir() if p.is_dir()]
    for cand in candidates:
        if all((cand / sub).exists() for sub in ["amazon", "dslr", "webcam"]):
            return cand
    return base


def _maybe_resolve_data_root(args) -> str:
    """
    Pick a dataset root based on user input, environment (Colab vs. local), and defaults.
    - Respect an existing explicit --data_root.
    - On Colab, download via KaggleHub.
    - Otherwise, fall back to repository defaults.
    """
    if args.data_root:
        explicit = Path(args.data_root)
        if explicit.exists():
            return str(explicit)
        print(f"[DATA][WARN] Provided data_root does not exist: {explicit}. Falling back to defaults.")

    name = _normalize_dataset_name(args.dataset_name)

    if is_colab():
        try:
            import kagglehub  # type: ignore
        except ImportError:
            import os

            os.system("pip install kagglehub")  # best-effort install in Colab
            import kagglehub  # type: ignore

        if name == "officehome":
            root_path = Path(kagglehub.dataset_download("lhrrraname/officehome"))
            root_path = _resolve_officehome_root_from(root_path)
        elif name == "office31":
            root_path = Path(kagglehub.dataset_download("xixuhu/office31"))
            root_path = _resolve_office31_root_from(root_path)
        else:
            raise ValueError(f"Unknown dataset_name '{args.dataset_name}' for KaggleHub resolution.")
        print(f"[DATA] Resolved dataset '{args.dataset_name}' via KaggleHub at: {root_path}")
        return str(root_path)

    if name == "officehome":
        return str(Path(DEFAULT_OFFICE_HOME_ROOT))
    if name == "office31":
        return str(Path(DEFAULT_OFFICE31_ROOT))
    raise ValueError(f"Unknown dataset_name '{args.dataset_name}'. Expected 'office_home' or 'office31'.")


def _class_probs_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, mode: str, num_classes: int
) -> torch.Tensor:
    """Return P(Ĉ|x) to use in IIS constraints."""
    if mode == "softmax":
        return F.softmax(logits, dim=1)
    if mode == "onehot":
        return F.one_hot(labels.to(torch.int64), num_classes=num_classes).float().to(logits.device)
    raise ValueError(f"Unknown source_prob_mode '{mode}'")


@torch.no_grad()
def build_pseudo_label_dataset(
    model: nn.Module,
    target_loader: DataLoader,
    device: torch.device,
    conf_thresh: float,
    max_ratio: float,
    num_source_samples: int,
) -> Tuple[List[int], List[int]]:
    """
    Generate pseudo labels for target samples with confidence above conf_thresh.
    Returns indices relative to target_loader.dataset and corresponding labels.
    """
    was_training = model.training
    model.eval()
    candidates: List[Tuple[float, int, int]] = []

    max_keep = None
    if max_ratio >= 0:
        max_keep = int(max_ratio * float(num_source_samples))
        if max_keep <= 0:
            if was_training:
                model.train()
            return [], []

    running_idx = 0
    for batch in target_loader:
        if len(batch) == 2:
            images, _ = batch
        else:
            images = batch[0]
        images = images.to(device)
        logits, _ = model(images, return_features=False)
        probs = F.softmax(logits, dim=1)
        max_prob, pred = probs.max(dim=1)
        bs = images.size(0)
        for i in range(bs):
            dataset_idx = running_idx + i
            if float(max_prob[i].item()) >= conf_thresh:
                candidates.append((float(max_prob[i].item()), dataset_idx, int(pred[i].item())))
        running_idx += bs

    if was_training:
        model.train()

    if not candidates:
        return [], []

    candidates.sort(key=lambda x: x[0], reverse=True)
    if max_keep is not None and max_keep > 0:
        candidates = candidates[:max_keep]

    pseudo_indices = [c[1] for c in candidates]
    pseudo_labels = [c[2] for c in candidates]
    return pseudo_indices, pseudo_labels


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
    """
    One adaptation epoch combining weighted source loss with optional pseudo-labeled target loss.
    Returns average loss, source accuracy, batches seen, pseudo samples used, and pseudo samples available.
    """
    model.train()
    total_loss, total_acc, total_src = 0.0, 0.0, 0
    batches_seen = 0
    pseudo_used = 0
    pseudo_total = len(pseudo_loader.dataset) if pseudo_loader is not None else 0
    pseudo_iter = iter(pseudo_loader) if pseudo_loader is not None else None

    for images, labels, idxs in tqdm(source_loader, desc="Adapt", leave=False):
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
        total_loss += loss.item() * (bs + pseudo_bs)
        total_acc += acc * bs
        total_src += bs
        batches_seen += 1
        if max_batches > 0 and batches_seen >= max_batches:
            break

    denom = total_src + pseudo_used if (pseudo_loader is not None and (total_src + pseudo_used) > 0) else total_src
    avg_loss = total_loss / denom if denom > 0 else 0.0
    src_acc = total_acc / total_src if total_src > 0 else 0.0
    return avg_loss, src_acc, batches_seen, pseudo_used, pseudo_total


def weighted_train(
    model: nn.Module,
    loader: DataLoader,
    weights: torch.Tensor,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, float, int]:
    loss, acc, batches, _, _ = adapt_epoch(
        model=model,
        optimizer=optimizer,
        source_loader=loader,
        source_weights_vec=weights,
        device=device,
        max_batches=max_batches,
        pseudo_loader=None,
        pseudo_loss_weight=1.0,
    )
    return loss, acc, batches


def save_iis_history(path: Path, weights: torch.Tensor, history: List[IISIterationStats]) -> None:
    _save_npz_safe(
        {
            "weights": weights.cpu().numpy(),
            "delta_norm": [h.delta_norm for h in history],
            "kl": [h.kl_moments for h in history],
            "moment_max": [h.max_moment_error for h in history],
            "moment_mean": [h.mean_moment_error for h in history],
            "moment_l2": [h.l2_moment_error for h in history],
            "w_min": [h.weight_min for h in history],
            "w_max": [h.weight_max for h in history],
            "w_entropy": [h.weight_entropy for h in history],
            "feature_mass_min": [h.feature_mass_min for h in history],
            "feature_mass_max": [h.feature_mass_max for h in history],
            "feature_mass_mean": [h.feature_mass_mean for h in history],
            "feature_mass_std": [h.feature_mass_std for h in history],
        },
        path,
    )


def adapt_me_iis(args) -> None:
    args.data_root = _maybe_resolve_data_root(args)
    data_root = Path(args.data_root) if args.data_root else None
    if data_root is not None and not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    print(f"[Seed] Using seed {args.seed} (deterministic={args.deterministic})")
    if args.dry_run_max_batches > 0:
        print(f"[DRY RUN] Limiting feature extraction and adaptation to {args.dry_run_max_batches} batches.")
    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device(deterministic=args.deterministic)
    data_generator = make_generator(args.seed)
    worker_init = make_worker_init_fn(args.seed)
    feature_layers = _parse_feature_layers(args.feature_layers)
    components_map = _build_component_map(feature_layers, args.components_per_layer, args.num_latent_styles)
    components_str = ",".join([str(components_map[layer]) for layer in feature_layers])
    total_components = sum(components_map.values())
    layer_tag = "-".join(feature_layers)
    adapt_ckpt_path = _build_adapt_ckpt_path(args, layer_tag)
    if getattr(args, "resume_adapt_from", None) in (None, "") and adapt_ckpt_path.exists():
        args.resume_adapt_from = str(adapt_ckpt_path)
        print(f"[CKPT] Auto-resume: found existing ME-IIS checkpoint at {adapt_ckpt_path}")
    print(
        f"[Config] dataset={args.dataset_name} layers={layer_tag} components_per_layer={components_str} "
        f"finetune_backbone={args.finetune_backbone} backbone_lr_scale={args.backbone_lr_scale} "
        f"classifier_lr={args.classifier_lr} source_prob_mode={args.source_prob_mode} "
        f"iis_iters={args.iis_iters} iis_tol={args.iis_tol} "
        f"gmm_selection_mode={args.gmm_selection_mode} bic_range=[{args.gmm_bic_min_components},"
        f"{args.gmm_bic_max_components}] "
        f"dry_run_max_samples={args.dry_run_max_samples} dry_run_max_batches={args.dry_run_max_batches}"
    )
    print(
        f"[ADAPT] Starting ME-IIS adaptation: dataset={args.dataset_name}, "
        f"source={args.source_domain}, target={args.target_domain}, "
        f"dry_run_max_batches={args.dry_run_max_batches}"
    )

    source_loader, _, target_eval_loader = get_domain_loaders(
        dataset_name=args.dataset_name,
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        batch_size=args.batch_size,
        root=str(data_root) if data_root is not None else None,
        num_workers=args.num_workers,
        debug_classes=False,
        max_samples_per_domain=args.dry_run_max_samples if args.dry_run_max_samples > 0 else None,
        generator=data_generator,
        worker_init_fn=worker_init,
    )

    source_train_raw = source_loader.dataset
    target_eval_raw = target_eval_loader.dataset

    if not hasattr(source_train_raw, "class_to_idx") or not hasattr(target_eval_raw, "class_to_idx"):
        raise ValueError("Domain datasets must expose class_to_idx for ME-IIS.")
    if source_train_raw.class_to_idx != target_eval_raw.class_to_idx:  # type: ignore
        raise ValueError("Source and target class mappings differ from canonical mapping.")
    eval_transform = getattr(target_eval_raw, "transform", None)
    if eval_transform is None:
        raise ValueError("Target evaluation dataset is missing a transform.")
    source_feat_raw = datasets.ImageFolder(source_train_raw.root, transform=eval_transform)
    if source_feat_raw.class_to_idx != source_train_raw.class_to_idx:  # type: ignore
        raise ValueError("Source eval dataset mapping does not match canonical mapping.")
    target_feat_raw = target_eval_raw  # deterministic transforms only

    sample_cap: Optional[int] = args.dry_run_max_samples if args.dry_run_max_samples > 0 else None
    if args.dry_run_max_batches > 0:
        batch_cap = args.dry_run_max_batches * args.batch_size
        sample_cap = batch_cap if sample_cap is None else min(sample_cap, batch_cap)
    if sample_cap is not None:
        sample_cap = max(1, sample_cap)
        print(f"[DryRun] Limiting datasets to first {sample_cap} samples.")
        src_idx = list(range(min(sample_cap, len(source_train_raw))))
        tgt_idx = list(range(min(sample_cap, len(target_eval_raw))))
        if not src_idx or not tgt_idx:
            raise ValueError("Dry-run sample cap produced an empty dataset; increase limits.")
        source_train_ds = Subset(source_train_raw, src_idx)
        source_feat_ds = Subset(source_feat_raw, src_idx)
        target_eval_ds = Subset(target_eval_raw, tgt_idx)
        target_feat_ds = Subset(target_feat_raw, tgt_idx)
    else:
        source_train_ds = source_train_raw
        source_feat_ds = source_feat_raw
        target_eval_ds = target_eval_raw
        target_feat_ds = target_feat_raw

    num_classes = _num_classes_from_dataset(source_train_ds)
    model = build_model(num_classes=num_classes, pretrained=True).to(device)
    source_ckpt = torch.load(args.checkpoint, map_location=device)
    model.backbone.load_state_dict(source_ckpt["backbone"])
    model.classifier.load_state_dict(source_ckpt["classifier"])

    target_eval_loader = build_loader(
        target_eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        generator=data_generator,
        drop_last=False,
    )
    baseline_acc, _ = evaluate(model, target_eval_loader, device)
    print(f"Baseline source-only target acc: {baseline_acc:.2f}")

    print("[Audit] Target labels are ignored during IIS and adaptation training.")
    print("[Audit] Target constraint loader uses deterministic transforms (train=False).")
    source_feat_loader = build_loader(
        source_feat_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        generator=data_generator,
        drop_last=False,
    )
    target_feat_loader = build_loader(
        target_feat_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        generator=data_generator,
        drop_last=False,
    )

    with torch.no_grad():
        source_feats, source_logits, source_labels = extract_features(
            model,
            source_feat_loader,
            device,
            feature_layers,
            max_batches=args.dry_run_max_batches,
        )
        target_feats, target_logits, target_labels = extract_features(
            model,
            target_feat_loader,
            device,
            feature_layers,
            max_batches=args.dry_run_max_batches,
        )
        source_probs = _class_probs_from_logits(
            logits=source_logits.to(device),
            labels=source_labels.to(device),
            mode=args.source_prob_mode,
            num_classes=num_classes,
        )
        target_probs = F.softmax(target_logits.to(device), dim=1)
        if target_labels.numel() > 0:
            print("[Audit] Dropping target labels during IIS (unsupervised adaptation).")
        del target_labels

    adapter = MaxEntAdapter(
        num_classes=num_classes,
        layers=list(feature_layers),
        components_per_layer=components_map,
        device=device,
        seed=args.seed,
        gmm_selection_mode=args.gmm_selection_mode,
        gmm_bic_min_components=args.gmm_bic_min_components,
        gmm_bic_max_components=args.gmm_bic_max_components,
    )
    adapter.fit_target_structure({k: v.to(device) for k, v in target_feats.items()})
    components_map = dict(adapter.components_per_layer)
    components_str = ",".join([str(components_map[layer]) for layer in feature_layers])
    total_components = sum(components_map.values())
    print(f"[GMM] Final components_per_layer after selection: {components_str}")
    weights, history = adapter.solve_iis(
        source_layer_feats={k: v.to(device) for k, v in source_feats.items()},
        source_class_probs=source_probs,
        target_layer_feats={k: v.to(device) for k, v in target_feats.items()},
        target_class_probs=target_probs,
        max_iter=args.iis_iters,
        iis_tol=args.iis_tol,
    )
    weights = weights.cpu()

    if len(weights) != len(source_train_ds):
        raise ValueError("IIS weight vector length does not match source training set.")
    print("[ADAPT] IIS solve completed, saving artifacts and running weighted fine-tuning...")
    history_path = (
        Path("results") / f"me_iis_weights_{args.source_domain}_to_{args.target_domain}_{layer_tag}_seed{args.seed}.npz"
    )
    save_iis_history(history_path, weights, history)

    if not args.finetune_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    backbone_lr = args.classifier_lr * args.backbone_lr_scale
    backbone_lr_value = backbone_lr if args.finetune_backbone else 0.0
    print(f"[Adapt] classifier_lr={args.classifier_lr:.4e}, backbone_lr={backbone_lr_value:.4e}")
    def _make_optimizer_and_scheduler(current_model: nn.Module):
        param_groups = [
            {"params": [p for p in current_model.backbone.parameters() if p.requires_grad], "lr": backbone_lr},
            {"params": list(current_model.classifier.parameters()), "lr": args.classifier_lr},
        ]
        param_groups = [p for p in param_groups if len(p["params"]) > 0]
        opt = optim.SGD(param_groups, momentum=0.9, weight_decay=args.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.adapt_epochs))
        return opt, sched

    optimizer, scheduler = _make_optimizer_and_scheduler(model)

    weighted_source_ds = source_train_ds
    weighted_loader = build_loader(
        IndexedDataset(weighted_source_ds),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        generator=data_generator,
        drop_last=False,
    )
    pseudo_loader: Optional[DataLoader] = None
    if args.use_pseudo_labels:
        pseudo_indices, pseudo_labels = build_pseudo_label_dataset(
            model=model,
            target_loader=target_feat_loader,
            device=device,
            conf_thresh=args.pseudo_conf_thresh,
            max_ratio=args.pseudo_max_ratio,
            num_source_samples=len(weighted_source_ds),
        )
        if len(pseudo_indices) > 0:
            pseudo_ds = PseudoLabeledDataset(target_feat_loader.dataset, pseudo_indices, pseudo_labels)
            pseudo_loader = build_loader(
                pseudo_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                seed=args.seed,
                generator=data_generator,
                drop_last=False,
            )
            print(f"[PL] Using {len(pseudo_indices)} pseudo-labelled target samples with conf_thresh={args.pseudo_conf_thresh}.")
        else:
            print("[PL] No pseudo-labelled target samples passed the confidence threshold.")
    acc = 0.0
    start_epoch = 0
    last_completed_epoch = -1
    adapt_batches_seen = 0
    scheduler_loaded = False
    resume_success = False
    if args.resume_adapt_from:
        resume_path = Path(args.resume_adapt_from)
        if resume_path.exists():
            try:
                resume_ckpt = torch.load(resume_path, map_location=device)
                backbone_state = resume_ckpt.get("backbone")
                classifier_state = resume_ckpt.get("classifier")
                if backbone_state is None or classifier_state is None:
                    raise RuntimeError("Adaptation checkpoint missing backbone/classifier state.")
                backbone_res = model.backbone.load_state_dict(backbone_state, strict=False)
                classifier_res = model.classifier.load_state_dict(classifier_state, strict=False)
                if backbone_res.missing_keys or backbone_res.unexpected_keys:
                    raise RuntimeError(
                        f"Backbone state incompatible (missing: {backbone_res.missing_keys}, "
                        f"unexpected: {backbone_res.unexpected_keys})"
                    )
                if classifier_res.missing_keys or classifier_res.unexpected_keys:
                    raise RuntimeError(
                        f"Classifier state incompatible (missing: {classifier_res.missing_keys}, "
                        f"unexpected: {classifier_res.unexpected_keys})"
                    )
                if "optimizer" in resume_ckpt:
                    optimizer.load_state_dict(resume_ckpt["optimizer"])
                if "scheduler" in resume_ckpt:
                    try:
                        scheduler.load_state_dict(resume_ckpt["scheduler"])
                        scheduler_loaded = True
                    except Exception as sched_err:
                        print(f"[CKPT][WARN] Could not load adaptation scheduler state: {sched_err}")
                adapt_batches_seen = int(resume_ckpt.get("adapt_batches_seen", adapt_batches_seen))
                last_completed_epoch = int(resume_ckpt.get("epoch", last_completed_epoch))
                acc = float(resume_ckpt.get("source_acc", acc))
                start_epoch = max(last_completed_epoch + 1, 0)
                resume_success = True
                print(
                    f"[CKPT] Resuming adaptation from {resume_path} "
                    f"(start_epoch={start_epoch + 1}, adapt_batches_seen={adapt_batches_seen})"
                )
            except Exception as exc:
                print(f"[CKPT][WARN] Failed to load adaptation checkpoint '{resume_path}': {exc}. Starting fresh.")
                traceback.print_exc()
        else:
            print(f"[CKPT][WARN] Adaptation resume checkpoint not found: {resume_path}. Starting fresh.")
    if not resume_success and args.resume_adapt_from:
        model = build_model(num_classes=num_classes, pretrained=True).to(device)
        model.backbone.load_state_dict(source_ckpt["backbone"])
        model.classifier.load_state_dict(source_ckpt["classifier"])
        if not args.finetune_backbone:
            for p in model.backbone.parameters():
                p.requires_grad = False
        optimizer, scheduler = _make_optimizer_and_scheduler(model)
        start_epoch = 0
        last_completed_epoch = -1
        adapt_batches_seen = 0
        scheduler_loaded = False
        acc = 0.0
    if not scheduler_loaded and start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    def _build_adapt_checkpoint(epoch_idx: int) -> Dict[str, Any]:
        return {
            "backbone": model.backbone.state_dict(),
            "classifier": model.classifier.state_dict(),
            "weights": weights.cpu(),
            "history": [h.__dict__ for h in history],
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch_idx,
            "adapt_batches_seen": adapt_batches_seen,
            "source_acc": acc,
            "args": vars(args),
        }

    writer = SummaryWriter(log_dir="runs/adapt_me_iis")
    log_dir = writer.log_dir
    try:
        for epoch in tqdm(range(start_epoch, args.adapt_epochs), desc="Adapt Epoch", leave=True):
            remaining_batches = 0
            if args.dry_run_max_batches > 0:
                remaining_batches = max(args.dry_run_max_batches - adapt_batches_seen, 0)
                if remaining_batches == 0:
                    print("[DRY RUN] Reached adaptation batch budget; stopping early.")
                    break
            loss, acc, batches_used, pseudo_used, pseudo_total = adapt_epoch(
                model=model,
                optimizer=optimizer,
                source_loader=weighted_loader,
                source_weights_vec=weights,
                device=device,
                max_batches=remaining_batches,
                pseudo_loader=pseudo_loader,
                pseudo_loss_weight=args.pseudo_loss_weight,
            )
            adapt_batches_seen += batches_used
            scheduler.step()
            target_acc, _ = evaluate(model, target_eval_loader, device)
            writer.add_scalar("Loss/adapt", loss, epoch)
            writer.add_scalar("Accuracy/source", acc, epoch)
            writer.add_scalar("Accuracy/target", target_acc, epoch)
            pseudo_frac = pseudo_used / pseudo_total if pseudo_total > 0 else 0.0
            if pseudo_total > 0:
                writer.add_scalar("Pseudo/fraction_used", pseudo_frac, epoch)
            print(
                f"[Adapt] Epoch {epoch+1}/{args.adapt_epochs} | Loss {loss:.4f} | "
                f"Source Acc {acc:.2f} | Target Acc {target_acc:.2f} | "
                f"Pseudo frac {pseudo_frac:.2f} (w={args.pseudo_loss_weight:.2f})"
            )
            last_completed_epoch = epoch
            if args.save_adapt_every > 0 and (epoch + 1) % args.save_adapt_every == 0:
                epoch_ckpt = _build_adapt_ckpt_path(args, layer_tag, epoch=epoch + 1)
                _save_checkpoint_safe(_build_adapt_checkpoint(last_completed_epoch), epoch_ckpt)
            if args.dry_run_max_batches > 0 and adapt_batches_seen >= args.dry_run_max_batches:
                print("[DRY RUN] Exhausted adaptation batch budget; exiting after this epoch.")
                break

        adapted_acc, _ = evaluate(model, target_eval_loader, device)
        print(f"Adapted target acc: {adapted_acc:.2f}")
        final_moment_err = history[-1].max_moment_error if history else float("nan")
        final_entropy = history[-1].weight_entropy if history else float("nan")
        print(
            f"[Summary] Baseline target acc: {baseline_acc:.2f} | Adapted target acc: {adapted_acc:.2f} | "
            f"Max IIS moment err: {final_moment_err:.4e} | Entropy H(Q): {final_entropy:.4f} | "
            f"Layers={layer_tag} comps={components_str}"
        )

        adapt_ckpt = _build_adapt_ckpt_path(args, layer_tag)
        _save_checkpoint_safe(_build_adapt_checkpoint(last_completed_epoch), adapt_ckpt)

        dataset_field = _dataset_tag(args.dataset_name)
        method_tag = "me_iis"
        if args.use_pseudo_labels:
            method_tag = "me_iis_pl"
        elif args.gmm_selection_mode == "bic":
            method_tag = "me_iis_bic"
        _append_csv_safe(
            row={
                "dataset": dataset_field,
                "source": args.source_domain,
                "target": args.target_domain,
                "seed": args.seed,
                "method": method_tag,
                "target_acc": round(adapted_acc, 4),
                "source_acc": round(acc, 4),
                "num_latent": total_components,
                "layers": layer_tag,
                "components_per_layer": components_str,
                "iis_iters": args.iis_iters,
                "iis_tol": args.iis_tol,
                "adapt_epochs": args.adapt_epochs,
                "finetune_backbone": args.finetune_backbone,
                "backbone_lr_scale": args.backbone_lr_scale,
                "classifier_lr": args.classifier_lr,
                "source_prob_mode": args.source_prob_mode,
            },
            path=Path("results/office_home_me_iis.csv"),
            fieldnames=OFFICE_HOME_ME_IIS_FIELDS,
        )
        print("[ADAPT] Weighted adaptation complete.")
    finally:
        writer.close()
        print(f"[TENSORBOARD] Logs written to {log_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="ME-IIS domain adaptation on Office-Home or Office-31.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="office_home",
        choices=["office_home", "office31"],
        help="Which benchmark to use (Office-Home or Office-31).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=(
            f"Path to dataset root (defaults: Office-Home -> {DEFAULT_OFFICE_HOME_ROOT}, "
            f"Office-31 -> {DEFAULT_OFFICE31_ROOT})."
        ),
    )
    parser.add_argument("--source_domain", type=str, required=True)
    parser.add_argument("--target_domain", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to source-only checkpoint.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_latent_styles", type=int, default=5)
    parser.add_argument(
        "--components_per_layer",
        type=str,
        default=None,
        help="Optional comma-separated overrides 'layer:count,...' for per-layer mixture components.",
    )
    parser.add_argument(
        "--gmm_selection_mode",
        type=str,
        default="fixed",
        choices=["fixed", "bic"],
        help="How to choose the number of GMM components per layer for ME-IIS.",
    )
    parser.add_argument(
        "--gmm_bic_min_components",
        type=int,
        default=2,
        help="Minimum number of mixture components per layer when using BIC selection.",
    )
    parser.add_argument(
        "--gmm_bic_max_components",
        type=int,
        default=8,
        help="Maximum number of mixture components per layer when using BIC selection.",
    )
    parser.add_argument(
        "--feature_layers",
        type=str,
        default="layer3,layer4,avgpool",
        help="Comma-separated ResNet-50 layers to include in ME-IIS constraints.",
    )
    parser.add_argument(
        "--source_prob_mode",
        type=str,
        default="softmax",
        choices=["softmax", "onehot"],
        help="How to form P(Ĉ|x) on source for constraints. Default uses model softmax (spec), "
        "onehot uses ground-truth labels for stability.",
    )
    parser.add_argument("--iis_iters", type=int, default=15)
    parser.add_argument("--iis_tol", type=float, default=1e-3, help="Tolerance for IIS max abs moment error.")
    parser.add_argument("--adapt_epochs", type=int, default=10)
    parser.add_argument(
        "--resume_adapt_from",
        type=str,
        default=None,
        help="Optional ME-IIS adaptation checkpoint to resume from.",
    )
    parser.add_argument(
        "--save_adapt_every",
        type=int,
        default=0,
        help="If >0, save ME-IIS adaptation checkpoint every N epochs.",
    )
    parser.add_argument(
        "--finetune_backbone", action="store_true", help="Fine-tune backbone during adaptation (otherwise frozen)."
    )
    parser.add_argument(
        "--backbone_lr_scale",
        type=float,
        default=0.1,
        help="Backbone LR is classifier_lr * backbone_lr_scale when finetuning.",
    )
    parser.add_argument("--classifier_lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument(
        "--use_pseudo_labels",
        action="store_true",
        help="If set, include high-confidence pseudo-labelled target samples during ME-IIS adaptation.",
    )
    parser.add_argument(
        "--pseudo_conf_thresh",
        type=float,
        default=0.9,
        help="Confidence threshold for creating pseudo labels on target (max softmax >= this).",
    )
    parser.add_argument(
        "--pseudo_max_ratio",
        type=float,
        default=1.0,
        help="Max size of pseudo-labelled target set as a ratio of the source sample count.",
    )
    parser.add_argument(
        "--pseudo_loss_weight",
        type=float,
        default=1.0,
        help="Multiplicative weight for the pseudo-labelled target loss term.",
    )
    parser.add_argument("--dry_run_max_samples", type=int, default=0, help="Limit samples for quick dry-runs.")
    parser.add_argument(
        "--dry_run_max_batches",
        type=int,
        default=0,
        help="If >0, limit both feature extraction and adaptation training to this many batches per phase.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic/cuDNN safe settings (pair with --seed for reproducibility).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    adapt_me_iis(args)
