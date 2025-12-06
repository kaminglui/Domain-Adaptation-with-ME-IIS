import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.domain_loaders import DEFAULT_OFFICE31_ROOT, DEFAULT_OFFICE_HOME_ROOT, get_domain_loaders
from eval import evaluate
from models.classifier import build_model
from utils.data_utils import build_loader, make_generator
from utils.logging_utils import OFFICE_HOME_ME_IIS_FIELDS, append_csv
from utils.env_utils import is_colab
from utils.seed_utils import get_device, set_seed


def _save_checkpoint_safe(checkpoint: Dict[str, Any], path: Path) -> None:
    """
    Save a checkpoint with robust logging and basic error handling.
    Ensures the parent directory exists and prints clear status messages.
    """
    path = Path(path)
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


def _build_source_ckpt_path(args: argparse.Namespace) -> Path:
    # Use Windows-safe filenames: replace "src->tgt" with "src_to_tgt"
    return Path("checkpoints") / f"source_only_{args.source_domain}_to_{args.target_domain}_seed{args.seed}.pth"


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == labels).float().mean().item() * 100.0)


def _infer_num_classes(loader: DataLoader) -> int:
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset  # type: ignore
    if hasattr(dataset, "classes"):
        return len(dataset.classes)  # type: ignore
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
    if not base.exists() or not base.is_dir():
        return base
    candidates = [base] + [p for p in base.iterdir() if p.is_dir()]
    for cand in candidates:
        if all((cand / sub).exists() for sub in ["amazon", "dslr", "webcam"]):
            return cand
    return base


def _maybe_resolve_data_root(args) -> str:
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, float, int]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    batches_seen = 0
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        if batch_idx == 0:
            print(f"[TRAIN] Got first batch: batch_idx={batch_idx}, batch_size={images.size(0)}")
            sys.stdout.flush()
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(images, return_features=False)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_acc += compute_accuracy(logits.detach(), labels) * batch_size
        total += batch_size
        batches_seen += 1
        if max_batches > 0 and batches_seen >= max_batches:
            break
    if total == 0:
        return 0.0, 0.0, batches_seen
    return running_loss / total, running_acc / total, batches_seen


def train_source(args) -> None:
    args.data_root = _maybe_resolve_data_root(args)
    data_root = Path(args.data_root) if args.data_root else None
    if data_root is not None and not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    ckpt_path = _build_source_ckpt_path(args)
    if getattr(args, "resume_from", None) in (None, "") and ckpt_path.exists():
        args.resume_from = str(ckpt_path)
        print(f"[CKPT] Auto-resume: found existing source checkpoint at {ckpt_path}")

    print(f"[Seed] Using seed {args.seed} (deterministic={args.deterministic})")
    if args.dry_run_max_batches > 0:
        print(f"[DRY RUN] Limiting training to {args.dry_run_max_batches} total batches.")
    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device(deterministic=args.deterministic)
    data_generator = make_generator(args.seed)
    source_loader, _, target_eval_loader = get_domain_loaders(
        dataset_name=args.dataset_name,
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        batch_size=args.batch_size,
        root=str(data_root) if data_root is not None else None,
        num_workers=args.num_workers,
        debug_classes=False,
        max_samples_per_domain=args.dry_run_max_samples if args.dry_run_max_samples > 0 else None,
    )
    source_ds = source_loader.dataset
    target_eval_ds = target_eval_loader.dataset
    source_loader = build_loader(
        source_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        generator=data_generator,
        drop_last=False,
    )
    target_eval_loader = build_loader(
        target_eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        generator=data_generator,
        drop_last=False,
    )
    print(f"[DEBUG] Built loaders: len(source_loader)={len(source_loader)}, len(target_eval_loader)={len(target_eval_loader)}")
    sys.stdout.flush()

    num_classes = _infer_num_classes(source_loader)
    model = build_model(num_classes=num_classes, pretrained=True).to(device)

    def _make_optimizer_and_scheduler(current_model: nn.Module):
        params = [
            {"params": current_model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": current_model.classifier.parameters(), "lr": args.lr_classifier},
        ]
        opt = optim.SGD(params, momentum=0.9, weight_decay=args.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)
        return opt, sched

    optimizer, scheduler = _make_optimizer_and_scheduler(model)
    writer = SummaryWriter(log_dir="runs/source_only")
    log_dir = writer.log_dir

    start_epoch = 0
    last_completed_epoch = -1
    scheduler_loaded = False
    best_target_acc = 0.0
    final_source_acc = 0.0
    num_batches_seen = 0
    resume_success = False
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            try:
                ckpt = torch.load(resume_path, map_location=device)
                backbone_state = ckpt.get("backbone")
                classifier_state = ckpt.get("classifier")
                if backbone_state is None or classifier_state is None:
                    raise RuntimeError("Checkpoint missing backbone/classifier state.")
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
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt:
                    try:
                        scheduler.load_state_dict(ckpt["scheduler"])
                        scheduler_loaded = True
                    except Exception as sched_err:
                        print(f"[CKPT][WARN] Could not load scheduler state: {sched_err}")
                best_target_acc = float(ckpt.get("best_target_acc", best_target_acc))
                final_source_acc = float(ckpt.get("source_acc", final_source_acc))
                last_completed_epoch = int(ckpt.get("epoch", last_completed_epoch))
                start_epoch = max(last_completed_epoch + 1, 0)
                resume_success = True
                print(
                    f"[CKPT] Resuming from checkpoint: {resume_path} "
                    f"(start_epoch={start_epoch + 1}, best_target_acc={best_target_acc:.2f})"
                )
            except Exception as exc:
                print(f"[CKPT][WARN] Failed to load checkpoint '{resume_path}': {exc}. Starting from scratch.")
                traceback.print_exc()
        else:
            print(f"[CKPT][WARN] Resume checkpoint not found: {resume_path}. Starting from scratch.")
    if not resume_success and args.resume_from:
        model = build_model(num_classes=num_classes, pretrained=True).to(device)
        optimizer, scheduler = _make_optimizer_and_scheduler(model)
        start_epoch = 0
        last_completed_epoch = -1
        best_target_acc = 0.0
        scheduler_loaded = False
    if not scheduler_loaded and start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    print(
        f"[TRAIN] Starting source-only training: dataset={args.dataset_name}, "
        f"source={args.source_domain}, target={args.target_domain}, "
        f"dry_run_max_batches={args.dry_run_max_batches}"
    )
    sys.stdout.flush()
    try:
        for epoch in tqdm(range(start_epoch, args.num_epochs), desc="Epoch", leave=True):
            remaining_batches = 0
            if args.dry_run_max_batches > 0:
                remaining_batches = max(args.dry_run_max_batches - num_batches_seen, 0)
                if remaining_batches == 0:
                    print("[DRY RUN] Reached batch budget; stopping training loop early.")
                    break
            print(f"[TRAIN] Entering epoch {epoch + 1}/{args.num_epochs}")
            sys.stdout.flush()
            loss, acc, batches_used = train_one_epoch(
                model, source_loader, optimizer, device, max_batches=remaining_batches
            )
            num_batches_seen += batches_used
            scheduler.step()
            target_acc, _ = evaluate(model, target_eval_loader, device)
            best_target_acc = max(best_target_acc, target_acc)
            final_source_acc = acc
            last_completed_epoch = epoch
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Accuracy/source", acc, epoch)
            writer.add_scalar("Accuracy/target", target_acc, epoch)
            print(
                f"Epoch {epoch+1}/{args.num_epochs} | Loss {loss:.4f} | "
                f"Source Acc {acc:.2f} | Target Acc {target_acc:.2f}"
            )
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                epoch_ckpt_path = (
                    Path("checkpoints")
                    / f"source_only_{args.source_domain}_to_{args.target_domain}_seed{args.seed}_epoch{epoch+1}.pth"
                )
                checkpoint = {
                    "backbone": model.backbone.state_dict(),
                    "classifier": model.classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": last_completed_epoch,
                    "best_target_acc": best_target_acc,
                    "source_acc": final_source_acc,
                    "args": vars(args),
                }
                _save_checkpoint_safe(checkpoint, epoch_ckpt_path)
            if args.dry_run_max_batches > 0 and num_batches_seen >= args.dry_run_max_batches:
                print("[DRY RUN] Exhausted batch budget; exiting after this epoch.")
                break

        print("[TRAIN] Finished training loop, preparing checkpoint...")
        sys.stdout.flush()
        ckpt_path = _build_source_ckpt_path(args)
        checkpoint = {
            "backbone": model.backbone.state_dict(),
            "classifier": model.classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": last_completed_epoch,
            "best_target_acc": best_target_acc,
            "source_acc": final_source_acc,
            "args": vars(args),
        }
        _save_checkpoint_safe(checkpoint, ckpt_path)
        dataset_field = _dataset_tag(args.dataset_name)
        append_csv(
            path="results/office_home_me_iis.csv",
            fieldnames=OFFICE_HOME_ME_IIS_FIELDS,
            row={
                "dataset": dataset_field,
                "source": args.source_domain,
                "target": args.target_domain,
                "seed": args.seed,
                "method": "source_only",
                "target_acc": round(best_target_acc, 4),
                "source_acc": round(final_source_acc, 4),
                "num_latent": 0,
                "layers": "",
                "components_per_layer": "",
                "iis_iters": 0,
                "iis_tol": 0.0,
                "adapt_epochs": args.num_epochs,
                "finetune_backbone": True,
                "backbone_lr_scale": 1.0,
                "classifier_lr": args.lr_classifier,
                "source_prob_mode": "",
            },
        )
    finally:
        writer.close()
        print(f"[TENSORBOARD] Logs written to {log_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Source-only training for Office-Home or Office-31.")
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
    parser.add_argument("--source_domain", type=str, required=True, help="Source domain e.g., Ar.")
    parser.add_argument("--target_domain", type=str, required=True, help="Target domain e.g., Cl.")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Optional path to a source-only checkpoint to resume training from.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="If >0, save an intermediate checkpoint every N epochs (in addition to the final one).",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_backbone", type=float, default=1e-3)
    parser.add_argument("--lr_classifier", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic/cuDNN safe settings.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry_run_max_batches",
        type=int,
        default=0,
        help="If >0, limit training to this many batches total (for quick sanity checks).",
    )
    parser.add_argument(
        "--dry_run_max_samples",
        type=int,
        default=0,
        help="If >0, limit the number of samples per domain (for very fast dry-runs).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_source(args)
