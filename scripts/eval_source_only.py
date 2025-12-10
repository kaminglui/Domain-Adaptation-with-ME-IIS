import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.domain_loaders import DEFAULT_OFFICE31_ROOT, DEFAULT_OFFICE_HOME_ROOT, get_domain_loaders
from eval import evaluate
from models.classifier import build_model
from utils.data_utils import make_generator, make_worker_init_fn
from utils.env_utils import is_colab
from utils.experiment_utils import dataset_tag, normalize_dataset_name
from utils.logging_utils import OFFICE_HOME_ME_IIS_FIELDS, append_csv
from utils.seed_utils import get_device, set_seed


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
    - Respect an explicit --data_root if it exists.
    - On Colab, download via KaggleHub.
    - Otherwise, fall back to repository defaults.
    """
    if args.data_root:
        explicit = Path(args.data_root)
        if explicit.exists():
            return str(explicit)
        print(f"[DATA][WARN] Provided data_root does not exist: {explicit}. Falling back to defaults.")

    name = normalize_dataset_name(args.dataset_name)

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


def _infer_num_classes_from_loader(loader) -> int:
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset  # type: ignore
    if hasattr(dataset, "classes"):
        return len(dataset.classes)  # type: ignore
    raise ValueError("Unable to infer number of classes from dataset.")


def _load_model_from_checkpoint(
    checkpoint_path: Path, num_classes: int, device: torch.device
) -> torch.nn.Module:
    model = build_model(num_classes=num_classes, pretrained=False).to(device)
    ckpt: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    backbone_state = ckpt.get("backbone")
    classifier_state = ckpt.get("classifier")
    if backbone_state is None or classifier_state is None:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} is missing backbone/classifier weights. "
            "Expected keys: 'backbone' and 'classifier'."
        )
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
    return model


def eval_source_only(args) -> float:
    """
    Evaluate a source-only checkpoint on the same domain (e.g., Art -> Art).
    Returns the top-1 accuracy (%).
    """
    args.data_root = _maybe_resolve_data_root(args)
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device(deterministic=args.deterministic)
    data_generator = make_generator(args.seed)
    worker_init = make_worker_init_fn(args.seed)
    # Reuse the aligned domain loader logic by pointing source and target to the same domain.
    _, _, eval_loader = get_domain_loaders(
        dataset_name=args.dataset_name,
        source_domain=args.domain,
        target_domain=args.domain,
        batch_size=args.batch_size,
        root=str(data_root),
        num_workers=args.num_workers,
        debug_classes=False,
        max_samples_per_domain=None,
        generator=data_generator,
        worker_init_fn=worker_init,
    )
    num_classes = _infer_num_classes_from_loader(eval_loader)
    model = _load_model_from_checkpoint(Path(args.checkpoint), num_classes=num_classes, device=device)

    acc, _ = evaluate(model, eval_loader, device)
    print(f"[EVAL] {args.dataset_name}, domain={args.domain}, source-only accuracy={acc:.2f}%")

    if args.append_results:
        append_csv(
            path=args.results_csv,
            fieldnames=OFFICE_HOME_ME_IIS_FIELDS,
            row={
                "dataset": dataset_tag(args.dataset_name),
                "source": args.domain,
                "target": args.domain,
                "seed": args.seed,
                "method": "source_only_self",
                "target_acc": round(acc, 4),
                "source_acc": round(acc, 4),
                "num_latent": 0,
                "layers": "",
                "components_per_layer": "",
                "iis_iters": 0,
                "iis_tol": 0.0,
                "adapt_epochs": 0,
                "finetune_backbone": False,
                "backbone_lr_scale": 1.0,
                "classifier_lr": "",
                "source_prob_mode": "",
            },
        )
    return acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a source-only checkpoint on the source domain.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["office_home", "office31"])
    parser.add_argument("--data_root", type=str, default=None, help="Path to dataset root.")
    parser.add_argument("--domain", type=str, required=True, help="Domain to evaluate (e.g., Ar for Office-Home).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the source-only checkpoint.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--results_csv",
        type=str,
        default=str(Path("results") / "office_home_me_iis.csv"),
        help="Optional CSV to append the source-self accuracy.",
    )
    parser.add_argument(
        "--append_results",
        action="store_true",
        help="If set, append a source_only_self row to the results CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        eval_source_only(args)
    except Exception as exc:  # pragma: no cover - keep visible in CLI usage
        print(f"[EVAL][ERROR] Failed to evaluate source-only checkpoint: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
