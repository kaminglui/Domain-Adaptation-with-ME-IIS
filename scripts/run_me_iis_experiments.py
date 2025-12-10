"""
Experiment driver for ME–IIS.
Families:
- layers: ablates which feature layers supply the style×class constraints.
- gmm: varies GMM complexity and BIC-based component selection for style modeling.
- me_iis: compares source-only vs ME–IIS reweighting vs ME–IIS with pseudo-labels.
These runs stitch together the ME–IIS pipeline (constraint construction, IIS reweighting,
weighted fine-tuning, optional pseudo-label loss) without re-implementing the core math.
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.adapt_me_iis as adapt_me_iis
import scripts.train_source as train_source


SUMMARY_FIELDS = [
    "dataset",
    "source",
    "target",
    "seed",
    "experiment_family",
    "method",
    "feature_layers",
    "gmm_selection_mode",
    "components_per_layer",
    "baseline_target_acc",
    "adapted_target_acc",
    "pseudo_conf_thresh",
    "pseudo_max_ratio",
    "pseudo_loss_weight",
]


@dataclass
class ResultRow:
    dataset: str
    source: str
    target: str
    seed: int
    method: str
    target_acc: float
    source_acc: float
    layers: str
    components_per_layer: str
    gmm_selection_mode: str


def parse_seeds(seed_str: str) -> List[int]:
    seeds = []
    for part in seed_str.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("At least one seed must be provided (e.g., --seeds 0,1,2).")
    return seeds


def _dataset_tag(name: str) -> str:
    norm = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if norm == "officehome":
        return "office-home"
    if norm == "office31":
        return "office-31"
    return name


def _parse_feature_layers(layer_str: str) -> List[str]:
    layers = [l.strip() for l in layer_str.split(",") if l.strip()]
    if not layers:
        raise ValueError("feature_layers must contain at least one layer name.")
    return layers


def _normalize_components_override(override: Optional[str], layers: Sequence[str]) -> Optional[str]:
    """
    Accept either colon-based overrides (layer:count,...) or plain comma-separated
    counts that align with the provided layers.
    """
    if not override:
        return None
    override = override.strip()
    if not override:
        return None
    if ":" in override:
        return override
    counts = [c.strip() for c in override.split(",") if c.strip()]
    if counts and len(counts) != len(layers):
        raise ValueError(
            f"components_per_layer override must match number of layers ({len(layers)}), got {len(counts)} entries."
        )
    mapped = [f"{layer}:{counts[idx]}" for idx, layer in enumerate(layers)]
    return ",".join(mapped)


def _uniform_component_override(count: int, layers: Sequence[str]) -> str:
    return ",".join(f"{layer}:{count}" for layer in layers)


def read_results_csv(path: Path) -> List[ResultRow]:
    if not path.exists():
        return []
    rows: List[ResultRow] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                ResultRow(
                    dataset=row.get("dataset", ""),
                    source=row.get("source", ""),
                    target=row.get("target", ""),
                    seed=int(row.get("seed", 0)),
                    method=row.get("method", ""),
                    target_acc=float(row.get("target_acc", 0.0)),
                    source_acc=float(row.get("source_acc", 0.0)),
                    layers=row.get("layers", ""),
                    components_per_layer=row.get("components_per_layer", ""),
                    gmm_selection_mode="",  # filled from method downstream if needed
                )
            )
        return rows


def latest_result(
    rows: Iterable[ResultRow],
    dataset: str,
    source: str,
    target: str,
    seed: int,
    method: str,
) -> Optional[ResultRow]:
    for row in reversed(list(rows)):
        if (
            row.dataset == dataset
            and row.source == source
            and row.target == target
            and row.seed == seed
            and row.method == method
        ):
            return row
    return None


def append_summary_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_source_ckpt_path(source_domain: str, target_domain: str, seed: int) -> Path:
    # Mirrors scripts/train_source.py naming for re-use.
    return Path("checkpoints") / f"source_only_{source_domain}_to_{target_domain}_seed{seed}.pth"


def run_source_training(base_args: argparse.Namespace, seed: int) -> ResultRow:
    train_args = argparse.Namespace(
        dataset_name=base_args.dataset_name,
        data_root=base_args.base_data_root,
        source_domain=base_args.source_domain,
        target_domain=base_args.target_domain,
        num_epochs=base_args.num_epochs,
        resume_from=None,
        save_every=0,
        batch_size=base_args.batch_size,
        lr_backbone=base_args.lr_backbone,
        lr_classifier=base_args.lr_classifier,
        weight_decay=base_args.weight_decay,
        num_workers=base_args.num_workers,
        deterministic=base_args.deterministic,
        seed=seed,
        dry_run_max_batches=base_args.dry_run_max_batches,
        dry_run_max_samples=base_args.dry_run_max_samples,
    )
    print(f"[Driver] Training/resuming source-only model for seed={seed}")
    train_source.train_source(train_args)
    dataset_field = _dataset_tag(base_args.dataset_name)
    rows = read_results_csv(Path("results") / "office_home_me_iis.csv")
    baseline = latest_result(
        rows=rows,
        dataset=dataset_field,
        source=base_args.source_domain,
        target=base_args.target_domain,
        seed=seed,
        method="source_only",
    )
    if baseline is None:
        raise RuntimeError("Baseline source-only result not found after training.")
    return baseline


def run_adaptation(
    base_args: argparse.Namespace,
    seed: int,
    feature_layers: Sequence[str],
    gmm_selection_mode: str,
    components_override: Optional[str],
    use_pseudo_labels: bool = False,
    pseudo_conf_thresh: Optional[float] = None,
    pseudo_max_ratio: Optional[float] = None,
    pseudo_loss_weight: Optional[float] = None,
    bic_min: Optional[int] = None,
    bic_max: Optional[int] = None,
    method_override: Optional[str] = None,
) -> Tuple[ResultRow, str]:
    ckpt_path = build_source_ckpt_path(base_args.source_domain, base_args.target_domain, seed)
    adapt_args = argparse.Namespace(
        dataset_name=base_args.dataset_name,
        data_root=base_args.base_data_root,
        source_domain=base_args.source_domain,
        target_domain=base_args.target_domain,
        checkpoint=str(ckpt_path),
        batch_size=base_args.batch_size,
        num_workers=base_args.num_workers,
        num_latent_styles=base_args.num_latent_styles,
        components_per_layer=components_override,
        gmm_selection_mode=gmm_selection_mode,
        gmm_bic_min_components=bic_min if bic_min is not None else base_args.gmm_bic_min_components,
        gmm_bic_max_components=bic_max if bic_max is not None else base_args.gmm_bic_max_components,
        feature_layers=",".join(feature_layers),
        source_prob_mode=base_args.source_prob_mode,
        iis_iters=base_args.iis_iters,
        iis_tol=base_args.iis_tol,
        adapt_epochs=base_args.adapt_epochs,
        resume_adapt_from=None,
        save_adapt_every=0,
        finetune_backbone=base_args.finetune_backbone,
        backbone_lr_scale=base_args.backbone_lr_scale,
        classifier_lr=base_args.classifier_lr,
        weight_decay=base_args.weight_decay,
        use_pseudo_labels=use_pseudo_labels,
        pseudo_conf_thresh=pseudo_conf_thresh if pseudo_conf_thresh is not None else base_args.pseudo_conf_thresh,
        pseudo_max_ratio=pseudo_max_ratio if pseudo_max_ratio is not None else base_args.pseudo_max_ratio,
        pseudo_loss_weight=pseudo_loss_weight if pseudo_loss_weight is not None else base_args.pseudo_loss_weight,
        dry_run_max_samples=base_args.dry_run_max_samples,
        dry_run_max_batches=base_args.dry_run_max_batches,
        deterministic=base_args.deterministic,
        seed=seed,
    )
    print(
        f"[Driver] Adapting seed={seed} layers={feature_layers} "
        f"gmm_mode={gmm_selection_mode} components_override={components_override} pseudo={use_pseudo_labels}"
    )
    adapt_me_iis.adapt_me_iis(adapt_args)
    dataset_field = _dataset_tag(base_args.dataset_name)
    rows = read_results_csv(Path("results") / "office_home_me_iis.csv")
    expected_method = method_override
    if expected_method is None:
        if use_pseudo_labels:
            expected_method = "me_iis_pl"
        elif gmm_selection_mode == "bic":
            expected_method = "me_iis_bic"
        else:
            expected_method = "me_iis"
    result = latest_result(
        rows=rows,
        dataset=dataset_field,
        source=base_args.source_domain,
        target=base_args.target_domain,
        seed=seed,
        method=expected_method,
    )
    if result is None:
        raise RuntimeError(f"Adaptation result (method={expected_method}) not found in CSV log.")
    return result, result.components_per_layer


def run_exp_layers(args: argparse.Namespace, seeds: List[int]) -> None:
    layer_configs: List[Tuple[str, ...]] = [
        ("layer4",),
        ("avgpool",),
        ("layer3", "layer4"),
        ("layer3", "layer4", "avgpool"),
    ]
    for seed in seeds:
        baseline = run_source_training(args, seed)
        baseline_acc = baseline.target_acc
        print(f"[Exp-layers] Seed={seed} baseline target acc={baseline_acc:.2f}")
        for layers in layer_configs:
            feature_layers = list(layers)
            comp_override = _normalize_components_override(args.components_per_layer, feature_layers)
            if comp_override is None and args.gmm_selection_mode == "fixed":
                comp_override = _uniform_component_override(args.num_latent_styles, feature_layers)
            adapt_row, comp_str = run_adaptation(
                base_args=args,
                seed=seed,
                feature_layers=feature_layers,
                gmm_selection_mode=args.gmm_selection_mode,
                components_override=comp_override,
                use_pseudo_labels=False,
            )
            print(
                f"[Exp-layers] Seed={seed} layers={feature_layers} "
                f"baseline={baseline_acc:.2f} adapted={adapt_row.target_acc:.2f} comps={comp_str}"
            )
            append_summary_row(
                Path(args.output_csv),
                {
                    "dataset": _dataset_tag(args.dataset_name),
                    "source": args.source_domain,
                    "target": args.target_domain,
                    "seed": seed,
                    "experiment_family": "layers",
                    "method": adapt_row.method,
                    "feature_layers": ",".join(feature_layers),
                    "gmm_selection_mode": args.gmm_selection_mode,
                    "components_per_layer": comp_str,
                    "baseline_target_acc": baseline_acc,
                    "adapted_target_acc": adapt_row.target_acc,
                    "pseudo_conf_thresh": "",
                    "pseudo_max_ratio": "",
                    "pseudo_loss_weight": "",
                },
            )


def run_exp_gmm(args: argparse.Namespace, seeds: List[int]) -> None:
    best_layers = _parse_feature_layers(args.feature_layers)
    gmm_configs = [
        {"mode": "fixed", "components": 2},
        {"mode": "fixed", "components": 5},
        {"mode": "fixed", "components": 10},
        {"mode": "bic", "components": None, "bic_min": args.gmm_bic_min_components, "bic_max": args.gmm_bic_max_components},
    ]
    for seed in seeds:
        baseline = run_source_training(args, seed)
        baseline_acc = baseline.target_acc
        print(f"[Exp-gmm] Seed={seed} baseline target acc={baseline_acc:.2f}")
        for cfg in gmm_configs:
            mode = cfg["mode"]
            comp_override = None
            bic_min = cfg.get("bic_min")
            bic_max = cfg.get("bic_max")
            if mode == "fixed":
                comp_override = _uniform_component_override(int(cfg["components"]), best_layers)
            adapt_row, comp_str = run_adaptation(
                base_args=args,
                seed=seed,
                feature_layers=best_layers,
                gmm_selection_mode=mode,
                components_override=comp_override,
                use_pseudo_labels=False,
                bic_min=bic_min,
                bic_max=bic_max,
            )
            print(
                f"[Exp-gmm] Seed={seed} mode={mode} layers={best_layers} "
                f"baseline={baseline_acc:.2f} adapted={adapt_row.target_acc:.2f} comps={comp_str}"
            )
            append_summary_row(
                Path(args.output_csv),
                {
                    "dataset": _dataset_tag(args.dataset_name),
                    "source": args.source_domain,
                    "target": args.target_domain,
                    "seed": seed,
                    "experiment_family": "gmm",
                    "method": adapt_row.method,
                    "feature_layers": ",".join(best_layers),
                    "gmm_selection_mode": mode,
                    "components_per_layer": comp_str,
                    "baseline_target_acc": baseline_acc,
                    "adapted_target_acc": adapt_row.target_acc,
                    "pseudo_conf_thresh": "",
                    "pseudo_max_ratio": "",
                    "pseudo_loss_weight": "",
                },
            )


def run_exp_me_iis(args: argparse.Namespace, seeds: List[int]) -> None:
    best_layers = _parse_feature_layers(args.feature_layers)
    for seed in seeds:
        baseline = run_source_training(args, seed)
        baseline_acc = baseline.target_acc
        print(f"[Exp-me_iis] Seed={seed} baseline target acc={baseline_acc:.2f}")
        # Source-only summary row
        append_summary_row(
            Path(args.output_csv),
            {
                "dataset": _dataset_tag(args.dataset_name),
                "source": args.source_domain,
                "target": args.target_domain,
                "seed": seed,
                "experiment_family": "me_iis",
                "method": "source_only",
                "feature_layers": ",".join(best_layers),
                "gmm_selection_mode": "none",
                "components_per_layer": "",
                "baseline_target_acc": baseline_acc,
                "adapted_target_acc": baseline_acc,
                "pseudo_conf_thresh": "",
                "pseudo_max_ratio": "",
                "pseudo_loss_weight": "",
            },
        )

        comp_override = _normalize_components_override(args.components_per_layer, best_layers)
        if comp_override is None and args.gmm_selection_mode == "fixed":
            comp_override = _uniform_component_override(args.num_latent_styles, best_layers)
        adapt_row, comp_str = run_adaptation(
            base_args=args,
            seed=seed,
            feature_layers=best_layers,
            gmm_selection_mode=args.gmm_selection_mode,
            components_override=comp_override,
            use_pseudo_labels=False,
        )
        append_summary_row(
            Path(args.output_csv),
            {
                "dataset": _dataset_tag(args.dataset_name),
                "source": args.source_domain,
                "target": args.target_domain,
                "seed": seed,
                "experiment_family": "me_iis",
                "method": adapt_row.method,
                "feature_layers": ",".join(best_layers),
                "gmm_selection_mode": args.gmm_selection_mode,
                "components_per_layer": comp_str,
                "baseline_target_acc": baseline_acc,
                "adapted_target_acc": adapt_row.target_acc,
                "pseudo_conf_thresh": "",
                "pseudo_max_ratio": "",
                "pseudo_loss_weight": "",
            },
        )

        adapt_pl_row, comp_pl_str = run_adaptation(
            base_args=args,
            seed=seed,
            feature_layers=best_layers,
            gmm_selection_mode=args.gmm_selection_mode,
            components_override=comp_override,
            use_pseudo_labels=True,
        )
        append_summary_row(
            Path(args.output_csv),
            {
                "dataset": _dataset_tag(args.dataset_name),
                "source": args.source_domain,
                "target": args.target_domain,
                "seed": seed,
                "experiment_family": "me_iis",
                "method": adapt_pl_row.method,
                "feature_layers": ",".join(best_layers),
                "gmm_selection_mode": args.gmm_selection_mode,
                "components_per_layer": comp_pl_str,
                "baseline_target_acc": baseline_acc,
                "adapted_target_acc": adapt_pl_row.target_acc,
                "pseudo_conf_thresh": args.pseudo_conf_thresh,
                "pseudo_max_ratio": args.pseudo_max_ratio,
                "pseudo_loss_weight": args.pseudo_loss_weight,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ME–IIS ablation suites.")
    parser.add_argument("--dataset_name", type=str, default="office_home", choices=["office_home", "office31"])
    parser.add_argument("--source_domain", type=str, required=True)
    parser.add_argument("--target_domain", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0", help='Comma-separated seeds, e.g., "0,1,2".')
    parser.add_argument("--experiment_family", type=str, required=True, choices=["layers", "gmm", "me_iis"])
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(Path("results") / "me_iis_experiments_summary.csv"),
        help="Where to write compact experiment summaries.",
    )
    parser.add_argument(
        "--base_data_root",
        type=str,
        default=None,
        help="Optional override for dataset root (otherwise reuse defaults in loaders).",
    )
    parser.add_argument("--num_epochs", type=int, default=50, help="Source-only epochs (matches train_source default).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_backbone", type=float, default=1e-3)
    parser.add_argument("--lr_classifier", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_latent_styles", type=int, default=5, help="Default components per layer when fixed.")
    parser.add_argument("--components_per_layer", type=str, default=None, help="Optional comma-separated override.")
    parser.add_argument(
        "--gmm_selection_mode",
        type=str,
        default="fixed",
        choices=["fixed", "bic"],
        help="How to choose GMM components when adapting.",
    )
    parser.add_argument("--gmm_bic_min_components", type=int, default=2)
    parser.add_argument("--gmm_bic_max_components", type=int, default=15)
    parser.add_argument("--feature_layers", type=str, default="layer3,layer4,avgpool")
    parser.add_argument("--source_prob_mode", type=str, default="softmax", choices=["softmax", "onehot"])
    parser.add_argument("--iis_iters", type=int, default=15)
    parser.add_argument("--iis_tol", type=float, default=1e-3)
    parser.add_argument("--adapt_epochs", type=int, default=10)
    parser.add_argument("--finetune_backbone", action="store_true")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1)
    parser.add_argument("--classifier_lr", type=float, default=1e-2)
    parser.add_argument("--pseudo_conf_thresh", type=float, default=0.9)
    parser.add_argument("--pseudo_max_ratio", type=float, default=0.3)
    parser.add_argument("--pseudo_loss_weight", type=float, default=0.5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dry_run_max_samples", type=int, default=0)
    parser.add_argument("--dry_run_max_batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    print(
        f"[Driver] Running family={args.experiment_family} seeds={seeds} "
        f"dataset={args.dataset_name} {args.source_domain}->{args.target_domain}"
    )
    if args.experiment_family == "layers":
        run_exp_layers(args, seeds)
    elif args.experiment_family == "gmm":
        run_exp_gmm(args, seeds)
    elif args.experiment_family == "me_iis":
        run_exp_me_iis(args, seeds)
    else:
        raise ValueError(f"Unknown experiment_family {args.experiment_family}")
    print(f"[Driver] Summary written to {args.output_csv}")


if __name__ == "__main__":
    main()
