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
from src.cli.args import ExperimentConfig, build_experiments_parser, dump_config
from src.experiments.legacy_results import (
    legacy_adapt_payload,
    legacy_run_id_and_config_json,
    legacy_train_payload,
)
from utils.experiment_utils import (
    build_components_map,
    build_source_ckpt_path,
    dataset_tag,
    parse_feature_layers,
)


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
    run_id: str
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


def _components_map_to_override(feature_layers: Sequence[str], comp_map: Dict[str, int]) -> str:
    return ",".join(f"{layer}:{comp_map[layer]}" for layer in feature_layers)


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
                    run_id=row.get("run_id", ""),
                    target_acc=float(row.get("target_acc", 0.0)),
                    source_acc=float(row.get("source_acc", 0.0)),
                    layers=row.get("layers", ""),
                    components_per_layer=row.get("components_per_layer", ""),
                    gmm_selection_mode="",  # filled from method downstream if needed
                )
            )
        return rows


def result_by_run_id(rows: Iterable[ResultRow], run_id: str) -> Optional[ResultRow]:
    for row in rows:
        if row.run_id == run_id:
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
    train_run_id, _cfg_json = legacy_run_id_and_config_json(legacy_train_payload(vars(train_args)))
    rows = read_results_csv(Path("results") / "office_home_me_iis.csv")
    baseline = result_by_run_id(rows=rows, run_id=train_run_id)
    if baseline is None:
        raise RuntimeError(f"Baseline source-only result not found after training (run_id={train_run_id}).")
    return baseline


def run_adaptation(
    base_args: argparse.Namespace,
    seed: int,
    feature_layers: Sequence[str],
    gmm_selection_mode: str,
    default_components: int,
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
    components_map = build_components_map(feature_layers, default_components, components_override)
    components_override_str: Optional[str] = None
    if components_override is not None or gmm_selection_mode == "fixed":
        components_override_str = _components_map_to_override(feature_layers, components_map)
    adapt_args = argparse.Namespace(
        dataset_name=base_args.dataset_name,
        data_root=base_args.base_data_root,
        source_domain=base_args.source_domain,
        target_domain=base_args.target_domain,
        checkpoint=str(ckpt_path),
        batch_size=base_args.batch_size,
        num_workers=base_args.num_workers,
        num_latent_styles=default_components,
        components_per_layer=components_override_str,
        gmm_selection_mode=gmm_selection_mode,
        gmm_bic_min_components=bic_min if bic_min is not None else base_args.gmm_bic_min_components,
        gmm_bic_max_components=bic_max if bic_max is not None else base_args.gmm_bic_max_components,
        cluster_backend=base_args.cluster_backend,
        vmf_kappa=base_args.vmf_kappa,
        cluster_clean_ratio=base_args.cluster_clean_ratio,
        kmeans_n_init=base_args.kmeans_n_init,
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
        f"gmm_mode={gmm_selection_mode} components_override={components_override_str or components_override} "
        f"pseudo={use_pseudo_labels}"
    )
    adapt_me_iis.adapt_me_iis(adapt_args)
    rows = read_results_csv(Path("results") / "office_home_me_iis.csv")
    adapt_run_id, _cfg_json = legacy_run_id_and_config_json(legacy_adapt_payload(vars(adapt_args)))
    result = result_by_run_id(rows=rows, run_id=adapt_run_id)
    if result is None:
        raise RuntimeError(f"Adaptation result not found in CSV log (run_id={adapt_run_id}).")
    return result, result.components_per_layer


def run_exp_layers(args: argparse.Namespace, seeds: List[int]) -> None:
    layer_configs: List[Tuple[str, ...]] = [
        ("layer4",),
        ("layer3", "layer4"),
    ]
    for seed in seeds:
        baseline = run_source_training(args, seed)
        baseline_acc = baseline.target_acc
        print(f"[Exp-layers] Seed={seed} baseline target acc={baseline_acc:.2f}")
        for layers in layer_configs:
            feature_layers = list(layers)
            comp_override = args.components_per_layer
            adapt_row, comp_str = run_adaptation(
                base_args=args,
                seed=seed,
                feature_layers=feature_layers,
                gmm_selection_mode=args.gmm_selection_mode,
                default_components=args.num_latent_styles,
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
                    "dataset": dataset_tag(args.dataset_name),
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
    best_layers = parse_feature_layers(args.feature_layers)
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
            default_components = int(cfg["components"]) if cfg["components"] is not None else args.num_latent_styles
            if mode == "fixed":
                comp_map = build_components_map(best_layers, default_components, None)
                comp_override = _components_map_to_override(best_layers, comp_map)
            adapt_row, comp_str = run_adaptation(
                base_args=args,
                seed=seed,
                feature_layers=best_layers,
                gmm_selection_mode=mode,
                default_components=default_components,
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
                    "dataset": dataset_tag(args.dataset_name),
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
    best_layers = parse_feature_layers(args.feature_layers)
    for seed in seeds:
        baseline = run_source_training(args, seed)
        baseline_acc = baseline.target_acc
        print(f"[Exp-me_iis] Seed={seed} baseline target acc={baseline_acc:.2f}")
        # Source-only summary row
        append_summary_row(
            Path(args.output_csv),
            {
                "dataset": dataset_tag(args.dataset_name),
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

        comp_override = args.components_per_layer
        adapt_row, comp_str = run_adaptation(
            base_args=args,
            seed=seed,
            feature_layers=best_layers,
            gmm_selection_mode=args.gmm_selection_mode,
            default_components=args.num_latent_styles,
            components_override=comp_override,
            use_pseudo_labels=False,
        )
        append_summary_row(
            Path(args.output_csv),
            {
                "dataset": dataset_tag(args.dataset_name),
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
            default_components=args.num_latent_styles,
            components_override=comp_override,
            use_pseudo_labels=True,
        )
        append_summary_row(
            Path(args.output_csv),
            {
                "dataset": dataset_tag(args.dataset_name),
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
    return build_experiments_parser().parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(**vars(args))
    dump_config(cfg, cfg.dump_config)
    seeds = parse_seeds(cfg.seeds)
    print(
        f"[Driver] Running family={cfg.experiment_family} seeds={seeds} "
        f"dataset={cfg.dataset_name} {cfg.source_domain}->{cfg.target_domain}"
    )
    if cfg.experiment_family == "layers":
        run_exp_layers(cfg, seeds)
    elif cfg.experiment_family == "gmm":
        run_exp_gmm(cfg, seeds)
    elif cfg.experiment_family == "me_iis":
        run_exp_me_iis(cfg, seeds)
    else:
        raise ValueError(f"Unknown experiment_family {cfg.experiment_family}")
    print(f"[Driver] Summary written to {cfg.output_csv}")


if __name__ == "__main__":
    main()
