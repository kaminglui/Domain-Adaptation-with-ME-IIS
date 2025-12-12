"""
Single source of truth for ME-IIS CLI arguments.

Parsers are grouped by purpose (dataset, model/backbone, clustering, IIS, training,
logging/output, reproducibility) and feed dataclass configs so scripts stay thin.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

DEFAULT_RESULTS_CSV = str(Path("results") / "office_home_me_iis.csv")
DEFAULT_EXPERIMENTS_CSV = str(Path("results") / "me_iis_experiments_summary.csv")


def _add_dump_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dump_config",
        type=str,
        nargs="?",
        const="-",
        default=None,
        help=(
            "If set, dump resolved config as JSON to stdout (default) or to the provided path. "
            "Directories are created if needed."
        ),
    )


@dataclass
class TrainConfig:
    dataset_name: str = "office_home"
    data_root: Optional[str] = None
    source_domain: str = ""
    target_domain: str = ""
    num_epochs: int = 50
    resume_from: Optional[str] = None
    save_every: int = 0
    batch_size: int = 32
    lr_backbone: float = 1e-3
    lr_classifier: float = 1e-2
    weight_decay: float = 1e-3
    num_workers: int = 4
    deterministic: bool = False
    seed: int = 0
    dry_run_max_batches: int = 0
    dry_run_max_samples: int = 0
    eval_on_source_self: bool = False
    eval_results_csv: str = DEFAULT_RESULTS_CSV
    dump_config: Optional[str] = None


@dataclass
class AdaptConfig:
    dataset_name: str = "office_home"
    data_root: Optional[str] = None
    source_domain: str = ""
    target_domain: str = ""
    checkpoint: str = ""
    batch_size: int = 32
    num_workers: int = 4
    num_latent_styles: int = 5
    components_per_layer: Optional[str] = None
    gmm_selection_mode: str = "fixed"
    gmm_bic_min_components: int = 2
    gmm_bic_max_components: int = 8
    cluster_backend: str = "gmm"
    vmf_kappa: float = 20.0
    cluster_clean_ratio: float = 1.0
    kmeans_n_init: int = 10
    feature_layers: str = "layer3,layer4"
    source_prob_mode: str = "softmax"
    iis_iters: int = 15
    iis_tol: float = 1e-3
    adapt_epochs: int = 10
    resume_adapt_from: Optional[str] = None
    save_adapt_every: int = 0
    finetune_backbone: bool = False
    backbone_lr_scale: float = 0.1
    classifier_lr: float = 1e-2
    weight_decay: float = 1e-3
    use_pseudo_labels: bool = False
    pseudo_conf_thresh: float = 0.9
    pseudo_max_ratio: float = 1.0
    pseudo_loss_weight: float = 1.0
    dry_run_max_samples: int = 0
    dry_run_max_batches: int = 0
    deterministic: bool = False
    seed: int = 0
    dump_config: Optional[str] = None


@dataclass
class ExperimentConfig:
    dataset_name: str = "office_home"
    source_domain: str = ""
    target_domain: str = ""
    seeds: str = "0"
    experiment_family: str = ""
    output_csv: str = DEFAULT_EXPERIMENTS_CSV
    base_data_root: Optional[str] = None
    num_epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4
    lr_backbone: float = 1e-3
    lr_classifier: float = 1e-2
    weight_decay: float = 1e-3
    num_latent_styles: int = 5
    components_per_layer: Optional[str] = None
    gmm_selection_mode: str = "fixed"
    gmm_bic_min_components: int = 2
    gmm_bic_max_components: int = 15
    cluster_backend: str = "gmm"
    vmf_kappa: float = 20.0
    cluster_clean_ratio: float = 1.0
    kmeans_n_init: int = 10
    feature_layers: str = "layer3,layer4"
    source_prob_mode: str = "softmax"
    iis_iters: int = 15
    iis_tol: float = 1e-3
    adapt_epochs: int = 10
    finetune_backbone: bool = False
    backbone_lr_scale: float = 0.1
    classifier_lr: float = 1e-2
    pseudo_conf_thresh: float = 0.9
    pseudo_max_ratio: float = 0.3
    pseudo_loss_weight: float = 0.5
    deterministic: bool = False
    dry_run_max_samples: int = 0
    dry_run_max_batches: int = 0
    dump_config: Optional[str] = None


def _make_parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


def build_train_parser() -> argparse.ArgumentParser:
    parser = _make_parser("Source-only training for Office-Home or Office-31.")
    ds = parser.add_argument_group("Dataset / Domains")
    ds.add_argument(
        "--dataset_name",
        type=str,
        default="office_home",
        choices=["office_home", "office31"],
        help="Benchmark to use.",
    )
    ds.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to dataset root (no existence check during parsing).",
    )
    ds.add_argument("--source_domain", type=str, required=True, help="Source domain, e.g., Ar.")
    ds.add_argument("--target_domain", type=str, required=True, help="Target domain, e.g., Cl.")

    train = parser.add_argument_group("Training")
    train.add_argument("--num_epochs", type=int, default=50)
    train.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Optional source-only checkpoint to resume from.",
    )
    train.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="If >0, save an intermediate checkpoint every N epochs.",
    )
    train.add_argument("--batch_size", type=int, default=32)
    train.add_argument("--lr_backbone", type=float, default=1e-3)
    train.add_argument("--lr_classifier", type=float, default=1e-2)
    train.add_argument("--weight_decay", type=float, default=1e-3)
    train.add_argument("--num_workers", type=int, default=4)

    repro = parser.add_argument_group("Reproducibility")
    repro.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic/cuDNN-safe settings (pair with --seed).",
    )
    repro.add_argument("--seed", type=int, default=0)

    debug = parser.add_argument_group("Debug / Speed")
    debug.add_argument(
        "--dry_run_max_batches",
        type=int,
        default=0,
        help="If >0, limit training to this many batches total.",
    )
    debug.add_argument(
        "--dry_run_max_samples",
        type=int,
        default=0,
        help="If >0, cap the number of samples per domain.",
    )

    logging = parser.add_argument_group("Logging / Output")
    logging.add_argument(
        "--eval_on_source_self",
        action="store_true",
        help="If set, evaluate source-only checkpoint on the source domain.",
    )
    logging.add_argument(
        "--eval_results_csv",
        type=str,
        default=DEFAULT_RESULTS_CSV,
        help="CSV to append source-self evaluation when enabled.",
    )
    _add_dump_config_arg(logging)
    return parser


def build_adapt_parser() -> argparse.ArgumentParser:
    parser = _make_parser("ME-IIS domain adaptation on Office-Home or Office-31.")
    ds = parser.add_argument_group("Dataset / Domains")
    ds.add_argument(
        "--dataset_name",
        type=str,
        default="office_home",
        choices=["office_home", "office31"],
        help="Benchmark to use.",
    )
    ds.add_argument("--data_root", type=str, default=None, help="Path to dataset root (not validated here).")
    ds.add_argument("--source_domain", type=str, required=True, help="Source domain, e.g., Ar.")
    ds.add_argument("--target_domain", type=str, required=True, help="Target domain, e.g., Cl.")
    ds.add_argument("--checkpoint", type=str, required=True, help="Path to source-only checkpoint.")

    model = parser.add_argument_group("Model / Backbone")
    model.add_argument("--batch_size", type=int, default=32)
    model.add_argument("--num_workers", type=int, default=4)
    model.add_argument("--feature_layers", type=str, default="layer3,layer4", help="Comma-separated feature layers.")
    model.add_argument(
        "--source_prob_mode",
        type=str,
        default="softmax",
        choices=["softmax", "onehot"],
        help="How to form P(Ĉ|x) on source for constraints.",
    )

    cluster = parser.add_argument_group("Clustering Backend")
    cluster.add_argument(
        "--cluster_backend",
        type=str,
        default="gmm",
        choices=["gmm", "vmf_softmax"],
        help="Latent probability backend for styles.",
    )
    cluster.add_argument(
        "--gmm_selection_mode",
        type=str,
        default="fixed",
        choices=["fixed", "bic"],
        help="How to choose GMM components per layer.",
    )
    cluster.add_argument("--gmm_bic_min_components", type=int, default=2)
    cluster.add_argument("--gmm_bic_max_components", type=int, default=8)
    cluster.add_argument("--num_latent_styles", type=int, default=5, help="Default components per layer when fixed.")
    cluster.add_argument(
        "--components_per_layer",
        type=str,
        default=None,
        help="Optional comma-separated overrides 'layer:count,...'.",
    )
    cluster.add_argument("--vmf_kappa", type=float, default=20.0, help="Concentration for vmf_softmax.")
    cluster.add_argument(
        "--cluster_clean_ratio",
        type=float,
        default=1.0,
        help="Keep-ratio for lowest-entropy target samples when fitting clustering.",
    )
    cluster.add_argument("--kmeans_n_init", type=int, default=10, help="n_init for KMeans (vmf_softmax).")

    iis = parser.add_argument_group("ME-IIS / IIS")
    iis.add_argument("--iis_iters", type=int, default=15, help="Number of IIS iterations.")
    iis.add_argument("--iis_tol", type=float, default=1e-3, help="Tolerance on max abs moment error.")

    train = parser.add_argument_group("Training")
    train.add_argument("--adapt_epochs", type=int, default=10)
    train.add_argument(
        "--resume_adapt_from",
        type=str,
        default=None,
        help="Optional adaptation checkpoint to resume from.",
    )
    train.add_argument(
        "--save_adapt_every",
        type=int,
        default=0,
        help="If >0, save ME-IIS adaptation checkpoint every N epochs.",
    )
    train.add_argument("--finetune_backbone", action="store_true", help="Fine-tune backbone during adaptation.")
    train.add_argument(
        "--backbone_lr_scale",
        type=float,
        default=0.1,
        help="Backbone LR is classifier_lr * backbone_lr_scale when finetuning.",
    )
    train.add_argument("--classifier_lr", type=float, default=1e-2)
    train.add_argument("--weight_decay", type=float, default=1e-3)

    pseudo = parser.add_argument_group("Pseudo-Labels")
    pseudo.add_argument("--use_pseudo_labels", action="store_true", help="Include pseudo-labelled target samples.")
    pseudo.add_argument("--pseudo_conf_thresh", type=float, default=0.9)
    pseudo.add_argument("--pseudo_max_ratio", type=float, default=1.0)
    pseudo.add_argument("--pseudo_loss_weight", type=float, default=1.0)

    repro = parser.add_argument_group("Reproducibility")
    repro.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic/cuDNN-safe settings (pair with --seed).",
    )
    repro.add_argument("--seed", type=int, default=0)

    debug = parser.add_argument_group("Debug / Speed")
    debug.add_argument(
        "--dry_run_max_samples",
        type=int,
        default=0,
        help="If >0, limit samples per domain for quick dry-runs.",
    )
    debug.add_argument(
        "--dry_run_max_batches",
        type=int,
        default=0,
        help="If >0, limit feature extraction and adaptation to this many batches.",
    )

    logging = parser.add_argument_group("Logging / Output")
    _add_dump_config_arg(logging)
    return parser


def build_experiments_parser() -> argparse.ArgumentParser:
    parser = _make_parser("Run ME–IIS ablation suites.")
    ds = parser.add_argument_group("Dataset / Domains")
    ds.add_argument("--dataset_name", type=str, default="office_home", choices=["office_home", "office31"])
    ds.add_argument("--source_domain", type=str, required=True)
    ds.add_argument("--target_domain", type=str, required=True)
    ds.add_argument("--base_data_root", type=str, default=None, help="Optional override for dataset root.")

    exp = parser.add_argument_group("Experiments")
    exp.add_argument("--seeds", type=str, default="0", help='Comma-separated seeds, e.g., "0,1,2".')
    exp.add_argument(
        "--experiment_family",
        type=str,
        required=True,
        choices=["layers", "gmm", "me_iis"],
        help="Which ablation family to run.",
    )
    exp.add_argument(
        "--output_csv",
        type=str,
        default=DEFAULT_EXPERIMENTS_CSV,
        help="Where to write compact experiment summaries.",
    )

    train = parser.add_argument_group("Training")
    train.add_argument("--num_epochs", type=int, default=50, help="Source-only epochs.")
    train.add_argument("--batch_size", type=int, default=32)
    train.add_argument("--num_workers", type=int, default=4)
    train.add_argument("--lr_backbone", type=float, default=1e-3)
    train.add_argument("--lr_classifier", type=float, default=1e-2)
    train.add_argument("--weight_decay", type=float, default=1e-3)

    cluster = parser.add_argument_group("Clustering Backend")
    cluster.add_argument("--num_latent_styles", type=int, default=5, help="Default components per layer when fixed.")
    cluster.add_argument("--components_per_layer", type=str, default=None, help="Optional comma-separated override.")
    cluster.add_argument(
        "--gmm_selection_mode",
        type=str,
        default="fixed",
        choices=["fixed", "bic"],
        help="How to choose GMM components when adapting.",
    )
    cluster.add_argument("--gmm_bic_min_components", type=int, default=2)
    cluster.add_argument("--gmm_bic_max_components", type=int, default=15)
    cluster.add_argument("--cluster_backend", type=str, default="gmm", choices=["gmm", "vmf_softmax"])
    cluster.add_argument("--vmf_kappa", type=float, default=20.0)
    cluster.add_argument("--cluster_clean_ratio", type=float, default=1.0)
    cluster.add_argument("--kmeans_n_init", type=int, default=10)
    cluster.add_argument("--feature_layers", type=str, default="layer3,layer4")

    iis = parser.add_argument_group("ME-IIS / IIS")
    iis.add_argument("--source_prob_mode", type=str, default="softmax", choices=["softmax", "onehot"])
    iis.add_argument("--iis_iters", type=int, default=15)
    iis.add_argument("--iis_tol", type=float, default=1e-3)
    iis.add_argument("--adapt_epochs", type=int, default=10)
    iis.add_argument("--finetune_backbone", action="store_true")
    iis.add_argument("--backbone_lr_scale", type=float, default=0.1)
    iis.add_argument("--classifier_lr", type=float, default=1e-2)

    pseudo = parser.add_argument_group("Pseudo-Labels")
    pseudo.add_argument("--pseudo_conf_thresh", type=float, default=0.9)
    pseudo.add_argument("--pseudo_max_ratio", type=float, default=0.3)
    pseudo.add_argument("--pseudo_loss_weight", type=float, default=0.5)

    repro = parser.add_argument_group("Reproducibility / Debug")
    repro.add_argument("--deterministic", action="store_true")
    repro.add_argument("--dry_run_max_samples", type=int, default=0)
    repro.add_argument("--dry_run_max_batches", type=int, default=0)

    logging = parser.add_argument_group("Logging / Output")
    _add_dump_config_arg(logging)
    return parser


def dump_config(config_obj, path: Optional[str]) -> None:
    """
    Print config to stdout, and optionally write to a path (if not "-" or None).
    """
    if path is None:
        return
    payload = json.dumps(asdict(config_obj), indent=2, sort_keys=True)
    print(payload)
    if path != "-":
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)
        print(f"[CONFIG] Saved config to {out_path}")


def parse_train_config(argv: Optional[list[str]] = None) -> TrainConfig:
    args = build_train_parser().parse_args(argv)
    return TrainConfig(**vars(args))


def parse_adapt_config(argv: Optional[list[str]] = None) -> AdaptConfig:
    args = build_adapt_parser().parse_args(argv)
    return AdaptConfig(**vars(args))


def parse_experiments_config(argv: Optional[list[str]] = None) -> ExperimentConfig:
    args = build_experiments_parser().parse_args(argv)
    return ExperimentConfig(**vars(args))

