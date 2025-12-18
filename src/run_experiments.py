from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

from src.algorithms import DANN, ERM, MEIIS, MEIISConfig
from src.datasets import build_camelyon17_loaders, build_camelyon17_transforms
from src.models import build_backbone
from src.train import train
from src.utils.run_id import encode_config_to_run_id


def _jsonable(v: Any) -> Any:
    if is_dataclass(v):
        return asdict(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)


def _extract_acc(metrics: Mapping[str, Any]) -> Optional[float]:
    for key in ("acc_avg", "accuracy", "acc"):
        if key in metrics:
            try:
                return float(metrics[key])
            except Exception:
                continue
    return None


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "camelyon17"
    split_mode: str = "uda_target"  # uda_target | align_val
    algorithm: str = "ERM"  # ERM | DANN | MEIIS
    backbone: str = "densenet121"
    seed: int = 0

    # Training
    epochs: int = 5
    batch_size: int = 64
    grad_accum_steps: int = 1
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0
    scheduler: str = "none"  # none | cosine | step
    early_stop_patience: int = 5
    amp: bool = True
    deterministic: bool = True

    # Data/loading
    download: bool = True
    unlabeled: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    augment: bool = False
    color_jitter: bool = False
    imagenet_normalize: bool = True

    # Backbone opts
    pretrained: bool = False
    replace_batchnorm_with_instancenorm: bool = False

    # DANN
    dann_penalty_weight: float = 1.0
    grl_lambda: float = 1.0

    # ME-IIS
    meiis: Optional[MEIISConfig] = None


def _to_flat_runid_cfg(cfg: ExperimentConfig) -> Dict[str, Any]:
    meiis = cfg.meiis
    base: Dict[str, Any] = {
        "dataset": cfg.dataset,
        "split_mode": cfg.split_mode,
        "algorithm": cfg.algorithm,
        "backbone": cfg.backbone,
        "seed": cfg.seed,
        "pretrained": cfg.pretrained,
        "replace_batchnorm_with_instancenorm": cfg.replace_batchnorm_with_instancenorm,
        "optimizer": cfg.optimizer,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "epochs": cfg.epochs,
        "dann_penalty_weight": cfg.dann_penalty_weight if cfg.algorithm.upper() == "DANN" else None,
        "grl_lambda": cfg.grl_lambda if cfg.algorithm.upper() == "DANN" else None,
    }
    if meiis is not None and cfg.algorithm.upper() == "MEIIS":
        base.update(
            {
                "meiis_K": meiis.K,
                "meiis_tau": meiis.confidence_threshold,
                "meiis_step": meiis.iis_step_size,
                "meiis_damp": meiis.iis_damping,
                "meiis_ema": meiis.ema_constraints,
            }
        )
    return {k: v for k, v in base.items() if v is not None}


def run_experiments(
    configs: Iterable[ExperimentConfig],
    *,
    data_root: str,
    ckpt_root: str,
    out_root: str,
    device: Optional[str] = None,
    summary_name: str = "summary.csv",
) -> Path:
    out_root_p = Path(out_root)
    out_root_p.mkdir(parents=True, exist_ok=True)
    summary_path = out_root_p / summary_name

    rows: List[Dict[str, Any]] = []

    for cfg in configs:
        runid_cfg = _to_flat_runid_cfg(cfg)
        run_id = encode_config_to_run_id(runid_cfg)
        run_dir = Path(ckpt_root) / run_id

        train_tf, eval_tf = build_camelyon17_transforms(
            augment=bool(cfg.augment), color_jitter=bool(cfg.color_jitter), imagenet_normalize=bool(cfg.imagenet_normalize)
        )

        loaders = build_camelyon17_loaders(
            data_root=str(data_root),
            download=bool(cfg.download),
            unlabeled=bool(cfg.unlabeled),
            split_mode=str(cfg.split_mode),
            train_transform=train_tf,
            eval_transform=eval_tf,
            batch_size=int(cfg.batch_size),
            unlabeled_batch_size=int(cfg.batch_size),
            include_indices_in_train=cfg.algorithm.upper() == "MEIIS",
            num_workers=int(cfg.num_workers),
            pin_memory=bool(cfg.pin_memory),
            persistent_workers=bool(cfg.persistent_workers),
            prefetch_factor=int(cfg.prefetch_factor),
        )

        bb = build_backbone(
            cfg.backbone,
            pretrained=bool(cfg.pretrained),
            replace_batchnorm_with_instancenorm=bool(cfg.replace_batchnorm_with_instancenorm),
        )

        algo_name = cfg.algorithm.upper()
        if algo_name == "ERM":
            algorithm = ERM(featurizer=bb.model, feature_dim=bb.feature_dim, num_classes=2)
            unlabeled_loader = None
        elif algo_name == "DANN":
            algorithm = DANN(
                featurizer=bb.model,
                feature_dim=bb.feature_dim,
                num_classes=2,
                dann_penalty_weight=float(cfg.dann_penalty_weight),
                grl_lambda=float(cfg.grl_lambda),
            )
            unlabeled_loader = loaders.unlabeled_loader
        elif algo_name == "MEIIS":
            meiis_cfg = cfg.meiis or MEIISConfig()
            algorithm = MEIIS(
                featurizer=bb.model,
                feature_dim=bb.feature_dim,
                num_classes=2,
                seed=int(cfg.seed),
                config=meiis_cfg,
            )
            unlabeled_loader = loaders.unlabeled_loader
        else:
            raise ValueError(f"Unknown algorithm '{cfg.algorithm}'.")

        train_cfg = {
            **runid_cfg,
            "run_id": run_id,
            "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "epochs": int(cfg.epochs),
            "batch_size": int(cfg.batch_size),
            "grad_accum_steps": int(cfg.grad_accum_steps),
            "optimizer": str(cfg.optimizer),
            "lr": float(cfg.lr),
            "weight_decay": float(cfg.weight_decay),
            "scheduler": str(cfg.scheduler),
            "early_stop_patience": int(cfg.early_stop_patience),
            "amp": bool(cfg.amp),
            "deterministic": bool(cfg.deterministic),
            "resume": True,
        }

        results = train(
            cfg=train_cfg,
            run_dir=run_dir,
            algorithm=algorithm,
            wilds_dataset=loaders.dataset,
            train_loader=loaders.train_loader,
            unlabeled_loader=unlabeled_loader,
            val_loader=loaders.val_loader,
            test_loader=loaders.test_loader,
            id_val_loader=loaders.id_val_loader,
        )

        test_metrics = (results.get("metrics", {}) or {}).get("test", {}) or {}
        val_metrics = (results.get("metrics", {}) or {}).get("val", {}) or {}
        row = {
            "run_id": run_id,
            "algorithm": cfg.algorithm,
            "backbone": cfg.backbone,
            "split_mode": cfg.split_mode,
            "seed": cfg.seed,
            "val_acc": _extract_acc(val_metrics),
            "test_acc": _extract_acc(test_metrics),
            "status": results.get("status"),
            "run_dir": str(run_dir),
        }
        rows.append(row)

        # Write incremental summary for Colab friendliness.
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        # Save per-run results copy to OUT_ROOT for convenience.
        (out_root_p / "runs").mkdir(parents=True, exist_ok=True)
        (out_root_p / "runs" / f"{run_id}.json").write_text(
            json.dumps(_jsonable(results), indent=2, sort_keys=True), encoding="utf-8"
        )

    return summary_path


def default_camelyon17_configs(*, seed: int = 0, split_mode: str = "uda_target") -> list[ExperimentConfig]:
    return [
        ExperimentConfig(algorithm="ERM", seed=seed, split_mode=split_mode),
        ExperimentConfig(algorithm="DANN", seed=seed, split_mode=split_mode, dann_penalty_weight=1.0, grl_lambda=1.0),
        ExperimentConfig(
            algorithm="MEIIS",
            seed=seed,
            split_mode=split_mode,
            meiis=MEIISConfig(
                K=8,
                confidence_threshold=0.90,
                max_iis_iters=10,
                iis_step_size=1.0,
                iis_damping=0.0,
                ema_constraints=0.0,
                min_confident_samples=512,
            ),
        ),
    ]


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--ckpt_root", type=str, required=True)
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--split_mode", type=str, default="uda_target", choices=["uda_target", "align_val"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    summary = run_experiments(
        default_camelyon17_configs(seed=args.seed, split_mode=args.split_mode),
        data_root=args.data_root,
        ckpt_root=args.ckpt_root,
        out_root=args.out_root,
    )
    print(f"Wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

