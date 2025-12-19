from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass, is_dataclass, replace
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

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
    for key in ("acc_wg", "acc_avg", "accuracy", "acc"):
        if key in metrics:
            try:
                return float(metrics[key])
            except Exception:
                continue
    return None


def _extract_selection_score(metrics: Mapping[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    for key in ("acc_wg", "acc_avg", "accuracy", "acc"):
        if key in metrics:
            try:
                return float(metrics[key]), key
            except Exception:
                continue
    return None, None


def _loguniform(rng: random.Random, lo: float, hi: float) -> float:
    lo = float(lo)
    hi = float(hi)
    if not (lo > 0 and hi > 0 and hi >= lo):
        raise ValueError(f"loguniform bounds must be positive with hi>=lo; got lo={lo}, hi={hi}")
    return float(10.0 ** rng.uniform(math.log10(lo), math.log10(hi)))


def _stable_rng(seed: int, *, tag: str) -> random.Random:
    # Avoid Python's randomized hash for reproducibility across processes.
    payload = f"{int(seed)}::{str(tag)}".encode("utf-8")
    h = int(sha1(payload).hexdigest()[:8], 16)
    return random.Random(int(h))


def _print_hparams(cfg: ExperimentConfig) -> None:
    algo = str(cfg.algorithm).upper()
    base: Dict[str, Any] = {
        "dataset": cfg.dataset,
        "algorithm": algo,
        "backbone": cfg.backbone,
        "pretrained": bool(cfg.pretrained),
        "epochs": int(cfg.epochs),
        "batch_size": cfg.batch_size,
        "grad_accum_steps": int(cfg.grad_accum_steps),
        "optimizer": cfg.optimizer,
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "augment": bool(cfg.augment),
        "color_jitter": bool(cfg.color_jitter),
        "amp": bool(cfg.amp),
        "deterministic": bool(cfg.deterministic),
        "paper_match": bool(cfg.paper_match),
    }
    if algo == "DANN":
        base.update(
            {
                "dann_penalty_weight": float(cfg.dann_penalty_weight),
                "grl_lambda": float(cfg.grl_lambda),
                "dann_featurizer_lr_mult": float(cfg.dann_featurizer_lr_mult),
                "dann_discriminator_lr_mult": float(cfg.dann_discriminator_lr_mult),
            }
        )
    if algo == "MEIIS":
        base["meiis"] = _jsonable(cfg.meiis or MEIISConfig())
    print("[hparams] " + json.dumps(_jsonable(base), sort_keys=True))


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "camelyon17"
    split_mode: str = "uda_target"  # uda_target | align_val
    eval_split: str = "test"  # val | test (main reporting split)
    adapt_split: str = "test_unlabeled"  # val_unlabeled | test_unlabeled
    algorithm: str = "ERM"  # ERM | DANN | MEIIS
    backbone: str = "densenet121"
    seed: int = 0
    force_rerun: bool = False
    paper_match: bool = False

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
    dann_featurizer_lr_mult: float = 0.1
    dann_discriminator_lr_mult: float = 1.0

    # ME-IIS
    meiis: Optional[MEIISConfig] = None


def _to_flat_runid_cfg(cfg: ExperimentConfig) -> Dict[str, Any]:
    meiis = cfg.meiis
    base: Dict[str, Any] = {
        "dataset": cfg.dataset,
        "split_mode": cfg.split_mode,
        "eval_split": cfg.eval_split,
        "adapt_split": cfg.adapt_split,
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
        "dann_featurizer_lr_mult": cfg.dann_featurizer_lr_mult if cfg.algorithm.upper() == "DANN" else None,
        "dann_discriminator_lr_mult": cfg.dann_discriminator_lr_mult if cfg.algorithm.upper() == "DANN" else None,
    }
    if meiis is not None and cfg.algorithm.upper() == "MEIIS":
        base.update(
            {
                "meiis_K": meiis.K,
                "meiis_tau": meiis.target_conf_thresh,
                "meiis_step": meiis.iis_step_size,
                "meiis_damp": meiis.iis_damping,
                "meiis_ema": meiis.ema_constraints,
                # Ensure the run_id hash/fingerprint captures the full MEIISConfig, not only a subset.
                "meiis_cfg": _jsonable(meiis),
            }
        )
    return {k: v for k, v in base.items() if v is not None}


def wilds2_random_search_configs(
    *,
    seed: int = 0,
    split_mode: str = "uda_target",
    n: int = 10,
    force_rerun: bool = False,
) -> list[ExperimentConfig]:
    """
    WILDS2-like random search: sample `n` configs per method and select by OOD val.

    This function only generates configs; selection/reporting is handled by the caller.
    """
    base = default_camelyon17_configs(seed=seed, split_mode=split_mode, force_rerun=force_rerun)
    out: list[ExperimentConfig] = []
    for cfg0 in base:
        algo = str(cfg0.algorithm).upper()
        rng = _stable_rng(int(seed), tag=f"wilds2::{split_mode}::{algo}")
        for _ in range(int(n)):
            lr = _loguniform(rng, 1e-5, 5e-4)
            wd = 0.0 if rng.random() < 0.25 else _loguniform(rng, 1e-6, 1e-3)

            if algo == "ERM":
                out.append(replace(cfg0, lr=lr, weight_decay=wd))
                continue
            if algo == "DANN":
                out.append(
                    replace(
                        cfg0,
                        lr=lr,
                        weight_decay=wd,
                        dann_penalty_weight=_loguniform(rng, 1e-2, 10.0),
                        grl_lambda=_loguniform(rng, 1e-1, 1.0),
                        dann_featurizer_lr_mult=float(cfg0.dann_featurizer_lr_mult),
                        dann_discriminator_lr_mult=float(cfg0.dann_discriminator_lr_mult),
                    )
                )
                continue
            if algo == "MEIIS":
                meiis_base = cfg0.meiis or MEIISConfig()
                meiis = replace(
                    meiis_base,
                    K=int(rng.choice([4, 8, 16])),
                    target_conf_thresh=float(rng.choice([0.80, 0.90, 0.95])),
                    iis_step_size=float(rng.choice([0.5, 1.0, 2.0])),
                    iis_damping=float(rng.choice([0.0, 0.1])),
                    ema_constraints=float(rng.choice([0.0, 0.5])),
                    weight_mix_alpha=float(rng.choice([0.8, 0.9])),
                )
                out.append(replace(cfg0, lr=lr, weight_decay=wd, meiis=meiis))
                continue
            raise ValueError(f"Unsupported algorithm '{cfg0.algorithm}' for tuning.")
    return out


def write_best_by_val(
    *,
    configs: Iterable[ExperimentConfig],
    out_root: str,
    selection_split: str = "val",
) -> Path:
    """
    Select the best run per algorithm by validation score and write `best_by_val.json`.
    """
    out_root_p = Path(out_root)
    runs_dir = out_root_p / "runs"
    best: Dict[str, Dict[str, Any]] = {}

    for cfg in configs:
        run_id = encode_config_to_run_id(_to_flat_runid_cfg(cfg))
        run_path = runs_dir / f"{run_id}.json"
        if not run_path.exists():
            continue
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        metrics = (payload.get("metrics", {}) or {}).get(str(selection_split), {}) or {}
        score, key = _extract_selection_score(metrics)
        if score is None:
            continue
        algo = str(cfg.algorithm).upper()
        cur = best.get(algo)
        if cur is None or float(score) > float(cur["val_score"]):
            best[algo] = {
                "run_id": run_id,
                "val_score": float(score),
                "val_metric_key": key,
                "test_score": _extract_acc((payload.get("metrics", {}) or {}).get("test", {}) or {}),
                "config": _jsonable(cfg),
                "run_path": str(run_path),
            }

    out_path = out_root_p / "best_by_val.json"
    out_path.write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[tuning] wrote {out_path}")
    for algo, info in best.items():
        print(
            f"[tuning] best[{algo}] val({info.get('val_metric_key')})={info.get('val_score'):.4f} "
            f"test={info.get('test_score')}"
        )
    return out_path


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

    configs_list = list(configs)
    if configs_list and not any(bool(c.paper_match) for c in configs_list):
        fairness_keys = (
            "backbone",
            "pretrained",
            "augment",
            "color_jitter",
            "imagenet_normalize",
            "epochs",
            "batch_size",
            "grad_accum_steps",
        )
        ref = configs_list[0]
        ref_sig = tuple(getattr(ref, k) for k in fairness_keys)
        for cfg in configs_list[1:]:
            sig = tuple(getattr(cfg, k) for k in fairness_keys)
            if sig != ref_sig:
                print(
                    "[fairness][WARN] Mismatched backbone/compute settings across methods. "
                    "Set `paper_match=True` if this is intentional."
                )
                break

    rows: List[Dict[str, Any]] = []

    for cfg in configs_list:
        _print_hparams(cfg)
        runid_cfg = _to_flat_runid_cfg(cfg)
        run_id = encode_config_to_run_id(runid_cfg)
        run_dir = Path(ckpt_root) / run_id

        train_tf, eval_tf = build_camelyon17_transforms(
            augment=bool(cfg.augment), color_jitter=bool(cfg.color_jitter), imagenet_normalize=bool(cfg.imagenet_normalize)
        )

        loaders = build_camelyon17_loaders(
            {
                "data_root": str(data_root),
                "download": bool(cfg.download),
                "unlabeled": bool(cfg.unlabeled),
                "split_mode": str(cfg.split_mode),
                "eval_split": str(cfg.eval_split),
                "adapt_split": str(cfg.adapt_split),
                "train_transform": train_tf,
                "eval_transform": eval_tf,
                "batch_size": int(cfg.batch_size),
                "unlabeled_batch_size": int(cfg.batch_size),
                "include_indices_in_train": cfg.algorithm.upper() == "MEIIS",
                "num_workers": int(cfg.num_workers),
                "pin_memory": bool(cfg.pin_memory),
                "persistent_workers": bool(cfg.persistent_workers),
                "prefetch_factor": int(cfg.prefetch_factor),
            }
        )

        print(
            "[protocol] "
            f"dataset={cfg.dataset} split_mode={cfg.split_mode} adapt_split={cfg.adapt_split} eval_split={cfg.eval_split} "
            f"unlabeled={bool(cfg.unlabeled)}"
        )
        if str(cfg.split_mode) == "uda_target" and str(cfg.adapt_split) == "test_unlabeled":
            print(
                "[protocol][NOTE] uda_target: adapt=test_unlabeled (target unlabeled), select on val, report on test."
            )
        if str(cfg.split_mode) == "align_val":
            print("[protocol][NOTE] align_val: adapt=val_unlabeled, evaluate=val (debug/ablation only).")
        if loaders.get("unlabeled_loader") is not None:
            print(
                "[data] "
                f"train_n={len(loaders['train_loader'].dataset)} "
                f"adapt_n={len(loaders['unlabeled_loader'].dataset)} "
                f"val_n={len(loaders['val_loader'].dataset)} "
                f"test_n={len(loaders['test_loader'].dataset)}"
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
            unlabeled_loader = loaders["unlabeled_loader"]
        elif algo_name == "MEIIS":
            meiis_cfg = cfg.meiis or MEIISConfig()
            algorithm = MEIIS(
                featurizer=bb.model,
                feature_dim=bb.feature_dim,
                num_classes=2,
                seed=int(cfg.seed),
                config=meiis_cfg,
            )
            unlabeled_loader = loaders["unlabeled_loader"]
        else:
            raise ValueError(f"Unknown algorithm '{cfg.algorithm}'.")

        algorithm.set_backbone_info(
            backbone_name=str(bb.name),
            pretrained=bool(cfg.pretrained),
            feature_dim=int(bb.feature_dim),
            feature_layer="final_pool",
        )

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
            "force_rerun": bool(cfg.force_rerun),
        }
        if algo_name == "DANN":
            train_cfg["lr_overrides"] = {
                "featurizer": float(cfg.lr) * float(cfg.dann_featurizer_lr_mult),
                "discriminator": float(cfg.lr) * float(cfg.dann_discriminator_lr_mult),
            }
        if algo_name == "MEIIS":
            # Capture full MEIISConfig in the checkpoint fingerprint/config.json for config-safe skipping.
            train_cfg["meiis"] = _jsonable(cfg.meiis or MEIISConfig())

        results = train(
            cfg=train_cfg,
            run_dir=run_dir,
            algorithm=algorithm,
            wilds_dataset=loaders["dataset"],
            train_loader=loaders["train_loader"],
            unlabeled_loader=unlabeled_loader,
            val_loader=loaders["val_loader"],
            test_loader=loaders["test_loader"],
            id_val_loader=loaders["id_val_loader"],
        )

        test_metrics = (results.get("metrics", {}) or {}).get("test", {}) or {}
        val_metrics = (results.get("metrics", {}) or {}).get("val", {}) or {}
        eval_metrics = (results.get("metrics", {}) or {}).get(str(cfg.eval_split), {}) or {}
        row = {
            "run_id": run_id,
            "algorithm": cfg.algorithm,
            "backbone": cfg.backbone,
            "split_mode": cfg.split_mode,
            "adapt_split": cfg.adapt_split,
            "eval_split": cfg.eval_split,
            "seed": cfg.seed,
            "val_acc": _extract_acc(val_metrics),
            "test_acc": _extract_acc(test_metrics),
            "eval_acc": _extract_acc(eval_metrics),
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


def default_camelyon17_configs(
    *, seed: int = 0, split_mode: str = "uda_target", force_rerun: bool = False
) -> list[ExperimentConfig]:
    if split_mode == "align_val":
        eval_split = "val"
        adapt_split = "val_unlabeled"
    else:
        eval_split = "test"
        adapt_split = "test_unlabeled"
    return [
        ExperimentConfig(
            algorithm="ERM",
            seed=seed,
            split_mode=split_mode,
            eval_split=eval_split,
            adapt_split=adapt_split,
            force_rerun=bool(force_rerun),
        ),
        ExperimentConfig(
            algorithm="DANN",
            seed=seed,
            split_mode=split_mode,
            eval_split=eval_split,
            adapt_split=adapt_split,
            force_rerun=bool(force_rerun),
            dann_penalty_weight=1.0,
            grl_lambda=1.0,
            dann_featurizer_lr_mult=0.1,
            dann_discriminator_lr_mult=1.0,
        ),
        ExperimentConfig(
            algorithm="MEIIS",
            seed=seed,
            split_mode=split_mode,
            eval_split=eval_split,
            adapt_split=adapt_split,
            force_rerun=bool(force_rerun),
            meiis=MEIISConfig(
                K=8,
                target_conf_thresh=0.90,
                target_conf_mode="maxprob",
                target_conf_min_count=256,
                max_iis_iters=10,
                iis_step_size=1.0,
                iis_damping=0.0,
                ema_constraints=0.0,
                weight_clip_max=10.0,
                weight_mix_alpha=0.8,
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
    p.add_argument("--force_rerun", action="store_true")
    p.add_argument("--tuning_mode", type=str, default="none", choices=["none", "wilds2"])
    p.add_argument("--tuning_n", type=int, default=10)
    args = p.parse_args(argv)

    if args.tuning_mode == "wilds2":
        configs = wilds2_random_search_configs(
            seed=int(args.seed),
            split_mode=str(args.split_mode),
            n=int(args.tuning_n),
            force_rerun=bool(args.force_rerun),
        )
        print(f"[tuning] mode=wilds2 n={int(args.tuning_n)} selection_split=val")
    else:
        configs = default_camelyon17_configs(
            seed=int(args.seed),
            split_mode=str(args.split_mode),
            force_rerun=bool(args.force_rerun),
        )

    summary = run_experiments(
        configs,
        data_root=args.data_root,
        ckpt_root=args.ckpt_root,
        out_root=args.out_root,
    )
    print(f"Wrote {summary}")

    if args.tuning_mode == "wilds2":
        write_best_by_val(configs=configs, out_root=str(args.out_root), selection_split="val")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
