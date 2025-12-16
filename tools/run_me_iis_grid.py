from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.run_experiment import _mode_defaults, _resolve_data_root


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Small ME-IIS grid runner (seed=0 by default).")
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--data_root", default=None, type=str)
    p.add_argument("--src", required=True, type=str)
    p.add_argument("--tgt", required=True, type=str)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--mode", default="quick", choices=["quick", "full"])
    p.add_argument("--force_rerun", action="store_true")
    p.add_argument("--bottleneck_dim", default=256, type=int)
    p.add_argument("--freeze_backbone", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    from src.experiments.run_config import RunConfig
    from src.experiments.runner import run_one

    data_root = _resolve_data_root(args.dataset, args.data_root)
    defaults = _mode_defaults(args.mode)

    base_params: Dict[str, Any] = {
        "num_latent_styles": 5,
        "components_per_layer": None,
        "cluster_backend": "gmm",
        "gmm_selection_mode": "bic",
        "gmm_bic_min_components": 2,
        "gmm_bic_max_components": 8,
        "cluster_clean_ratio": 1.0,
        "source_prob_mode": "softmax",
        "iis_tol": 1e-3,
        "use_pseudo_labels": False,
        "weight_mix_alpha": 0.0,
    }

    layers_grid: List[List[str]] = [["layer2"], ["layer3"], ["layer4"], ["layer3", "layer4"]]
    reg_grid = [1e-6, 1e-4, 1e-3]
    iters_grid = [15, 50]
    clip_grid: List[Optional[float]] = [None, 10.0]

    results: List[Dict[str, Any]] = []
    for feature_layers, reg_covar, iis_iters, weight_clip_max in itertools.product(
        layers_grid, reg_grid, iters_grid, clip_grid
    ):
        method_params = dict(base_params)
        method_params.update(
            {
                "feature_layers": list(feature_layers),
                "gmm_reg_covar": float(reg_covar),
                "iis_iters": int(iis_iters),
                "weight_clip_max": weight_clip_max,
            }
        )
        cfg = RunConfig(
            dataset_name=str(args.dataset),
            data_root=str(data_root),
            source_domain=str(args.src),
            target_domain=str(args.tgt),
            method="me_iis",
            epochs_source=int(defaults["epochs_source"]),
            epochs_adapt=int(defaults["epochs_adapt"]),
            batch_size=int(defaults["batch_size"]),
            num_workers=int(defaults["num_workers"]),
            bottleneck_dim=int(args.bottleneck_dim),
            finetune_backbone=bool(not args.freeze_backbone),
            seed=int(args.seed),
            deterministic=bool(defaults.get("deterministic", False)),
            dry_run_max_samples=int(defaults.get("dry_run_max_samples", 0)),
            method_params=method_params,
        )
        res = run_one(cfg, force_rerun=bool(args.force_rerun), write_metrics=True)
        results.append(
            {
                "feature_layers": ",".join(feature_layers),
                "gmm_reg_covar": float(reg_covar),
                "iis_iters": int(iis_iters),
                "weight_clip_max": weight_clip_max,
                "run_id": cfg.run_id,
                "status": res.get("status"),
                "run_dir": res.get("run_dir"),
                "target_acc": res.get("target_acc_eval"),
            }
        )
        print(
            f"[GRID] layers={','.join(feature_layers)} reg={reg_covar:g} iters={iis_iters} clip={weight_clip_max} "
            f"status={res.get('status')} tgt_acc={res.get('target_acc_eval')}"
        )

    out = Path("outputs") / "me_iis_grid_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[GRID] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
