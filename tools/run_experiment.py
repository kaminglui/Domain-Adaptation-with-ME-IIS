from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.persist_paths import persistent_dataset_root


def _resolve_data_root(dataset_name: str, data_root: Optional[str]) -> str:
    if data_root is not None and str(data_root).strip():
        return str(Path(str(data_root)).expanduser())
    persistent = persistent_dataset_root(dataset_name)
    if persistent is not None:
        return str(persistent)
    # Repo-defaults match datasets/domain_loaders.py constants.
    name = dataset_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if name == "officehome":
        return str(Path("datasets") / "Office-Home")
    if name == "office31":
        return str(Path("datasets") / "Office-31")
    return str(Path("datasets") / dataset_name)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified experiment runner (debug harness).")
    p.add_argument("--dataset", required=True, type=str, help="e.g. office_home")
    p.add_argument("--data_root", default=None, type=str, help="Dataset root (optional; auto-resolved if omitted).")
    p.add_argument("--src", required=True, type=str, help="Source domain code (e.g. Ar).")
    p.add_argument("--tgt", required=True, type=str, help="Target domain code (e.g. Cl).")
    p.add_argument(
        "--method",
        required=True,
        type=str,
        choices=["source_only", "dann", "dan", "jan", "cdan", "me_iis", "pseudo_label"],
        help="Which baseline/method to run.",
    )
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--mode", default="full", choices=["quick", "full"])
    p.add_argument("--force_rerun", action="store_true")
    p.add_argument("--device", default="cuda", type=str, help="cuda|cpu|cuda:0 etc (best-effort).")
    p.add_argument("--bottleneck_dim", default=256, type=int)
    p.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone weights during adaptation (UDA methods).",
    )
    p.add_argument(
        "--one_batch_debug",
        action="store_true",
        help="Run a single forward/backward step and exit (method/loss sanity check).",
    )
    p.add_argument(
        "--cuda_launch_blocking",
        action="store_true",
        help="Set CUDA_LAUNCH_BLOCKING=1 for easier device-side assert debugging.",
    )
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable mixed precision on CUDA (default: enabled when using CUDA).",
    )
    return p.parse_args()


def _mode_defaults(mode: str) -> Dict[str, Any]:
    if str(mode).lower() == "quick":
        return {
            "epochs_source": 1,
            "epochs_adapt": 1,
            "batch_size": 8,
            "num_workers": 0,
            "deterministic": True,
            "dry_run_max_samples": 256,
        }
    return {"epochs_source": 50, "epochs_adapt": 10, "batch_size": 32, "num_workers": 4, "deterministic": True}


def main() -> int:
    args = _parse_args()
    os.environ["ME_IIS_DEVICE"] = str(args.device)
    if args.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    from src.experiments.metrics import get_git_sha
    from src.experiments.run_config import RunConfig, get_run_dir
    from src.experiments.runner import run_one

    data_root = Path(_resolve_data_root(args.dataset, args.data_root))
    try:
        from utils.resource_auto import maybe_cache_dataset_to_local

        cached_root, cache_info = maybe_cache_dataset_to_local(
            dataset_name=str(args.dataset),
            data_root=data_root,
        )
        if str(cached_root) != str(data_root):
            print(f"[DATA][CACHE] {cache_info.get('original_root')} -> {cached_root}")
        data_root = Path(cached_root)
    except Exception:
        data_root = Path(data_root)
    defaults = _mode_defaults(args.mode)
    method_params: Dict[str, Any] = {}
    if args.amp is not None:
        method_params["amp"] = bool(args.amp)

    cfg = RunConfig(
        dataset_name=str(args.dataset),
        data_root=str(data_root),
        source_domain=str(args.src),
        target_domain=str(args.tgt),
        method=str(args.method),
        epochs_source=int(defaults["epochs_source"]),
        epochs_adapt=int(defaults["epochs_adapt"]),
        batch_size=int(defaults["batch_size"]),
        num_workers=int(defaults["num_workers"]),
        bottleneck_dim=int(args.bottleneck_dim),
        finetune_backbone=bool(not args.freeze_backbone),
        method_params=method_params,
        seed=int(args.seed),
        deterministic=bool(defaults.get("deterministic", False)),
        dry_run_max_samples=int(defaults.get("dry_run_max_samples", 0)),
    )

    run_dir = get_run_dir(cfg, runs_root=None)
    sha = get_git_sha()
    print(f"[GIT] {sha}")
    print(f"[DATA] dataset={cfg.dataset_name} root={cfg.data_root}")
    print(f"[RUN] dir={run_dir}")
    print(f"[RUN] method={cfg.method} run_id={cfg.run_id}")

    if args.one_batch_debug:
        # The unified runner exposes one-batch debug via method_params; keep the CLI stable.
        cfg = replace(cfg, method_params={**cfg.method_params, "one_batch_debug": True, "device": str(args.device)})

    force = bool(args.force_rerun) or bool(args.one_batch_debug)
    res = run_one(cfg, force_rerun=force, runs_root=None, write_metrics=not args.one_batch_debug)
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
