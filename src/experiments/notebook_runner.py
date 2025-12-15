"""
Helpers for orchestrating ME-IIS experiments from notebooks without duplicating core logic.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.cli.args import build_adapt_parser, build_train_parser
from src.experiments.legacy_results import legacy_adapt_payload, legacy_run_id_and_config_json, legacy_train_payload

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    print("[RUN]", " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def _results_row_by_run_id(run_id: str) -> Dict[str, float]:
    csv_path = REPO_ROOT / "results" / "office_home_me_iis.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if "run_id" not in df.columns:
        return {}
    mask = df["run_id"].astype(str) == str(run_id)
    if not mask.any():
        return {}
    row = df[mask].iloc[-1]
    return {"target_acc": float(row.get("target_acc", float("nan"))), "source_acc": float(row.get("source_acc", float("nan")))}


def _npz_for_run_id(run_id: str, source: str, target: str) -> Optional[Path]:
    results_dir = REPO_ROOT / "results"
    candidates = [p for p in results_dir.glob("*.npz") if str(run_id) in p.name]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    pattern = f"me_iis_weights_{source}_to_{target}_"
    candidates = [p for p in results_dir.glob("*.npz") if pattern in p.name]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


@dataclass
class RunResult:
    backend: str
    checkpoint: Optional[str]
    adapted_target_acc: Optional[float]
    delta_acc: Optional[float]
    iis_final_max_err: Optional[float]
    iis_final_kl: Optional[float]
    weight_entropy: Optional[float]
    history_npz: Optional[str]
    feature_layers: str


def run_train(
    dataset_name: str,
    data_root: str,
    source_domain: str,
    target_domain: str,
    batch_size: int,
    num_epochs: int,
    seed: int,
    deterministic: bool,
    feature_layers: str,
    quick: bool = False,
) -> Dict[str, str]:
    args = [
        "python",
        "scripts/train_source.py",
        "--dataset_name",
        dataset_name,
        "--data_root",
        data_root,
        "--source_domain",
        source_domain,
        "--target_domain",
        target_domain,
        "--batch_size",
        str(batch_size),
        "--num_epochs",
        str(num_epochs),
        "--seed",
        str(seed),
    ]
    if deterministic:
        args.append("--deterministic")
    if quick:
        args += ["--dry_run_max_batches", "2", "--dry_run_max_samples", "4"]
    _run(args)
    parsed = build_train_parser().parse_args(args[2:])
    run_id, _cfg_json = legacy_run_id_and_config_json(legacy_train_payload(vars(parsed)))
    checkpoint = Path("checkpoints") / f"source_only_{source_domain}_to_{target_domain}_seed{seed}.pth"
    return {"checkpoint": str(checkpoint), "run_id": run_id}


def run_adapt(
    dataset_name: str,
    data_root: str,
    source_domain: str,
    target_domain: str,
    batch_size: int,
    seed: int,
    deterministic: bool,
    feature_layers: str,
    checkpoint: str,
    backend: str,
    adapt_epochs: int,
    gmm_selection_mode: str = "fixed",
    vmf_kappa: float = 20.0,
    cluster_clean_ratio: float = 1.0,
    quick: bool = False,
) -> Dict[str, Optional[float]]:
    args = [
        "python",
        "scripts/adapt_me_iis.py",
        "--dataset_name",
        dataset_name,
        "--data_root",
        data_root,
        "--source_domain",
        source_domain,
        "--target_domain",
        target_domain,
        "--checkpoint",
        checkpoint,
        "--batch_size",
        str(batch_size),
        "--feature_layers",
        feature_layers,
        "--cluster_backend",
        backend,
        "--gmm_selection_mode",
        gmm_selection_mode,
        "--vmf_kappa",
        str(vmf_kappa),
        "--cluster_clean_ratio",
        str(cluster_clean_ratio),
        "--adapt_epochs",
        str(adapt_epochs),
        "--seed",
        str(seed),
    ]
    if deterministic:
        args.append("--deterministic")
    if quick:
        args += ["--dry_run_max_batches", "2", "--dry_run_max_samples", "4", "--iis_iters", "3"]
    start = time.time()
    _run(args)
    runtime = time.time() - start
    parsed = build_adapt_parser().parse_args(args[2:])
    run_id, _cfg_json = legacy_run_id_and_config_json(legacy_adapt_payload(vars(parsed)))
    row = _results_row_by_run_id(run_id)
    target_acc = row.get("target_acc")
    source_acc = row.get("source_acc")
    delta_acc = target_acc - source_acc if target_acc is not None and source_acc is not None else None
    npz = _npz_for_run_id(run_id, source_domain, target_domain)
    iis_err = iis_kl = entropy = None
    if npz is not None:
        data = np.load(npz, allow_pickle=True)
        if "moment_max" in data:
            iis_err = float(data["moment_max"][-1])
        if "kl" in data:
            iis_kl = float(data["kl"][-1])
        if "w_entropy" in data:
            entropy = float(data["w_entropy"][-1])
    return {
        "backend": backend,
        "checkpoint": str(Path("checkpoints") / f"me_iis_{source_domain}_to_{target_domain}_{feature_layers.replace(',', '-')}_seed{seed}.pth"),
        "adapted_target_acc": target_acc,
        "delta_acc": delta_acc,
        "iis_final_max_err": iis_err,
        "iis_final_kl": iis_kl,
        "weight_entropy": entropy,
        "history_npz": str(npz) if npz else None,
        "feature_layers": feature_layers,
        "runtime_sec": runtime,
        "run_id": run_id,
    }


def run_vmf_sweep(
    base_checkpoint: str,
    grid: Dict[str, List],
    dataset_name: str,
    data_root: str,
    source_domain: str,
    target_domain: str,
    batch_size: int,
    seed: int,
    deterministic: bool,
    feature_layers: str,
    layers: str,
    adapt_epochs: int,
    quick: bool = False,
) -> pd.DataFrame:
    rows = []
    for K in grid.get("K", []):
        for kappa in grid.get("kappa", []):
            for clean in grid.get("clean_ratio", []):
                run = run_adapt(
                    dataset_name=dataset_name,
                    data_root=data_root,
                    source_domain=source_domain,
                    target_domain=target_domain,
                    batch_size=batch_size,
                    seed=seed,
                    deterministic=deterministic,
                    feature_layers=layers,
                    checkpoint=base_checkpoint,
                    backend="vmf_softmax",
                    adapt_epochs=adapt_epochs,
                    vmf_kappa=float(kappa),
                    cluster_clean_ratio=float(clean),
                    quick=quick,
                )
                run["K"] = K
                run["kappa"] = kappa
                run["clean_ratio"] = clean
                rows.append(run)
    return pd.DataFrame(rows)


def summarize_runs(runs: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(runs)
