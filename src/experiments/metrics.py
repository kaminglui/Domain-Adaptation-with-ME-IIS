from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.experiments.run_config import RunConfig
from utils.experiment_utils import dataset_tag


METRICS_FIELDS = [
    "dataset",
    "src",
    "tgt",
    "method",
    "run_id",
    "seed",
    "source_acc",
    "target_acc",
    "epochs_source",
    "epochs_adapt",
    "backbone",
    "batch_size",
    "lr",
    "timestamp",
    "git_sha",
    # Extra (useful for auditing fairness / reruns)
    "lr_backbone",
    "lr_classifier",
    "classifier_lr",
    "backbone_lr_scale",
    "weight_decay",
    "deterministic",
    "transforms_id",
    "method_params_json",
]


def get_git_sha(repo_root: Optional[Path] = None) -> str:
    try:
        cwd = str(repo_root) if repo_root is not None else None
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_metrics_row(
    config: RunConfig,
    source_acc: float,
    target_acc: float,
    git_sha: str = "",
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    ts = timestamp or utc_timestamp()
    return {
        "dataset": dataset_tag(config.dataset_name),
        "src": config.source_domain,
        "tgt": config.target_domain,
        "method": config.method,
        "run_id": config.run_id,
        "seed": config.seed,
        "source_acc": round(float(source_acc), 6),
        "target_acc": round(float(target_acc), 6),
        "epochs_source": int(config.epochs_source),
        "epochs_adapt": int(config.epochs_adapt),
        "backbone": config.backbone,
        "batch_size": int(config.batch_size),
        "lr": float(config.lr_classifier),
        "timestamp": ts,
        "git_sha": git_sha,
        "lr_backbone": float(config.lr_backbone),
        "lr_classifier": float(config.lr_classifier),
        "classifier_lr": float(config.classifier_lr),
        "backbone_lr_scale": float(config.backbone_lr_scale),
        "weight_decay": float(config.weight_decay),
        "deterministic": bool(config.deterministic),
        "transforms_id": config.transforms_id,
        "method_params_json": json.dumps(config.method_params, sort_keys=True),
    }


def write_metrics_csv(path: Path, row: Dict[str, Any], fieldnames: Iterable[str] = METRICS_FIELDS) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerow(row)

