import csv
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

OFFICE_HOME_ME_IIS_FIELDS = [
    "dataset",
    "source",
    "target",
    "seed",
    "method",
    "target_acc",
    "source_acc",
    "num_latent",
    "layers",
    "components_per_layer",
    "iis_iters",
    "iis_tol",
    "adapt_epochs",
    "finetune_backbone",
    "backbone_lr_scale",
    "classifier_lr",
    "source_prob_mode",
]

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def append_csv(path: str, fieldnames: Iterable[str], row: Dict) -> None:
    """Append a row to CSV, creating header if needed."""
    ensure_dir(os.path.dirname(path) or ".")
    file_exists = os.path.isfile(path)
    with open(path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


class TBLogger:
    """Lightweight TensorBoard logger wrapper."""

    def __init__(self, log_dir: str):
        ensure_dir(log_dir)
        if SummaryWriter is None:
            raise ImportError("TensorBoard not installed. Add tensorboard to requirements to use TBLogger.")
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalars(self, tag: str, scalar_dict: Dict[str, float], step: int) -> None:
        for key, val in scalar_dict.items():
            self.writer.add_scalar(f"{tag}/{key}", val, step)

    def close(self) -> None:
        self.writer.close()
