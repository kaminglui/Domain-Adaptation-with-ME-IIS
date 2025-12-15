from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from datasets.domain_loaders import get_domain_loaders
from src.experiments.run_config import RunConfig
from utils.data_utils import make_generator, make_worker_init_fn


@dataclass(frozen=True)
class BudgetEstimate:
    source_batches_per_epoch: int
    target_batches_per_epoch: int


def estimate_batches_per_epoch(config: RunConfig) -> BudgetEstimate:
    if config.data_root is None:
        raise ValueError("data_root must be set to estimate budget.")
    gen = make_generator(config.seed)
    worker_init = make_worker_init_fn(config.seed)
    src_loader, tgt_loader, _ = get_domain_loaders(
        dataset_name=config.dataset_name,
        source_domain=config.source_domain,
        target_domain=config.target_domain,
        batch_size=config.batch_size,
        root=str(Path(config.data_root)),
        num_workers=config.num_workers,
        debug_classes=False,
        max_samples_per_domain=config.dry_run_max_samples if config.dry_run_max_samples > 0 else None,
        generator=gen,
        worker_init_fn=worker_init,
    )
    return BudgetEstimate(
        source_batches_per_epoch=len(src_loader),
        target_batches_per_epoch=len(tgt_loader),
    )


def estimate_total_steps(config: RunConfig) -> Dict[str, int]:
    """
    Approximate optimizer-step counts for fairness auditing.

    Notes:
    - In this repo's unified runners, adaptation methods iterate over the source loader per epoch.
    - DANN/CORAL also consume target batches per step (cycled as needed), but the optimizer step count
      is still governed by source batches.
    """
    batches = estimate_batches_per_epoch(config)
    steps_source = int(config.epochs_source) * int(batches.source_batches_per_epoch)
    steps_adapt = 0 if config.method == "source_only" else int(config.epochs_adapt) * int(batches.source_batches_per_epoch)
    return {
        "source_batches_per_epoch": int(batches.source_batches_per_epoch),
        "target_batches_per_epoch": int(batches.target_batches_per_epoch),
        "steps_source": int(steps_source),
        "steps_adapt": int(steps_adapt),
        "steps_total": int(steps_source + steps_adapt),
    }

