from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from datasets.domain_loaders import get_domain_loaders
from src.experiments.run_config import RunConfig
from utils.data_utils import make_generator, make_worker_init_fn


@dataclass(frozen=True)
class BudgetEstimate:
    source_batches_per_epoch: int
    target_batches_per_epoch: int


@dataclass(frozen=True)
class StepBudget:
    source_batches_per_epoch: int
    target_batches_per_epoch: int
    adapt_steps_per_epoch: int
    epochs_source: int
    epochs_adapt: int
    steps_source: int
    steps_adapt: int
    steps_total: int


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


def compute_step_budget(
    *,
    method: str,
    epochs_source: int,
    epochs_adapt: int,
    source_batches_per_epoch: int,
    target_batches_per_epoch: int,
    adapt_steps_per_epoch_override: Optional[int] = None,
) -> StepBudget:
    method = str(method)
    src_batches = int(source_batches_per_epoch)
    tgt_batches = int(target_batches_per_epoch)
    epochs_source_i = int(epochs_source)
    epochs_adapt_i = int(epochs_adapt)
    if src_batches < 0 or tgt_batches < 0:
        raise ValueError("Batch counts must be non-negative.")

    steps_source = int(epochs_source_i) * int(src_batches)
    if method == "source_only":
        adapt_steps_per_epoch = 0
        steps_adapt = 0
    else:
        if tgt_batches <= 0 and adapt_steps_per_epoch_override is None:
            raise ValueError("target_batches_per_epoch must be > 0 for adaptation methods.")
        adapt_steps_per_epoch = (
            int(adapt_steps_per_epoch_override)
            if adapt_steps_per_epoch_override is not None
            else int(min(src_batches, tgt_batches))
        )
        steps_adapt = int(epochs_adapt_i) * int(adapt_steps_per_epoch)

    return StepBudget(
        source_batches_per_epoch=int(src_batches),
        target_batches_per_epoch=int(tgt_batches),
        adapt_steps_per_epoch=int(adapt_steps_per_epoch),
        epochs_source=int(epochs_source_i),
        epochs_adapt=int(epochs_adapt_i),
        steps_source=int(steps_source),
        steps_adapt=int(steps_adapt),
        steps_total=int(steps_source + steps_adapt),
    )


def estimate_total_steps(config: RunConfig) -> Dict[str, int]:
    """
    Approximate optimizer-step counts for fairness auditing.

    Notes:
    - In the unified runners, most adaptation methods consume paired (source, target) batches via zip(),
      so optimizer steps per epoch are min(len(source), len(target)).
    """
    batches = estimate_batches_per_epoch(config)
    budget = compute_step_budget(
        method=config.method,
        epochs_source=int(config.epochs_source),
        epochs_adapt=int(config.epochs_adapt),
        source_batches_per_epoch=int(batches.source_batches_per_epoch),
        target_batches_per_epoch=int(batches.target_batches_per_epoch),
    )
    return {
        "source_batches_per_epoch": int(budget.source_batches_per_epoch),
        "target_batches_per_epoch": int(budget.target_batches_per_epoch),
        "adapt_steps_per_epoch": int(budget.adapt_steps_per_epoch),
        "steps_source": int(budget.steps_source),
        "steps_adapt": int(budget.steps_adapt),
        "steps_total": int(budget.steps_total),
    }
