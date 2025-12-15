"""
Shared helpers for building reproducible DataLoaders.
"""
import inspect
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def make_worker_init_fn(base_seed: int):
    """Return a worker_init_fn that seeds python/np/torch for each worker."""
    def _init_fn(worker_id: int) -> None:
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return _init_fn


def make_generator(seed: int) -> torch.Generator:
    """Create a torch.Generator seeded for deterministic dataloading."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    drop_last: bool = False,
    generator: Optional[torch.Generator] = None,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    """Construct a DataLoader with optional shared generator and seeded workers."""
    worker_init = make_worker_init_fn(seed) if num_workers > 0 else None
    gen = generator if generator is not None else make_generator(seed)
    supported = set(inspect.signature(DataLoader).parameters.keys())
    kwargs = {}
    if "pin_memory" in supported:
        kwargs["pin_memory"] = bool(pin_memory)
    if num_workers > 0:
        if "persistent_workers" in supported:
            kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None and "prefetch_factor" in supported:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init,
        generator=gen,
        drop_last=drop_last,
        **kwargs,
    )
