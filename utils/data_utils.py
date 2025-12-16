"""
Shared helpers for building reproducible DataLoaders.
"""
import inspect
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.env_utils import is_colab


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

    def _make(nw: int) -> DataLoader:
        worker_init = make_worker_init_fn(seed) if nw > 0 else None
        gen = generator if generator is not None else make_generator(seed)
        supported = set(inspect.signature(DataLoader).parameters.keys())
        kwargs = {}
        if "pin_memory" in supported:
            kwargs["pin_memory"] = bool(pin_memory)
        if nw > 0:
            if "persistent_workers" in supported:
                kwargs["persistent_workers"] = bool(persistent_workers)
            if prefetch_factor is not None and "prefetch_factor" in supported:
                kwargs["prefetch_factor"] = int(prefetch_factor)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=nw,
            worker_init_fn=worker_init,
            generator=gen,
            drop_last=drop_last,
            **kwargs,
        )

    loader = _make(int(num_workers))

    # Colab stability: fall back to single-process loading when multiprocessing is flaky.
    if is_colab() and num_workers > 0:
        try:
            _ = next(iter(loader))
        except Exception as exc:
            msg = str(exc)
            patterns = (
                "DataLoader worker",
                "worker exited unexpectedly",
                "worker (pid",
                "pickle",
                "EOFError",
                "BrokenPipeError",
            )
            if any(p.lower() in msg.lower() for p in patterns):
                print(f"[DATALOADER][WARN] {msg} | falling back to num_workers=0")
                return _make(0)
            raise
    return loader
