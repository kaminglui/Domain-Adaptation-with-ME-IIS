"""
Compatibility shim for Office-31 dataset loaders.

All real logic now lives in datasets/domain_loaders.py.
"""

from typing import Tuple

from torch.utils.data import DataLoader

from .domain_loaders import DEFAULT_OFFICE31_ROOT, get_domain_loaders

DEFAULT_DATA_ROOT = DEFAULT_OFFICE31_ROOT


def get_office31_loaders(
    source_domain: str,
    target_domain: str,
    batch_size: int,
    root: str = DEFAULT_DATA_ROOT,
    num_workers: int = 4,
    debug_classes: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Backwards-compatible wrapper around get_domain_loaders for Office-31.
    """
    return get_domain_loaders(
        dataset_name="office31",
        source_domain=source_domain,
        target_domain=target_domain,
        batch_size=batch_size,
        root=root,
        num_workers=num_workers,
        debug_classes=debug_classes,
    )
