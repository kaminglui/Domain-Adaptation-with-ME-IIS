from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from utils.env_utils import is_colab
from utils.experiment_utils import normalize_dataset_name


DEFAULT_COLAB_MYDRIVE = Path("/content/drive/MyDrive")
DEFAULT_PERSIST_SUBDIR = "ME-IIS"


def resolve_persist_root() -> Optional[Path]:
    """
    Resolve a persistent root directory for large artifacts (datasets + checkpoints).

    Priority:
    1) `ME_IIS_PERSIST_ROOT` env var (any platform)
    2) Colab + mounted Drive: `/content/drive/MyDrive/ME-IIS`
    3) None (use repo-relative defaults)
    """
    env = os.getenv("ME_IIS_PERSIST_ROOT")
    if env:
        return Path(env).expanduser()
    if is_colab() and DEFAULT_COLAB_MYDRIVE.exists():
        return DEFAULT_COLAB_MYDRIVE / DEFAULT_PERSIST_SUBDIR
    return None


def legacy_checkpoints_dir() -> Path:
    root = resolve_persist_root()
    return (root / "checkpoints") if root is not None else Path("checkpoints")


def legacy_results_dir() -> Path:
    root = resolve_persist_root()
    return (root / "results") if root is not None else Path("results")


def persistent_datasets_dir() -> Optional[Path]:
    root = resolve_persist_root()
    return (root / "datasets") if root is not None else None


def persistent_dataset_root(dataset_name: str) -> Optional[Path]:
    """
    Default persistent dataset root directory under the persist root.

    Returns None when no persist root is configured.
    """
    datasets_dir = persistent_datasets_dir()
    if datasets_dir is None:
        return None
    name = normalize_dataset_name(dataset_name)
    if name == "officehome":
        return datasets_dir / "Office-Home"
    if name == "office31":
        return datasets_dir / "Office-31"
    return datasets_dir / dataset_name

