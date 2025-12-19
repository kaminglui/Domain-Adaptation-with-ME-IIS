from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Optional

from utils.env_utils import is_colab


class WithIndexDataset:
    """
    Wrap a WILDS subset to additionally return the sample index.

    Output becomes: (x, y, metadata, idx).
    """

    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, y, metadata = self.dataset[idx]
        return x, y, metadata, idx

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)


@dataclass(frozen=True)
class Camelyon17Splits:
    labeled_train: Any
    labeled_val: Any
    labeled_test: Any
    unlabeled_train: Optional[Any]
    unlabeled_val: Any
    unlabeled_test: Any
    labeled_id_val: Optional[Any] = None


@dataclass(frozen=True)
class Camelyon17Loaders:
    dataset: Any
    splits: Camelyon17Splits
    train_loader: Any
    unlabeled_loader: Optional[Any]
    unlabeled_train_loader: Optional[Any]
    val_loader: Any
    test_loader: Any
    id_val_loader: Optional[Any] = None


def _require_wilds() -> Any:
    try:
        import wilds  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency 'wilds'. In Colab: `pip install wilds` (or install wilds from source)."
        ) from exc
    return wilds


def _get_wilds_loaders() -> tuple[Any, Any]:
    try:
        from wilds.common.data_loaders import get_eval_loader, get_train_loader  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Unable to import WILDS dataloader helpers. Is `wilds` installed correctly?"
        ) from exc
    return get_train_loader, get_eval_loader


def build_camelyon17_transforms(
    *,
    augment: bool,
    color_jitter: bool = False,
    imagenet_normalize: bool = True,
) -> tuple[Any, Any]:
    """
    Centralized transforms for Camelyon17.

    WILDS returns PIL images; we keep the default eval transform minimal and put
    histology-friendly color jitter behind flags.
    """
    try:
        from torchvision import transforms as T  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Missing dependency 'torchvision'.") from exc

    ops = [T.ToTensor()]
    if imagenet_normalize:
        ops.append(T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    eval_transform = T.Compose(ops)

    if not augment:
        return eval_transform, eval_transform

    train_ops = [T.ToTensor()]
    if color_jitter:
        train_ops.append(
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05)
        )
    train_ops.append(T.RandomHorizontalFlip(p=0.5))
    if imagenet_normalize:
        train_ops.append(T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    train_transform = T.Compose(train_ops)
    return train_transform, eval_transform


def _maybe_get_subset(dataset: Any, split: str, *, transform: Any) -> Optional[Any]:
    try:
        return dataset.get_subset(split, transform=transform)
    except Exception:
        return None


def _call_wilds_loader(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """
    WILDS has had minor signature differences across versions; we keep a
    best-effort compatibility shim for performance kwargs.
    """
    try:
        return fn(*args, **kwargs)
    except TypeError:
        pass

    # Retry without common perf-only kwargs.
    for drop_key in ("prefetch_factor", "persistent_workers"):
        if drop_key in kwargs:
            retry = dict(kwargs)
            retry.pop(drop_key, None)
            try:
                return fn(*args, **retry)
            except TypeError:
                continue
    # Final retry: strip both.
    retry = dict(kwargs)
    retry.pop("prefetch_factor", None)
    retry.pop("persistent_workers", None)
    return fn(*args, **retry)


def get_camelyon17_dataset(*, root_dir: str, download: bool, unlabeled: bool) -> Any:
    wilds = _require_wilds()
    return wilds.get_dataset(dataset="camelyon17", root_dir=root_dir, download=download, unlabeled=unlabeled)


def build_camelyon17_splits(
    dataset: Any,
    *,
    train_transform: Any,
    eval_transform: Any,
) -> Camelyon17Splits:
    labeled_train = dataset.get_subset("train", transform=train_transform)
    labeled_val = dataset.get_subset("val", transform=eval_transform)
    labeled_test = dataset.get_subset("test", transform=eval_transform)

    unlabeled_train = _maybe_get_subset(dataset, "train_unlabeled", transform=eval_transform)
    unlabeled_val = dataset.get_subset("val_unlabeled", transform=eval_transform)
    unlabeled_test = dataset.get_subset("test_unlabeled", transform=eval_transform)

    labeled_id_val = _maybe_get_subset(dataset, "id_val", transform=eval_transform)

    return Camelyon17Splits(
        labeled_train=labeled_train,
        labeled_val=labeled_val,
        labeled_test=labeled_test,
        unlabeled_train=unlabeled_train,
        unlabeled_val=unlabeled_val,
        unlabeled_test=unlabeled_test,
        labeled_id_val=labeled_id_val,
    )

SplitMode = Literal["uda_target", "align_val"]
EvalSplit = Literal["val", "test"]
AdaptSplit = Literal["val_unlabeled", "test_unlabeled"]


def build_camelyon17_loaders(config: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Build Camelyon17 splits + dataloaders using the official WILDS library.

    Required keys (Phase 1 spec):
      - split_mode: "uda_target" | "align_val"
      - eval_split: "val" | "test"
      - adapt_split: "val_unlabeled" | "test_unlabeled"

    Returns a dict containing the dataset object, subset handles, and loaders.
    """
    data_root_raw = config.get("data_root", None)
    if data_root_raw is None or str(data_root_raw).strip() == "":
        data_root = os.environ.get("WILDS_DATA_ROOT")
        if not data_root:
            data_root = "/content/data/wilds" if is_colab() else os.path.join("datasets", "wilds")
    else:
        data_root = str(data_root_raw)
    download = bool(config.get("download", True))
    unlabeled = bool(config.get("unlabeled", True))

    split_mode: SplitMode = str(config.get("split_mode", "uda_target"))  # type: ignore[assignment]
    eval_split: EvalSplit = str(config.get("eval_split", "test"))  # type: ignore[assignment]
    adapt_split: AdaptSplit = str(config.get("adapt_split", "test_unlabeled"))  # type: ignore[assignment]

    if split_mode not in {"uda_target", "align_val"}:
        raise ValueError(f"Unknown split_mode='{split_mode}'. Expected 'uda_target' or 'align_val'.")
    if eval_split not in {"val", "test"}:
        raise ValueError(f"Unknown eval_split='{eval_split}'. Expected 'val' or 'test'.")
    if adapt_split not in {"val_unlabeled", "test_unlabeled"}:
        raise ValueError(
            f"Unknown adapt_split='{adapt_split}'. Expected 'val_unlabeled' or 'test_unlabeled'."
        )

    if split_mode == "align_val" and adapt_split != "val_unlabeled":
        raise ValueError("split_mode='align_val' requires adapt_split='val_unlabeled'.")
    if split_mode == "align_val" and eval_split != "val":
        raise ValueError("split_mode='align_val' requires eval_split='val' (debug/ablation only).")
    if split_mode == "uda_target" and adapt_split != "test_unlabeled":
        raise ValueError("split_mode='uda_target' requires adapt_split='test_unlabeled'.")
    if eval_split == "test" and adapt_split == "test_unlabeled" and split_mode != "uda_target":
        raise ValueError(
            "Refusing to adapt on test_unlabeled when eval_split='test' unless split_mode='uda_target'."
        )

    train_transform = config.get("train_transform", None)
    eval_transform = config.get("eval_transform", None)

    batch_size = int(config.get("batch_size", 0))
    if batch_size <= 0:
        raise ValueError("Missing/invalid required config['batch_size'] for Camelyon17.")
    unlabeled_batch_size = config.get("unlabeled_batch_size", None)
    unlabeled_bs = int(unlabeled_batch_size) if unlabeled_batch_size is not None else int(batch_size)

    include_indices_in_train = bool(config.get("include_indices_in_train", False))

    num_workers = int(config.get("num_workers", 8))
    pin_memory = bool(config.get("pin_memory", True))
    persistent_workers = bool(config.get("persistent_workers", True))
    prefetch_factor = int(config.get("prefetch_factor", 2))
    loader_kwargs = config.get("loader_kwargs", None) or {}

    dataset = get_camelyon17_dataset(root_dir=data_root, download=download, unlabeled=unlabeled)
    splits = build_camelyon17_splits(dataset, train_transform=train_transform, eval_transform=eval_transform)

    adapt_subset = splits.unlabeled_val if adapt_split == "val_unlabeled" else splits.unlabeled_test
    eval_subset = splits.labeled_val if eval_split == "val" else splits.labeled_test

    get_train_loader, get_eval_loader = _get_wilds_loaders()
    extra: Dict[str, Any] = dict(loader_kwargs)

    labeled_train_ds = WithIndexDataset(splits.labeled_train) if include_indices_in_train else splits.labeled_train

    common_train_kwargs: Dict[str, Any] = dict(
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    if int(num_workers) > 0:
        common_train_kwargs["persistent_workers"] = bool(persistent_workers)
        common_train_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = _call_wilds_loader(
        get_train_loader,
        "standard",
        labeled_train_ds,
        batch_size=int(batch_size),
        **common_train_kwargs,
        **extra,
    )

    unlabeled_loader = None
    if unlabeled:
        unlabeled_loader = _call_wilds_loader(
            get_train_loader,
            "standard",
            adapt_subset,
            batch_size=int(unlabeled_bs),
            **common_train_kwargs,
            **extra,
        )

    unlabeled_train_loader = None
    if unlabeled and splits.unlabeled_train is not None:
        unlabeled_train_loader = _call_wilds_loader(
            get_train_loader,
            "standard",
            splits.unlabeled_train,
            batch_size=int(unlabeled_bs),
            **common_train_kwargs,
            **extra,
        )

    common_eval_kwargs: Dict[str, Any] = dict(
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    if int(num_workers) > 0:
        common_eval_kwargs["persistent_workers"] = bool(persistent_workers)
        common_eval_kwargs["prefetch_factor"] = int(prefetch_factor)

    val_loader = _call_wilds_loader(
        get_eval_loader,
        "standard",
        splits.labeled_val,
        batch_size=int(batch_size),
        **common_eval_kwargs,
        **extra,
    )
    test_loader = _call_wilds_loader(
        get_eval_loader,
        "standard",
        splits.labeled_test,
        batch_size=int(batch_size),
        **common_eval_kwargs,
        **extra,
    )
    eval_loader = _call_wilds_loader(
        get_eval_loader,
        "standard",
        eval_subset,
        batch_size=int(batch_size),
        **common_eval_kwargs,
        **extra,
    )

    id_val_loader = None
    if splits.labeled_id_val is not None:
        id_val_loader = _call_wilds_loader(
            get_eval_loader,
            "standard",
            splits.labeled_id_val,
            batch_size=int(batch_size),
            **common_eval_kwargs,
            **extra,
        )

    return {
        "dataset": dataset,
        "splits": splits,
        "train_loader": train_loader,
        "unlabeled_loader": unlabeled_loader,
        "unlabeled_train_loader": unlabeled_train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "id_val_loader": id_val_loader,
        "adapt_split": adapt_split,
        "eval_split": eval_split,
        "split_mode": split_mode,
        "adapt_loader": unlabeled_loader,
        "eval_loader": eval_loader,
    }
