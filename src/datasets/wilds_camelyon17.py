from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


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
    unlabeled_val: Any
    unlabeled_test: Any
    labeled_id_val: Optional[Any] = None


@dataclass(frozen=True)
class Camelyon17Loaders:
    dataset: Any
    splits: Camelyon17Splits
    train_loader: Any
    unlabeled_loader: Optional[Any]
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

    unlabeled_val = dataset.get_subset("val_unlabeled", transform=eval_transform)
    unlabeled_test = dataset.get_subset("test_unlabeled", transform=eval_transform)

    labeled_id_val = _maybe_get_subset(dataset, "id_val", transform=eval_transform)

    return Camelyon17Splits(
        labeled_train=labeled_train,
        labeled_val=labeled_val,
        labeled_test=labeled_test,
        unlabeled_val=unlabeled_val,
        unlabeled_test=unlabeled_test,
        labeled_id_val=labeled_id_val,
    )


def build_camelyon17_loaders(
    *,
    data_root: str,
    download: bool,
    unlabeled: bool,
    split_mode: str,
    train_transform: Any,
    eval_transform: Any,
    batch_size: int,
    unlabeled_batch_size: Optional[int] = None,
    include_indices_in_train: bool = False,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    loader_kwargs: Optional[Mapping[str, Any]] = None,
) -> Camelyon17Loaders:
    """
    Build Camelyon17 splits + dataloaders using the official WILDS library.

    split_mode:
      - "uda_target": labeled train + unlabeled test_unlabeled (hosp5)
      - "align_val" : labeled train + unlabeled val_unlabeled  (hosp4)
    """
    dataset = get_camelyon17_dataset(root_dir=data_root, download=download, unlabeled=unlabeled)
    splits = build_camelyon17_splits(dataset, train_transform=train_transform, eval_transform=eval_transform)

    if split_mode not in {"uda_target", "align_val"}:
        raise ValueError(f"Unknown split_mode='{split_mode}'. Expected 'uda_target' or 'align_val'.")

    unlabeled_subset = splits.unlabeled_test if split_mode == "uda_target" else splits.unlabeled_val
    unlabeled_bs = int(unlabeled_batch_size) if unlabeled_batch_size is not None else int(batch_size)

    get_train_loader, get_eval_loader = _get_wilds_loaders()
    extra: Dict[str, Any] = dict(loader_kwargs or {})

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
    unlabeled_loader = _call_wilds_loader(
        get_train_loader,
        "standard",
        unlabeled_subset,
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

    return Camelyon17Loaders(
        dataset=dataset,
        splits=splits,
        train_loader=train_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        id_val_loader=id_val_loader,
    )
