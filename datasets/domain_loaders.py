from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_OFFICE_HOME_ROOT = "./datasets/Office-Home"
DEFAULT_OFFICE31_ROOT = "./datasets/Office-31"

OFFICE_HOME_DOMAIN_TO_SUBDIR: Dict[str, str] = {
    "Ar": "Art",
    "Cl": "Clipart",
    "Pr": "Product",
    "Rw": "RealWorld",
}
OFFICE_HOME_REALWORLD_CANDIDATES = ["RealWorld", "Real World", "Real_World", "Real"]

OFFICE31_DOMAIN_TO_SUBDIR: Dict[str, str] = {
    "A": "amazon",
    "D": "dslr",
    "W": "webcam",
}


def _build_transforms(train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _normalize_dataset_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


# --------------------- Office-Home helpers --------------------- #
def _oh_normalize_domain_code(domain: str) -> str:
    if domain in OFFICE_HOME_DOMAIN_TO_SUBDIR:
        return domain
    normalized = domain.lower().replace("_", "").replace(" ", "")
    for candidate in OFFICE_HOME_REALWORLD_CANDIDATES:
        if normalized == candidate.lower().replace("_", "").replace(" ", ""):
            return "Rw"
    for code, subdir in OFFICE_HOME_DOMAIN_TO_SUBDIR.items():
        if normalized == subdir.lower().replace(" ", ""):
            return code
    raise ValueError(f"Unknown Office-Home domain '{domain}'. Expected one of {list(OFFICE_HOME_DOMAIN_TO_SUBDIR.keys())}.")


def _oh_resolve_realworld_dir(root: Path) -> Path:
    for candidate in OFFICE_HOME_REALWORLD_CANDIDATES:
        candidate_dir = root / candidate
        if candidate_dir.exists():
            return candidate_dir
    raise FileNotFoundError(
        f"Could not find RealWorld domain under {root}. Tried: {', '.join(OFFICE_HOME_REALWORLD_CANDIDATES)}"
    )


def _oh_resolve_domain_dir(root: Path, domain: str) -> Path:
    code = _oh_normalize_domain_code(domain)
    subdir = OFFICE_HOME_DOMAIN_TO_SUBDIR[code]
    if subdir == "RealWorld":
        return _oh_resolve_realworld_dir(root)
    domain_dir = root / subdir
    if not domain_dir.exists():
        raise FileNotFoundError(f"Domain path does not exist: {domain_dir}")
    return domain_dir


def _oh_list_sorted_classes(domain_dir: Path) -> List[str]:
    classes = sorted([p.name for p in domain_dir.iterdir() if p.is_dir()])
    if not classes:
        raise ValueError(f"No class subfolders found in {domain_dir}")
    return classes


def _oh_build_shared_class_mapping(
    root: Path, source_domain: str, target_domain: str, debug_classes: bool = False
) -> Dict[str, int]:
    src_dir = _oh_resolve_domain_dir(root, source_domain)
    tgt_dir = _oh_resolve_domain_dir(root, target_domain)
    src_classes = _oh_list_sorted_classes(src_dir)
    tgt_classes = _oh_list_sorted_classes(tgt_dir)
    if set(src_classes) != set(tgt_classes):
        missing_src = sorted(set(tgt_classes) - set(src_classes))
        missing_tgt = sorted(set(src_classes) - set(tgt_classes))
        raise ValueError(
            "Source/target class folders do not match.\n"
            f"Missing in source: {missing_src}\n"
            f"Missing in target: {missing_tgt}"
        )
    class_to_idx = {cls: idx for idx, cls in enumerate(src_classes)}
    if debug_classes:
        print("Office-Home classes (sorted):")
        for idx, name in enumerate(src_classes):
            print(f"  {idx}: {name}")
    return class_to_idx


def _oh_get_loaders(
    root: str,
    source_domain: str,
    target_domain: str,
    batch_size: int,
    num_workers: int = 4,
    debug_classes: bool = False,
    max_samples_per_domain: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Office-Home root does not exist: {root_path}")

    class_to_idx = _oh_build_shared_class_mapping(root_path, source_domain, target_domain, debug_classes=debug_classes)

    src_dir = _oh_resolve_domain_dir(root_path, source_domain)
    tgt_dir = _oh_resolve_domain_dir(root_path, target_domain)

    source_ds = datasets.ImageFolder(src_dir.as_posix(), transform=_build_transforms(train=True))
    target_ds = datasets.ImageFolder(tgt_dir.as_posix(), transform=_build_transforms(train=True))
    target_eval_ds = datasets.ImageFolder(tgt_dir.as_posix(), transform=_build_transforms(train=False))

    def maybe_subset(ds):
        if max_samples_per_domain is not None and len(ds) > max_samples_per_domain:
            subset = Subset(ds, list(range(max_samples_per_domain)))
            for attr in ("class_to_idx", "classes", "root", "transform"):
                if hasattr(ds, attr):
                    setattr(subset, attr, getattr(ds, attr))
            return subset
        return ds

    for name, ds in [("source", source_ds), ("target_train", target_ds), ("target_eval", target_eval_ds)]:
        if set(ds.class_to_idx.keys()) != set(class_to_idx.keys()):
            missing_ds = sorted(set(class_to_idx.keys()) - set(ds.class_to_idx.keys()))
            missing_map = sorted(set(ds.class_to_idx.keys()) - set(class_to_idx.keys()))
            raise ValueError(
                f"Office-Home {name} class names differ from canonical mapping.\n"
                f"Missing in {name}: {missing_ds}\n"
                f"Missing in canonical: {missing_map}"
            )
        if ds.class_to_idx != class_to_idx:
            raise ValueError(
                f"Office-Home {name} class_to_idx ordering does not match canonical mapping.\n"
                f"{name} mapping: {ds.class_to_idx}\n"
                f"Canonical:      {class_to_idx}"
            )

    source_ds = maybe_subset(source_ds)
    target_ds = maybe_subset(target_ds)
    target_eval_ds = maybe_subset(target_eval_ds)

    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    target_eval_loader = DataLoader(
        target_eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    return source_loader, target_loader, target_eval_loader


# --------------------- Office-31 helpers --------------------- #
def _o31_resolve_domain_dir(root: Path, domain: str) -> Path:
    if domain not in OFFICE31_DOMAIN_TO_SUBDIR:
        raise ValueError(f"Unknown Office-31 domain '{domain}'. Expected one of {list(OFFICE31_DOMAIN_TO_SUBDIR.keys())}.")
    domain_dir = root / OFFICE31_DOMAIN_TO_SUBDIR[domain]
    if not domain_dir.exists():
        raise FileNotFoundError(f"Domain path does not exist: {domain_dir}")
    return domain_dir


def _o31_list_sorted_classes(domain_dir: Path) -> List[str]:
    classes = sorted([p.name for p in domain_dir.iterdir() if p.is_dir()])
    if not classes:
        raise ValueError(f"No class subfolders found in {domain_dir}")
    return classes


def _o31_build_shared_class_mapping(
    root: Path, source_domain: str, target_domain: str, debug_classes: bool = False
) -> Dict[str, int]:
    canonical_domain = "A" if "A" in {source_domain, target_domain} else source_domain
    canonical_dir = _o31_resolve_domain_dir(root, canonical_domain)
    canonical_classes = _o31_list_sorted_classes(canonical_dir)

    src_dir = _o31_resolve_domain_dir(root, source_domain)
    tgt_dir = _o31_resolve_domain_dir(root, target_domain)
    src_classes = _o31_list_sorted_classes(src_dir)
    tgt_classes = _o31_list_sorted_classes(tgt_dir)

    if set(canonical_classes) != set(src_classes) or set(canonical_classes) != set(tgt_classes):
        raise ValueError(
            "Office-31 class folders differ across domains. "
            "Ensure Amazon/DSLR/Webcam share the same 31 class names."
        )

    class_to_idx = {cls: idx for idx, cls in enumerate(canonical_classes)}
    if debug_classes:
        print("Office-31 classes (sorted):")
        for idx, name in enumerate(canonical_classes):
            print(f"  {idx}: {name}")
    return class_to_idx


def _o31_get_loaders(
    root: str,
    source_domain: str,
    target_domain: str,
    batch_size: int,
    num_workers: int = 4,
    debug_classes: bool = False,
    max_samples_per_domain: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Office-31 root does not exist: {root_path}")

    class_to_idx = _o31_build_shared_class_mapping(root_path, source_domain, target_domain, debug_classes=debug_classes)
    src_dir = _o31_resolve_domain_dir(root_path, source_domain)
    tgt_dir = _o31_resolve_domain_dir(root_path, target_domain)

    source_ds = datasets.ImageFolder(src_dir.as_posix(), transform=_build_transforms(train=True))
    target_ds = datasets.ImageFolder(tgt_dir.as_posix(), transform=_build_transforms(train=True))
    target_eval_ds = datasets.ImageFolder(tgt_dir.as_posix(), transform=_build_transforms(train=False))

    def maybe_subset(ds):
        if max_samples_per_domain is not None and len(ds) > max_samples_per_domain:
            subset = Subset(ds, list(range(max_samples_per_domain)))
            for attr in ("class_to_idx", "classes", "root", "transform"):
                if hasattr(ds, attr):
                    setattr(subset, attr, getattr(ds, attr))
            return subset
        return ds

    for name, ds in [("source", source_ds), ("target_train", target_ds), ("target_eval", target_eval_ds)]:
        if set(ds.class_to_idx.keys()) != set(class_to_idx.keys()):
            missing_ds = sorted(set(class_to_idx.keys()) - set(ds.class_to_idx.keys()))
            missing_map = sorted(set(ds.class_to_idx.keys()) - set(class_to_idx.keys()))
            raise ValueError(
                f"Office-31 {name} class names differ from canonical mapping.\n"
                f"Missing in {name}: {missing_ds}\n"
                f"Missing in canonical: {missing_map}"
            )
        if ds.class_to_idx != class_to_idx:
            raise ValueError(
                f"Office-31 {name} class_to_idx ordering does not match canonical mapping.\n"
                f"{name} mapping: {ds.class_to_idx}\n"
                f"Canonical:      {class_to_idx}"
            )

    source_ds = maybe_subset(source_ds)
    target_ds = maybe_subset(target_ds)
    target_eval_ds = maybe_subset(target_eval_ds)

    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    target_eval_loader = DataLoader(target_eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return source_loader, target_loader, target_eval_loader


def get_domain_loaders(
    dataset_name: str,
    source_domain: str,
    target_domain: str,
    batch_size: int,
    root: str | None = None,
    num_workers: int = 4,
    debug_classes: bool = False,
    max_samples_per_domain: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    name = _normalize_dataset_name(dataset_name)

    if name in ("officehome",):
        if root is None:
            root = DEFAULT_OFFICE_HOME_ROOT
        return _oh_get_loaders(
            root=root,
            source_domain=source_domain,
            target_domain=target_domain,
            batch_size=batch_size,
            num_workers=num_workers,
            debug_classes=debug_classes,
            max_samples_per_domain=max_samples_per_domain,
        )
    elif name in ("office31",):
        if root is None:
            root = DEFAULT_OFFICE31_ROOT
        return _o31_get_loaders(
            root=root,
            source_domain=source_domain,
            target_domain=target_domain,
            batch_size=batch_size,
            num_workers=num_workers,
            debug_classes=debug_classes,
            max_samples_per_domain=max_samples_per_domain,
        )
    else:
        raise ValueError(f"Unknown dataset_name '{dataset_name}'. Expected 'office_home' or 'office31'.")
