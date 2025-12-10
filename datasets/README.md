# datasets – Data Loaders and Roots

## Purpose
Holds dataset loader code for Office-Home and Office-31 and serves as the default root for downloaded datasets.

## Contents
- `domain_loaders.py` – Core aligned loaders with class mapping checks and transforms.
- `office_home.py`, `office31.py` – Dataset-specific helpers.
- `__init__.py` – Package marker.
- [Office-31](Office-31/README.md) – Optional dataset tree for Office-31.
- [Office-Home](Office-Home/README.md) – Optional dataset tree for Office-Home.

## Main APIs
### `get_domain_loaders(dataset_name, source_domain, target_domain, batch_size, root=None, num_workers=4, debug_classes=False, max_samples_per_domain=None, generator=None, worker_init_fn=None)`
- Builds aligned source/target loaders plus an eval loader for the target domain.
- Validates that class folders match exactly across domains.

Key parameters:
- `dataset_name` (str; "office_home" | "office31") – Benchmark selector.
- `source_domain`, `target_domain` (str) – Domain codes (Ar/Cl/Pr/Rw or A/D/W).
- `batch_size` (int; default 32) – Loader batch size.
- `root` (str|None) – Dataset root; defaults to `DEFAULT_OFFICE_HOME_ROOT` / `DEFAULT_OFFICE31_ROOT`.
- `num_workers` (int; default 4) – DataLoader workers.
- `debug_classes` (bool) – Print class mappings.
- `max_samples_per_domain` (int|None) – Optional subsampling cap.
- `generator`, `worker_init_fn` – Determinism helpers (seeds).

Returns: `(source_loader, target_loader, target_eval_loader)`; raises `ValueError` if class sets/order differ.

## Dependencies and Interactions
- Used by `scripts/train_source.py`, `scripts/adapt_me_iis.py`, `scripts/eval_source_only.py`.
- Torchvision `ImageFolder` for images and standard ImageNet transforms.

## Notes / Gotchas
- Office-Home Real World may be spelled differently; resolution handled internally.
- Class folder mismatches cause explicit errors to protect label alignment.
