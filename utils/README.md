# utils – Helper Modules

## Purpose
Shared utilities for data loading, seeding, feature extraction, experiment bookkeeping, logging, and tests.

## Contents
- `data_utils.py` – DataLoader builders with deterministic seeding.
- `env_utils.py` – Colab detection helper.
- `experiment_utils.py` – Feature-layer parsing, components map construction, checkpoint naming, dataset tagging.
- `feature_utils.py` – Feature extraction helpers for models.
- `logging_utils.py` – CSV/TensorBoard helpers; defines `OFFICE_HOME_ME_IIS_FIELDS`.
- `seed_utils.py` – Deterministic seed/device helpers.
- `test_utils.py` – Tiny models and synthetic dataset builders for tests.
- `__init__.py` – Package marker.

## Key APIs
- `build_loader(dataset, batch_size, shuffle, num_workers, seed, generator, drop_last)` – Wrapper for deterministic loaders.
- `make_generator(seed)`, `make_worker_init_fn(seed)` – Seed helpers for reproducibility.
- `parse_feature_layers(layers_str)` – Validated comma-separated layer string → list.
- `build_components_map(feature_layers, default_components, override_str)` – Per-layer GMM component mapping (supports `layer:count` or positional lists).
- `build_source_ckpt_path(source_domain, target_domain, seed, base_dir="checkpoints")` – Canonical source checkpoint path.
- `extract_features(model, loader, device, return_intermediates=False, feature_layers=None)` – Collects features/logits/labels.
- `append_csv(path, fieldnames, row)`, `TBLogger(log_dir)` – Logging utilities.
- `set_seed(seed, deterministic=True)`, `get_device(gpu_preference=None, deterministic=True)` – Seed/device utilities.
- Test helpers: `build_tiny_model`, `create_office_home_like`, `temporary_workdir`.

## Notes
- `parse_feature_layers` raises if no valid layers are provided.
- Component overrides validate positive integers and known layer names.
