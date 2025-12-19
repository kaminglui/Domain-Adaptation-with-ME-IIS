# Camelyon17 (WILDS) Readiness Audit (main branch)

Repo state: `main` @ `0fca5d1d30251220d9f42f3aafe243568ceb0c2a`

This is a **no-behavior-change** audit with **exact code pointers** for Camelyon17 (WILDS) runs, focusing on:
- dataset loading + split usage (label leakage avoidance),
- ME-IIS forward path (features → pseudo-probs → constraints → IIS → weights),
- method selection/routing,
- smoke-test hooks,
- efficiency hooks.

If something is not present in the Camelyon17 (WILDS) pipeline, it is explicitly marked **MISSING** with what to add.

---

## Where Camelyon17 is loaded via WILDS (module + function)

- `src/datasets/wilds_camelyon17.py::get_camelyon17_dataset`
  - calls `wilds.get_dataset(dataset="camelyon17", root_dir=..., download=..., unlabeled=...)`.

---

## Which WILDS splits are used

### Split construction

- `src/datasets/wilds_camelyon17.py::build_camelyon17_splits`
  - **Labeled training split**: `dataset.get_subset("train", transform=train_transform)`
  - **Evaluation splits**:
    - `dataset.get_subset("val", transform=eval_transform)`
    - `dataset.get_subset("test", transform=eval_transform)`
    - optional `dataset.get_subset("id_val", ...)` via `src/datasets/wilds_camelyon17.py::_maybe_get_subset`
  - **Unlabeled splits**:
    - `dataset.get_subset("val_unlabeled", transform=eval_transform)`
    - `dataset.get_subset("test_unlabeled", transform=eval_transform)`
    - optional `dataset.get_subset("train_unlabeled", ...)` via `src/datasets/wilds_camelyon17.py::_maybe_get_subset`

### Loader selection + leakage guards

- `src/datasets/wilds_camelyon17.py::build_camelyon17_loaders`
  - **Labeled training**: builds `train_loader` via `wilds.common.data_loaders.get_train_loader("standard", labeled_train_ds, ...)`
    - optional index injection for per-sample weighting: `src/datasets/wilds_camelyon17.py::WithIndexDataset` (enabled by `config["include_indices_in_train"]=True`)
  - **Unlabeled adaptation**: selects adaptation subset based on `config["adapt_split"]`:
    - `"val_unlabeled"` → `splits.unlabeled_val`
    - `"test_unlabeled"` → `splits.unlabeled_test`
  - **Evaluation**:
    - `val_loader` on `splits.labeled_val` via `wilds.common.data_loaders.get_eval_loader(...)`
    - `test_loader` on `splits.labeled_test` via `wilds.common.data_loaders.get_eval_loader(...)`
    - optional `id_val_loader` on `splits.labeled_id_val` via `wilds.common.data_loaders.get_eval_loader(...)`
  - **Split protocol enforcement**:
    - `split_mode` must be `"uda_target"` or `"align_val"`
    - `eval_split` must be `"val"` or `"test"`
    - `adapt_split` must be `"val_unlabeled"` or `"test_unlabeled"`
    - `"align_val"` requires `adapt_split="val_unlabeled"` and `eval_split="val"`
    - `"uda_target"` requires `adapt_split="test_unlabeled"`
    - refuses adapting on `test_unlabeled` unless `split_mode=="uda_target"`

### Evaluation metrics path

- `src/train/trainer.py::eval_wilds_split`
  - uses `dataset.eval(...)` when available (WILDS datasets expose it)
  - for classification, passes predicted labels via `argmax` over logits

---

## Where feature extraction happens (the exact forward path)

### What function produces f(x)

- Canonical feature API: `src/algorithms/base.py::Algorithm.extract_features`
  - computes `f(x) = self.featurizer(x)`
  - prints one-time log including backbone/feature_dim/`f(x)` shape

### Whether f(x) is final pooled backbone features or intermediate layers

- **Final pooled backbone features** (no intermediate taps in the Camelyon17 path):
  - `src/models/backbones.py::DenseNet121Backbone.forward`
    - `torchvision.models.densenet121(...).features` → ReLU → `adaptive_avg_pool2d(...,(1,1))` → flatten
  - `src/models/backbones.py::ResNet50Backbone.forward`
    - `torchvision.models.resnet50(...); net.fc = Identity()` → returns final pooled vector

### Shape of f(x)

- `src/algorithms/base.py::Algorithm.extract_features` returns the featurizer output tensor.
- For batch size `B`, `f(x)` has shape `(B, feature_dim)` where `feature_dim` comes from `src/models/backbones.py::build_backbone` (`BackboneOutput.feature_dim`).

---

## Where target pseudo-probabilities are computed (softmax(logits))

- `src/algorithms/me_iis.py::MEIIS.update_importance_weights`
  - `_collect_fit(...)`: `probs = torch.nn.functional.softmax(logits, dim=1)`
  - `_collect_target_moments(...)`: `probs = torch.nn.functional.softmax(logits, dim=1)`

---

## Where ME-IIS constraints Pg are computed

- Target moment collection (effective Pg for IIS in Camelyon17 path):
  - `src/algorithms/me_iis.py::MEIIS.update_importance_weights`
    - `_collect_target_moments(...)` builds target joint constraints via:
      - `models/me_iis_adapter.py::MaxEntAdapter.get_joint_features(...)`
    - flattens per-sample constraints with `joint.reshape(N, -1)` and computes:
      - `target_moments = target_moments_sum / target_count` (mean over **selected** target samples)
    - uses `target_moments_override=target_moments_used` when calling IIS, so this override is the Pg used by IIS.
- Under-the-hood helper (not the primary Camelyon17 computation path):
  - `models/iis_components.py::IISUpdater.compute_pg(flat_joint)` returns `flat_joint.mean(dim=0)`.

---

## Where IIS iterations happen and what objective is logged (per-iteration)

- IIS loop:
  - `models/me_iis_adapter.py::MaxEntAdapter.solve_iis`
    - uses `models/iis_components.py::IISUpdater.compute_pm(...)` and `IISUpdater.delta_lambda(...)` (Eq. 18)
    - updates + renormalizes weights inline (equivalent to `IISUpdater.update_weights`)
- Per-iteration objective:
  - in `models/me_iis_adapter.py::MaxEntAdapter.solve_iis`:
    - `objective = (lambda_vec * target_moments).sum() - logZ`
    - printed each iter as `[IIS] iter ... | obj ...`
    - stored in `models/me_iis_adapter.py::IISIterationStats.objective`
- Objective trace surfaced:
  - `src/algorithms/me_iis.py::MEIIS.update_importance_weights` returns `iis_objective` (list of floats) and `iis_objective_decreased`.

---

## Where source weights w_i are applied to the training loss

- Weight computation/storage:
  - `src/algorithms/me_iis.py::MEIIS.update_importance_weights`
    - calls `models/me_iis_adapter.py::MaxEntAdapter.solve_iis_from_joint(...)`
    - stores weights in `src/algorithms/me_iis.py::MEIIS.source_weights` (registered buffer)
- Weighted loss application:
  - `src/algorithms/me_iis.py::MEIIS.update`
    - per-sample CE: `torch.nn.functional.cross_entropy(..., reduction="none")`
    - weight lookup: `w = self.source_weights[batch.idx]`
    - weighted loss: `(w * per_sample).sum() / w.sum()`
  - `batch.idx` exists only when `src/datasets/wilds_camelyon17.py::WithIndexDataset` is used (via `include_indices_in_train=True`).

---

## Where “method selection” happens (how the repo chooses ERM/DANN/ME-IIS runs)

- Config generation (currently always returns all 3):
  - `src/run_experiments.py::default_camelyon17_configs`
- Routing to algorithm implementation:
  - `src/run_experiments.py::run_experiments`
    - selects algorithm by `cfg.algorithm.upper()` and instantiates:
      - `src/algorithms/erm.py::ERM`
      - `src/algorithms/dann.py::DANN`
      - `src/algorithms/me_iis.py::MEIIS`
    - then calls `src/train/trainer.py::train`
- Notebook orchestrator:
  - `notebooks/Run_Camelyon17_WILDS.ipynb` imports `src.run_experiments.default_camelyon17_configs` and `src.run_experiments.run_experiments` and iterates configs.

- **MISSING (Camelyon17/WILDS)**: a CLI entrypoint supporting `--methods ...` to run only a subset (e.g. `me_iis` only) without running other baselines.

---

## Where “small test” knobs exist (dry_run_max_batches/samples, quick mode, etc.)

- **MISSING (Camelyon17/WILDS)**:
  - No `--smoke_test`, `dry_run_max_batches`, `dry_run_max_samples` routing exists in `src/run_experiments.py` or `src/train/trainer.py`.
- Present in non-WILDS domain CLI:
  - `tools/run_experiment.py` has `--mode quick|full` and `--one_batch_debug`
  - `src/experiments/run_config.py::RunConfig` contains `dry_run_max_samples` and `dry_run_max_batches`
- Legacy (not wired into Camelyon17/WILDS runner):
  - `legacy/run_smoke_tests.py` and `legacy/scripts/*` include dry-run knobs.

---

## Where efficiency features exist (AMP, grad accumulation, dynamic batch size, caching)

- AMP:
  - `src/train/trainer.py::train` uses `torch.autocast(device_type="cuda", dtype=torch.float16)` + `torch.cuda.amp.GradScaler` when `cfg["amp"]` and CUDA are enabled.
- Gradient accumulation:
  - `src/train/trainer.py::train` uses `cfg["grad_accum_steps"]` to delay optimizer steps.
- Dynamic batch size probing:
  - `src/train/trainer.py::train` supports `cfg["batch_size"] = "auto"` and probes candidate batch sizes.
- Dataloader performance kwargs:
  - `src/datasets/wilds_camelyon17.py::build_camelyon17_loaders` forwards `num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers` (when workers > 0)
  - WILDS signature differences handled by `src/datasets/wilds_camelyon17.py::_call_wilds_loader`
- Optional compile:
  - `src/train/trainer.py::train` supports `cfg["compile"]` (guarded).
- Dataset caching:
  - WILDS caches under `root_dir` passed to `wilds.get_dataset` in `src/datasets/wilds_camelyon17.py::get_camelyon17_dataset`.
  - **MISSING (Camelyon17/WILDS)**: an explicit “prefer local SSD else Drive” dataset root selection/copy policy (beyond the current `WILDS_DATA_ROOT`/default-root behavior).
- **MISSING (Camelyon17/WILDS)**: per-run `stdout.log` capture inside each `run_dir` (prints currently go only to the console).

