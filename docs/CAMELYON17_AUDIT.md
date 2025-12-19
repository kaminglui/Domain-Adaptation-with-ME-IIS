# Camelyon17 (WILDS) Implementation Audit

- **Dataset loading**
  - `src/datasets/wilds_camelyon17.py::get_camelyon17_dataset` calls `wilds.get_dataset(dataset="camelyon17", root_dir=..., download=..., unlabeled=...)` and is invoked from `build_camelyon17_loaders`.

- **WILDS splits in use**
  - Labeled training: `src/datasets/wilds_camelyon17.py::build_camelyon17_splits` builds `labeled_train = dataset.get_subset("train", transform=train_transform)` and `build_camelyon17_loaders` feeds it to `get_train_loader` (optionally wrapped in `WithIndexDataset` when `include_indices_in_train=True` for ME-IIS).
  - Unlabeled adaptation: in `build_camelyon17_loaders`, `unlabeled_subset = splits.unlabeled_test` when `split_mode=="uda_target"` else `splits.unlabeled_val`; the loader is built with `get_train_loader("standard", unlabeled_subset, ...)`.
  - Evaluation: `build_camelyon17_loaders` creates `val_loader` on `splits.labeled_val`, `test_loader` on `splits.labeled_test`, and optional `id_val_loader` on `splits.labeled_id_val` (if the subset exists). Metrics are computed via `src/train/train_loop.py::eval_wilds_split`, which prefers `wilds_dataset.eval(...)` when available.

- **Feature extraction**
  - `src/models/backbones.py::DenseNet121Backbone.forward` returns global-average-pooled DenseNet features (`net.features` → ReLU → adaptive_avg_pool → flatten) with `out_features = net.classifier.in_features` (~1024).
  - `src/models/backbones.py::ResNet50Backbone.forward` uses torchvision ResNet-50 with `fc` set to `Identity`, returning the final avgpooled/flattened features of dimension 2048.
  - These backbones are passed as `featurizer` to algorithms: `ERM.forward`, `DANN.update`, and `MEIIS.forward` all call `self.featurizer(x)`; DANN also reuses the same feature tensor for the domain discriminator after `grad_reverse`. There is no separate `extract_features` API; features come from the final pooled backbone outputs.

- **Target pseudo-probabilities**
  - Computed in `src/algorithms/me_iis.py::MEIIS.update_importance_weights`: for each target batch, `logits = self.forward(x)` followed by `probs = F.softmax(logits, dim=1)`; confidence/entropy filters use these pseudo-probabilities.

- **ME-IIS constraint Pg construction**
  - In `MEIIS.update_importance_weights` (pass 2), confident target samples feed `self.adapter.get_joint_features({layer: feats_keep}, probs_keep)`; the flattened joints are averaged (`target_moments_sum/target_count`) to form the target constraint vector.
  - In `models/me_iis_adapter.py::MaxEntAdapter.solve_iis`, `iis.compute_pg(target_flat)` computes Pg from the flattened target joint (or an override).

- **IIS updates and logged objective**
  - `models/me_iis_adapter.py::MaxEntAdapter.solve_iis` performs the IIS iterations (delta_lambda, weight renormalization). The dual objective `objective = (lambda_vec * target_moments).sum() - logZ` is printed each iteration (`[IIS] iter ... obj ...`) and stored in `IISIterationStats.objective`; `MEIIS.update_importance_weights` surfaces this history via the returned `iis_objective` list.

- **Source weights applied to training loss**
  - `src/algorithms/me_iis.py::MEIIS.update` computes per-sample cross-entropy and, when `batch.idx` and `self.source_weights` are available, looks up `w = self.source_weights[idx_cpu]`, normalizes by `w.sum()` (batch-normalized even when `normalize_weights="global"`), and uses `(w * per_sample).sum() / denom`; otherwise defaults to an unweighted mean.

- **Run IDs, checkpoints, and skip logic**
  - Run IDs: `src/utils/run_id.py::encode_config_to_run_id` builds `k=v__...__h=HASH` strings from config fields; `decode_run_id_to_config` reverses them.
  - Run directory: `src/run_experiments.py::run_experiments` sets `run_dir = Path(ckpt_root) / run_id`.
  - Checkpoints/skip: `src/train/train_loop.py::train` writes `config.json`, checkpoints `best.pt`/`last.pt`, and `results.json`; if both `results.json` and `best.pt` exist and `force_rerun` is False, it returns the saved results without training. Resume pulls state from `last.pt` via `_load_ckpt`.

- **Existing efficiency features**
  - Mixed precision: `train` wraps forward/backward in `torch.autocast(device_type="cuda", dtype=torch.float16)` with `GradScaler` when `cfg["amp"]` and CUDA are enabled.
  - Gradient accumulation: `grad_accum_steps` in `train` scales loss and delays optimizer steps to achieve larger effective batches.
  - Dataloader performance: `src/datasets/wilds_camelyon17.py::build_camelyon17_loaders` forwards `num_workers`, `pin_memory`, `prefetch_factor`, and `persistent_workers` (when workers > 0) to WILDS `get_train_loader`/`get_eval_loader`, with a compatibility shim that retries without unsupported kwargs.
  - Early-stop/resume reduce wasted epochs via `early_stop_patience` and checkpoint reload in `train`.
  - NOT FOUND: explicit dataset caching location selection beyond WILDS defaults; dynamic batch sizing/auto OOM adjustment; caching of precomputed features or logits (would logically live in `src/datasets/wilds_camelyon17.py` or `src/train/train_loop.py` if added).
