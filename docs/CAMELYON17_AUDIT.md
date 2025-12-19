# Camelyon17 (WILDS) Implementation Audit

This document is a "what exists today" audit of the current main branch implementation for Camelyon17 runs.

## Dataset loading and splits

- **Where Camelyon17 is loaded**
  - `src/datasets/wilds_camelyon17.py::get_camelyon17_dataset` calls `wilds.get_dataset(dataset="camelyon17", root_dir=..., download=..., unlabeled=...)`.

- **Which WILDS splits are used**
  - **Labeled training**: `src/datasets/wilds_camelyon17.py::build_camelyon17_splits` constructs `labeled_train = dataset.get_subset("train", ...)`. `src/datasets/wilds_camelyon17.py::build_camelyon17_loaders` builds `train_loader` via `wilds.common.data_loaders.get_train_loader("standard", labeled_train_ds, ...)` (optionally wrapped by `src/datasets/wilds_camelyon17.py::WithIndexDataset` when `include_indices_in_train=True` for ME-IIS).
  - **Unlabeled adaptation**:
    - `src/datasets/wilds_camelyon17.py::build_camelyon17_splits` constructs `unlabeled_val = dataset.get_subset("val_unlabeled", ...)` and `unlabeled_test = dataset.get_subset("test_unlabeled", ...)`, and optionally `unlabeled_train = dataset.get_subset("train_unlabeled", ...)` when present in the installed WILDS version.
    - `src/datasets/wilds_camelyon17.py::build_camelyon17_loaders` selects the adaptation subset via config `adapt_split` ("val_unlabeled" or "test_unlabeled") and builds `unlabeled_loader` via `wilds.common.data_loaders.get_train_loader("standard", adapt_subset, ...)`.
    - Optional `unlabeled_train_loader` is built when `unlabeled_train` exists (for debugging/ablation; not used by the main trainer).
  - **Evaluation**: `src/datasets/wilds_camelyon17.py::build_camelyon17_loaders` builds `val_loader` on `splits.labeled_val`, `test_loader` on `splits.labeled_test`, and optional `id_val_loader` on `splits.labeled_id_val` using `wilds.common.data_loaders.get_eval_loader(...)`.
    - Metrics are computed in `src/train/trainer.py::eval_wilds_split`, which calls `dataset.eval(...)` when available (fallback: manual accuracy).
      - For classification, `eval_wilds_split` passes predicted labels (argmax over logits) into `dataset.eval(...)`.
      - If `dataset.eval(...)` returns `(results_dict, results_str)`, `eval_wilds_split` consumes only the dict.
    - Early stopping and "best checkpoint" selection use `val_loader` in `src/train/trainer.py::train`. Final reporting includes `val`, `test`, and optional `id_val` in `src/train/trainer.py::train`.
  - **Protocol config and enforcement**: `src/datasets/wilds_camelyon17.py::build_camelyon17_loaders`
    - requires `split_mode` in {"uda_target","align_val"}
    - requires `eval_split` in {"val","test"} and `adapt_split` in {"val_unlabeled","test_unlabeled"}
    - enforces that adapting on `test_unlabeled` is only allowed when `split_mode=="uda_target"` (to avoid accidental test leakage).

## Feature extraction (f(x))

- **Which class/function produces f(x)**
  - Canonical API is `src/algorithms/base.py::Algorithm.extract_features`, which computes `f(x) = self.featurizer(x)`.
  - Used by:
    - `src/algorithms/erm.py::ERM.forward`
    - `src/algorithms/dann.py::DANN.forward` (and `src/algorithms/dann.py::DANN.update` reuses the same `feats_all` for the domain discriminator after `grad_reverse`)
    - `src/algorithms/me_iis.py::MEIIS.forward` and `src/algorithms/me_iis.py::MEIIS.update_importance_weights`

- **What layer is used**
  - Features are the final pooled backbone outputs (no bottleneck/projection head in the Camelyon17 path).
    - `src/models/backbones.py::DenseNet121Backbone.forward` uses torchvision DenseNet features -> ReLU -> `adaptive_avg_pool2d(..., (1,1))` -> flatten.
    - `src/models/backbones.py::ResNet50Backbone.forward` uses torchvision ResNet-50 with `fc = Identity()`, returning the final pooled feature vector.

- **Shape of f(x)**
  - For batch size `B`, `f(x)` has shape `(B, feature_dim)` where `feature_dim` is `BackboneOutput.feature_dim` from `src/models/backbones.py::build_backbone` (DenseNet: `DenseNet121Backbone.out_features = int(net.classifier.in_features)`; ResNet-50: `ResNet50Backbone.out_features = 2048`).

- **Feature logging (once per run)**
  - `src/algorithms/base.py::Algorithm.extract_features` prints (once) the backbone name, pretrained flag, `feat_layer`, `feature_dim`, `f(x)` shape, and module mode ("train"/"eval") at first feature extraction.

- **Model mode during ME-IIS feature/pseudo-prob extraction**
  - `src/algorithms/me_iis.py::MEIIS.update_importance_weights` switches the model to eval mode for stable pseudo-probabilities and restores the prior mode in a `finally:` block.

## Target pseudo-probabilities

- `src/algorithms/me_iis.py::MEIIS.update_importance_weights` computes pseudo-probabilities via:
  - `logits = self.forward(x)` then `probs = torch.nn.functional.softmax(logits, dim=1)`
  - confidence: `conf = probs.max(dim=1).values`
  - entropy: `ent = -(probs * log(probs)).sum(dim=1)`

## ME-IIS constraints Pg (target moments)

- **Where target constraints are computed**
  - `src/algorithms/me_iis.py::MEIIS.update_importance_weights` (pass 2) builds joint constraints on confident target samples:
    - `joint = models/me_iis_adapter.py::MaxEntAdapter.get_joint_features({layer: feats_keep}, probs_keep)[layer]` with shape `(N_keep, J, K)`
    - `flat = joint.reshape(N_keep, -1)` and the target constraint vector is `target_moments = flat.mean(dim=0)` (optionally EMA-smoothed into `self.target_moments_ema`).

- **Where Pg is formed/consumed in IIS**
  - `models/iis_components.py::IISUpdater.compute_pg(flat_joint)` computes Pg as `flat_joint.mean(dim=0)` when used.
  - `models/me_iis_adapter.py::MaxEntAdapter.solve_iis` calls `compute_pg(...)` on the provided target joint, but if `target_moments_override` is passed it replaces Pg with that override vector.
  - In the Camelyon17 ME-IIS path, `src/algorithms/me_iis.py::MEIIS.update_importance_weights` passes `target_moments_override=target_moments_used` to `models/me_iis_adapter.py::MaxEntAdapter.solve_iis_from_joint` -> `solve_iis`, so the effective Pg used for IIS updates is the override computed from all confident target samples (not just the small `target_joint` passed for validation/mass checks).

## IIS updates and logged objective

- **Where IIS updates happen**
  - IIS iterations are implemented in `models/me_iis_adapter.py::MaxEntAdapter.solve_iis` using helpers from `models/iis_components.py::IISUpdater`:
    - `pm = IISUpdater.compute_pm(weights, source_flat)`
    - `delta = IISUpdater.delta_lambda(pg, pm)` (Eq. 18, with denominator `Nd+Nc` exposed by `IISUpdater.mass_constant`)
    - weight updates + renormalization happen inline in `solve_iis` (equivalent to `IISUpdater.update_weights`)

- **What objective is logged**
  - Per-iteration objective in `models/me_iis_adapter.py::MaxEntAdapter.solve_iis`:
    - `objective = (lambda_vec * target_moments).sum() - logZ`
  - Printed every iteration (`[IIS] iter ... obj ...`) and stored in `models/me_iis_adapter.py::IISIterationStats.objective`.
  - `src/algorithms/me_iis.py::MEIIS.update_importance_weights` returns this trace as `iis_objective` (list of floats).

- **Monotonicity check**
  - `models/me_iis_adapter.py::MaxEntAdapter.solve_iis` prints `[IIS][WARN] Dual objective decreased ...` if the objective drops by more than `1e-10` between iterations.
  - `src/algorithms/me_iis.py::MEIIS.update_importance_weights` sets `iis_objective_decreased` and, when `MEIISConfig.debug=True` and `run_dir` is provided by the trainer, writes a small debug dump JSON file into the run directory.

## Source weights applied to training loss

- **Where weights are computed and stored**
  - `src/algorithms/me_iis.py::MEIIS.update_importance_weights` calls `models/me_iis_adapter.py::MaxEntAdapter.solve_iis_from_joint` and stores the resulting weight vector in `src/algorithms/me_iis.py::MEIIS.source_weights`.
    - Optional stabilizers: `MEIISConfig.weight_clip_max` (clip + renormalize) and `MEIISConfig.weight_mix_alpha` (mix with uniform + renormalize).

- **Where weights are applied**
  - `src/algorithms/me_iis.py::MEIIS.update` computes per-sample cross-entropy (`reduction="none"`), looks up `w = self.source_weights[batch.idx]`, and uses a batch-normalized weighted average `(w * per_sample).sum() / w.sum()`.
  - `batch.idx` is provided only when `src/datasets/wilds_camelyon17.py::build_camelyon17_loaders(..., include_indices_in_train=True)` wraps the training subset with `src/datasets/wilds_camelyon17.py::WithIndexDataset`.

## Run IDs, checkpoints, and skip logic

- **Run ID construction**
  - `src/utils/run_id.py::encode_config_to_run_id` encodes a flat config dict into `k=v__...__h=HASH` (HASH is `sha1(json.dumps(config, sort_keys=True))[:8]`).
  - `src/run_experiments.py::run_experiments` uses this to choose `run_dir = Path(ckpt_root) / run_id`.

- **Checkpoint saving and resume**
  - `src/train/trainer.py::_save_ckpt` writes `best.pt` and `last.pt`, including algorithm/optimizer/scaler state and RNG state.
  - `src/train/trainer.py::_load_ckpt` restores those states (best/last).

- **Skip logic ("already done")**
  - `src/train/trainer.py::train` skips only if `results.json` and `best.pt` exist and `config_fingerprint.txt` matches the fingerprint of the provided config (computed by `src/utils/run_id.py::fingerprint_config`).
  - If the fingerprints mismatch and `force_rerun` is false, `src/train/trainer.py::train` raises (to avoid silently mixing artifacts).
  - `src/train/trainer.py::train` writes `config.json` only after the skip check passes (so a skipped run does not overwrite config metadata).

## Existing efficiency features

- **Mixed precision (AMP)**: `src/train/trainer.py::train` uses `torch.autocast(device_type="cuda", dtype=torch.float16)` + `torch.cuda.amp.GradScaler` when `cfg["amp"]` and CUDA are enabled.
- **Gradient accumulation**: `src/train/trainer.py::train` implements `grad_accum_steps` by scaling loss and stepping the optimizer every N steps.
- **DataLoader performance**: `src/datasets/wilds_camelyon17.py::build_camelyon17_loaders` forwards `num_workers`, `pin_memory`, `prefetch_factor`, and `persistent_workers` (when workers > 0) to WILDS `get_train_loader` / `get_eval_loader`, via a compatibility shim `src/datasets/wilds_camelyon17.py::_call_wilds_loader` that retries without unsupported kwargs.
- **Built-in dataset caching**: WILDS caches under `root_dir` provided to `wilds.get_dataset` in `src/datasets/wilds_camelyon17.py::get_camelyon17_dataset`.
- **Dynamic batch sizing**: `src/train/trainer.py::train` supports `cfg["batch_size"] = "auto"` (CUDA-only) and probes a list of candidate batch sizes to find the largest that does not OOM.
- **Optional torch.compile**: `src/train/trainer.py::train` supports `cfg["compile"]` (guarded; disabled by default).
- **NOT FOUND**: caching of precomputed features/logits for ME-IIS updates (would likely live in `src/algorithms/me_iis.py::MEIIS.update_importance_weights` and/or around the epoch loop in `src/train/trainer.py::train` if added).
