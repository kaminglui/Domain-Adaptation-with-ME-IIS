# ME-IIS Parameter Tuning Report (Factual + Evidence-Based)

This report summarizes tunable parameters for ME-IIS in this repo, and ties observed behaviors to the available logs/artifacts in the repository.

## D1) Tunable parameters and roles (as implemented)

Parameters are exposed in two places:
- Legacy scripts: `src/cli/args.py:AdaptConfig` / `scripts/adapt_me_iis.py`
- Unified runner (notebook): `src/experiments/run_config.py:RunConfig.method_params` consumed by `src/experiments/methods/me_iis.py`

### Constraint feature definition
- `feature_layers`
  - Legacy: `--feature_layers` parsed by `utils/experiment_utils.py:parse_feature_layers(...)`
  - Unified: `method_params["feature_layers"]` (or `RunConfig.feature_layers`)
  - Effect: chooses which backbone layers contribute latent “style” variables and thus which constraints exist (changes `N_c` and `C_total`)

### Latent style model / clustering
- `num_latent_styles` / `components_per_layer`
  - Legacy: `--num_latent_styles`, `--components_per_layer`
  - Unified: `method_params["num_latent_styles"]`, `method_params["components_per_layer"]`
  - Effect: number of mixture components per layer (changes constraint dimensionality; too large can increase “unachievable constraints” risk)
- `gmm_selection_mode`, `gmm_bic_min_components`, `gmm_bic_max_components`
  - Effect: fixed components vs. per-layer BIC selection (see `clustering/gmm_backend.py`)
- `cluster_backend`
  - `gmm` (`clustering/gmm_backend.py:GMMBackend`) vs `vmf_softmax` (`clustering/vmf_softmax_backend.py:VMFSoftmaxBackend`)
- `vmf_kappa`, `kmeans_n_init`
  - Only for `vmf_softmax`: controls softmax concentration and prototype fitting stability
- `cluster_clean_ratio`
  - If `<1`, fits clustering using only the lowest-entropy target predictions (`models/me_iis_adapter.py:fit_target_structure`)

### Source probability mode (for joint style×class constraints)
- `source_prob_mode` (`softmax` vs `onehot`)
  - Legacy: `--source_prob_mode`
  - Unified: `method_params["source_prob_mode"]`
  - Effect: whether source class probabilities in the joint constraint use model posteriors (soft) or true one-hot labels

### IIS solver hyperparameters
- `iis_iters` / `iis_tol`
  - Controls max iterations and early stop tolerance in `models/me_iis_adapter.py:MaxEntAdapter.solve_iis`

### Adaptation (weighted fine-tuning) hyperparameters
- `adapt_epochs`
  - Number of weighted fine-tuning epochs
- `finetune_backbone`, `backbone_lr_scale`, `classifier_lr`
  - Controls whether the backbone is updated, and the LR ratio between backbone and classifier
- `batch_size`, `weight_decay`, (and base `lr_*` in the unified runner)

### Optional pseudo-label stage (if enabled)
- `use_pseudo_labels`, `pseudo_conf_thresh`, `pseudo_max_ratio`, `pseudo_loss_weight`
  - Legacy: `--use_pseudo_labels` + `--pseudo_*`
  - Unified: `method_params["use_pseudo_labels"]` + `method_params["pseudo_*"]`
  - Effect: adds a target pseudo-label loss term during adaptation (see `scripts/adapt_me_iis.py:adapt_epoch` and `src/experiments/methods/me_iis.py:adapt_epoch`)

## D2) Observed behaviors / failure modes in available artifacts

### (1) Colab notebook: seed=0 metrics table shows ME-IIS underperforming baselines
The committed notebook output in `notebooks/Run_All_Experiments.ipynb` contains, for Office-Home `Ar→Cl` (seed=0), the following displayed metrics (from existing `metrics.csv` files under `outputs/runs/office-home/Ar2Cl/...` at the time of execution):

| method | seed | source_acc | target_acc | run_id |
|---|---:|---:|---:|---|
| `source_only` | 0 | 78.038731 | 30.148912 | f2fc29f097 |
| `me_iis` | 0 | 76.967450 | 28.568156 | d3b3fbdc39 |
| `dann` | 0 | 80.428513 | 33.127148 | ab83d40821 |
| `coral` | 0 | 80.428513 | 33.127148 | ffc0b32a86 |

In that table, ME-IIS (`me_iis`) is lower than source-only and lower than DANN/CORAL for that single reported seed.

### (2) Colab notebook: repeated failures with CUDA “device-side assert”
The notebook output also shows multiple run attempts failing with:
- `AcceleratorError: CUDA error: device-side assert triggered`

Those failures occur across methods/seeds in the stored output and indicate that (in that Colab execution) training/adaptation crashed before producing new `metrics.csv` for the attempted `run_id`s.

### (3) Legacy IIS artifact in this repo shows non-converging max moment error and weight concentration
This repo contains a legacy IIS artifact:
- `results/me_iis_weights_Ar_to_Cl_layer3-layer4-avgpool_seed0.npz`

Its stored iteration traces include (final values):
- `moment_max` (max abs moment error): `0.7249999`
- `moment_l2`: `1.0382177`
- `kl`: `26.557272`
- `w_entropy`: `2.5852497` (entropy decreases over iterations)
- `w_max`: `0.39320147` and `w_min`: `0.005336082`
- `feature_mass_mean`: `3.0` with `feature_mass_std ≈ 5.4e-08` (mass condition holds for 3 layers)

In the same run’s checkpoint:
- `checkpoints/source_only_Ar_to_Cl_seed0.pth` reports it was a dry run (`num_epochs=1`, `dry_run_max_samples=64`, `dry_run_max_batches=5`), and
- `checkpoints/me_iis_Ar_to_Cl_layer3-layer4-avgpool_seed0.pth` includes the same IIS history with `max_moment_error ≈ 0.725` at the end.

Interpretable implications from these logged values:
- The IIS solver is updating weights (entropy decreases; `w_max` increases), but the **maximum moment error does not decrease** meaningfully across iterations in that artifact.
- In `models/me_iis_adapter.py:MaxEntAdapter.solve_iis`, a constant, large max moment error is consistent with at least one “unachievable constraint” (a target moment with positive mass that has zero mass in the source joint features).

## D3) Minimal tuning plan (seed=0 only; fair budget)

The notebook already encodes a “fairness guard” and shared epoch budgets across methods. A minimal feasibility sweep (seed=0 only) that stays within that structure:

1. `num_latent_styles`: `[3, 5, 8]`
2. `feature_layers`: `[["layer4"], ["layer3","layer4"]]`
3. `source_prob_mode`: `["onehot", "softmax"]`
4. `finetune_backbone`: `[False, True]` with `backbone_lr_scale` kept small when `True` (e.g., `0.1`)

Keep fixed across all configs:
- `EPOCHS_SOURCE`, `EPOCHS_ADAPT`, batch size, and optimizer schedule (same training budget for all methods)
- `cluster_backend` fixed initially (e.g., `gmm` with `gmm_selection_mode="bic"` or `fixed`) to reduce sweep size

Logging recommendation for the sweep:
- Use the unified runner (as in the notebook) so each config has a deterministic `run_id` and a per-run `metrics.csv`.
- Export one aggregated CSV via the notebook export cell (writes `outputs/runs/{dataset_tag}/{src}2{tgt}/all_metrics.csv`) which includes `run_id` and `method_params_json`.

