# scripts – Training, Adaptation, Evaluation, and Utilities

## Purpose
CLI entrypoints for source training, ME–IIS adaptation, evaluation, ablations, demos, plotting, and sanity checks.

## Contents
- `adapt_me_iis.py` – ME–IIS adaptation pipeline.
- `train_source.py` – Source-only training with optional source-self eval.
- `eval_source_only.py` – Evaluate a source checkpoint on its own domain.
- `run_me_iis_experiments.py` – Ablation driver (layers, GMM, ME–IIS variants).
- `demo_me_iis_toy.py` – Toy IIS demo.
- `plot_iis_dynamics.py` – Plot IIS convergence metrics.
- `run_office_home.sh` – Shell helper.
- `test_me_iis_sanity.py` – Quick IIS/adaptation sanity check.
- `__init__.py` – Package marker.

## Key CLIs

### adapt_me_iis.py
Run ME–IIS adaptation from a source checkpoint.
- Example:
  ```bash
  python scripts/adapt_me_iis.py --dataset_name office_home --data_root datasets/Office-Home \
    --source_domain Ar --target_domain Cl --checkpoint checkpoints/source_only_Ar_to_Cl_seed0.pth \
    --feature_layers "layer3,layer4" --gmm_selection_mode bic --adapt_epochs 10
  ```
- Important options:
  - `--dataset_name` (office_home|office31; default office_home)
  - `--data_root` (path|None; defaults per dataset)
  - `--source_domain`, `--target_domain` (required)
  - `--checkpoint` (required) – Source-only checkpoint
  - `--batch_size` (32), `--num_workers` (4)
  - `--feature_layers` ("layer3,layer4"), `--num_latent_styles` (5), `--components_per_layer` (override string)
  - `--gmm_selection_mode` (fixed|bic; default fixed), `--gmm_bic_min_components` (2), `--gmm_bic_max_components` (8)
  - `--source_prob_mode` (softmax|onehot; default softmax)
  - `--iis_iters` (15), `--iis_tol` (1e-3)
  - `--adapt_epochs` (10), `--resume_adapt_from` (path|None), `--save_adapt_every` (int; 0)
  - `--finetune_backbone` (flag), `--backbone_lr_scale` (0.1), `--classifier_lr` (1e-2), `--weight_decay` (1e-3)
  - Pseudo-labels: `--use_pseudo_labels` (flag), `--pseudo_conf_thresh` (0.9), `--pseudo_max_ratio` (1.0), `--pseudo_loss_weight` (1.0)
  - Dry run: `--dry_run_max_samples` (0), `--dry_run_max_batches` (0)
  - `--deterministic` (flag), `--seed` (0)

### train_source.py
Train a source-only classifier; optional source-self evaluation.
- Example:
  ```bash
  python scripts/train_source.py --dataset_name office_home --data_root datasets/Office-Home \
    --source_domain Ar --target_domain Cl --num_epochs 50 --batch_size 32 --eval_on_source_self
  ```
- Important options:
  - Dataset/domain: `--dataset_name` (office_home|office31), `--data_root`, `--source_domain`, `--target_domain`
  - Training: `--num_epochs` (50), `--batch_size` (32), `--lr_backbone` (1e-3), `--lr_classifier` (1e-2), `--weight_decay` (1e-3), `--num_workers` (4)
  - Checkpointing: `--resume_from` (path|None), `--save_every` (0)
  - Determinism: `--deterministic` (flag), `--seed` (0)
  - Dry runs: `--dry_run_max_batches` (0), `--dry_run_max_samples` (0)
  - Eval: `--eval_on_source_self` (flag), `--eval_results_csv` (path)

### eval_source_only.py
Evaluate a source checkpoint on its own domain.
- Example:
  ```bash
  python scripts/eval_source_only.py --dataset_name office_home --data_root datasets/Office-Home \
    --domain Ar --checkpoint checkpoints/source_only_Ar_to_Cl_seed0.pth --append_results
  ```
- Options: `--dataset_name` (office_home|office31), `--data_root`, `--domain`, `--checkpoint`, `--batch_size` (32), `--num_workers` (4), `--seed` (0), `--deterministic`, `--results_csv`, `--append_results`.

### run_me_iis_experiments.py
Ablation driver for layers/GMM/ME–IIS variants.
- Example (layers family):
  ```bash
  python scripts/run_me_iis_experiments.py --experiment_family layers \
    --dataset_name office_home --source_domain Ar --target_domain Cl --seeds 0,1 \
    --feature_layers "layer3,layer4"
  ```
- Important options:
  - `--experiment_family` (layers|gmm|me_iis; required)
  - `--dataset_name`, `--source_domain`, `--target_domain`, `--seeds`
  - Paths: `--output_csv`, `--base_data_root`
  - Training: `--num_epochs`, `--batch_size`, `--num_workers`, `--lr_backbone`, `--lr_classifier`, `--weight_decay`
  - ME–IIS: `--feature_layers`, `--num_latent_styles`, `--components_per_layer`, `--gmm_selection_mode`, `--gmm_bic_min_components`, `--gmm_bic_max_components`, `--source_prob_mode`, `--iis_iters`, `--iis_tol`, `--adapt_epochs`, `--finetune_backbone`, `--backbone_lr_scale`, `--classifier_lr`
  - Pseudo-label knobs: `--pseudo_conf_thresh`, `--pseudo_max_ratio`, `--pseudo_loss_weight`
  - Dry-run: `--dry_run_max_samples`, `--dry_run_max_batches`
  - Determinism: `--deterministic`

### Other scripts
- `demo_me_iis_toy.py` – Toy IIS demo; run directly.
- `plot_iis_dynamics.py` – Plot IIS metrics from saved `.npz`.
- `run_office_home.sh` – Example shell wrapper.
- `test_me_iis_sanity.py` – Quick IIS/adaptation sanity script; used by smoke tests.

## Dependencies and Interactions
- Scripts depend on `models/`, `datasets/domain_loaders.py`, and `utils/` helpers.
- CSV logging under `results/`; checkpoints under `checkpoints/`; TensorBoard logs under `runs/`.
