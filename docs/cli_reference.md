# CLI Reference (auto-generated)

Generated from argparse metadata. Regenerate with:
`python scripts/generate_cli_reference.py --out docs/cli_reference.md`

## Train (source-only)

Usage: `usage: python scripts/train_source.py [-h]
                                      [--dataset_name {office_home,office31}]
                                      [--data_root DATA_ROOT]
                                      --source_domain SOURCE_DOMAIN
                                      --target_domain TARGET_DOMAIN
                                      [--num_epochs NUM_EPOCHS]
                                      [--resume_from RESUME_FROM]
                                      [--save_every SAVE_EVERY]
                                      [--batch_size BATCH_SIZE]
                                      [--lr_backbone LR_BACKBONE]
                                      [--lr_classifier LR_CLASSIFIER]
                                      [--weight_decay WEIGHT_DECAY]
                                      [--num_workers NUM_WORKERS]
                                      [--deterministic] [--seed SEED]
                                      [--dry_run_max_batches DRY_RUN_MAX_BATCHES]
                                      [--dry_run_max_samples DRY_RUN_MAX_SAMPLES]
                                      [--eval_on_source_self]
                                      [--eval_results_csv EVAL_RESULTS_CSV]
                                      [--dump_config [DUMP_CONFIG]]`

### options

### Dataset / Domains
- `--dataset_name` (type: str; default: office_home; choices: ['office_home', 'office31']) — Benchmark to use.
- `--data_root` (type: str; default: None) — Path to dataset root (no existence check during parsing).
- `--source_domain` (type: str; default: None) — Source domain, e.g., Ar.
- `--target_domain` (type: str; default: None) — Target domain, e.g., Cl.

### Training
- `--num_epochs` (type: int; default: 50) — 
- `--resume_from` (type: str; default: None) — Optional source-only checkpoint to resume from.
- `--save_every` (type: int; default: 0) — If >0, save an intermediate checkpoint every N epochs.
- `--batch_size` (type: int; default: 32) — 
- `--lr_backbone` (type: float; default: 0.001) — 
- `--lr_classifier` (type: float; default: 0.01) — 
- `--weight_decay` (type: float; default: 0.001) — 
- `--num_workers` (type: int; default: 4) — 

### Reproducibility
- `--deterministic` (type: flag; default: False) — Force deterministic/cuDNN-safe settings (pair with --seed).
- `--seed` (type: int; default: 0) — 

### Debug / Speed
- `--dry_run_max_batches` (type: int; default: 0) — If >0, limit training to this many batches total.
- `--dry_run_max_samples` (type: int; default: 0) — If >0, cap the number of samples per domain.

### Logging / Output
- `--eval_on_source_self` (type: flag; default: False) — If set, evaluate source-only checkpoint on the source domain.
- `--eval_results_csv` (type: str; default: results\office_home_me_iis.csv) — CSV to append source-self evaluation when enabled.
- `--dump_config` (type: str; default: None) — If set, dump resolved config as JSON to stdout (default) or to the provided path. Directories are created if needed.

## Adapt (ME-IIS)

Usage: `usage: python scripts/adapt_me_iis.py [-h]
                                      [--dataset_name {office_home,office31}]
                                      [--data_root DATA_ROOT]
                                      --source_domain SOURCE_DOMAIN
                                      --target_domain TARGET_DOMAIN
                                      --checkpoint CHECKPOINT
                                      [--batch_size BATCH_SIZE]
                                      [--num_workers NUM_WORKERS]
                                      [--feature_layers FEATURE_LAYERS]
                                      [--source_prob_mode {softmax,onehot}]
                                      [--cluster_backend {gmm,vmf_softmax}]
                                      [--gmm_selection_mode {fixed,bic}]
                                      [--gmm_bic_min_components GMM_BIC_MIN_COMPONENTS]
                                      [--gmm_bic_max_components GMM_BIC_MAX_COMPONENTS]
                                      [--num_latent_styles NUM_LATENT_STYLES]
                                      [--components_per_layer COMPONENTS_PER_LAYER]
                                      [--vmf_kappa VMF_KAPPA]
                                      [--cluster_clean_ratio CLUSTER_CLEAN_RATIO]
                                      [--kmeans_n_init KMEANS_N_INIT]
                                      [--iis_iters IIS_ITERS]
                                      [--iis_tol IIS_TOL]
                                      [--adapt_epochs ADAPT_EPOCHS]
                                      [--resume_adapt_from RESUME_ADAPT_FROM]
                                      [--save_adapt_every SAVE_ADAPT_EVERY]
                                      [--finetune_backbone]
                                      [--backbone_lr_scale BACKBONE_LR_SCALE]
                                      [--classifier_lr CLASSIFIER_LR]
                                      [--weight_decay WEIGHT_DECAY]
                                      [--use_pseudo_labels]
                                      [--pseudo_conf_thresh PSEUDO_CONF_THRESH]
                                      [--pseudo_max_ratio PSEUDO_MAX_RATIO]
                                      [--pseudo_loss_weight PSEUDO_LOSS_WEIGHT]
                                      [--deterministic] [--seed SEED]
                                      [--dry_run_max_samples DRY_RUN_MAX_SAMPLES]
                                      [--dry_run_max_batches DRY_RUN_MAX_BATCHES]
                                      [--dump_config [DUMP_CONFIG]]`

### options

### Dataset / Domains
- `--dataset_name` (type: str; default: office_home; choices: ['office_home', 'office31']) — Benchmark to use.
- `--data_root` (type: str; default: None) — Path to dataset root (not validated here).
- `--source_domain` (type: str; default: None) — Source domain, e.g., Ar.
- `--target_domain` (type: str; default: None) — Target domain, e.g., Cl.
- `--checkpoint` (type: str; default: None) — Path to source-only checkpoint.

### Model / Backbone
- `--batch_size` (type: int; default: 32) — 
- `--num_workers` (type: int; default: 4) — 
- `--feature_layers` (type: str; default: layer3,layer4) — Comma-separated feature layers.
- `--source_prob_mode` (type: str; default: softmax; choices: ['softmax', 'onehot']) — How to form P(Ĉ|x) on source for constraints.

### Clustering Backend
- `--cluster_backend` (type: str; default: gmm; choices: ['gmm', 'vmf_softmax']) — Latent probability backend for styles.
- `--gmm_selection_mode` (type: str; default: fixed; choices: ['fixed', 'bic']) — How to choose GMM components per layer.
- `--gmm_bic_min_components` (type: int; default: 2) — 
- `--gmm_bic_max_components` (type: int; default: 8) — 
- `--num_latent_styles` (type: int; default: 5) — Default components per layer when fixed.
- `--components_per_layer` (type: str; default: None) — Optional comma-separated overrides 'layer:count,...'.
- `--vmf_kappa` (type: float; default: 20.0) — Concentration for vmf_softmax.
- `--cluster_clean_ratio` (type: float; default: 1.0) — Keep-ratio for lowest-entropy target samples when fitting clustering.
- `--kmeans_n_init` (type: int; default: 10) — n_init for KMeans (vmf_softmax).

### ME-IIS / IIS
- `--iis_iters` (type: int; default: 15) — Number of IIS iterations.
- `--iis_tol` (type: float; default: 0.001) — Tolerance on max abs moment error.

### Training
- `--adapt_epochs` (type: int; default: 10) — 
- `--resume_adapt_from` (type: str; default: None) — Optional adaptation checkpoint to resume from.
- `--save_adapt_every` (type: int; default: 0) — If >0, save ME-IIS adaptation checkpoint every N epochs.
- `--finetune_backbone` (type: flag; default: False) — Fine-tune backbone during adaptation.
- `--backbone_lr_scale` (type: float; default: 0.1) — Backbone LR is classifier_lr * backbone_lr_scale when finetuning.
- `--classifier_lr` (type: float; default: 0.01) — 
- `--weight_decay` (type: float; default: 0.001) — 

### Pseudo-Labels
- `--use_pseudo_labels` (type: flag; default: False) — Include pseudo-labelled target samples.
- `--pseudo_conf_thresh` (type: float; default: 0.9) — 
- `--pseudo_max_ratio` (type: float; default: 1.0) — 
- `--pseudo_loss_weight` (type: float; default: 1.0) — 

### Reproducibility
- `--deterministic` (type: flag; default: False) — Force deterministic/cuDNN-safe settings (pair with --seed).
- `--seed` (type: int; default: 0) — 

### Debug / Speed
- `--dry_run_max_samples` (type: int; default: 0) — If >0, limit samples per domain for quick dry-runs.
- `--dry_run_max_batches` (type: int; default: 0) — If >0, limit feature extraction and adaptation to this many batches.

### Logging / Output
- `--dump_config` (type: str; default: None) — If set, dump resolved config as JSON to stdout (default) or to the provided path. Directories are created if needed.

## Experiment Runner

Usage: `usage: python scripts/run_me_iis_experiments.py [-h]
                                                [--dataset_name {office_home,office31}]
                                                --source_domain SOURCE_DOMAIN
                                                --target_domain TARGET_DOMAIN
                                                [--base_data_root BASE_DATA_ROOT]
                                                [--seeds SEEDS]
                                                --experiment_family {layers,gmm,me_iis}
                                                [--output_csv OUTPUT_CSV]
                                                [--num_epochs NUM_EPOCHS]
                                                [--batch_size BATCH_SIZE]
                                                [--num_workers NUM_WORKERS]
                                                [--lr_backbone LR_BACKBONE]
                                                [--lr_classifier LR_CLASSIFIER]
                                                [--weight_decay WEIGHT_DECAY]
                                                [--num_latent_styles NUM_LATENT_STYLES]
                                                [--components_per_layer COMPONENTS_PER_LAYER]
                                                [--gmm_selection_mode {fixed,bic}]
                                                [--gmm_bic_min_components GMM_BIC_MIN_COMPONENTS]
                                                [--gmm_bic_max_components GMM_BIC_MAX_COMPONENTS]
                                                [--cluster_backend {gmm,vmf_softmax}]
                                                [--vmf_kappa VMF_KAPPA]
                                                [--cluster_clean_ratio CLUSTER_CLEAN_RATIO]
                                                [--kmeans_n_init KMEANS_N_INIT]
                                                [--feature_layers FEATURE_LAYERS]
                                                [--source_prob_mode {softmax,onehot}]
                                                [--iis_iters IIS_ITERS]
                                                [--iis_tol IIS_TOL]
                                                [--adapt_epochs ADAPT_EPOCHS]
                                                [--finetune_backbone]
                                                [--backbone_lr_scale BACKBONE_LR_SCALE]
                                                [--classifier_lr CLASSIFIER_LR]
                                                [--pseudo_conf_thresh PSEUDO_CONF_THRESH]
                                                [--pseudo_max_ratio PSEUDO_MAX_RATIO]
                                                [--pseudo_loss_weight PSEUDO_LOSS_WEIGHT]
                                                [--deterministic]
                                                [--dry_run_max_samples DRY_RUN_MAX_SAMPLES]
                                                [--dry_run_max_batches DRY_RUN_MAX_BATCHES]
                                                [--dump_config [DUMP_CONFIG]]`

### options

### Dataset / Domains
- `--dataset_name` (type: str; default: office_home; choices: ['office_home', 'office31']) — 
- `--source_domain` (type: str; default: None) — 
- `--target_domain` (type: str; default: None) — 
- `--base_data_root` (type: str; default: None) — Optional override for dataset root.

### Experiments
- `--seeds` (type: str; default: 0) — Comma-separated seeds, e.g., "0,1,2".
- `--experiment_family` (type: str; default: None; choices: ['layers', 'gmm', 'me_iis']) — Which ablation family to run.
- `--output_csv` (type: str; default: results\me_iis_experiments_summary.csv) — Where to write compact experiment summaries.

### Training
- `--num_epochs` (type: int; default: 50) — Source-only epochs.
- `--batch_size` (type: int; default: 32) — 
- `--num_workers` (type: int; default: 4) — 
- `--lr_backbone` (type: float; default: 0.001) — 
- `--lr_classifier` (type: float; default: 0.01) — 
- `--weight_decay` (type: float; default: 0.001) — 

### Clustering Backend
- `--num_latent_styles` (type: int; default: 5) — Default components per layer when fixed.
- `--components_per_layer` (type: str; default: None) — Optional comma-separated override.
- `--gmm_selection_mode` (type: str; default: fixed; choices: ['fixed', 'bic']) — How to choose GMM components when adapting.
- `--gmm_bic_min_components` (type: int; default: 2) — 
- `--gmm_bic_max_components` (type: int; default: 15) — 
- `--cluster_backend` (type: str; default: gmm; choices: ['gmm', 'vmf_softmax']) — 
- `--vmf_kappa` (type: float; default: 20.0) — 
- `--cluster_clean_ratio` (type: float; default: 1.0) — 
- `--kmeans_n_init` (type: int; default: 10) — 
- `--feature_layers` (type: str; default: layer3,layer4) — 

### ME-IIS / IIS
- `--source_prob_mode` (type: str; default: softmax; choices: ['softmax', 'onehot']) — 
- `--iis_iters` (type: int; default: 15) — 
- `--iis_tol` (type: float; default: 0.001) — 
- `--adapt_epochs` (type: int; default: 10) — 
- `--finetune_backbone` (type: flag; default: False) — 
- `--backbone_lr_scale` (type: float; default: 0.1) — 
- `--classifier_lr` (type: float; default: 0.01) — 

### Pseudo-Labels
- `--pseudo_conf_thresh` (type: float; default: 0.9) — 
- `--pseudo_max_ratio` (type: float; default: 0.3) — 
- `--pseudo_loss_weight` (type: float; default: 0.5) — 

### Reproducibility / Debug
- `--deterministic` (type: flag; default: False) — 
- `--dry_run_max_samples` (type: int; default: 0) — 
- `--dry_run_max_batches` (type: int; default: 0) — 

### Logging / Output
- `--dump_config` (type: str; default: None) — If set, dump resolved config as JSON to stdout (default) or to the provided path. Directories are created if needed.
