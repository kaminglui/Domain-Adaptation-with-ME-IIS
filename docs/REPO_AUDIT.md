# Repo Audit (Factual) — ME-IIS UDA Repository

This document is a factual inspection of the current repository code paths (no assumptions). Paths and function names are referenced exactly as implemented.

## 1.1 Repo Map (key directories/files)

### Training scripts
- `scripts/train_source.py`: source-only training entrypoint (`train_source(cfg)`), checkpoint save/resume logic, and CSV logging.

### Adaptation scripts (ME-IIS)
- `scripts/adapt_me_iis.py`: ME-IIS adaptation entrypoint (`adapt_me_iis(cfg)`), feature extraction, clustering + IIS reweighting, weighted fine-tuning, optional pseudo-label loss, checkpoint save/resume logic, and CSV logging.

### Dataset loaders
- `datasets/domain_loaders.py`: dataset loading for Office-Home/Office-31 with aligned class mappings and train/eval transforms via `_build_transforms(train=...)`; public entrypoint `get_domain_loaders(...)`.
- `datasets/office_home.py`, `datasets/office31.py`: dataset-specific helpers (not used by the main scripts directly; loaders are primarily in `datasets/domain_loaders.py`).
- `datasets/Office-Home/`, `datasets/Office-31/`: optional on-disk dataset roots (the code can also resolve KaggleHub downloads in Colab).

### Model definitions
- `models/backbone.py`: `ResNet50Backbone` (torchvision ResNet-50) with `forward_intermediates(...)` returning pooled activations for `SUPPORTED_LAYERS`, and `FullModel` wrapper.
- `models/classifier.py`: `ClassifierHead` and `build_model(num_classes, pretrained=True)` creating `FullModel(ResNet50Backbone, ClassifierHead)`.
- `models/me_iis_adapter.py`: `MaxEntAdapter` implementing per-layer clustering (`fit_target_structure`) and IIS reweighting (`solve_iis`).
- `models/iis_components.py`: modular IIS pieces: `JointConstraintBuilder` (builds style×class joint constraints), `TargetEntropyFilter`, and `IISUpdater` (Pal & Miller fractional IIS update rules).

### Clustering backends
- `clustering/factory.py`: `create_backend(...)` (dispatch to concrete backends).
- `clustering/gmm_backend.py`: `GMMBackend` (sklearn `GaussianMixture`, optional BIC selection).
- `clustering/vmf_softmax_backend.py`: `VMFSoftmaxBackend` (KMeans prototypes on L2-normalized features + softmax responsibilities).
- `clustering/base.py`: `LatentBackend` interface.

### Checkpoint utilities
- `utils/experiment_utils.py`: `build_source_ckpt_path(...)` (canonical source checkpoint naming), plus dataset tagging and layer/component parsing helpers.
- `scripts/train_source.py`: checkpoint save/resume is implemented directly in-script; checkpoint contents include model/optimizer/scheduler/epoch.
- `scripts/adapt_me_iis.py`: ME-IIS adaptation checkpoint naming `_build_adapt_ckpt_path(...)` and save/resume are implemented directly in-script; checkpoint contents include model/optimizer/scheduler/epoch plus IIS weights/history.

### Evaluation code
- `eval.py`: `evaluate(model, loader, device)` computes top-1 accuracy (%) + confusion matrix; `predict_features(...)` extracts penultimate features/logits/labels.
- `scripts/eval_source_only.py`: loads a checkpoint and evaluates it on a specified domain (e.g., Ar→Ar) using `eval.evaluate(...)`.

### Experiment runners
- `scripts/run_me_iis_experiments.py`: experiment driver stitching `scripts/train_source.py` + `scripts/adapt_me_iis.py` for ablation families (`layers`, `gmm`, `me_iis`).
- `src/experiments/notebook_runner.py`: notebook-oriented wrappers that shell out to `scripts/train_source.py` / `scripts/adapt_me_iis.py` and parse `results/office_home_me_iis.csv`.
- `src/experiments/runner.py`: unified in-process runner used by the new notebook; enforces deterministic `run_id` run directories, skip/resume, log capture, and unified `metrics.csv` writing.

### CLI/config parsing
- `src/cli/args.py`: dataclass configs (`TrainConfig`, `AdaptConfig`, `ExperimentConfig`) and arg parsers (`build_train_parser`, `build_adapt_parser`, `build_experiments_parser`), plus `dump_config(...)`.
- `src/experiments/run_config.py`: `RunConfig` (hashed to a deterministic `run_id`) and run directory helpers.

### Notebooks
- `ME_IIS_Colab.ipynb`: Colab-oriented workflow that runs scripts, maintains its own run directories, and summarizes results.
- `notebooks/Run_All_Experiments.ipynb`: new Colab workflow that runs ME-IIS + baselines via `src/experiments/runner.py` and writes per-run `metrics.csv`.

### New baselines (UDA)
- `models/dann.py`: gradient reversal + domain discriminator head for DANN.
- `src/experiments/methods/source_only.py`: unified source-only training with deterministic run directories + checkpoints.
- `src/experiments/methods/me_iis.py`: ME-IIS adaptation implemented with the unified checkpointing/run_id system.
- `src/experiments/methods/dann.py`: DANN adaptation implemented with the unified checkpointing/run_id system.
- `src/experiments/methods/coral.py`: CORAL adaptation implemented with the unified checkpointing/run_id system.
- `src/experiments/methods/pseudo_label.py`: pseudo-label self-training baseline implemented with the unified checkpointing/run_id system.

### Tests
- `tests/test_checkpoints_resume.py`: trains for 1 epoch, saves, resumes; adapts for 1 epoch, saves, resumes (patches model to a tiny model for speed).
- `tests/test_domain_loaders.py`: validates class mapping and loader behavior for synthetic Office-Home/Office-31-like folder trees.
- `tests/test_clustering_backends.py`: sanity checks clustering backend shapes and behavior.
- `tests/test_experiment_utils.py`: tests parsing and component-map construction utilities.
- CLI flag validation: `tests/test_cli_flags_*.py`, `tests/test_cli_invalid_args.py`.

## 1.2 End-to-End Execution Paths (no guessing)

### Source-only training

**Entrypoint**
- Script: `scripts/train_source.py`
- Main: `__main__` calls `parse_args()` → constructs `TrainConfig` → `dump_config(cfg, cfg.dump_config)` → `train_source(cfg)`.

**Dataset loading**
- `train_source(args)` calls `datasets/domain_loaders.get_domain_loaders(...)` to obtain:
  - `source_loader` (train transforms, shuffled),
  - `target_loader` (train transforms, shuffled; unused in source-only),
  - `target_eval_loader` (eval transforms, not shuffled).
  See `datasets/domain_loaders.py:get_domain_loaders(...)`.

**Transforms**
- Implemented in `datasets/domain_loaders.py:_build_transforms(train=True|False)`:
  - Train: `Resize(256)`, `RandomResizedCrop(224)`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize(IMAGENET_MEAN, IMAGENET_STD)`.
  - Eval: `Resize(256)`, `CenterCrop(224)`, `ToTensor()`, `Normalize(...)`.

**Model initialization**
- `models.classifier.build_model(num_classes, pretrained=True)` → `models.backbone.ResNet50Backbone(pretrained=True)` uses `torchvision.models.resnet50(weights=IMAGENET1K_V1)` (ImageNet pretrained).

**Optimizer + scheduler**
- In `scripts/train_source.py:train_source(...)`:
  - Optimizer: `torch.optim.SGD` with two param groups:
    - backbone params at `lr=args.lr_backbone`,
    - classifier params at `lr=args.lr_classifier`,
    - `momentum=0.9`, `weight_decay=args.weight_decay`.
  - Scheduler: `torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.num_epochs)`.

**Epochs / batch size / LR (defaults)**
- Defaults in `src/cli/args.py:TrainConfig`:
  - `num_epochs=50`, `batch_size=32`, `lr_backbone=1e-3`, `lr_classifier=1e-2`, `weight_decay=1e-3`.

**What gets saved**
- Final checkpoint path: `utils.experiment_utils.build_source_ckpt_path(...)` →
  `checkpoints/source_only_{source}_to_{target}_seed{seed}.pth`.
- Checkpoint content keys (see `scripts/train_source.py`):
  - `"backbone"`, `"classifier"`, `"optimizer"`, `"scheduler"`, `"epoch"`, `"best_target_acc"`, `"source_acc"`, `"args"`.
- Optional epoch checkpoints when `--save_every > 0`:
  - `checkpoints/source_only_{source}_to_{target}_seed{seed}_epoch{epoch}.pth`.

**Resume behavior**
- `scripts/train_source.py:train_source(...)`:
  - If `--resume_from` is not set and the final checkpoint exists, it sets `args.resume_from` to that path (“Auto-resume”).
  - If `--resume_from` points to a valid checkpoint, it restores model weights and (if present) optimizer/scheduler state, then resumes at `start_epoch = last_completed_epoch + 1`.

**Evaluation during training**
- After each epoch, it evaluates on `target_eval_loader` using `eval.py:evaluate(...)`.
- Metrics: top-1 accuracy (%) and confusion matrix (confusion matrix is returned but not logged to CSV).

### Adaptation stage (ME-IIS)

**Entrypoint**
- Script: `scripts/adapt_me_iis.py`
- Main: `__main__` calls `parse_args()` → constructs `AdaptConfig` → `dump_config(cfg, cfg.dump_config)` → `adapt_me_iis(cfg)`.

**Source checkpoint loading**
- `adapt_me_iis(args)` loads the source-only checkpoint via `torch.load(args.checkpoint)` and applies:
  - `model.backbone.load_state_dict(source_ckpt["backbone"])`
  - `model.classifier.load_state_dict(source_ckpt["classifier"])`
  See `scripts/adapt_me_iis.py`.

**Feature extraction (which layers, how)**
- Layer selection:
  - `feature_layers = utils.experiment_utils.parse_feature_layers(args.feature_layers)` (comma-separated string).
- Extraction:
  - `utils.feature_utils.extract_features(model, loader, device, feature_layers, ...)` calls
    `model.forward_with_intermediates(images, feature_layers=feature_layers)`.
  - `models.backbone.ResNet50Backbone.forward_intermediates(...)` returns pooled activations per requested layer:
    - For `layer1`–`layer4`: `adaptive_avg_pool2d(...)->flatten`.
    - For `avgpool`: the penultimate 2048-d vector.

**Deterministic transforms for IIS constraints**
- `scripts/adapt_me_iis.py` builds a *deterministic* dataset for constraint computation:
  - `eval_transform = target_eval_raw.transform` (which is `train=False` transforms from `datasets/domain_loaders.py`).
  - `source_feat_raw = torchvision.datasets.ImageFolder(source_train_raw.root, transform=eval_transform)`.
  - `target_feat_raw = target_eval_raw` (already deterministic).
  This makes IIS feature extraction use center-crop eval transforms (not random train augmentations).

**Clustering method details**
- Implemented in `models/me_iis_adapter.py:MaxEntAdapter.fit_target_structure(...)`.
- Per-layer latent backend is created via `clustering/factory.py:create_backend(...)`:
  - `"gmm"`: `clustering/gmm_backend.py:GMMBackend` (sklearn `GaussianMixture`), `covariance_type="diag"`, `reg_covar=1e-6`; optional BIC selection (`selection_mode="bic"`) over `[gmm_bic_min_components, gmm_bic_max_components]` with subsampling (`max_subsample=20000`).
  - `"vmf_softmax"`: `clustering/vmf_softmax_backend.py:VMFSoftmaxBackend` using KMeans prototypes + softmax responsibilities with scale `kappa` (`--vmf_kappa`).
- Optional “clean clustering”:
  - If `--cluster_clean_ratio < 1.0`, it selects low-entropy target samples via `models/iis_components.py:TargetEntropyFilter`, which uses `utils/entropy.py`.

**How constraints are formed (style × class)**
- Implemented by `models/iis_components.py:JointConstraintBuilder.build_joint(...)`:
  - For each layer: responsibilities `P[M_i=j | a_i(t)]` from the fitted backend.
  - Combined with class posteriors per sample (`class_probs`) to form joint features:
    `joint_{i,j,c}(t) = P[M_i=j|a_i(t)] * P(C=c|x(t))`.
- Source `P(C=c|x_s)`:
  - `scripts/adapt_me_iis.py:_class_probs_from_logits(...)`:
    - `"softmax"` uses model softmax outputs.
    - `"onehot"` uses ground-truth source labels (one-hot).
- Target `P(C=c|x_t)`:
  - Always `softmax(target_logits)` in `scripts/adapt_me_iis.py`.

**How IIS is solved**
- Implemented in `models/me_iis_adapter.py:MaxEntAdapter.solve_iis(...)` using `models/iis_components.py:IISUpdater`.
- Iteration:
  - Initialize weights `q` uniform over source samples.
  - Compute target moments `P_g` as mean over target joint constraints (`IISUpdater.compute_pg`).
  - Compute model moments `P_m` as weighted sum over source joint constraints (`IISUpdater.compute_pm`).
  - Update `Δλ = log(P_g / P_m) / (N_d + N_c)` (`IISUpdater.delta_lambda`), with `N_d=0`, `N_c=len(layers)`.
  - Update weights: `q_new ∝ q * exp(sum_k f_k(t) * Δλ_k)` then renormalize (`IISUpdater.update_weights`).
- Stopping:
  - Runs for `max_iter=args.iis_iters`, early-stops if `max_abs_moment_error < args.iis_tol` (when `iis_tol > 0`).
- Logs iteration stats (`IISIterationStats`) including moment errors and weight entropy.

**How weights are applied in training**
- Weighted adaptation epoch is `scripts/adapt_me_iis.py:adapt_epoch(...)`:
  - Loader yields `(image, label, idx)` via `IndexedDataset`.
  - Source loss uses per-sample CE with `reduction="none"` then reweighted:
    `loss_src = (w_i * ce_i).sum() / (w.sum() + 1e-8)`.
  - Optional pseudo-labeled target loss is added (if enabled) as an additional CE term.

**Backbone frozen vs fine-tuned**
- `scripts/adapt_me_iis.py`:
  - If `--finetune_backbone` is **not** set, backbone params are set `requires_grad=False`.
  - Optimizer param groups:
    - Backbone group only includes params with `requires_grad=True`, LR=`classifier_lr * backbone_lr_scale`.
    - Classifier group LR=`classifier_lr`.

**What gets saved**
- Checkpoint naming in `scripts/adapt_me_iis.py:_build_adapt_ckpt_path(...)`:
  - `checkpoints/me_iis_{source}_to_{target}_{layer_tag}_seed{seed}.pth`
  - Epoch checkpoints: add `_epoch{epoch}`.
- Adaptation checkpoint contents (see `scripts/adapt_me_iis.py:_build_adapt_checkpoint(...)`):
  - `"backbone"`, `"classifier"`, `"weights"`, `"history"`, `"optimizer"`, `"scheduler"`, `"epoch"`, `"adapt_batches_seen"`, `"source_acc"`, `"args"`.
- IIS artifacts:
  - `results/me_iis_weights_{source}_to_{target}_{layer_tag}_seed{seed}.npz` via `scripts/adapt_me_iis.py:save_iis_history(...)`.

### Evaluation

**Single evaluation function**
- `eval.py:evaluate(model, loader, device)` is used by both:
  - `scripts/train_source.py` (target evaluation each epoch),
  - `scripts/adapt_me_iis.py` (baseline + per-epoch + final evaluation).

**Target labels usage**
- `eval.py:evaluate(...)` consumes labels from the provided DataLoader for metric computation (accuracy + confusion matrix).
- In ME-IIS constraint construction, target labels are explicitly not used:
  - `scripts/adapt_me_iis.py` calls `extract_features(...)` on a target dataset, then immediately `del target_labels` after computing softmax probabilities, printing an audit line: `"[Audit] Dropping target labels during IIS (unsupervised adaptation)."`

**Metrics computed**
- Top-1 accuracy (%) and confusion matrix (sklearn `confusion_matrix`).

## Unified Run System (new, notebook-driven)

The notebook `notebooks/Run_All_Experiments.ipynb` runs methods through `src/experiments/runner.py:run_one(...)` using `src/experiments/run_config.py:RunConfig`:
- A deterministic `run_id` is computed as `sha1(json.dumps(config, sort_keys=True))[:10]` (see `src/experiments/run_config.py:compute_run_id(...)`).
- Run directories follow: `outputs/runs/{dataset}/{src}2{tgt}/{method}/{run_id}/` (see `src/experiments/run_config.py:get_run_dir(...)`).
- Each run writes:
  - `config.json` (canonical config),
  - `state.json` + deterministic checkpoints under `checkpoints/` (see `src/experiments/checkpointing.py`),
  - `logs/stdout.txt` + `logs/stderr.txt` (see `src/experiments/stream_capture.py`),
  - `metrics.csv` written by a single evaluation harness (`src/experiments/eval_harness.py` + `src/experiments/metrics.py`).
