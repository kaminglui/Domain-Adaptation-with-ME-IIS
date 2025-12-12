# ME-IIS Domain Adaptation
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaminglui/Domain-Adaptation-with-ME-IIS/blob/main/ME_IIS_Colab.ipynb)

## Overview
ME-IIS implements max-entropy importance sampling for unsupervised domain adaptation on Office-Home and Office-31. It trains a ResNet-50 source classifier, reweights source samples via IIS using style x class constraints, and optionally fine-tunes with pseudo-labels. A Colab notebook and an experiment driver provide end-to-end pipelines, including source-only health checks.

### Background
The method follows the probabilistic-instance IIS extension of maximum entropy as in Pal & Miller (2007): style/class joint feature masses are treated as fractional constraints, and IIS updates apply Δλ = (1/(N_d+N_c)) log(P_g / P_m) without changing the math. Latent probabilities can come from either GMM or spherical vMF-softmax prototypes; both yield valid P[M_i=j | a_i].

## Key Features
- Source-only training and source-self evaluation with automatic dataset resolution.
- ME-IIS adaptation with pluggable style clustering (GMM by default, vMF-softmax prototypes optional), deterministic seeds, and resumable checkpoints.
- Optional warm-start pseudo-labeling via two-stage runs in the Colab.
- Experiment driver for layer/GMM ablations with CSV logging.
- Smoke tests and IIS sanity checks for quick regression detection.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Menu](#cli-menu)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Layout](#project-layout)
- [Folder-level Documentation](#folder-level-documentation)
- [CLI Reference](#cli-reference)

## Installation
- Python 3.10+ recommended.
- Dependencies: PyTorch + torchvision (GPU optional), scikit-learn, tqdm, tensorboard, kagglehub (for Colab).
- Install:
  ```bash
  git clone https://github.com/kaminglui/Domain-Adaptation-with-ME-IIS.git
  cd Domain-Adaptation-with-ME-IIS
  pip install -r requirements.txt           # minimal
  pip install -r env/requirements_colab.txt # Colab-friendly set
  ```
- Datasets: place Office-Home under `datasets/Office-Home` or Office-31 under `datasets/Office-31`, or let KaggleHub fetch them on Colab.

## Quick Start
- Colab: open the two-stage pseudo-label notebook [ME_IIS_Colab.ipynb](https://colab.research.google.com/github/kaminglui/Domain-Adaptation-with-ME-IIS/blob/main/ME_IIS_Colab.ipynb).
- Train source (Office-Home Art → Clipart):
  ```bash
  python scripts/train_source.py --dataset_name office_home --data_root datasets/Office-Home \
    --source_domain Ar --target_domain Cl --num_epochs 50
  ```
- Adapt with ME-IIS (GMM backend):
  ```bash
  python scripts/adapt_me_iis.py --dataset_name office_home --data_root datasets/Office-Home \
    --source_domain Ar --target_domain Cl --checkpoint checkpoints/source_only_Ar_to_Cl_seed0.pth \
    --feature_layers layer3,layer4 --gmm_selection_mode bic --adapt_epochs 10
  ```
- Adapt with vMF-softmax prototypes:
  ```bash
  python scripts/adapt_me_iis.py --dataset_name office_home --data_root datasets/Office-Home \
    --source_domain Ar --target_domain Cl --checkpoint checkpoints/source_only_Ar_to_Cl_seed0.pth \
    --feature_layers layer3,layer4 --cluster_backend vmf_softmax --vmf_kappa 20.0 \
    --cluster_clean_ratio 0.8 --kmeans_n_init 10
  ```
- Run an ablation sweep (all seeds for GMM family):
  ```bash
  python scripts/run_me_iis_experiments.py --dataset_name office_home --source_domain Ar --target_domain Cl \
    --experiment_family gmm --seeds 0,1,2
  ```

## Usage
- Source training: `scripts/train_source.py`
- Source-self eval: `scripts/eval_source_only.py`
- Adaptation: `scripts/adapt_me_iis.py`
- Ablations: `scripts/run_me_iis_experiments.py`
- Smoke tests: `python run_smoke_tests.py`
- Notebook: `ME_IIS_Colab.ipynb` (two-stage pseudo-label warm start)

## Configuration
- Controlled via CLI flags (see per-folder READMEs).
- Dataset roots default to `datasets/Office-Home` / `datasets/Office-31` or auto-resolve on Colab.
- Clustering backends: `--cluster_backend gmm` (default) or `vmf_softmax`. For the vMF-like backend, `--vmf_kappa` controls concentration and `--kmeans_n_init` controls prototype restarts. `--cluster_clean_ratio <1` fits clustering on the lowest-entropy target predictions but still computes target moments on all samples.
- Default feature layers: `layer3,layer4` (avgpool is opt-in).
- CSV logs under `results/`.

## CLI Menu
### Train (source-only)
- Command: `python scripts/train_source.py [required flags] [options]`
- Required: `--source_domain`, `--target_domain` (defaults cover dataset name/root for Office-Home/Office-31)
- Common options: `--num_epochs`, `--batch_size`, `--lr_backbone`, `--lr_classifier`, `--save_every`
- Examples:
  - `python scripts/train_source.py --dataset_name office_home --data_root datasets/Office-Home --source_domain Ar --target_domain Cl`
  - `python scripts/train_source.py --dataset_name office31 --source_domain amazon --target_domain webcam --num_epochs 30 --deterministic`

### Adapt (ME-IIS)
- Command: `python scripts/adapt_me_iis.py [required flags] [options]`
- Required: `--source_domain`, `--target_domain`, `--checkpoint`
- Backend selection:
  - `--cluster_backend gmm`
  - `--cluster_backend vmf_softmax --vmf_kappa 20 --cluster_clean_ratio 0.8`
- Common IIS options: `--feature_layers layer3,layer4`, `--gmm_selection_mode {fixed,bic}`, `--iis_iters`, `--iis_tol`, `--adapt_epochs`
- Examples:
  - Office-Home Art→Clipart (GMM/BIC):  
    `python scripts/adapt_me_iis.py --dataset_name office_home --data_root datasets/Office-Home --source_domain Ar --target_domain Cl --checkpoint checkpoints/source_only_Ar_to_Cl_seed0.pth --gmm_selection_mode bic`
  - Office-Home Product→Real (vMF-softmax):  
    `python scripts/adapt_me_iis.py --dataset_name office_home --data_root datasets/Office-Home --source_domain Pr --target_domain Rw --checkpoint checkpoints/source_only_Pr_to_Rw_seed0.pth --cluster_backend vmf_softmax --vmf_kappa 25 --cluster_clean_ratio 0.8 --kmeans_n_init 10`

### Experiment Runner
- Command: `python scripts/run_me_iis_experiments.py --experiment_family {layers,gmm,me_iis} --source_domain ... --target_domain ...`
- Run all pairs/seeds: `--seeds 0,1,2` (comma separated)
- Control outputs/logging: `--output_csv results/me_iis_experiments_summary.csv`
- Layer or GMM sweeps reuse the same ME-IIS runner flags (`--gmm_selection_mode`, `--cluster_backend`, `--feature_layers`, `--pseudo_*`, etc.).

### Reproducibility
- `--seed` sets RNG seeds; pair with `--deterministic` for cuDNN-safe determinism.
- `--dump_config [path]` prints resolved configs as JSON (to stdout by default) and writes to the provided path when given.

### Logging / Outputs
- Checkpoints: `checkpoints/source_only_*` and `checkpoints/me_iis_*`.
- CSV summaries: `results/office_home_me_iis.csv` and experiment sweeps under `results/me_iis_experiments_summary.csv`.
- IIS weights/history: saved via `_save_npz_safe` under `results/` during adaptation.

## Project Layout
- `scripts/` - Training, adaptation, evaluation, ablations, sanity checks.
- `models/` - ResNet-50 backbone, classifier head, ME-IIS adapter.
- `clustering/` - Pluggable clustering backends (GMM, vMF-softmax prototypes).
- `utils/` - Data, feature, logging, seed helpers.
- `datasets/` - Dataset loaders and (optional) dataset trees.
- `tests/` - Unit and integration tests.
- `env/` - Requirements files.
- `results/` - CSV and IIS artifacts.
- `checkpoints/` - Saved source/adapted weights.
- `ME_IIS_Colab.ipynb` - Colab pipeline.

## Folder-level Documentation
- checkpoints
- [datasets](datasets/README.md)
- [env](env/README.md)
- [models](models/README.md)
- results
- [scripts](scripts/README.md)
- [tests](tests/README.md)
- [utils](utils/README.md)

## CLI Reference
- Auto-generated parser details live in [docs/cli_reference.md](docs/cli_reference.md). Regenerate after changing flags:  
  `python scripts/generate_cli_reference.py --out docs/cli_reference.md`

## Colab Experiments
- Notebook: `ME_IIS_Colab.ipynb`
  - Includes source-only, ME-IIS GMM, and new `vmf_softmax` experiments plus sweeps (kappa/K/entropy filtering) and layer-placement ablations.
  - Sections cover setup, data configuration, debugging checks (PMF/joint mass/IIS diagnostics), and recommended next-step sweeps.
  - Switch to vMF runs by setting `--cluster_backend vmf_softmax` (with `--vmf_kappa`, `--cluster_clean_ratio`, `--kmeans_n_init`) in the config cell.
- Uses the same CLI flags as the scripts; see [docs/cli_reference.md](docs/cli_reference.md) for flag details.
