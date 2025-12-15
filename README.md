# ME-IIS Domain Adaptation
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaminglui/ME-IIS/blob/main/notebooks/Run_All_Experiments.ipynb)

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
- Dependencies: PyTorch + torchvision (GPU optional), scikit-learn, tqdm, tensorboard, pandas/matplotlib (notebook summaries), kagglehub (for Colab).
- Install:
  ```bash
  git clone https://github.com/kaminglui/ME-IIS.git
  cd ME-IIS
  pip install -r requirements.txt           # minimal
  pip install -r env/requirements_colab.txt # Colab-friendly set
  ```
- Datasets: place Office-Home under `datasets/Office-Home` or Office-31 under `datasets/Office-31`, or let KaggleHub fetch them on Colab.

## Quick Start
- Colab (recommended): open `notebooks/Run_All_Experiments.ipynb` to run ME-IIS + fair UDA baselines with deterministic `run_id`, skip/resume, and unified `metrics.csv`.
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
- Notebook: `notebooks/Run_All_Experiments.ipynb` (ME-IIS + baselines, deterministic run_id, skip/resume, summaries/plots)

## Unified run system (recommended)
The notebook uses the unified runner in `src/experiments/runner.py` and writes each run to:

`outputs/runs/{dataset}/{src}2{tgt}/{method}/{run_id}/`
- `config.json` (canonical config used to hash `run_id`)
- `logs/stdout.txt`, `logs/stderr.txt`
- `checkpoints/` (deterministic filenames include `run_id`)
- `metrics.csv` (single-row CSV with unified columns for fair comparisons)

Run one experiment locally from Python:
```python
from src.experiments.run_config import RunConfig
from src.experiments.runner import run_one

cfg = RunConfig(
    dataset_name="office_home",
    data_root="datasets/Office-Home",
    source_domain="Ar",
    target_domain="Cl",
    method="me_iis",  # source_only | me_iis | dann | coral | pseudo_label
    epochs_source=50,
    epochs_adapt=10,
    batch_size=32,
    method_params={"feature_layers": ["layer3", "layer4"], "iis_iters": 15, "cluster_backend": "gmm"},
    seed=0,
    deterministic=True,
)
run_one(cfg, force_rerun=False)
```

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
- Deterministic runs (recommended): `outputs/runs/{dataset}/{src}2{tgt}/{method}/{run_id}/` with `config.json`, `logs/`, `checkpoints/`, and `metrics.csv`.
- Legacy scripts: checkpoints under `checkpoints/` and CSV summaries under `results/` (e.g., `results/office_home_me_iis.csv`).
- ME-IIS IIS weights/history: saved during adaptation (legacy scripts write under `results/`; unified runs write under each run directory).

## Project Layout
- `scripts/` - Training, adaptation, evaluation, ablations, sanity checks.
- `models/` - ResNet-50 backbone, classifier head, ME-IIS adapter.
- `notebooks/` - Colab notebooks (recommended entrypoint: `notebooks/Run_All_Experiments.ipynb`).
- `src/experiments/` - Deterministic config/run_id system, checkpointing, baselines (DANN/CORAL), unified evaluation + metrics logging.
- `clustering/` - Pluggable clustering backends (GMM, vMF-softmax prototypes).
- `utils/` - Data, feature, logging, seed helpers.
- `datasets/` - Dataset loaders and (optional) dataset trees.
- `tests/` - Unit and integration tests.
- `env/` - Requirements files.
- `results/` - CSV and IIS artifacts.
- `checkpoints/` - Saved source/adapted weights.
- `outputs/` - Deterministic run directories (`outputs/runs/...`).

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
- Notebook: `notebooks/Run_All_Experiments.ipynb`
  - Runs: `source_only`, `me_iis`, `dann`, `coral`, optional `pseudo_label`.
  - Enforces a shared training budget (configured once) and logs each run to `outputs/runs/.../{run_id}/metrics.csv`.

### Adding a new baseline
1. Add a method under `src/experiments/methods/` that saves `backbone` + `classifier` in its checkpoint.
2. Register it in `src/experiments/runner.py:run_one(...)`.
3. Put method-specific knobs in `RunConfig.method_params` (these are hashed into `run_id`).
4. Add a small unit test under `tests/` (forward pass + one training step).
