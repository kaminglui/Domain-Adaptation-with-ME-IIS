# ME‑IIS Domain Adaptation

## Overview
ME‑IIS implements max-entropy importance sampling for unsupervised domain adaptation on Office-Home and Office-31. It trains a ResNet-50 source classifier, reweights source samples via IIS using style×class constraints, and optionally fine-tunes with pseudo-labels. A Colab notebook and an experiment driver provide end-to-end pipelines, including source-only health checks.

## Key Features
- Source-only training and source-self evaluation with automatic dataset resolution.
- ME–IIS adaptation with GMM-based style clustering, deterministic seeds, and resumable checkpoints.
- Optional warm-start pseudo-labeling via two-stage runs in the Colab.
- Experiment driver for layer/GMM ablations with CSV logging.
- Smoke tests and IIS sanity checks for quick regression detection.

## Table of Contents
- [Installation](#installation)
- [Quickstart / Getting Started](#quickstart--getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Layout](#project-layout)
- [Folder-level Documentation](#folder-level-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact / Support](#contact--support)

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

## Quickstart / Getting Started
Train source model (Ar→Cl):
```bash
python scripts/train_source.py --dataset_name office_home \
  --data_root datasets/Office-Home --source_domain Ar --target_domain Cl --num_epochs 50
```
Evaluate source-only on its own domain:
```bash
python scripts/eval_source_only.py --dataset_name office_home \
  --data_root datasets/Office-Home --domain Ar \
  --checkpoint checkpoints/source_only_Ar_to_Cl_seed0.pth --append_results
```
Adapt with ME–IIS:
```bash
python scripts/adapt_me_iis.py --dataset_name office_home --data_root datasets/Office-Home \
  --source_domain Ar --target_domain Cl --checkpoint checkpoints/source_only_Ar_to_Cl_seed0.pth \
  --feature_layers "layer3,layer4" --gmm_selection_mode bic --adapt_epochs 10
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
- Default feature layers: `layer3,layer4` (avgpool is opt-in).
- CSV logs under `results/`.

## Project Layout
- `scripts/` – Training, adaptation, evaluation, ablations, sanity checks.
- `models/` – ResNet-50 backbone, classifier head, ME–IIS adapter.
- `utils/` – Data, feature, logging, seed helpers.
- `datasets/` – Dataset loaders and (optional) dataset trees.
- `tests/` – Unit and integration tests.
- `env/` – Requirements files.
- `results/` – CSV and IIS artifacts.
- `checkpoints/` – Saved source/adapted weights.
- `ME_IIS_Colab.ipynb` – Colab pipeline.

## Folder-level Documentation
- [checkpoints](checkpoints/README.md)
- [datasets](datasets/README.md)
- [datasets/Office-31](datasets/Office-31/README.md)
- [datasets/Office-Home](datasets/Office-Home/README.md)
- [env](env/README.md)
- [models](models/README.md)
- [results](results/README.md)
- [scripts](scripts/README.md)
- [tests](tests/README.md)
- [utils](utils/README.md)

## Contributing
Pull requests welcome. Run `python run_smoke_tests.py` before submitting; keep changes deterministic when possible.

## License
TODO: clarify license (no LICENSE file present).

## Contact / Support
Open an issue on the repository’s GitHub issues page for questions or bug reports.

- Default `layers` experiments sweep `layer4` and `layer3,layer4`. The `layer3+layer4` combo kept spatial/style cues and performed best in our tests. Adding `avgpool` by default hurt accuracy, so treat it as an opt-in variant if you need it.
