# ME-IIS Domain Adaptation
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaminglui/Domain-Adaptation-with-ME-IIS/blob/main/notebooks/Run_All_Experiments.ipynb)

## Overview
This repo provides a correctness-first, reproducible UDA pipeline for:
- **Camelyon17 (WILDS)**: ERM, DANN, and ME-IIS baselines using the official `wilds` dataset/splits/metrics
- **Office-Home / Office-31** (legacy notebook cells + CLI) with:
- **ME-IIS** (max-entropy importance sampling) + diagnostics
- Standard UDA baselines: **source_only, dann, dan, jan, cdan** (optional `pseudo_label`)
- A single unified runner that enforces fair comparisons and writes a `signature.json` guard per run

## Install
- Python 3.10+ recommended.
- Install minimal deps: `pip install -r requirements.txt`

## Datasets
- Office-Home: `datasets/Office-Home/` (domains: `Art`, `Clipart`, `Product`, `RealWorld`)
- Office-31: `datasets/Office-31/` (domains: `amazon`, `dslr`, `webcam`)

## Unified CLI (recommended)
Single entrypoint (debug harness): `tools/run_experiment.py`.

Examples (Office-Home Ar→Cl):
- Source-only: `python tools/run_experiment.py --dataset office_home --data_root datasets/Office-Home --src Ar --tgt Cl --method source_only --seed 0 --mode full`
- DANN: `python tools/run_experiment.py --dataset office_home --data_root datasets/Office-Home --src Ar --tgt Cl --method dann --seed 0 --mode full`
- ME-IIS: `python tools/run_experiment.py --dataset office_home --data_root datasets/Office-Home --src Ar --tgt Cl --method me_iis --seed 0 --mode full`
- One-batch sanity step: `python tools/run_experiment.py --dataset office_home --data_root datasets/Office-Home --src Ar --tgt Cl --method dann --seed 0 --mode quick --one_batch_debug`

Outputs are written to:
`outputs/runs/{dataset}/{src}2{tgt}/{method}/{run_id}/`
- `signature.json` (method routing guard + enabled losses + resolved dataloader + step budget)
- `logs/stdout.txt`, `logs/stderr.txt`
- `checkpoints/*_final.pth` (and `*_last.pth` for resume)
- `metrics.csv` (unified evaluation row)

## Notebook
- `notebooks/Run_All_Experiments.ipynb` runs the full suite (quick/full), handles failed runs explicitly, and reports paper-style 12-transfer averages.

## Camelyon17 Quickstart (WILDS)
Recommended entrypoint: `experiment.py` (method-only runs + smoke tests).

Examples:
- ERM smoke: `python experiment.py --dataset camelyon17 --methods source_only --smoke_test --seed 0`
- ME-IIS smoke (auto-runs source-only dependency if needed): `python experiment.py --dataset camelyon17 --methods me_iis --smoke_test --seed 0 --split_mode align_val`
- Full ME-IIS only run (UDA target protocol): `python experiment.py --dataset camelyon17 --methods me_iis --seed 0 --split_mode uda_target --batch_size auto --num_workers auto`

Notes:
- Default WILDS root selection:
  - Uses `--data_root` if provided, else `WILDS_DATA_ROOT` if set.
  - In Colab (when unset), chooses `/content/data/wilds` if local disk has headroom, otherwise `/content/drive/MyDrive/data/wilds`, and prints disk usage.
- Resume/skip: rerunning the same command reuses the existing run directory unless `--force_rerun` is set.
- Outputs are written under `outputs/runs/camelyon17/{split_mode}/{method}/{run_id}/` with `config.json`, `fingerprint.txt`, `metrics.json`, `stdout.log`, `best.pt`, `last.pt`.
- Label-leakage protocol note: `docs/CAMELYON17_LABEL_LEAKAGE.md`

## Camelyon17 (WILDS) in Colab
- Open `notebooks/Run_All_Experiments.ipynb` and run the top “Camelyon17 (WILDS)” section.
- Set `PROJECT_FOLDER` in the `[A1] Drive + Paths` cell to control where checkpoints/results are written in Drive.
- Dataset download root defaults to local SSD (`/content/data/wilds`) for performance; change `WILDS_DATA_ROOT` if you need Drive persistence.
- Outputs:
  - Checkpoints: `CKPT_ROOT/<run_id>/{best.pt,last.pt}`
  - Metrics: `CKPT_ROOT/<run_id>/results.json`
  - Summary table: `OUT_ROOT/summary.csv`

## Development
- Tests: `pip install -r requirements-dev.txt && python -m pytest -q`
- Unused-code audit: `python tools/audit_unused.py` (writes `docs/UNUSED_CODE_REPORT.md`)

## Legacy
Older script entrypoints and CLI parsers were moved to `legacy/` to keep the current pipeline minimal and unambiguous.
