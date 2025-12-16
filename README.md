# ME-IIS Domain Adaptation
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaminglui/Domain-Adaptation-with-ME-IIS/blob/main/notebooks/Run_All_Experiments.ipynb)

## Overview
This repo provides a correctness-first, reproducible UDA pipeline for Office-Home / Office-31 with:
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

## Development
- Tests: `python -m pytest -q`
- Unused-code audit: `python tools/audit_unused.py` (writes `docs/UNUSED_CODE_REPORT.md`)

## Legacy
Older script entrypoints and CLI parsers were moved to `legacy/` to keep the current pipeline minimal and unambiguous.
