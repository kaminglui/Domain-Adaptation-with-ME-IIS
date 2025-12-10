# tests – Automated Checks

## Purpose
Unit and integration tests for loaders, checkpoint resume logic, experiment utilities, and IIS adaptation behavior.

## Contents
- `test_checkpoints_resume.py` – Validates resume logic for checkpoints.
- `test_domain_loaders.py` – Ensures class mapping alignment and loader correctness.
- `test_experiment_utils.py` – Tests feature-layer parsing and component map construction.
- `test_me_iis_additional.py` – Additional IIS/adaptation behavior tests.
- `__init__.py` – Package marker.

## Usage
Run the full suite:
```bash
python -m unittest
```

Smoke harness:
```bash
python run_smoke_tests.py
```

## Notes
- Tests create synthetic datasets and tiny models via `utils.test_utils`.
- Some tests patch `build_model` to keep runtime short.
