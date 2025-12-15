# Colab Notebook Audit (Factual)

Notebook audited:
- `notebooks/Run_All_Experiments.ipynb`

This notebook is a Python-orchestrated (in-process) runner. It does **not** primarily execute shell `python scripts/*.py` command cells; instead it imports `src/experiments/runner.py` and calls into the unified runner.

## B1) What the notebook runs (cells that execute commands / runs)

### Setup + install
- Repo checkout (code cell 2):
  - Uses `git clone` when `REPO_DIR` does not exist.
  - Otherwise uses `git pull --ff-only` (best-effort) and prints `git rev-parse HEAD`.
- Dependency install (code cell 3):
  - Runs `pip install -r requirements.txt` via `subprocess.run([sys.executable, "-m", "pip", ...])`.

### Config cell (single place to edit)
- Defines the experiment configuration (code cell 5):
  - Dataset selection: `DATASET_NAME`, `DATA_ROOT`
  - Mode: `MODE = "QUICK" | "FULL"`
  - Seeds: `SEEDS = [0] if MODE == "QUICK" else [0, 1, 2]`
  - Shared training budget: `EPOCHS_SOURCE`, `EPOCHS_ADAPT`, `BATCH_SIZE`, `NUM_WORKERS`
  - Method knobs in dicts: `ME_IIS_PARAMS`, `DANN_PARAMS`, `CORAL_PARAMS`, `PL_PARAMS`
  - Output root: `RUNS_ROOT = Path("outputs") / "runs"`

### Data cell (download/verify)
- Code cell 7:
  - If `DATA_ROOT` is blank, downloads the dataset via KaggleHub:
    - Office-Home: `kagglehub.dataset_download("lhrrraname/officehome")`
    - Office-31: `kagglehub.dataset_download("xixuhu/office31")`
  - Attempts to “find” the dataset root by scanning for expected domain folders.
  - Sets `DATA_ROOT = Path(data_root_path).resolve()`.

### Run cells (unified in-process runner)
- Code cell 9 defines:
  - `make_cfg(method, seed, method_params)` → builds `src/experiments/run_config.py:RunConfig`
  - `run_one_nb(cfg, ...)` → notebook-safe wrapper around `src/experiments/runner.py:run_one(...)`
  - A fairness guard printing an “optimizer-step budget” table via `src/experiments/budget.py:estimate_total_steps(...)`
  - Loops over `SEEDS` and runs:
    - `method="source_only"`
- Code cell 11:
  - Runs `method="me_iis"` for each seed via `run_one_nb(...)`
- Code cell 13:
  - Runs baselines for each seed via `run_one_nb(...)`:
    - `method="dann"`, `method="coral"`, `method="pseudo_label"`

### Output layout expected by the notebook
The notebook constructs `RunConfig.run_id` and expects each run to live under:
- `outputs/runs/{dataset_tag}/{src}2{tgt}/{method}/{run_id}/`

When a run succeeds, the unified runner writes:
- `config.json`, `state.json`, `logs/stdout.txt`, `logs/stderr.txt`, checkpoints, and `metrics.csv`.

## B1) Stored notebook outputs (what the committed notebook shows)

The notebook file includes stored outputs from an execution in a Colab-like environment (paths such as `/content/...` and `/kaggle/input/...` are printed).

### Observed environment + data resolution
- Repo HEAD printed by the notebook output:
  - `057fb70446f4eb365f5e8e814e4b0f3714d1c939`
- Dataset resolution printed by the notebook output:
  - `DATASET_NAME: office_home`
  - `DATA_ROOT: /kaggle/input/officehome/datasets/OfficeHomeDataset_10072016`

### Observed run outcomes and warnings/errors
- The run cells (source-only, ME-IIS, and baselines) print failures of the form:
  - `AcceleratorError: CUDA error: device-side assert triggered`
  - The notebook prints a `run_dir` for each failed run (example pattern):
    - `outputs/runs/office-home/Ar2Cl/source_only/<run_id>`

### Observed “summary table” output in the notebook
The notebook output contains a displayed table built from existing `metrics.csv` files found under `outputs/runs/office-home/Ar2Cl` at the time of execution:

```
        method  seed  source_acc  target_acc      run_id
0        coral     0   80.428513   33.127148  ffc0b32a86
1         dann     0   80.428513   33.127148  ab83d40821
2       me_iis     0   76.967450   28.568156  d3b3fbdc39
3  source_only     0   78.038731   30.148912  f2fc29f097
```

The notebook output also prints:
- `[Summary] Found 4 metrics.csv files under outputs/runs/office-home/Ar2Cl`
- `[Export] Wrote: outputs/runs/office-home/Ar2Cl/all_metrics.csv`

## B2) Output integrity diagnosis and fix (run_id-based aggregation)

### What the notebook previously did
In its original summary/export cells, the notebook:
- globbed for `metrics.csv` files under `RUNS_ROOT/dataset_tag/pair/*/*/metrics.csv`, then
- concatenated them into a DataFrame and pivoted by `(method, seed)`.

This is vulnerable to stale runs:
- If a new attempted run fails before producing `metrics.csv`, but older `metrics.csv` from previous runs still exist on disk, the summary can show values for runs that did **not** complete in the current session/config.

### What was changed
This repo now includes a run-id-based notebook summarizer:
- `src/experiments/notebook_summary.py:collect_expected_runs(...)`

And the notebook summary/export cells were updated to:
- build the **expected** list of `RunConfig`s (methods × seeds) using the same `make_cfg(...)` used for running, then
- locate each run by its deterministic `run_id` directory (no glob ambiguity), and
- display `target_acc` for `status=="OK"`, otherwise display an explicit status string (e.g., `FAILED`, `NOT RUN`).

### Legacy (non-unified) CSV logging robustness
Separately, legacy scripts write `results/office_home_me_iis.csv`. This repo now:
- adds a `run_id` column and uses `utils/logging_utils.py:upsert_csv_row(...)` keyed by `run_id` so repeated resumes do not append duplicate rows.

