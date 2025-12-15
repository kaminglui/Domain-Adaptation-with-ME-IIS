# ME-IIS Feasibility Determination (Evidence-Based)

This feasibility assessment is based on:
- code/math alignment (`docs/ME_IIS_MATH_ALIGNMENT.md`),
- notebook outputs (`notebooks/Run_All_Experiments.ipynb`),
- on-disk legacy artifacts in this repo under `results/` and `checkpoints/`, and
- unit tests (`python -m pytest`).

## Is the method implemented correctly relative to Pal & Miller IIS?

**Yes, at the IIS solver level.**

Evidence:
- The included paper `Iterative Scaling.pdf` (Pal & Miller, 2007) states the probabilistic-instance latent constraint moments (Eq. 14–15) and the IIS parameter increment form (Eq. 18) with denominator `(N_d + N_c)` on page 7.
- The repo’s IIS implementation matches this structure:
  - `models/iis_components.py` implements `P_g = mean(flat_joint)` and `P_m = Σ_t q(t) f(t)`, and `Δλ = log(P_g/P_m)/(N_d+N_c)` (`IISUpdater.compute_pg`, `.compute_pm`, `.delta_lambda`).
  - The weight update `q_new ∝ q * exp(Σ_k f_k(t) Δλ_k)` is implemented in `models/iis_components.py:IISUpdater.update_weights`.
  - The ME-IIS adapter sets `N_d=0` and `N_c=len(layers)` in `models/me_iis_adapter.py` via `IISUpdater(num_latent=len(layers), num_discrete=0)`.
- Deterministic unit tests verify:
  - Eq. (18) denominator/update form (`tests/test_clustering_backends.py::test_iis_delta_lambda_matches_eq18`)
  - known-solution IIS convergence (`tests/test_me_iis_additional.py` and `tests/test_iis_two_class_solution.py`)

## Is the method empirically promising in this implementation (with observed results)?

**Not yet, based on the repository’s available representative outputs.**

Evidence from the stored notebook metrics table (Office-Home `Ar→Cl`, seed=0):
- `source_only` target_acc: `30.148912`
- `me_iis` target_acc: `28.568156`
- `dann` / `coral` target_acc: `33.127148`

In that table, ME-IIS is below source-only and below the baselines for that single reported seed.

Additionally, the stored notebook outputs show multiple runs failing before logging metrics (CUDA “device-side assert triggered”), reducing the amount of reliable multi-seed evidence available in-repo.

## Likely cause category (based on logs/artifacts)

The repo contains a legacy IIS artifact for a dry-run config:
- `results/me_iis_weights_Ar_to_Cl_layer3-layer4-avgpool_seed0.npz`

That file’s IIS diagnostics show:
- a large, nearly constant `moment_max ≈ 0.725` over 15 iterations
- decreasing entropy and increasing max weight (`w_entropy` decreases; `w_max` rises to ~0.39)
- feature-mass condition holds (`feature_mass_mean = 3.0`, tiny std)

This combination is consistent with **constraints that are not feasible** (e.g., at least one “unachievable constraint” where the target has positive mass on a (style, class) bin that has zero mass under all source samples). In `models/me_iis_adapter.py`, such constraints are detected and warned about, and the count is now recorded per iteration (`num_unachievable_constraints`).

## Is ME-IIS still feasible to evaluate in this repo?

**Feasible to evaluate and debug, but not yet supported by strong in-repo evidence of gains.**

Reasons:
- The IIS implementation is unit-tested and aligns with the Pal & Miller update.
- The unified runner logs deterministic `run_id` runs to per-run directories with `metrics.csv`.
- The notebook summary/export is now `run_id`-based (expected configs → deterministic run paths), so it does not silently pick up stale runs and it reports `FAILED` / `NOT RUN` explicitly.

However:
- The only representative accuracy table stored in the notebook shows ME-IIS underperforming.
- The dry-run IIS artifact in this repo shows non-converging max moment error and weight concentration, which can undermine the intended “match target moments” goal.

## Next steps (minimal, seed=0)

For a quick feasibility check without multi-seed cost, run the minimal sweep described in `docs/ME_IIS_TUNING_REPORT.md` (seed=0 only) using the unified runner/notebook. The key question to resolve is whether tuning can:
- reduce unachievable constraints,
- avoid weight collapse, and
- beat source-only on at least one representative task under a fixed training budget.

