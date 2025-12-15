# ME-IIS Math ↔ Code Alignment (Pal & Miller IIS)

This document maps the repository’s IIS implementation to the Pal & Miller (2007) “probabilistic instances” IIS extension, using the on-disk paper `Iterative Scaling.pdf` and the current code in `models/`.

## C1) Constraints used in this repo (feature functions)

### What this repo treats as the “feature functions” `f_k(t)`

For each selected backbone layer `i` and for each latent component `j` and class `c`, this repo constructs a **joint latent×class fractional feature**:

`f_{i,j,c}(t) = P[M_i = j | a_i(t)] * P[C = c | x(t)]`

where:
- `a_i(t)` is the pooled activation vector extracted from layer `i` for sample `t`.
- `P[M_i=j | a_i(t)]` is the latent responsibility from a clustering backend fit on *target* activations.
- `P[C=c | x(t)]` is the classifier posterior (`softmax`) or (for source only) optionally a one-hot distribution.

**Code construction**
- Responsibilities `P[M_i=j | a_i(t)]`:
  - Built in `models/iis_components.py:162` (`JointConstraintBuilder.build_joint`) by calling `LatentBackend.predict_proba(...)` on the layer activations.
- Class posteriors `P[C=c | x(t)]`:
  - Legacy ME-IIS script: `scripts/adapt_me_iis.py` builds:
    - source `P[C|x_s]` via `scripts/adapt_me_iis.py:_class_probs_from_logits(...)` (“softmax” or “onehot”)
    - target `P[C|x_t]` via `softmax(target_logits)` and then deletes `target_labels`
  - Unified ME-IIS method: `src/experiments/methods/me_iis.py:_class_probs_from_logits(...)`
- Joint tensor:
  - `joint[layer] = gamma.unsqueeze(2) * probs.unsqueeze(1)` in `models/iis_components.py:162`

**Shapes**
- For a given layer `i`:
  - `gamma`: `(N, J_i)` (responsibilities over `J_i` components)
  - `probs`: `(N, K)` (class probs over `K` classes; row-normalized)
  - `joint[i]`: `(N, J_i, K)`

The flattening index `k` corresponds to `(layer i, component j, class c)` and is implemented by:
- `models/iis_components.py:43` (`ConstraintIndexer`) and `models/iis_components.py:57` (`ConstraintIndexer.flatten`)

## C1) What are “ground-truth” vs “model” moments in this setting?

The Pal & Miller paper defines (for latent-variable constraints) expected constraint probabilities `P_g[...]` (“ground truth” constraint moments) and model moments `P_m[...]`.

In this repo:
- **Target moments** (“ground truth moments” for the ME constraints) are computed from unlabeled target samples using *model-predicted* class posteriors:
  - `P_g` is computed as the mean of the target joint features:
    - `target_moments = IISUpdater.compute_pg(target_flat)` in `models/me_iis_adapter.py:221` (`MaxEntAdapter.solve_iis`)
- **Model moments** are computed as a weighted average of the source joint features under the current weights `q`:
  - `P_m = Σ_t q(t) f(t)`:
    - `pm = IISUpdater.compute_pm(weights, source_flat)` in `models/iis_components.py:243`

**Shapes**
- `target_flat`: `(N_t, C_total)`
- `source_flat`: `(N_s, C_total)`
- `target_moments` (`P_g`): `(C_total,)`
- `pm` (`P_m`): `(C_total,)`
- `weights` (`q`): `(N_s,)`, non-negative, sums to 1

## C1) Pal & Miller IIS update and how this repo implements it

### Paper reference (Pal & Miller, 2007)
The included paper `Iterative Scaling.pdf` contains, on page 7, the latent-variable constraint moments:
- Eq. (14): `P_g[C=c, M_i=j] = (1/T) Σ_t I(c(t)=c) P[M_i=j | a_i(t)]`
- Eq. (15): `P_m[C=c, M_i=j] = (1/T) Σ_t P[C=c | f(t), a(t)] P[M_i=j | a_i(t)]`
and the IIS update:
- Eq. (18): `Δλ = (1/(N_d + N_c)) * log(P_g / P_m)` (applied to latent constraints)

### Code mapping

**`P_g` / `P_m`**
- `P_g`: `models/iis_components.py:237` (`IISUpdater.compute_pg`) returns `flat_joint.mean(dim=0)`
- `P_m`: `models/iis_components.py:243` (`IISUpdater.compute_pm`) returns `Σ_t weights[t] * flat_joint[t]`

**Update constant `1/(N_d + N_c)`**
- Pal & Miller denominator corresponds to `models/iis_components.py:230` (`IISUpdater.mass_constant`)
- In this repo, `MaxEntAdapter` sets `num_discrete=0` and `num_latent=len(layers)`:
  - `models/me_iis_adapter.py:83` (`self.iis_updater = IISUpdater(num_latent=len(self.layers), num_discrete=0)`)
  - Thus the denominator is `N_d + N_c = 0 + len(layers)`

**`Δλ`**
- `models/iis_components.py:251` (`IISUpdater.delta_lambda`) implements:
  - `log(clamp(pg)/clamp(pm)) / (N_d + N_c)`

**Weight update**
- `models/iis_components.py:257` (`IISUpdater.update_weights`) implements:
  - `q_new(t) ∝ q(t) * exp( Σ_k f_k(t) * Δλ_k )` then renormalizes to sum 1

**IIS iteration loop**
- `models/me_iis_adapter.py:221` (`MaxEntAdapter.solve_iis`) calls `IISUpdater.step(...)` each iteration:
  - `models/iis_components.py:267` (`IISUpdater.step`) computes `pm`, `delta`, and updated weights

## C2) Pal & Miller “feature mass” condition (constant-mass validation)

### Does this repo’s `f_k(t)` design guarantee constant mass?

Yes (by construction, per layer), because:
- `Σ_j P[M_i=j | a_i(t)] = 1` (responsibilities are normalized)
- `Σ_c P[C=c | x(t)] = 1` (class probs are row-normalized)
Therefore, for each layer `i`:
- `Σ_{j,c} f_{i,j,c}(t) = 1`

Summing across all selected layers:
- `Σ_{i,j,c} f_{i,j,c}(t) = N_c = len(layers)` (constant across samples `t`)

### What the code does (validation + warnings)
- Per-layer and total mass checks:
  - `models/iis_components.py:124` (`JointConstraintBuilder._validate_joint`) checks:
    - per-layer mass close to 1
    - total mass close to `expected_feature_mass = len(layers)` (`models/iis_components.py:101`)
- Additional diagnostic print:
  - `models/me_iis_adapter.py:296` checks the empirical `feature_mass` mean/std and prints a warning if relative std exceeds `f_mass_rel_tol`

### Correction implemented
To reduce numerical drift from clustering backends, this repo now explicitly renormalizes responsibilities:
- `models/iis_components.py:162` and `models/iis_components.py:184`
  - clamps responsibilities to `>=0`
  - divides each row by its row-sum

This enforces the per-layer mass condition assumed by the standard IIS derivation (and matches the validation logic already present in `_validate_joint`).

## C3) Unit-level correctness checks (what exists in `tests/`)

Deterministic IIS tests in this repo include:
- Pal & Miller Eq. (18) denominator/update form:
  - `tests/test_clustering_backends.py::TestLatentBackendAndConstraints::test_iis_delta_lambda_matches_eq18`
- Constraint mass property:
  - `tests/test_clustering_backends.py::test_joint_constraint_mass_is_Nc`
- Synthetic known-solution IIS:
  - `tests/test_me_iis_additional.py::TestIISKnownSolution::test_exact_solution_known_joint`
  - `tests/test_iis_two_class_solution.py::TestIISKnownSolutionTwoClasses::test_exact_solution_two_components_two_classes`
- Unachievable-constraint detection (legacy solver prints warnings and records indices):
  - `tests/test_me_iis_additional.py::TestIISConvergenceProperties::test_unreachable_constraint`
