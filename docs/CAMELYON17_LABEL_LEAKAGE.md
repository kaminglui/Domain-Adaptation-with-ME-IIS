# Camelyon17 (WILDS): How We Avoid Label Leakage

This repo uses the official WILDS Camelyon17 dataset splits and keeps a strict separation between:
- **labeled source training**,  
- **unlabeled target adaptation**, and  
- **labeled evaluation (never used for training)**.

## Splits we use

WILDS provides (at minimum) these split names for Camelyon17:
- Labeled: `train`, `val`, `test`
- Unlabeled: `val_unlabeled`, `test_unlabeled`
- Optional (WILDS-version dependent): `train_unlabeled`, `id_val`

Split construction is centralized in `src/datasets/wilds_camelyon17.py::build_camelyon17_splits`.

## Protocols

### 1) `split_mode=uda_target` (intended for real UDA runs)

- **Labeled training**: `train`
- **Unlabeled adaptation**: `test_unlabeled`
- **Evaluation**:
  - selection/early stopping: `val` (labeled)
  - final reporting: `test` (labeled)

Important: `test` labels are never used during training; only `test_unlabeled` is used for adaptation.

### 2) `split_mode=align_val` (debug/ablation only)

- **Labeled training**: `train`
- **Unlabeled adaptation**: `val_unlabeled`
- **Evaluation**: `val` (labeled)

This mode is useful for quick debugging because adaptation and evaluation occur on the same target domain, but it is **not** the protocol used for final test reporting.

## Code-level safeguards

`src/datasets/wilds_camelyon17.py::build_camelyon17_loaders` enforces:
- `split_mode="uda_target"` ⇒ `adapt_split="test_unlabeled"`
- `split_mode="align_val"` ⇒ `adapt_split="val_unlabeled"` and `eval_split="val"`
- refusing to adapt on `test_unlabeled` unless `split_mode="uda_target"`

## ME-IIS: target labels are never accessed

ME-IIS uses only:
- `P(ŷ|x_t)` from the current model (softmax over logits) on the **unlabeled** target split, and
- source labels `y_s` for supervised training with importance weights.

See `src/algorithms/me_iis.py::MEIIS.update_importance_weights` (target pseudo-probs) and `src/algorithms/me_iis.py::MEIIS.update` (weighted supervised loss).

