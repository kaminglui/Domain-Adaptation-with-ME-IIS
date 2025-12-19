from __future__ import annotations

# Backwards-compatible import path.
#
# The canonical Camelyon17 trainer implementation lives in `src/train/trainer.py`.

from .trainer import eval_wilds_split, set_seed, train

__all__ = ["train", "eval_wilds_split", "set_seed"]

