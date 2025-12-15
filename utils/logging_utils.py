import hashlib
import json
import csv
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

OFFICE_HOME_ME_IIS_FIELDS = [
    "dataset",
    "source",
    "target",
    "seed",
    "method",
    "run_id",
    "status",
    "error",
    "target_acc",
    "source_acc",
    "num_latent",
    "layers",
    "components_per_layer",
    "iis_iters",
    "iis_tol",
    "adapt_epochs",
    "finetune_backbone",
    "backbone_lr_scale",
    "classifier_lr",
    "source_prob_mode",
    "config_json",
    "timestamp_utc",
]

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def append_csv(path: str, fieldnames: Iterable[str], row: Dict) -> None:
    """Append a row to CSV, creating header if needed."""
    ensure_dir(os.path.dirname(path) or ".")
    file_exists = os.path.isfile(path)
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _sha1_10(blob: bytes) -> str:
    return hashlib.sha1(blob).hexdigest()[:10]


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_csv(path: Path) -> Tuple[list[str], list[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def upsert_csv_row(
    path: str,
    fieldnames: Iterable[str],
    row: Dict[str, Any],
    unique_key: str = "run_id",
) -> None:
    """
    Upsert a single CSV row, ensuring the file contains at most one row per `unique_key`.

    - If the CSV doesn't exist, it is created with `fieldnames`.
    - If the CSV exists, it is rewritten with the union of existing headers + `fieldnames`.
    - If the CSV exists but lacks `unique_key`, it is migrated by computing a deterministic
      `unique_key` for each existing row as sha1(json(row))[:10]. This migration is best-effort
      and primarily intended to deduplicate legacy runs that were appended multiple times.
    """
    path_obj = Path(path)
    ensure_dir(os.path.dirname(path) or ".")

    run_id = str(row.get(unique_key, "")).strip()
    if not run_id:
        raise ValueError(f"upsert_csv_row requires a non-empty '{unique_key}' in row.")

    desired_fields = list(fieldnames)
    if unique_key not in desired_fields:
        desired_fields.insert(0, unique_key)

    if not path_obj.exists():
        _write_csv(path_obj, desired_fields, [row])
        return

    existing_fields, rows = _load_csv(path_obj)
    if unique_key not in existing_fields:
        migrated: list[Dict[str, Any]] = []
        for r in rows:
            rid = _sha1_10(_canonical_json(r).encode("utf-8"))
            r[unique_key] = rid
            migrated.append(r)
        rows = migrated
        existing_fields = existing_fields + [unique_key]

    # Best-effort dedupe: keep the last occurrence for each unique_key.
    deduped: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for r in reversed(rows):
        rid = str(r.get(unique_key, "")).strip()
        if not rid:
            rid = _sha1_10(_canonical_json(r).encode("utf-8"))
            r[unique_key] = rid
        if rid in seen:
            continue
        seen.add(rid)
        deduped.append(r)
    rows = list(reversed(deduped))

    out_fields = desired_fields + [f for f in existing_fields if f not in desired_fields]

    kept = [r for r in rows if str(r.get(unique_key, "")).strip() != run_id]
    kept.append(row)
    _write_csv(path_obj, out_fields, kept)


class TBLogger:
    """Lightweight TensorBoard logger wrapper."""

    def __init__(self, log_dir: str):
        ensure_dir(log_dir)
        if SummaryWriter is None:
            raise ImportError("TensorBoard not installed. Add tensorboard to requirements to use TBLogger.")
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalars(self, tag: str, scalar_dict: Dict[str, float], step: int) -> None:
        for key, val in scalar_dict.items():
            self.writer.add_scalar(f"{tag}/{key}", val, step)

    def close(self) -> None:
        self.writer.close()
