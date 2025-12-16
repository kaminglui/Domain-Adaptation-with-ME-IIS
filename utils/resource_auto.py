"""
Colab-first resource helpers.

This module is the public surface for:
- machine resource detection (CPU/RAM/GPU/disk),
- dataloader auto-tuning defaults,
- optional dataset caching from Google Drive -> local SSD.

Implementation reuses the existing primitives in `utils.resource_utils` to avoid
duplicating logic across the codebase.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from utils.env_utils import is_colab
from utils.resource_utils import (
    CheckpointTuning,
    DataLoaderTuning,
    ResourceSnapshot,
    auto_tune_dataloader,
    detect_resources as _detect_resources,
    format_bytes,
    tune_checkpoint_saving,
    write_resource_snapshot,
)


def detect_resources(
    *,
    disk_path: Path,
    data_path: Optional[Path] = None,
    cuda_device: int = 0,
) -> ResourceSnapshot:
    """
    Detect CPU/GPU/disk resources.

    Uses `psutil` when available, otherwise falls back to OS-specific methods.
    """
    snap = _detect_resources(disk_path=disk_path, data_path=data_path, cuda_device=cuda_device)
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return ResourceSnapshot(
            cpu_total_bytes=int(getattr(vm, "total", snap.cpu_total_bytes or 0)) or snap.cpu_total_bytes,
            cpu_available_bytes=int(getattr(vm, "available", snap.cpu_available_bytes or 0)) or snap.cpu_available_bytes,
            cpu_method="psutil:virtual_memory",
            cpu_count=snap.cpu_count,
            cuda_available=snap.cuda_available,
            cuda_device=snap.cuda_device,
            cuda_name=snap.cuda_name,
            cuda_total_bytes=snap.cuda_total_bytes,
            cuda_free_bytes=snap.cuda_free_bytes,
            disk_path=snap.disk_path,
            disk_total_bytes=snap.disk_total_bytes,
            disk_free_bytes=snap.disk_free_bytes,
            data_path=snap.data_path,
            data_disk_total_bytes=snap.data_disk_total_bytes,
            data_disk_free_bytes=snap.data_disk_free_bytes,
        )
    except Exception:
        return snap


def _is_drive_path(path: Path) -> bool:
    raw = str(path).replace("\\", "/")
    return raw.startswith("/content/drive/")


def _dir_size_bytes(path: Path, stop_at_bytes: Optional[int] = None) -> Optional[int]:
    try:
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                fp = Path(root) / name
                try:
                    total += int(fp.stat().st_size)
                except Exception:
                    continue
                if stop_at_bytes is not None and total >= int(stop_at_bytes):
                    return int(total)
        return int(total)
    except Exception:
        return None


def maybe_cache_dataset_to_local(
    *,
    dataset_name: str,
    data_root: Path,
    local_base: Path = Path("/content/datasets"),
    headroom_bytes: int = 5 * 1024**3,
) -> Tuple[Path, Dict[str, Any]]:
    """
    On Colab, if `data_root` is on Google Drive, copy it once to local SSD and return the local path.
    Returns (resolved_root, info_dict).
    """
    info: Dict[str, Any] = {
        "dataset": str(dataset_name),
        "original_root": str(data_root),
        "cached": False,
        "cache_root": None,
        "reason": None,
    }
    if not is_colab():
        info["reason"] = "not_colab"
        return data_root, info

    data_root = Path(data_root).expanduser()
    if not data_root.exists():
        info["reason"] = "missing_data_root"
        return data_root, info
    if not _is_drive_path(data_root):
        info["reason"] = "not_drive_path"
        return data_root, info

    local_base = Path(local_base)
    local_base.mkdir(parents=True, exist_ok=True)
    target = local_base / data_root.name
    marker = target / ".me_iis_cached_ok"
    info["cache_root"] = str(target)

    if marker.exists():
        info["cached"] = True
        info["reason"] = "already_cached"
        return target, info

    try:
        free = int(shutil.disk_usage(str(local_base)).free)
    except Exception:
        info["reason"] = "disk_usage_failed"
        return data_root, info

    budget = max(0, int(free) - int(headroom_bytes))
    if budget <= 0:
        info["reason"] = "insufficient_disk_headroom"
        return data_root, info

    est_size = _dir_size_bytes(data_root, stop_at_bytes=budget + 1)
    if est_size is None:
        info["reason"] = "size_scan_failed"
        return data_root, info
    info["estimated_bytes"] = int(est_size)
    info["estimated_pretty"] = format_bytes(int(est_size))

    if est_size > budget:
        info["reason"] = f"insufficient_disk_free (need~{format_bytes(est_size)}, have~{format_bytes(budget)})"
        return data_root, info

    print(
        f"[CACHE] Copying dataset from Drive to local SSD: {data_root} -> {target} "
        f"(size~{format_bytes(est_size)} free~{format_bytes(free)})"
    )
    shutil.copytree(data_root, target, dirs_exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")
    info["cached"] = True
    info["reason"] = "copied"
    return target, info


def write_resources_with_tuning(
    *,
    run_dir: Path,
    resources: ResourceSnapshot,
    dataloader_tuning: Optional[DataLoaderTuning] = None,
    checkpoint_tuning: Optional[CheckpointTuning] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    tuning: Dict[str, Any] = {}
    if dataloader_tuning is not None:
        tuning["dataloader"] = asdict(dataloader_tuning)
    if checkpoint_tuning is not None:
        tuning["checkpoint"] = asdict(checkpoint_tuning)
    if extra:
        tuning.update(extra)
    return write_resource_snapshot(run_dir, resources, tuning=tuning if tuning else None)

