"""
Resource detection + lightweight auto-tuning helpers.

Goal: make training runs adapt to the current machine (CPU RAM / CUDA VRAM / disk)
without hard-coding batch sizes or dataloader worker counts.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = str(raw).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def auto_resources_enabled(default: bool = False) -> bool:
    """
    Global switch for resource-aware auto-tuning.

    Enable by setting environment variable:
      - ME_IIS_AUTO_RESOURCES=1
    """
    return _env_flag("ME_IIS_AUTO_RESOURCES", default=default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if value < 1024.0 or unit == "PB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}PB"


def _windows_memory_bytes() -> Tuple[int, int]:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
    if not ok:
        raise OSError("GlobalMemoryStatusEx failed")
    return int(stat.ullTotalPhys), int(stat.ullAvailPhys)


def _proc_meminfo_bytes() -> Tuple[Optional[int], Optional[int]]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None, None
    total_kb = available_kb = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                total_kb = int(parts[1])
        elif line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                available_kb = int(parts[1])
    total = int(total_kb * 1024) if total_kb is not None else None
    available = int(available_kb * 1024) if available_kb is not None else None
    return total, available


def cpu_memory_bytes() -> Tuple[Optional[int], Optional[int], str]:
    """
    Return (total_bytes, available_bytes, method).
    """
    system = platform.system().lower()
    if system.startswith("windows"):
        try:
            total, avail = _windows_memory_bytes()
            return total, avail, "windows:GlobalMemoryStatusEx"
        except Exception:
            return None, None, "windows:failed"

    # Prefer /proc/meminfo on Linux for MemAvailable.
    total, avail = _proc_meminfo_bytes()
    if total is not None:
        return total, avail, "procfs:/proc/meminfo"

    # Generic POSIX fallback.
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))  # type: ignore[attr-defined]
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))  # type: ignore[attr-defined]
        total = page_size * total_pages
        avail_pages = None
        try:
            avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))  # type: ignore[attr-defined]
        except Exception:
            avail_pages = None
        avail = page_size * avail_pages if avail_pages is not None else None
        return int(total), int(avail) if avail is not None else None, "posix:sysconf"
    except Exception:
        return None, None, "posix:failed"


def disk_usage_bytes(path: Path) -> Tuple[int, int, int]:
    usage = shutil.disk_usage(str(path))
    return int(usage.total), int(usage.used), int(usage.free)


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_total_bytes: Optional[int]
    cpu_available_bytes: Optional[int]
    cpu_method: str
    cpu_count: Optional[int]

    cuda_available: bool
    cuda_device: Optional[int]
    cuda_name: Optional[str]
    cuda_total_bytes: Optional[int]
    cuda_free_bytes: Optional[int]

    disk_path: str
    disk_total_bytes: Optional[int]
    disk_free_bytes: Optional[int]

    data_path: Optional[str] = None
    data_disk_total_bytes: Optional[int] = None
    data_disk_free_bytes: Optional[int] = None

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Keep JSON small + stable.
        payload["cpu_total_pretty"] = format_bytes(self.cpu_total_bytes)
        payload["cpu_available_pretty"] = format_bytes(self.cpu_available_bytes)
        payload["cuda_total_pretty"] = format_bytes(self.cuda_total_bytes)
        payload["cuda_free_pretty"] = format_bytes(self.cuda_free_bytes)
        payload["disk_total_pretty"] = format_bytes(self.disk_total_bytes)
        payload["disk_free_pretty"] = format_bytes(self.disk_free_bytes)
        payload["data_disk_total_pretty"] = format_bytes(self.data_disk_total_bytes)
        payload["data_disk_free_pretty"] = format_bytes(self.data_disk_free_bytes)
        return payload


def _cuda_mem_get_info(device_index: int) -> Tuple[Optional[int], Optional[int]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        try:
            free, total = torch.cuda.mem_get_info(device_index)  # type: ignore[arg-type]
            return int(free), int(total)
        except TypeError:
            # Older torch: mem_get_info() has no device arg.
            current = int(torch.cuda.current_device())
            torch.cuda.set_device(int(device_index))
            try:
                free, total = torch.cuda.mem_get_info()
                return int(free), int(total)
            finally:
                torch.cuda.set_device(current)
    except Exception:
        return None, None


def detect_resources(
    disk_path: Path,
    data_path: Optional[Path] = None,
    cuda_device: int = 0,
) -> ResourceSnapshot:
    cpu_total, cpu_avail, cpu_method = cpu_memory_bytes()
    cpu_count = os.cpu_count()

    cuda_available = False
    cuda_name = None
    cuda_total = None
    cuda_free = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            props = torch.cuda.get_device_properties(int(cuda_device))
            cuda_name = str(getattr(props, "name", "cuda"))
            cuda_total = int(getattr(props, "total_memory", 0)) or None
            free, total = _cuda_mem_get_info(int(cuda_device))
            cuda_free = free
            cuda_total = total if total is not None else cuda_total
    except Exception:
        cuda_available = False

    disk_total = disk_free = None
    try:
        total, _used, free = disk_usage_bytes(Path(disk_path))
        disk_total, disk_free = total, free
    except Exception:
        pass

    data_total = data_free = None
    if data_path is not None:
        try:
            total, _used, free = disk_usage_bytes(Path(data_path))
            data_total, data_free = total, free
        except Exception:
            data_total = data_free = None

    return ResourceSnapshot(
        cpu_total_bytes=cpu_total,
        cpu_available_bytes=cpu_avail,
        cpu_method=cpu_method,
        cpu_count=int(cpu_count) if cpu_count is not None else None,
        cuda_available=cuda_available,
        cuda_device=int(cuda_device) if cuda_available else None,
        cuda_name=cuda_name if cuda_available else None,
        cuda_total_bytes=cuda_total if cuda_available else None,
        cuda_free_bytes=cuda_free if cuda_available else None,
        disk_path=str(Path(disk_path)),
        disk_total_bytes=disk_total,
        disk_free_bytes=disk_free,
        data_path=str(Path(data_path)) if data_path is not None else None,
        data_disk_total_bytes=data_total,
        data_disk_free_bytes=data_free,
    )


@dataclass(frozen=True)
class DataLoaderTuning:
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: Optional[int]

    def as_loader_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "pin_memory": bool(self.pin_memory),
        }
        if int(self.num_workers) > 0:
            kwargs["persistent_workers"] = bool(self.persistent_workers)
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return kwargs


@dataclass(frozen=True)
class CheckpointTuning:
    save_every_epochs: int
    estimated_checkpoint_bytes: Optional[int]
    estimated_total_epoch_checkpoints: Optional[int]
    estimated_total_bytes: Optional[int]


def recommend_num_workers(
    requested: int,
    cpu_count: Optional[int],
    cpu_available_bytes: Optional[int],
    max_workers_cap: int = 16,
) -> int:
    """
    Heuristic: use as many workers as the machine can sustain.

    - Cap to avoid spawning hundreds of workers on high-core machines.
    - Also cap based on available RAM (1 worker ~= 2GB budget).
    """
    if cpu_count is None or cpu_count <= 0:
        return max(0, int(requested))
    target_by_cpu = max(0, int(cpu_count) - 1)
    target = min(int(max_workers_cap), int(target_by_cpu))
    if cpu_available_bytes is not None and cpu_available_bytes > 0:
        target_by_ram = int(cpu_available_bytes // (2 * 1024**3))
        target = min(target, max(0, target_by_ram))
    return max(int(requested), int(target))


def recommend_prefetch_factor(cpu_available_bytes: Optional[int], num_workers: int) -> Optional[int]:
    if int(num_workers) <= 0:
        return None
    if cpu_available_bytes is None:
        return 2
    # Increase prefetch when plenty of RAM is available.
    return 4 if cpu_available_bytes >= 24 * 1024**3 else 2


def estimate_checkpoint_bytes_sgd(model) -> Optional[int]:
    """
    Rough disk size estimate for a single checkpoint payload in this repo:
    params + optimizer momentum buffers + misc overhead.
    """
    try:
        import torch

        param_bytes = 0
        for p in model.parameters():
            param_bytes += int(p.numel()) * int(p.element_size())
        # weights + momentum + some overhead (scheduler/state dict metadata)
        return int(param_bytes * 2.4)
    except Exception:
        return None


def tune_checkpoint_saving(
    *,
    disk_free_bytes: Optional[int],
    total_epochs: int,
    save_every_epochs_requested: int,
    model,
    reserve_bytes: int = 5 * 1024**3,
) -> CheckpointTuning:
    est_ckpt = estimate_checkpoint_bytes_sgd(model)
    if disk_free_bytes is None or est_ckpt is None or est_ckpt <= 0:
        return CheckpointTuning(
            save_every_epochs=int(save_every_epochs_requested),
            estimated_checkpoint_bytes=est_ckpt,
            estimated_total_epoch_checkpoints=None,
            estimated_total_bytes=None,
        )

    # Unique files: epoch checkpoints + final checkpoint. "last" checkpoint overwrites.
    requested_every = int(save_every_epochs_requested)
    if requested_every <= 0:
        epoch_ckpts = 0
    else:
        epoch_ckpts = max(0, (int(total_epochs) + requested_every - 1) // requested_every)
    total_unique = epoch_ckpts + 1  # +final
    est_total = int(total_unique) * int(est_ckpt)

    budget = int(max(0, int(disk_free_bytes) - int(reserve_bytes)))
    if est_total <= budget:
        return CheckpointTuning(
            save_every_epochs=int(save_every_epochs_requested),
            estimated_checkpoint_bytes=int(est_ckpt),
            estimated_total_epoch_checkpoints=int(epoch_ckpts),
            estimated_total_bytes=int(est_total),
        )

    # Pick the largest save_every_epochs that fits; if none fits, disable epoch checkpoints.
    if total_epochs <= 0:
        return CheckpointTuning(
            save_every_epochs=0,
            estimated_checkpoint_bytes=int(est_ckpt),
            estimated_total_epoch_checkpoints=0,
            estimated_total_bytes=int(est_ckpt),
        )

    # Max number of epoch checkpoints we can afford (leave room for final).
    max_epoch_ckpts = max(0, (budget // int(est_ckpt)) - 1)
    if max_epoch_ckpts <= 0:
        return CheckpointTuning(
            save_every_epochs=0,
            estimated_checkpoint_bytes=int(est_ckpt),
            estimated_total_epoch_checkpoints=0,
            estimated_total_bytes=int(est_ckpt),
        )

    # Need at most max_epoch_ckpts across total_epochs: every=ceil(total_epochs/max_epoch_ckpts)
    tuned_every = max(1, (int(total_epochs) + int(max_epoch_ckpts) - 1) // int(max_epoch_ckpts))
    epoch_ckpts = max(0, (int(total_epochs) + tuned_every - 1) // tuned_every)
    total_unique = epoch_ckpts + 1
    est_total = int(total_unique) * int(est_ckpt)
    return CheckpointTuning(
        save_every_epochs=int(tuned_every),
        estimated_checkpoint_bytes=int(est_ckpt),
        estimated_total_epoch_checkpoints=int(epoch_ckpts),
        estimated_total_bytes=int(est_total),
    )


def _run_dummy_backward(model, device, batch_size: int, input_size: int, num_classes: int) -> int:
    import torch
    import torch.nn.functional as F

    if device.type != "cuda":
        raise ValueError("dummy backward is only supported for CUDA devices")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Use deterministic tensors (avoid consuming RNG state).
    images = torch.zeros(int(batch_size), 3, int(input_size), int(input_size), device=device, dtype=torch.float32)
    labels = torch.zeros(int(batch_size), device=device, dtype=torch.long)
    model.eval()
    logits, _ = model(images, return_features=False)
    if int(logits.shape[1]) != int(num_classes):
        # num_classes mismatch doesn't matter for memory, but keep labels in-range.
        num_classes = int(logits.shape[1])
    loss = F.cross_entropy(logits, labels.clamp(min=0, max=max(0, num_classes - 1)))
    loss.backward()
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    # Cleanup
    for p in model.parameters():
        p.grad = None
    del images, labels, logits, loss
    torch.cuda.empty_cache()
    return peak


def recommend_max_cuda_batch_size(
    *,
    model,
    device,
    input_size: int,
    num_classes: int,
    base_batch_size: int,
    target_utilization: float = 0.90,
    headroom_bytes: int = 512 * 1024**2,
    max_batch_size_cap: int = 4096,
) -> int:
    """
    Estimate a "near-max" batch size by measuring CUDA peak memory at two small batch sizes
    and extrapolating under the current free VRAM.

    Intended for throughput tuning, not as a scientific hyperparameter sweep.
    """
    import torch

    if device.type != "cuda" or not torch.cuda.is_available():
        return int(base_batch_size)

    # Respect env overrides for power users.
    target_utilization = float(_env_float("ME_IIS_GPU_UTIL", float(target_utilization)))
    max_batch_size_cap = int(_env_int("ME_IIS_MAX_BATCH", int(max_batch_size_cap)))
    headroom_bytes = int(_env_int("ME_IIS_GPU_HEADROOM_MB", int(headroom_bytes // 1024**2))) * 1024**2

    target_utilization = float(max(0.1, min(0.98, target_utilization)))
    max_batch_size_cap = int(max(1, max_batch_size_cap))

    # Warm-up to ensure kernels/allocators are initialized.
    try:
        _run_dummy_backward(model, device, batch_size=1, input_size=input_size, num_classes=num_classes)
    except Exception:
        return int(base_batch_size)

    try:
        m1 = _run_dummy_backward(model, device, batch_size=1, input_size=input_size, num_classes=num_classes)
        m2 = _run_dummy_backward(model, device, batch_size=2, input_size=input_size, num_classes=num_classes)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return max(1, int(min(base_batch_size, 1)))
    except Exception:
        return int(base_batch_size)

    slope = float(m2 - m1)
    if slope <= 0:
        return int(base_batch_size)

    # Current free VRAM is the true limiter (other processes, fragmentation, etc.).
    try:
        free_bytes, _total_bytes = _cuda_mem_get_info(int(device.index or 0))
    except Exception:
        free_bytes, _total_bytes = None, None
    if free_bytes is None:
        # Fallback: cannot see free memory; don't guess.
        return int(base_batch_size)

    budget = float(free_bytes) * float(target_utilization) - float(headroom_bytes)
    if budget <= float(m1):
        return max(1, int(min(base_batch_size, 1)))

    # mem(bs) ~= m1 + slope*(bs-1)
    est = 1.0 + (budget - float(m1)) / float(slope)
    bs = int(max(1, min(float(max_batch_size_cap), est)))

    # Validate and back off if needed.
    while bs > 1:
        try:
            _run_dummy_backward(model, device, batch_size=int(bs), input_size=input_size, num_classes=num_classes)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(1, int(float(bs) * 0.8))
        except Exception:
            return int(base_batch_size)

    return int(max(1, bs))


def auto_tune_dataloader(
    *,
    base_batch_size: int,
    base_num_workers: int,
    device,
    resources: ResourceSnapshot,
    model=None,
    input_size: int = 224,
    num_classes: int = 1000,
) -> DataLoaderTuning:
    """
    Auto-tune dataloader settings for throughput, gated by env var:
      - ME_IIS_AUTO_RESOURCES=1 enables auto tuning.
    """
    if not auto_resources_enabled(default=False):
        return DataLoaderTuning(
            batch_size=int(base_batch_size),
            num_workers=int(base_num_workers),
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
        )

    tuned_workers = recommend_num_workers(
        requested=int(base_num_workers),
        cpu_count=resources.cpu_count,
        cpu_available_bytes=resources.cpu_available_bytes,
        max_workers_cap=int(_env_int("ME_IIS_MAX_WORKERS", 16)),
    )
    tuned_prefetch = recommend_prefetch_factor(resources.cpu_available_bytes, tuned_workers)
    pin_memory = bool(device.type == "cuda")
    persistent_workers = bool(tuned_workers > 0)

    tuned_bs = int(base_batch_size)
    if model is not None and device.type == "cuda":
        tuned_bs = recommend_max_cuda_batch_size(
            model=model,
            device=device,
            input_size=int(input_size),
            num_classes=int(num_classes),
            base_batch_size=int(base_batch_size),
        )

    return DataLoaderTuning(
        batch_size=int(max(1, tuned_bs)),
        num_workers=int(max(0, tuned_workers)),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
        prefetch_factor=tuned_prefetch,
    )


def write_resource_snapshot(run_dir: Path, snapshot: ResourceSnapshot, tuning: Optional[Dict[str, Any]] = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"resources": snapshot.to_json_dict()}
    if tuning is not None:
        payload["tuning"] = tuning
    path = run_dir / "resources.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
