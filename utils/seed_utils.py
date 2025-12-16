import os
import random
from typing import Optional

import numpy as np
import torch

_DEVICE_CACHE: Optional[torch.device] = None
_DEVICE_KEY: Optional[str] = None
_DEVICE_LOGGED_KEYS: set[str] = set()


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    # Enable cuDNN autotune for speed when determinism is not required and CUDA is present.
    torch.backends.cudnn.benchmark = torch.cuda.is_available() and not deterministic
    # Make hash seed stable for Python
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(
    gpu_preference: Optional[int] = None,
    deterministic: bool = True,
    device: Optional[str] = None,
) -> torch.device:
    """
    Return CUDA device if available else CPU, with a single clear log line.
    Optionally enables cuDNN benchmarking when determinism is not requested.
    """
    global _DEVICE_CACHE, _DEVICE_KEY, _DEVICE_LOGGED_KEYS

    device_spec = device
    if device_spec is None:
        device_spec = os.environ.get("ME_IIS_DEVICE")
    device_spec = str(device_spec).strip() if device_spec is not None else ""
    cache_key = device_spec or f"auto:{gpu_preference if gpu_preference is not None else 0}"

    if _DEVICE_CACHE is None or _DEVICE_KEY != cache_key:
        resolved: Optional[torch.device] = None
        device_desc = "cpu"

        if device_spec:
            spec = device_spec.lower()
            if spec == "cpu":
                resolved = torch.device("cpu")
                device_desc = "cpu"
            elif spec.startswith("cuda"):
                if not torch.cuda.is_available():
                    print(f"[DEVICE][WARN] Requested '{device_spec}' but CUDA is unavailable. Falling back to CPU.")
                    resolved = torch.device("cpu")
                    device_desc = "cpu"
                else:
                    idx = 0
                    if ":" in spec:
                        _, raw_idx = spec.split(":", 1)
                        try:
                            idx = int(raw_idx)
                        except Exception:
                            idx = 0
                    if idx >= int(torch.cuda.device_count()):
                        print(
                            f"[DEVICE][WARN] Requested '{device_spec}' but only {torch.cuda.device_count()} CUDA device(s) found. "
                            "Using cuda:0."
                        )
                        idx = 0
                    resolved = torch.device(f"cuda:{idx}")
                    device_desc = f"cuda (GPU) {idx}"
            else:
                try:
                    resolved = torch.device(device_spec)
                    device_desc = str(resolved)
                except Exception as exc:
                    print(f"[DEVICE][WARN] Failed to parse device '{device_spec}': {exc}. Falling back to CPU.")
                    resolved = torch.device("cpu")
                    device_desc = "cpu"
        else:
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                device_idx = gpu_preference if gpu_preference is not None else 0
                try:
                    resolved = torch.device(f"cuda:{device_idx}")
                    device_desc = f"cuda (GPU) {device_idx}"
                except Exception as exc:
                    print(f"[DEVICE][WARN] Failed to select CUDA device {device_idx}: {exc}. Falling back to CPU.")
                    resolved = torch.device("cpu")
                    device_desc = "cpu"
            else:
                resolved = torch.device("cpu")
                device_desc = "cpu"

        _DEVICE_CACHE = resolved
        _DEVICE_KEY = cache_key
        if cache_key not in _DEVICE_LOGGED_KEYS:
            print(f"[DEVICE] Using {device_desc}")
            _DEVICE_LOGGED_KEYS.add(cache_key)

    # Always apply deterministic/benchmark flags (may differ per run).
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = torch.cuda.is_available() and not deterministic
    return _DEVICE_CACHE
