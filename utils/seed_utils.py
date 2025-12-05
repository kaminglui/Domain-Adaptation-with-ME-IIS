import os
import random
from typing import Optional

import numpy as np
import torch

_DEVICE_CACHE: Optional[torch.device] = None
_DEVICE_LOGGED = False


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


def get_device(gpu_preference: Optional[int] = None, deterministic: bool = True) -> torch.device:
    """
    Return CUDA device if available else CPU, with a single clear log line.
    Optionally enables cuDNN benchmarking when determinism is not requested.
    """
    global _DEVICE_CACHE, _DEVICE_LOGGED
    if _DEVICE_CACHE is None:
        use_cuda = torch.cuda.is_available()
        device_desc = "cpu"
        if use_cuda:
            device_idx = gpu_preference if gpu_preference is not None else 0
            try:
                _DEVICE_CACHE = torch.device(f"cuda:{device_idx}")
                device_desc = f"cuda (GPU) {device_idx}"
                if not deterministic:
                    torch.backends.cudnn.benchmark = True
            except Exception as exc:
                print(f"[DEVICE][WARN] Failed to select CUDA device {device_idx}: {exc}. Falling back to CPU.")
                _DEVICE_CACHE = torch.device("cpu")
                device_desc = "cpu"
        if _DEVICE_CACHE is None:
            _DEVICE_CACHE = torch.device("cpu")
        torch.backends.cudnn.deterministic = deterministic
        if not _DEVICE_LOGGED:
            print(f"[DEVICE] Using {device_desc}")
            _DEVICE_LOGGED = True
    return _DEVICE_CACHE
