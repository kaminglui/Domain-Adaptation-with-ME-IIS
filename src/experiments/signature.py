from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn


def _json_canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def hash_state_dict(state: Mapping[str, Any]) -> str:
    """
    Deterministic content hash for a torch state_dict-like mapping.

    Intended for run signatures (detecting identical initial weights), not cryptographic security.
    """
    h = hashlib.sha1()
    for key in sorted(state.keys()):
        h.update(str(key).encode("utf-8"))
        value = state[key]
        if torch.is_tensor(value):
            t = value.detach().to("cpu").contiguous()
            h.update(str(t.dtype).encode("utf-8"))
            h.update(str(tuple(t.shape)).encode("utf-8"))
            h.update(t.numpy().tobytes())
        else:
            h.update(_json_canonical(str(value)))
    return h.hexdigest()[:12]


def hash_model_state(model: nn.Module, *, keys: Optional[list[str]] = None) -> str:
    """
    Hash selected submodule states (default: backbone/bottleneck/classifier if present).
    """
    selected = keys or ["backbone", "bottleneck", "classifier"]
    merged: Dict[str, Any] = {}
    for name in selected:
        if not hasattr(model, name):
            continue
        sub = getattr(model, name)
        if isinstance(sub, nn.Module):
            for k, v in sub.state_dict().items():
                merged[f"{name}.{k}"] = v
    if not merged:
        merged = dict(model.state_dict())
    return hash_state_dict(merged)


def signature_fingerprint(signature: Mapping[str, Any]) -> str:
    """
    Comparison fingerprint that intentionally ignores human labels like method_name.
    """
    payload = {
        "model_state_hash": signature.get("model_state_hash", ""),
        "source_checkpoint": signature.get("source_checkpoint"),
        "loss_terms_enabled": signature.get("loss_terms_enabled", {}),
        "model_components": signature.get("model_components", {}),
    }
    return hashlib.sha1(_json_canonical(payload)).hexdigest()[:12]


def write_signature(run_dir: Path, signature: Dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    signature = dict(signature)
    signature["comparison_fingerprint"] = signature_fingerprint(signature)
    path = run_dir / "signature.json"
    path.write_text(json.dumps(signature, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path

