from __future__ import annotations

import json
import re
from hashlib import sha1
from typing import Any, Mapping


_SAFE_CHARS_RE = re.compile(r"[^a-zA-Z0-9_.=\\-]+")


def _sanitize(value: str) -> str:
    value = value.strip()
    value = value.replace(" ", "")
    value = _SAFE_CHARS_RE.sub("-", value)
    return value


def _format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        # Stable compact float formatting.
        s = f"{v:.6g}"
        s = s.replace("+", "")
        return s
    return str(v)


_KEY_ORDER: list[tuple[str, str]] = [
    ("dataset", "ds"),
    ("split_mode", "split"),
    ("eval_split", "eval"),
    ("adapt_split", "adapt"),
    ("algorithm", "alg"),
    ("backbone", "bb"),
    ("seed", "seed"),
    ("pretrained", "pt"),
    ("replace_batchnorm_with_instancenorm", "in"),
    ("optimizer", "opt"),
    ("lr", "lr"),
    ("weight_decay", "wd"),
    ("batch_size", "bs"),
    ("grad_accum_steps", "ga"),
    ("epochs", "ep"),
    # DANN
    ("dann_penalty_weight", "dann"),
    ("grl_lambda", "grl"),
    ("dann_featurizer_lr_mult", "fmult"),
    ("dann_discriminator_lr_mult", "dmult"),
    # deepCORAL
    ("coral_penalty_weight", "coral"),
    # ME-IIS
    ("meiis_K", "K"),
    ("meiis_tau", "tau"),
    ("meiis_step", "iis"),
    ("meiis_damp", "damp"),
    ("meiis_ema", "ema"),
]


def encode_config_to_run_id(config: Mapping[str, Any]) -> str:
    """
    Encode a flat config mapping into a stable, path-safe run ID.

    Format: `k=v__k=v__...` with a fixed key order plus an 8-char hash of the
    full config to reduce collision risk.
    """
    parts: list[str] = []
    for key, short in _KEY_ORDER:
        if key not in config or config[key] is None:
            continue
        val = _sanitize(_format_value(config[key]))
        parts.append(f"{short}={val}")

    # Add a short hash to make collisions extremely unlikely while keeping the ID decodable.
    payload = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    h = sha1(payload.encode("utf-8")).hexdigest()[:8]
    parts.append(f"h={h}")
    return "__".join(parts)


def decode_run_id_to_config(run_id: str) -> dict[str, Any]:
    """
    Best-effort decode of a run ID produced by `encode_config_to_run_id`.
    """
    inv = {short: key for key, short in _KEY_ORDER}
    out: dict[str, Any] = {}
    for token in run_id.split("__"):
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        key = inv.get(k, k)
        out[key] = v
    return out


def fingerprint_config(config: Mapping[str, Any]) -> str:
    """
    Compute a stable fingerprint for config-safe checkpoint skipping.

    Uses a canonical JSON dump with sorted keys and sha1 hashing.
    """
    payload = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return sha1(payload.encode("utf-8")).hexdigest()
