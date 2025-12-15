from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from utils.experiment_utils import dataset_tag
from utils.persist_paths import resolve_persist_root


def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def config_to_canonical_dict(config_obj: Any) -> Dict[str, Any]:
    if hasattr(config_obj, "__dataclass_fields__"):
        payload = asdict(config_obj)
    elif isinstance(config_obj, Mapping):
        payload = dict(config_obj)
    else:
        raise TypeError(f"Unsupported config type: {type(config_obj)}")
    return _json_safe(payload)


def compute_run_id(config_obj: Any) -> str:
    canonical = config_to_canonical_dict(config_obj)
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]


def default_runs_root() -> Path:
    override = os.getenv("ME_IIS_RUNS_ROOT")
    if override:
        return Path(override)
    persist_root = resolve_persist_root()
    if persist_root is not None:
        return Path(persist_root) / "outputs" / "runs"
    return Path("outputs") / "runs"


def get_run_dir(config: "RunConfig", runs_root: Optional[Path] = None) -> Path:
    root = runs_root or default_runs_root()
    ds = dataset_tag(config.dataset_name)
    pair = f"{config.source_domain}2{config.target_domain}"
    return root / ds / pair / config.method / config.run_id


def save_config(run_dir: Path, config: "RunConfig") -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.json"
    path.write_text(
        json.dumps(config_to_canonical_dict(config), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True)
class RunConfig:
    dataset_name: str
    data_root: Optional[str]
    source_domain: str
    target_domain: str

    method: str  # e.g. "source_only", "me_iis", "dann", "coral", "pseudo_label"

    backbone: str = "resnet50"
    backbone_pretrained: bool = True
    input_size: int = 224
    transforms_id: str = "domain_loaders_v1"

    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 1e-3
    scheduler: str = "cosine"

    epochs_source: int = 50
    epochs_adapt: int = 10
    batch_size: int = 32
    num_workers: int = 4

    lr_backbone: float = 1e-3
    lr_classifier: float = 1e-2

    finetune_backbone: bool = False
    backbone_lr_scale: float = 0.1
    classifier_lr: float = 1e-2

    feature_layers: tuple[str, ...] = ("layer3", "layer4")

    method_params: Dict[str, Any] = field(default_factory=dict)

    seed: int = 0
    deterministic: bool = False

    dry_run_max_samples: int = 0
    dry_run_max_batches: int = 0

    @property
    def run_id(self) -> str:
        return compute_run_id(self)
