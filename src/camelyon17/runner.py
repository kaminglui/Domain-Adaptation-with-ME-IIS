from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import torch

from src.algorithms import DANN, ERM, MEIIS, MEIISConfig
from src.datasets import build_camelyon17_loaders, build_camelyon17_transforms
from src.experiments.stream_capture import tee_std_streams
from src.models import build_backbone
from src.train import train
from src.utils.run_id import encode_config_to_run_id, fingerprint_config

Method = Literal["source_only", "dann", "me_iis"]


@dataclass(frozen=True)
class Camelyon17SplitConfig:
    split_mode: Literal["uda_target", "align_val"] = "uda_target"
    adapt_split: Literal["val_unlabeled", "test_unlabeled"] = "test_unlabeled"
    eval_split: Literal["val", "test"] = "test"


@dataclass(frozen=True)
class Camelyon17RunnerConfig:
    # Dataset
    dataset: str = "camelyon17"
    data_root: Optional[str] = None
    download: bool = True
    unlabeled: bool = True

    # Outputs
    runs_root: Path = Path("outputs") / "runs"

    # Repro + compute
    seed: int = 0
    deterministic: bool = True
    amp: bool = True
    device: Optional[str] = None

    # DataLoader
    num_workers: int | str = "auto"
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Model/training
    backbone: Literal["densenet121", "resnet50"] = "densenet121"
    pretrained: bool = False
    replace_batchnorm_with_instancenorm: bool = False
    augment: bool = False
    color_jitter: bool = False
    imagenet_normalize: bool = True

    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0
    scheduler: str = "none"
    early_stop_patience: int = 5
    grad_accum_steps: int = 1

    # Epoch budgets (source-only vs adaptation)
    epochs_source: int = 5
    epochs_adapt: int = 5

    # Dry-run / smoke-test knobs
    dry_run_max_batches: int = 0
    dry_run_max_samples: int = 0
    disable_checkpointing: bool = False

    # DANN
    dann_penalty_weight: float = 1.0
    grl_lambda: float = 1.0
    dann_featurizer_lr_mult: float = 0.1
    dann_discriminator_lr_mult: float = 1.0

    # ME-IIS
    meiis: MEIISConfig = MEIISConfig(
        K=8,
        target_conf_thresh=0.90,
        target_conf_mode="maxprob",
        target_conf_min_count=256,
        max_iis_iters=10,
        iis_step_size=1.0,
        iis_damping=0.0,
        ema_constraints=0.0,
        weight_clip_max=10.0,
        weight_mix_alpha=0.8,
    )


def _parse_methods(flags: Sequence[str]) -> List[Method]:
    if not flags:
        raise ValueError("Missing required --methods (e.g. --methods me_iis).")
    items: List[str] = []
    for raw in flags:
        for tok in str(raw).split(","):
            tok = tok.strip()
            if tok:
                items.append(tok)

    out: List[Method] = []
    seen: set[str] = set()
    alias = {"erm": "source_only", "source": "source_only", "meiis": "me_iis"}
    for m in items:
        key = alias.get(m.lower(), m.lower())
        if key in seen:
            continue
        if key not in {"source_only", "dann", "me_iis"}:
            raise ValueError(f"Unknown method '{m}'. Supported: source_only,dann,me_iis.")
        out.append(key)  # type: ignore[arg-type]
        seen.add(key)
    return out


def _run_dir(*, runs_root: Path, split: Camelyon17SplitConfig, method: Method, run_id: str) -> Path:
    return Path(runs_root) / "camelyon17" / str(split.split_mode) / str(method) / str(run_id)


def _flatten_meiis_cfg(cfg: MEIISConfig) -> Dict[str, Any]:
    return {
        "meiis_K": int(cfg.K),
        "meiis_tau": float(cfg.target_conf_thresh),
        "meiis_step": float(cfg.iis_step_size),
        "meiis_damp": float(cfg.iis_damping),
        "meiis_ema": float(cfg.ema_constraints),
        # Make run_id + fingerprint capture full config (not just selected fields).
        "meiis_cfg": asdict(cfg),
    }


def _base_runid_cfg(
    cfg: Camelyon17RunnerConfig, split: Camelyon17SplitConfig, *, batch_size: int | str
) -> Dict[str, Any]:
    return {
        "dataset": str(cfg.dataset),
        "split_mode": str(split.split_mode),
        "eval_split": str(split.eval_split),
        "adapt_split": str(split.adapt_split),
        "backbone": str(cfg.backbone),
        "seed": int(cfg.seed),
        "pretrained": bool(cfg.pretrained),
        "replace_batchnorm_with_instancenorm": bool(cfg.replace_batchnorm_with_instancenorm),
        "optimizer": str(cfg.optimizer),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "batch_size": (str(batch_size) if isinstance(batch_size, str) else int(batch_size)),
        "grad_accum_steps": int(cfg.grad_accum_steps),
        "dry_run_max_batches": int(cfg.dry_run_max_batches),
        "dry_run_max_samples": int(cfg.dry_run_max_samples),
        "disable_checkpointing": bool(cfg.disable_checkpointing),
    }


def _source_only_runid_cfg(
    cfg: Camelyon17RunnerConfig, split: Camelyon17SplitConfig, *, batch_size: int | str
) -> Dict[str, Any]:
    # Intentionally strip adaptation-only knobs + ME-IIS/DANN knobs so the source checkpoint is reusable.
    out = _base_runid_cfg(cfg, split, batch_size=batch_size)
    out.pop("eval_split", None)
    out.pop("adapt_split", None)
    # Source-only is used as an initialization dependency; keep it checkpointable even if
    # the caller disabled checkpointing for adaptation runs.
    out["disable_checkpointing"] = False
    out["algorithm"] = "ERM"
    out["epochs"] = int(cfg.epochs_source)
    return out


def _dann_runid_cfg(
    cfg: Camelyon17RunnerConfig, split: Camelyon17SplitConfig, *, batch_size: int | str
) -> Dict[str, Any]:
    out = _base_runid_cfg(cfg, split, batch_size=batch_size)
    out["algorithm"] = "DANN"
    out["epochs"] = int(cfg.epochs_adapt)
    out.update(
        {
            "dann_penalty_weight": float(cfg.dann_penalty_weight),
            "grl_lambda": float(cfg.grl_lambda),
            "dann_featurizer_lr_mult": float(cfg.dann_featurizer_lr_mult),
            "dann_discriminator_lr_mult": float(cfg.dann_discriminator_lr_mult),
        }
    )
    return out


def _me_iis_runid_cfg(
    cfg: Camelyon17RunnerConfig, split: Camelyon17SplitConfig, *, batch_size: int | str
) -> Dict[str, Any]:
    out = _base_runid_cfg(cfg, split, batch_size=batch_size)
    out["algorithm"] = "MEIIS"
    out["epochs"] = int(cfg.epochs_adapt)
    out.update(_flatten_meiis_cfg(cfg.meiis))
    return out


def _load_algorithm_weights(*, algorithm: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("algorithm", None)
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format (missing 'algorithm' state_dict): {checkpoint_path}")
    missing, unexpected = algorithm.load_state_dict(state, strict=False)
    if missing:
        print(f"[init][WARN] missing keys when loading init checkpoint: {missing}")
    if unexpected:
        print(f"[init][WARN] unexpected keys when loading init checkpoint: {unexpected}")


class Camelyon17Runner:
    def __init__(self, *, cfg: Camelyon17RunnerConfig, split: Camelyon17SplitConfig):
        if str(cfg.dataset) != "camelyon17":
            raise ValueError(f"Unsupported dataset '{cfg.dataset}' for Camelyon17Runner.")
        self.cfg = cfg
        self.split = split

    def _build_loaders(self, *, batch_size: int, include_indices_in_train: bool) -> Dict[str, Any]:
        train_tf, eval_tf = build_camelyon17_transforms(
            augment=bool(self.cfg.augment),
            color_jitter=bool(self.cfg.color_jitter),
            imagenet_normalize=bool(self.cfg.imagenet_normalize),
        )
        return build_camelyon17_loaders(
            {
                "data_root": None if self.cfg.data_root is None else str(self.cfg.data_root),
                "download": bool(self.cfg.download),
                "unlabeled": bool(self.cfg.unlabeled),
                "split_mode": str(self.split.split_mode),
                "eval_split": str(self.split.eval_split),
                "adapt_split": str(self.split.adapt_split),
                "train_transform": train_tf,
                "eval_transform": eval_tf,
                "batch_size": int(batch_size),
                "unlabeled_batch_size": int(batch_size),
                "include_indices_in_train": bool(include_indices_in_train),
                "num_workers": self.cfg.num_workers,
                "pin_memory": bool(self.cfg.pin_memory),
                "persistent_workers": bool(self.cfg.persistent_workers),
                "prefetch_factor": int(self.cfg.prefetch_factor),
            }
        )

    def _build_backbone(self):
        return build_backbone(
            str(self.cfg.backbone),
            pretrained=bool(self.cfg.pretrained),
            replace_batchnorm_with_instancenorm=bool(self.cfg.replace_batchnorm_with_instancenorm),
        )

    def _train_cfg(
        self,
        *,
        runid_cfg: Dict[str, Any],
        run_id: str,
        force_rerun: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        out = dict(runid_cfg)
        out.update(
            {
                "run_id": str(run_id),
                "device": str(device),
                "batch_size": runid_cfg["batch_size"],
                "epochs": int(runid_cfg["epochs"]),
                "grad_accum_steps": int(self.cfg.grad_accum_steps),
                "optimizer": str(self.cfg.optimizer),
                "lr": float(self.cfg.lr),
                "weight_decay": float(self.cfg.weight_decay),
                "scheduler": str(self.cfg.scheduler),
                "early_stop_patience": int(self.cfg.early_stop_patience),
                "amp": bool(self.cfg.amp),
                "deterministic": bool(self.cfg.deterministic),
                "seed": int(self.cfg.seed),
                "resume": not bool(force_rerun),
                "force_rerun": bool(force_rerun),
                "dry_run_max_batches": int(self.cfg.dry_run_max_batches),
                "dry_run_max_samples": int(self.cfg.dry_run_max_samples),
                "disable_checkpointing": bool(self.cfg.disable_checkpointing),
            }
        )
        if extra:
            out.update(extra)
        return out

    def _completed_checkpoint(self, run_dir: Path) -> Optional[Path]:
        best = Path(run_dir) / "best.pt"
        results = Path(run_dir) / "results.json"
        if best.exists() and results.exists():
            return best
        return None

    def _read_fingerprint(self, run_dir: Path) -> Optional[str]:
        for name in ("fingerprint.txt", "config_fingerprint.txt"):
            p = Path(run_dir) / name
            if p.exists():
                return p.read_text(encoding="utf-8").strip()
        return None

    def ensure_source_only(
        self, *, loaders: Dict[str, Any], batch_size: int | str, force_rerun: bool
    ) -> Tuple[Path, Dict[str, Any]]:
        # Always include indices so the same loaders can be reused for ME-IIS.
        runid_cfg = _source_only_runid_cfg(self.cfg, self.split, batch_size=batch_size)
        run_id = encode_config_to_run_id(runid_cfg)
        run_dir = _run_dir(runs_root=self.cfg.runs_root, split=self.split, method="source_only", run_id=run_id)

        # Keep checkpoints for source_only since it is a dependency for ME-IIS initialization.
        train_cfg = self._train_cfg(
            runid_cfg=runid_cfg,
            run_id=run_id,
            force_rerun=bool(force_rerun),
            extra={"disable_checkpointing": False},
        )
        expected_fp = fingerprint_config(train_cfg)

        ckpt = self._completed_checkpoint(run_dir)
        if ckpt is not None and not force_rerun:
            existing_fp = self._read_fingerprint(run_dir)
            if existing_fp is None:
                raise RuntimeError(f"Missing fingerprint file in existing run_dir: {run_dir}")
            if existing_fp != expected_fp:
                raise RuntimeError(
                    "Found an existing source_only run_dir but its fingerprint does not match the requested config. "
                    "Use --force_rerun (or delete the run_dir) to regenerate.\n"
                    f"run_dir={run_dir}\nexpected_fp={expected_fp}\nexisting_fp={existing_fp}"
                )
            return ckpt, {"status": "exists", "run_dir": str(run_dir), "run_id": run_id, "fingerprint": existing_fp}

        bb = self._build_backbone()
        algorithm = ERM(featurizer=bb.model, feature_dim=bb.feature_dim, num_classes=2)
        algorithm.set_backbone_info(
            backbone_name=str(bb.name),
            pretrained=bool(self.cfg.pretrained),
            feature_dim=int(bb.feature_dim),
            feature_layer="final_pool",
        )

        run_dir.mkdir(parents=True, exist_ok=True)
        with tee_std_streams(run_dir / "stdout.log", run_dir / "stderr.log", mode="a"):
            print("[run] method=source_only run_id=" + str(run_id))
            res = train(
                cfg=train_cfg,
                run_dir=run_dir,
                algorithm=algorithm,
                wilds_dataset=loaders["dataset"],
                train_loader=loaders["train_loader"],
                unlabeled_loader=None,
                val_loader=loaders["val_loader"],
                test_loader=loaders["test_loader"],
                id_val_loader=loaders.get("id_val_loader"),
            )

        ckpt = self._completed_checkpoint(run_dir)
        if ckpt is None:
            raise RuntimeError(f"source_only did not produce a completed checkpoint at: {run_dir}")
        return ckpt, {"status": res.get("status"), "run_dir": str(run_dir), "run_id": run_id, "fingerprint": expected_fp}

    def run(
        self,
        *,
        methods: Sequence[str],
        batch_size: int | str,
        force_rerun: bool,
    ) -> Dict[str, Any]:
        resolved_methods = _parse_methods(methods)

        # Build loaders once and reuse across methods. Use indices so ME-IIS can weight per-sample losses.
        batch_size_cfg = batch_size
        loader_batch_size = (
            16
            if isinstance(batch_size_cfg, str) and batch_size_cfg.strip().lower() == "auto"
            else int(batch_size_cfg)
        )
        loaders = self._build_loaders(batch_size=int(loader_batch_size), include_indices_in_train=True)

        results: Dict[str, Any] = {}

        source_ckpt: Optional[Path] = None
        source_meta: Optional[Dict[str, Any]] = None

        for method in resolved_methods:
            if method == "source_only":
                source_ckpt, source_meta = self.ensure_source_only(
                    loaders=loaders, batch_size=batch_size_cfg, force_rerun=bool(force_rerun)
                )
                results[method] = dict(source_meta)
                continue

            if method == "me_iis":
                if source_ckpt is None:
                    # Dependency (do not force rerun unless the user explicitly asked for source_only).
                    source_ckpt, source_meta = self.ensure_source_only(
                        loaders=loaders, batch_size=batch_size_cfg, force_rerun=False
                    )
                assert source_meta is not None

                runid_cfg = _me_iis_runid_cfg(self.cfg, self.split, batch_size=batch_size_cfg)
                run_id = encode_config_to_run_id(runid_cfg)
                run_dir = _run_dir(runs_root=self.cfg.runs_root, split=self.split, method=method, run_id=run_id)

                bb = self._build_backbone()
                algorithm = MEIIS(
                    featurizer=bb.model,
                    feature_dim=bb.feature_dim,
                    num_classes=2,
                    seed=int(self.cfg.seed),
                    config=self.cfg.meiis,
                )
                algorithm.set_backbone_info(
                    backbone_name=str(bb.name),
                    pretrained=bool(self.cfg.pretrained),
                    feature_dim=int(bb.feature_dim),
                    feature_layer="final_pool",
                )

                extra = {
                    "meiis": asdict(self.cfg.meiis),
                    "source_run_id": str(source_meta.get("run_id")),
                    "source_fingerprint": str(source_meta.get("fingerprint")),
                }
                train_cfg = self._train_cfg(
                    runid_cfg=runid_cfg,
                    run_id=run_id,
                    force_rerun=bool(force_rerun),
                    extra=extra,
                )
                if not force_rerun and (run_dir / "last.pt").exists():
                    init_from_source = False
                else:
                    init_from_source = True

                run_dir.mkdir(parents=True, exist_ok=True)
                with tee_std_streams(run_dir / "stdout.log", run_dir / "stderr.log", mode="a"):
                    print("[run] method=me_iis run_id=" + str(run_id))
                    if init_from_source:
                        print(f"[dependency] initializing ME-IIS from source checkpoint: {source_ckpt}")
                        _load_algorithm_weights(algorithm=algorithm, checkpoint_path=source_ckpt)

                    res = train(
                        cfg=train_cfg,
                        run_dir=run_dir,
                        algorithm=algorithm,
                        wilds_dataset=loaders["dataset"],
                        train_loader=loaders["train_loader"],
                        unlabeled_loader=loaders.get("unlabeled_loader"),
                        val_loader=loaders["val_loader"],
                        test_loader=loaders["test_loader"],
                        id_val_loader=loaders.get("id_val_loader"),
                    )

                results[method] = {
                    "status": res.get("status"),
                    "run_dir": str(run_dir),
                    "run_id": run_id,
                    "source_checkpoint": str(source_ckpt),
                }
                continue

            if method == "dann":
                runid_cfg = _dann_runid_cfg(self.cfg, self.split, batch_size=batch_size_cfg)
                run_id = encode_config_to_run_id(runid_cfg)
                run_dir = _run_dir(runs_root=self.cfg.runs_root, split=self.split, method=method, run_id=run_id)

                bb = self._build_backbone()
                algorithm = DANN(
                    featurizer=bb.model,
                    feature_dim=bb.feature_dim,
                    num_classes=2,
                    dann_penalty_weight=float(self.cfg.dann_penalty_weight),
                    grl_lambda=float(self.cfg.grl_lambda),
                )
                algorithm.set_backbone_info(
                    backbone_name=str(bb.name),
                    pretrained=bool(self.cfg.pretrained),
                    feature_dim=int(bb.feature_dim),
                    feature_layer="final_pool",
                )

                train_cfg = self._train_cfg(runid_cfg=runid_cfg, run_id=run_id, force_rerun=bool(force_rerun))
                train_cfg["lr_overrides"] = {
                    "featurizer": float(self.cfg.lr) * float(self.cfg.dann_featurizer_lr_mult),
                    "discriminator": float(self.cfg.lr) * float(self.cfg.dann_discriminator_lr_mult),
                }

                run_dir.mkdir(parents=True, exist_ok=True)
                with tee_std_streams(run_dir / "stdout.log", run_dir / "stderr.log", mode="a"):
                    print("[run] method=dann run_id=" + str(run_id))
                    res = train(
                        cfg=train_cfg,
                        run_dir=run_dir,
                        algorithm=algorithm,
                        wilds_dataset=loaders["dataset"],
                        train_loader=loaders["train_loader"],
                        unlabeled_loader=loaders.get("unlabeled_loader"),
                        val_loader=loaders["val_loader"],
                        test_loader=loaders["test_loader"],
                        id_val_loader=loaders.get("id_val_loader"),
                    )

                results[method] = {"status": res.get("status"), "run_dir": str(run_dir), "run_id": run_id}
                continue

            raise AssertionError(f"Unhandled method: {method}")

        # Write a small top-level record for convenience.
        out_dir = Path(self.cfg.runs_root) / "camelyon17" / str(self.split.split_mode)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "last_run.json"
        summary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return results
