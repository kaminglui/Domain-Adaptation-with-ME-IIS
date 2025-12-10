"""
Lightweight smoke-test harness.

Runs three quick checks in order (compileall, IIS sanity script, dry-run train+adapt)
to catch obvious regressions. Defaults mirror the existing scripts; extra flags make
it easy to skip steps or inspect artifacts without changing the core behavior.
"""

import argparse
import compileall
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional
from unittest import mock

import scripts.adapt_me_iis as adapt_me_iis
import scripts.train_source as train_source
from utils.test_utils import build_tiny_model, create_office_home_like, temporary_workdir


@dataclass
class SmokeConfig:
    dataset_name: str = "office_home"
    source_domain: str = "Ar"
    target_domain: str = "Cl"
    num_epochs: int = 1
    adapt_epochs: int = 1
    batch_size: int = 2
    lr_backbone: float = 1e-3
    lr_classifier: float = 1e-2
    weight_decay: float = 1e-3
    num_workers: int = 0
    seed: int = 0
    dry_run_max_batches: int = 2
    dry_run_max_samples: int = 6
    images_per_class: int = 4
    image_size: int = 256
    feature_layers: str = "layer3,layer4"
    gmm_selection_mode: str = "fixed"
    gmm_bic_min_components: int = 2
    gmm_bic_max_components: int = 8
    num_latent_styles: int = 2
    source_prob_mode: str = "onehot"
    iis_iters: int = 3
    iis_tol: float = 0.1
    finetune_backbone: bool = False
    backbone_lr_scale: float = 0.1
    save_every: int = 1
    save_adapt_every: int = 1
    use_pseudo_labels: bool = False
    pseudo_conf_thresh: float = 0.9
    pseudo_max_ratio: float = 1.0
    pseudo_loss_weight: float = 1.0

    def source_args(self, data_root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            dataset_name=self.dataset_name,
            data_root=str(data_root),
            source_domain=self.source_domain,
            target_domain=self.target_domain,
            num_epochs=self.num_epochs,
            resume_from=None,
            save_every=self.save_every,
            batch_size=self.batch_size,
            lr_backbone=self.lr_backbone,
            lr_classifier=self.lr_classifier,
            weight_decay=self.weight_decay,
            num_workers=self.num_workers,
            deterministic=True,
            seed=self.seed,
            dry_run_max_batches=self.dry_run_max_batches,
            dry_run_max_samples=self.dry_run_max_samples,
        )

    def adapt_args(self, data_root: Path, checkpoint: Path) -> argparse.Namespace:
        return argparse.Namespace(
            dataset_name=self.dataset_name,
            data_root=str(data_root),
            source_domain=self.source_domain,
            target_domain=self.target_domain,
            checkpoint=str(checkpoint),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            num_latent_styles=self.num_latent_styles,
            components_per_layer=None,
            gmm_selection_mode=self.gmm_selection_mode,
            gmm_bic_min_components=self.gmm_bic_min_components,
            gmm_bic_max_components=self.gmm_bic_max_components,
            feature_layers=self.feature_layers,
            source_prob_mode=self.source_prob_mode,
            iis_iters=self.iis_iters,
            iis_tol=self.iis_tol,
            adapt_epochs=self.adapt_epochs,
            resume_adapt_from=None,
            save_adapt_every=self.save_adapt_every,
            finetune_backbone=self.finetune_backbone,
            backbone_lr_scale=self.backbone_lr_scale,
            classifier_lr=self.lr_classifier,
            weight_decay=self.weight_decay,
            dry_run_max_samples=self.dry_run_max_samples,
            dry_run_max_batches=self.dry_run_max_batches,
            deterministic=True,
            seed=self.seed,
            use_pseudo_labels=self.use_pseudo_labels,
            pseudo_conf_thresh=self.pseudo_conf_thresh,
            pseudo_max_ratio=self.pseudo_max_ratio,
            pseudo_loss_weight=self.pseudo_loss_weight,
        )


@dataclass
class SmokeStep:
    name: str
    fn: Callable[[SmokeConfig], None]


def _run_step(step: SmokeStep, cfg: SmokeConfig) -> bool:
    print(f"[SMOKE] {step.name} ...", flush=True)
    try:
        step.fn(cfg)
        print(f"[SMOKE] {step.name} PASS", flush=True)
        return True
    except Exception as exc:  # pragma: no cover - best-effort smoke script
        print(f"[SMOKE][FAIL] {step.name}: {exc}", flush=True)
        return False


def _run_compile(_: SmokeConfig) -> None:
    success = compileall.compile_dir(".", quiet=1)
    if not success:
        raise RuntimeError("compileall returned failure status.")


def _run_sanity_script(_: SmokeConfig) -> None:
    subprocess.run([sys.executable, "scripts/test_me_iis_sanity.py"], check=True)


def _create_data_tree(cfg: SmokeConfig, root: Path) -> Path:
    return create_office_home_like(
        root,
        domains=(cfg.source_domain, cfg.target_domain),
        images_per_class=cfg.images_per_class,
        image_size=cfg.image_size,
    )


def _run_dry_runs(cfg: SmokeConfig, workdir: Path) -> None:
    data_root = workdir / "data"
    _create_data_tree(cfg, data_root)

    source_args = cfg.source_args(data_root)
    with temporary_workdir(workdir):
        with mock.patch("scripts.train_source.build_model", build_tiny_model):
            train_source.train_source(source_args)
    ckpt = workdir / "checkpoints" / f"source_only_{cfg.source_domain}_to_{cfg.target_domain}_seed{cfg.seed}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Source-only checkpoint missing: {ckpt}")

    adapt_args = cfg.adapt_args(data_root, checkpoint=ckpt)
    with temporary_workdir(workdir):
        with mock.patch("scripts.adapt_me_iis.build_model", build_tiny_model):
            adapt_me_iis.adapt_me_iis(adapt_args)
    layer_tag = "-".join([layer.strip() for layer in cfg.feature_layers.split(",") if layer.strip()])
    adapt_ckpt = workdir / "checkpoints" / f"me_iis_{cfg.source_domain}_to_{cfg.target_domain}_{layer_tag}_seed{cfg.seed}.pth"
    if not adapt_ckpt.exists():
        raise FileNotFoundError(f"Adaptation checkpoint missing: {adapt_ckpt}")


def _run_dry_step(cfg: SmokeConfig, keep_workdir: bool, workdir_override: Optional[Path]) -> None:
    if workdir_override:
        workdir_override.mkdir(parents=True, exist_ok=True)
        _run_dry_runs(cfg, workdir_override)
        print(f"[SMOKE] Workdir preserved at {workdir_override}", flush=True)
        return

    tmpdir = tempfile.mkdtemp()
    workdir = Path(tmpdir)
    try:
        _run_dry_runs(cfg, workdir)
        if keep_workdir:
            print(f"[SMOKE] Workdir preserved at {workdir}", flush=True)
    finally:
        if not keep_workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ME-IIS smoke tests.")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compileall check.")
    parser.add_argument("--skip-sanity", action="store_true", help="Skip scripts/test_me_iis_sanity.py.")
    parser.add_argument("--skip-dry-run", action="store_true", help="Skip dry-run train/adapt stage.")
    parser.add_argument("--keep-workdir", action="store_true", help="Keep the temporary dry-run directory for inspection.")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Optional existing directory for dry-run artifacts (skips temp creation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SmokeConfig()

    steps: List[SmokeStep] = []
    if not args.skip_compile:
        steps.append(SmokeStep("Compile modules", _run_compile))
    if not args.skip_sanity:
        steps.append(SmokeStep("IIS sanity test", _run_sanity_script))
    if not args.skip_dry_run:
        steps.append(
            SmokeStep(
                "Dry-run train/adapt",
                lambda c: _run_dry_step(c, keep_workdir=args.keep_workdir, workdir_override=args.workdir),
            )
        )

    if not steps:
        print("[SMOKE] No steps selected; exiting.", flush=True)
        return

    ok = all(_run_step(step, cfg) for step in steps)
    if not ok:
        sys.exit(1)
    print("[SMOKE] All smoke tests passed.", flush=True)


if __name__ == "__main__":
    main()
