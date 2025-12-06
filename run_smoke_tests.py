import argparse
import compileall
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import scripts.adapt_me_iis as adapt_me_iis
import scripts.train_source as train_source
from utils.test_utils import build_tiny_model, create_office_home_like, temporary_workdir


def _run_step(name: str, func) -> bool:
    print(f"[SMOKE] {name} ...", flush=True)
    try:
        func()
        print(f"[SMOKE] {name} PASS", flush=True)
        return True
    except Exception as exc:  # pragma: no cover - best-effort smoke script
        print(f"[SMOKE][FAIL] {name}: {exc}", flush=True)
        return False


def _run_compile() -> None:
    success = compileall.compile_dir(".", quiet=1)
    if not success:
        raise RuntimeError("compileall returned failure status.")


def _run_sanity_script() -> None:
    subprocess.run([sys.executable, "scripts/test_me_iis_sanity.py"], check=True)


def _source_args(data_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_name="office_home",
        data_root=str(data_root),
        source_domain="Ar",
        target_domain="Cl",
        num_epochs=1,
        resume_from=None,
        save_every=1,
        batch_size=2,
        lr_backbone=1e-3,
        lr_classifier=1e-2,
        weight_decay=1e-3,
        num_workers=0,
        deterministic=True,
        seed=0,
        dry_run_max_batches=2,
        dry_run_max_samples=6,
    )


def _adapt_args(data_root: Path, checkpoint: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_name="office_home",
        data_root=str(data_root),
        source_domain="Ar",
        target_domain="Cl",
        checkpoint=str(checkpoint),
        batch_size=2,
        num_workers=0,
        num_latent_styles=2,
        components_per_layer=None,
        feature_layers="layer3,layer4",
        source_prob_mode="onehot",
        iis_iters=3,
        iis_tol=0.1,
        adapt_epochs=1,
        resume_adapt_from=None,
        save_adapt_every=1,
        finetune_backbone=False,
        backbone_lr_scale=0.1,
        classifier_lr=1e-2,
        weight_decay=1e-3,
        dry_run_max_samples=6,
        dry_run_max_batches=2,
        deterministic=True,
        seed=0,
    )


def _run_dry_runs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        data_root = workdir / "data"
        create_office_home_like(data_root, domains=("Ar", "Cl"), images_per_class=4, image_size=256)

        source_args = _source_args(data_root)
        with temporary_workdir(workdir):
            with mock.patch("scripts.train_source.build_model", build_tiny_model):
                train_source.train_source(source_args)
        ckpt = workdir / "checkpoints" / "source_only_Ar_to_Cl_seed0.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Source-only checkpoint missing: {ckpt}")

        adapt_args = _adapt_args(data_root, checkpoint=ckpt)
        with temporary_workdir(workdir):
            with mock.patch("scripts.adapt_me_iis.build_model", build_tiny_model):
                adapt_me_iis.adapt_me_iis(adapt_args)
        adapt_ckpt = workdir / "checkpoints" / "me_iis_Ar_to_Cl_layer3-layer4_seed0.pth"
        if not adapt_ckpt.exists():
            raise FileNotFoundError(f"Adaptation checkpoint missing: {adapt_ckpt}")


def main() -> None:
    steps = [
        ("Compile modules", _run_compile),
        ("IIS sanity test", _run_sanity_script),
        ("Dry-run train/adapt", _run_dry_runs),
    ]
    ok = all(_run_step(name, fn) for name, fn in steps)
    if not ok:
        sys.exit(1)
    print("[SMOKE] All smoke tests passed.", flush=True)


if __name__ == "__main__":
    main()
