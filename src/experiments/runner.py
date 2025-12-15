from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from src.experiments.eval_harness import evaluate_source_and_target
from src.experiments.run_artifacts import RunArtifacts
from src.experiments.checkpointing import ensure_run_dirs
from src.experiments.metrics import build_metrics_row, get_git_sha, write_metrics_csv
from src.experiments.methods import coral, dann, me_iis, pseudo_label, source_only
from src.experiments.run_config import RunConfig, get_run_dir
from src.experiments.stream_capture import redirect_std_streams


def derive_source_only_config(config: RunConfig) -> RunConfig:
    """
    Build the canonical source-only config used as the shared dependency for all adaptation methods.

    Important: this intentionally strips adaptation-only knobs so the source checkpoint is reused
    across methods (fair + no redundant retraining).
    """
    return replace(
        config,
        method="source_only",
        epochs_adapt=0,
        method_params={},
        finetune_backbone=False,
        backbone_lr_scale=0.1,
        classifier_lr=1e-2,
        feature_layers=("layer3", "layer4"),
    )


def _stage_for_method(method: str) -> str:
    return "source" if method == "source_only" else "adapt"


def _run_with_logs(config: RunConfig, fn, *args, runs_root: Optional[Path], **kwargs):
    run_dir = get_run_dir(config, runs_root=runs_root)
    artifacts = RunArtifacts(
        run_dir=run_dir,
        run_id=config.run_id,
        stage=_stage_for_method(config.method),
        method=config.method,
    )
    ensure_run_dirs(artifacts)
    with redirect_std_streams(artifacts.stdout_path, artifacts.stderr_path, mode="a"):
        return fn(*args, **kwargs)


def run_one(
    config: RunConfig,
    force_rerun: bool = False,
    runs_root: Optional[Path] = None,
    write_metrics: bool = True,
    raise_on_error: bool = True,
) -> Dict[str, Any]:
    """
    Run a single method for a single seed, with:
    - deterministic run_dir + checkpoint naming (run_id),
    - skip/resume behavior,
    - unified evaluation + metrics.csv writing.
    """
    try:
        if config.method == "source_only":
            run_res = _run_with_logs(
                config,
                source_only.run,
                config,
                force_rerun=force_rerun,
                runs_root=runs_root,
            )
            ckpt_path = Path(run_res["checkpoint"])
        else:
            src_cfg = derive_source_only_config(config)
            src_res = _run_with_logs(
                src_cfg,
                source_only.run,
                src_cfg,
                force_rerun=False,
                runs_root=runs_root,
            )
            src_ckpt_path = Path(src_res["checkpoint"])

            if config.method in {"me_iis", "me_iis_pl"}:
                run_res = _run_with_logs(
                    config,
                    me_iis.run,
                    config,
                    source_checkpoint=src_ckpt_path,
                    force_rerun=force_rerun,
                    runs_root=runs_root,
                )
            elif config.method == "dann":
                run_res = _run_with_logs(
                    config,
                    dann.run,
                    config,
                    source_checkpoint=src_ckpt_path,
                    force_rerun=force_rerun,
                    runs_root=runs_root,
                )
            elif config.method == "coral":
                run_res = _run_with_logs(
                    config,
                    coral.run,
                    config,
                    source_checkpoint=src_ckpt_path,
                    force_rerun=force_rerun,
                    runs_root=runs_root,
                )
            elif config.method == "pseudo_label":
                run_res = _run_with_logs(
                    config,
                    pseudo_label.run,
                    config,
                    source_checkpoint=src_ckpt_path,
                    force_rerun=force_rerun,
                    runs_root=runs_root,
                )
            else:
                raise ValueError(f"Unknown method '{config.method}'")
            ckpt_path = Path(run_res["checkpoint"])
    except Exception as exc:
        run_dir = get_run_dir(config, runs_root=runs_root)
        artifacts = RunArtifacts(
            run_dir=run_dir,
            run_id=config.run_id,
            stage=_stage_for_method(config.method),
            method=config.method,
        )
        ensure_run_dirs(artifacts)
        import traceback

        artifacts.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        with artifacts.stderr_path.open("a", encoding="utf-8") as f:
            f.write("\n" + traceback.format_exc() + "\n")
        if raise_on_error:
            raise
        return {
            "status": "failed",
            "run_dir": str(run_dir),
            "checkpoint": None,
            "metrics_csv": None,
            "error": f"{type(exc).__name__}: {exc}",
            "stdout_path": str(artifacts.stdout_path),
            "stderr_path": str(artifacts.stderr_path),
        }

    if not write_metrics:
        return run_res

    if config.data_root is None:
        raise ValueError("data_root must be set to evaluate.")
    data_root = Path(config.data_root)
    src_acc, tgt_acc = evaluate_source_and_target(
        checkpoint_path=ckpt_path,
        dataset_name=config.dataset_name,
        data_root=data_root,
        source_domain=config.source_domain,
        target_domain=config.target_domain,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        deterministic=config.deterministic,
    )

    run_dir = get_run_dir(config, runs_root=runs_root)
    metrics_row = build_metrics_row(
        config=config,
        source_acc=src_acc,
        target_acc=tgt_acc,
        git_sha=get_git_sha(),
    )
    write_metrics_csv(run_dir / "metrics.csv", metrics_row)
    run_res = dict(run_res)
    run_res.update({"source_acc_eval": src_acc, "target_acc_eval": tgt_acc, "metrics_csv": str(run_dir / "metrics.csv")})
    return run_res
