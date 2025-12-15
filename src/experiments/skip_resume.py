from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.experiments.checkpointing import find_existing_completed_checkpoint, find_resume_checkpoint
from src.experiments.run_artifacts import RunArtifacts
from src.experiments.run_config import RunConfig, get_run_dir


def _default_stage_for_method(method: str) -> str:
    return "source" if method == "source_only" else "adapt"


def build_artifacts(config: RunConfig, stage: Optional[str] = None) -> RunArtifacts:
    run_dir = get_run_dir(config)
    stage_value = stage or _default_stage_for_method(config.method)
    return RunArtifacts(run_dir=run_dir, run_id=config.run_id, stage=stage_value, method=config.method)


def find_existing_checkpoint(config: RunConfig, stage: Optional[str] = None) -> Optional[Path]:
    """
    Return a completed checkpoint for the given config, if present.
    This is the canonical "skip if trained" hook.
    """
    return find_existing_completed_checkpoint(build_artifacts(config, stage=stage))


def find_partial_checkpoint(config: RunConfig, stage: Optional[str] = None) -> Optional[Path]:
    """Return a resumable (incomplete) checkpoint for the given config, if present."""
    return find_resume_checkpoint(build_artifacts(config, stage=stage))
