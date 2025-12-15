from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.experiments.run_artifacts import RunArtifacts


@dataclass(frozen=True)
class RunState:
    completed: bool
    stage: str
    method: str
    run_id: str
    last_completed_epoch: int
    total_epochs: int
    final_checkpoint: str
    last_checkpoint: str


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_state(path: Path) -> Optional[RunState]:
    payload = _load_json(path)
    if not payload:
        return None
    try:
        return RunState(
            completed=bool(payload.get("completed", False)),
            stage=str(payload.get("stage", "")),
            method=str(payload.get("method", "")),
            run_id=str(payload.get("run_id", "")),
            last_completed_epoch=int(payload.get("last_completed_epoch", -1)),
            total_epochs=int(payload.get("total_epochs", 0)),
            final_checkpoint=str(payload.get("final_checkpoint", "")),
            last_checkpoint=str(payload.get("last_checkpoint", "")),
        )
    except Exception:
        return None


def save_state(path: Path, state: RunState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "completed": state.completed,
                "stage": state.stage,
                "method": state.method,
                "run_id": state.run_id,
                "last_completed_epoch": state.last_completed_epoch,
                "total_epochs": state.total_epochs,
                "final_checkpoint": state.final_checkpoint,
                "last_checkpoint": state.last_checkpoint,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def ensure_run_dirs(artifacts: RunArtifacts) -> None:
    artifacts.run_dir.mkdir(parents=True, exist_ok=True)
    artifacts.logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts.checkpoints_dir.mkdir(parents=True, exist_ok=True)


def save_checkpoint(path: Path, checkpoint: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, map_location: Optional[torch.device] = None) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def find_existing_completed_checkpoint(artifacts: RunArtifacts) -> Optional[Path]:
    state = load_state(artifacts.state_path)
    if state and state.completed:
        candidate = artifacts.run_dir / state.final_checkpoint
        if candidate.exists():
            return candidate
    if artifacts.final_checkpoint_path.exists():
        return artifacts.final_checkpoint_path
    return None


def find_resume_checkpoint(artifacts: RunArtifacts) -> Optional[Path]:
    state = load_state(artifacts.state_path)
    if state and not state.completed:
        candidate = artifacts.run_dir / state.last_checkpoint
        if candidate.exists():
            return candidate
    if artifacts.last_checkpoint_path.exists():
        return artifacts.last_checkpoint_path
    return None

