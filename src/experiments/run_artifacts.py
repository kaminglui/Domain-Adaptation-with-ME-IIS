from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunArtifacts:
    """
    Canonical filesystem layout for a single run directory.

    run_dir/
      config.json
      state.json
      logs/
        stdout.txt
        stderr.txt
      checkpoints/
        ...
      metrics.csv
    """

    run_dir: Path
    run_id: str
    stage: str  # e.g. "source" | "adapt"
    method: str  # e.g. "source_only" | "me_iis" | "dann" | "coral" | "pseudo_label"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def stdout_path(self) -> Path:
        return self.logs_dir / "stdout.txt"

    @property
    def stderr_path(self) -> Path:
        return self.logs_dir / "stderr.txt"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.csv"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoints_dir / f"{self.stage}_{self.method}_{self.run_id}_last.pth"

    @property
    def final_checkpoint_path(self) -> Path:
        return self.checkpoints_dir / f"{self.stage}_{self.method}_{self.run_id}_final.pth"

    def epoch_checkpoint_path(self, epoch_1based: int) -> Path:
        return self.checkpoints_dir / f"{self.stage}_{self.method}_{self.run_id}_epoch{int(epoch_1based)}.pth"

