from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.experiments.run_config import RunConfig, get_run_dir


def _read_single_row_csv(path: Path) -> Optional[Dict[str, str]]:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return dict(row)
    return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tail_text(path: Path, max_chars: int = 800) -> str:
    try:
        if not path.exists():
            return ""
        txt = path.read_text(encoding="utf-8", errors="replace")
        if len(txt) <= max_chars:
            return txt
        return txt[-max_chars:]
    except Exception:
        return ""


def collect_expected_runs(configs: Iterable[RunConfig], runs_root: Optional[Path]) -> List[Dict[str, Any]]:
    """
    Collect a deterministic summary table for a list of expected configs using run_id-based paths.

    Status values:
    - "OK": metrics.csv exists and is readable
    - "FAILED": stderr.txt has content, but metrics.csv missing
    - "INCOMPLETE": run_dir exists or state.json exists, but no metrics.csv and no stderr content
    - "NOT RUN": run_dir does not exist
    - "MISSING_METRICS": state.json says completed but metrics.csv missing
    """
    rows: List[Dict[str, Any]] = []
    for cfg in configs:
        run_dir = get_run_dir(cfg, runs_root=runs_root)
        metrics_path = run_dir / "metrics.csv"
        state_path = run_dir / "state.json"
        stderr_path = run_dir / "logs" / "stderr.txt"

        metrics_row = _read_single_row_csv(metrics_path)
        state = _read_json(state_path)
        stderr_tail = _tail_text(stderr_path)

        status = "NOT RUN"
        if metrics_row is not None:
            status = "OK"
        elif state is not None:
            if bool(state.get("completed", False)):
                status = "MISSING_METRICS"
            else:
                status = "INCOMPLETE"
        elif run_dir.exists():
            status = "INCOMPLETE"

        if status != "OK" and stderr_tail.strip():
            status = "FAILED"

        row: Dict[str, Any] = {
            "dataset": cfg.dataset_name,
            "src": cfg.source_domain,
            "tgt": cfg.target_domain,
            "method": cfg.method,
            "seed": cfg.seed,
            "run_id": cfg.run_id,
            "status": status,
            "run_dir": str(run_dir),
            "metrics_csv": str(metrics_path) if metrics_path.exists() else "",
            "stderr_tail": stderr_tail,
        }
        if metrics_row is not None:
            row.update(metrics_row)
        rows.append(row)
    return rows

