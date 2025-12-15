from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


class _Tee(TextIO):
    def __init__(self, primary: TextIO, secondary: TextIO):
        self._primary = primary
        self._secondary = secondary

    def write(self, s: str) -> int:  # type: ignore[override]
        n1 = self._primary.write(s)
        self._secondary.write(s)
        return n1

    def flush(self) -> None:  # type: ignore[override]
        self._primary.flush()
        self._secondary.flush()


@contextmanager
def tee_std_streams(stdout_path: Path, stderr_path: Path, mode: str = "a") -> Iterator[None]:
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    prev_out = sys.stdout
    prev_err = sys.stderr
    with stdout_path.open(mode, encoding="utf-8") as out_f, stderr_path.open(mode, encoding="utf-8") as err_f:
        sys.stdout = _Tee(prev_out, out_f)  # type: ignore[assignment]
        sys.stderr = _Tee(prev_err, err_f)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = prev_out
            sys.stderr = prev_err


@contextmanager
def redirect_std_streams(stdout_path: Path, stderr_path: Path, mode: str = "a") -> Iterator[None]:
    """
    Redirect stdout/stderr to files (no console output) while running a block.
    Useful for notebooks where we want minimal clutter but persistent logs.
    """
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    prev_out = sys.stdout
    prev_err = sys.stderr
    with stdout_path.open(mode, encoding="utf-8") as out_f, stderr_path.open(mode, encoding="utf-8") as err_f:
        sys.stdout = out_f  # type: ignore[assignment]
        sys.stderr = err_f  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = prev_out
            sys.stderr = prev_err
