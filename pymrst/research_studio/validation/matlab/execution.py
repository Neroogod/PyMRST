"""Safe, shell-free MATLAB batch execution for qualification generators."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time

from .schemas import MatlabExecutionResult, MatlabFixtureSpec


DEFAULT_MACOS_MATLAB = Path("/Applications/MATLAB_R2021b.app/bin/matlab")


def discover_matlab(explicit: Path | str | None = None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    configured = os.environ.get("PYMRST_MATLAB_EXECUTABLE")
    if configured:
        candidates.append(Path(configured))
    located = shutil.which("matlab")
    if located:
        candidates.append(Path(located))
    candidates.append(DEFAULT_MACOS_MATLAB)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(
        "MATLAB executable not found; set PYMRST_MATLAB_EXECUTABLE explicitly")


def _matlab_quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def generation_command(spec: MatlabFixtureSpec, executable: Path) -> tuple[str, ...]:
    folder = _matlab_quote(spec.generator_path.parent)
    expression = (
        f"fprintf('MATLAB_RELEASE=%s\\n', version('-release')); "
        f"addpath('{folder}'); {spec.generator_function}()"
    )
    return (str(executable), "-batch", expression)


def run_generator(spec: MatlabFixtureSpec, *, executable: Path | str | None = None,
                  timeout_seconds: float = 1800.0) -> MatlabExecutionResult:
    matlab = discover_matlab(executable)
    command = generation_command(spec, matlab)
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=spec.generator_path.parent, capture_output=True, text=True,
        timeout=float(timeout_seconds), check=False,
    )
    return MatlabExecutionResult(
        command=command, returncode=completed.returncode,
        stdout=completed.stdout, stderr=completed.stderr,
        elapsed_seconds=time.perf_counter() - started,
    )


__all__ = ["DEFAULT_MACOS_MATLAB", "discover_matlab", "generation_command",
           "run_generator"]
