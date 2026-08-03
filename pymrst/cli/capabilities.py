"""What this installation can actually do, without importing PyMRST.

`pymrst doctor` has to work when something is broken, which is the only time
anyone runs it. So nothing here imports a solver, a model or a grid: it probes
for the optional dependency behind each capability and reports it. A doctor
that raises during a diagnosis is worse than no doctor.

Capabilities are named after `pyproject.toml`'s extras so the remedy is
obvious - a missing one tells you the exact `pip install` that fixes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
import os
import shutil
import sys


@dataclass(frozen=True)
class Capability:
    """One thing the installation either can or cannot do."""

    name: str
    modules: tuple[str, ...]
    extra: str | None
    detail: str

    @property
    def available(self) -> bool:
        return all(_installed(module) for module in self.modules)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(m for m in self.modules if not _installed(m))

    @property
    def remedy(self) -> str:
        if self.available or self.extra is None:
            return ""
        return f"pip install 'pymrst[{self.extra}]'"


def _installed(module: str) -> bool:
    """Is `module` importable, without importing it.

    `find_spec` avoids paying for a heavy optional dependency - torch in
    particular - just to report that it exists. It also avoids a broken
    optional package taking the whole report down with it.
    """
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# Ordered so the required things come first and a reader stops at the first
# real problem rather than scanning a list of optional extras.
CAPABILITIES: tuple[Capability, ...] = (
    Capability("Core (numpy, scipy)", ("numpy", "scipy"), None,
               "required - PyMRST cannot run without these"),
    Capability("Algebraic multigrid", ("pyamg",), "amg",
               "CPR's elliptic stage; without it that stage falls back to an "
               "incomplete factorisation"),
    Capability("3D plotting", ("pyvista",), "plotting",
               "grid and result rendering"),
    Capability("Interactive inspector", ("pyvista", "pyvistaqt", "PyQt5"), "gui",
               "plot_toolbar; degrades to a plain blocking window without it"),
    Capability("Browser plots", ("plotly",), "browser", "notebook figures"),
    Capability("Result store", ("h5py",), "studio",
               "HDF5 ResultStore, checkpoints and restart"),
    Capability("MATLAB v7.3 files", ("h5py",), "mat73",
               "reading -v7.3 .mat files; SciPy covers earlier versions"),
    Capability("Graph partitioner", ("pymetis",), "metis",
               "coarsegrid.partition_metis"),
    Capability("Machine learning", ("torch",), "ml", "surrogate components"),
    Capability("Web app", ("streamlit", "pandas", "plotly"), "webapp",
               "the compositional flash predictor front end"),
)


def cuda_status() -> str:
    """Whether a GPU is visible, reported without importing torch eagerly."""
    if not _installed("torch"):
        return "not checked (torch not installed)"
    try:
        import torch
    except Exception as error:                          # pragma: no cover
        return f"torch present but failed to import ({type(error).__name__})"
    if not torch.cuda.is_available():
        return "no CUDA device visible"
    try:
        return f"{torch.cuda.device_count()} device(s), {torch.cuda.get_device_name(0)}"
    except Exception:                                   # pragma: no cover
        return f"{torch.cuda.device_count()} device(s)"


def writable_paths() -> list[tuple[str, Path, bool, str]]:
    """Directories PyMRST writes to, and whether it actually can.

    Reported because the failure they cause appears much later and looks
    unrelated - a dataset fetch or a result store failing on a read-only or
    missing home directory.
    """
    home = Path(os.environ.get("PYMRST_DATA_DIR", Path.home()/".pymrst"))
    entries = [("Dataset cache", home/"datasets"), ("Output", Path.cwd())]
    report = []
    for label, path in entries:
        ok, note = _writable(path)
        report.append((label, path, ok, note))
    return report


def _writable(path: Path) -> tuple[bool, str]:
    probe = path if path.exists() else path.parent
    if not probe.exists():
        return False, "does not exist and cannot be created"
    if not os.access(probe, os.W_OK):
        return False, "not writable"
    if path.exists():
        free = shutil.disk_usage(probe).free/1e9
        return True, f"{free:.1f} GB free"
    return True, "will be created on first use"


def environment() -> dict[str, str]:
    return {
        "PyMRST": _version(),
        "Python": sys.version.split()[0],
        "Platform": f"{sys.platform}",
        "Executable": sys.executable,
    }


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("pymrst")
    except Exception:                                   # pragma: no cover
        return "unknown (not installed as a distribution)"


__all__ = ["CAPABILITIES", "Capability", "cuda_status", "environment",
           "writable_paths"]
