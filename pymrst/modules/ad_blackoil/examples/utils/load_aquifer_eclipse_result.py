"""Port guard for MRST ``loadAquiferEclipseResult.m``."""

from __future__ import annotations

from pathlib import Path


def load_aquifer_eclipse_result(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    raise NotImplementedError(
        "PyMRST does not yet parse Eclipse summary/result files for the "
        "aquifertest comparison. Use the PyMRST analytic aquifer example or "
        "the phase48/48b MATLAB .mat validations."
    )


__all__ = ["load_aquifer_eclipse_result"]
