"""MRST-shaped diagnostics utility landing modules."""

from pymrst.modules.diagnostics.utils.compute_f_and_phi import compute_f_phi
from pymrst.modules.diagnostics.utils.compute_lorenz import lorenz_coefficient
from pymrst.modules.diagnostics.utils.compute_sweep import compute_sweep
from pymrst.modules.diagnostics.utils.compute_time_of_flight import (
    Diagnostics,
    compute_diagnostics,
)

__all__ = [
    "Diagnostics",
    "compute_diagnostics",
    "compute_f_phi",
    "compute_sweep",
    "lorenz_coefficient",
]
