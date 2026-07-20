"""Flow diagnostics: time-of-flight, tracers, sweep (MRST modules/diagnostics)."""

from pymrst.modules.diagnostics.flow_diagnostics import (
    Diagnostics,
    compute_diagnostics,
    compute_f_phi,
    compute_sweep,
    lorenz_coefficient,
)

__all__ = [
    "Diagnostics", "compute_diagnostics", "compute_f_phi",
    "lorenz_coefficient", "compute_sweep",
]
