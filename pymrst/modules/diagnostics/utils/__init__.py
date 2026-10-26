"""MRST ``diagnostics/utils``: time of flight, tracers, well pairs and the
heterogeneity measures built on them."""

from pymrst.modules.diagnostics.utils.compute_f_and_phi import compute_f_and_phi, compute_f_phi
from pymrst.modules.diagnostics.utils.compute_f_and_phi_from_dist import (
    compute_f_and_phi_from_dist,
)
from pymrst.modules.diagnostics.utils.compute_lorenz import compute_lorenz, lorenz_coefficient
from pymrst.modules.diagnostics.utils.compute_sweep import compute_sweep, compute_sweep_ad
from pymrst.modules.diagnostics.utils.compute_time_of_flight import (
    Diagnostics,
    compute_diagnostics,
    compute_time_of_flight,
)
from pymrst.modules.diagnostics.utils.compute_tof_and_tracer import compute_tof_and_tracer
from pymrst.modules.diagnostics.utils.compute_well_pairs import compute_well_pairs
from pymrst.modules.diagnostics.utils.validate_state_for_diagnostics import (
    validate_state_for_diagnostics,
)

__all__ = [
    "Diagnostics",
    "compute_diagnostics",
    "compute_f_and_phi",
    "compute_f_and_phi_from_dist",
    "compute_f_phi",
    "compute_lorenz",
    "compute_sweep",
    "compute_sweep_ad",
    "compute_time_of_flight",
    "compute_tof_and_tracer",
    "compute_well_pairs",
    "lorenz_coefficient",
    "validate_state_for_diagnostics",
]
