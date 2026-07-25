"""MRST-style simulator entry points for `ad_core`."""

from pymrst.modules.ad_core.simulators.nonlinear_solver import (
    NonLinearSolver,
    SimulationReport,
    StepReport,
)
from pymrst.modules.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad

# The adjoint drivers (compute_gradient_adjoint_ad, compute_gradient_perturbation_ad)
# are NOT re-exported here on purpose: they pull in the optimization module, which
# imports back into ad_core, and barrel-exporting them at package-init time would
# create an import cycle (ad_core.utils -> simulators -> optimization -> ad_core).
# Import them from their submodule, as MRST files them:
#   from pymrst.modules.ad_core.simulators.compute_gradient_adjoint_ad import ...

__all__ = [
    "NonLinearSolver",
    "SimulationReport",
    "StepReport",
    "simulate_schedule_ad",
]
