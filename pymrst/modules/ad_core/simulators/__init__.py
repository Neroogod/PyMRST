"""MRST-style simulator entry points for `ad_core`."""

from pymrst.modules.ad_core.simulators.nonlinear_solver import (
    NonLinearSolver,
    SimulationReport,
    StepReport,
)
from pymrst.modules.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad

__all__ = [
    "NonLinearSolver",
    "SimulationReport",
    "StepReport",
    "simulate_schedule_ad",
]
