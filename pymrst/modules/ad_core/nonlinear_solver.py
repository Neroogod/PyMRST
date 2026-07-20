"""Backward-compatible import path for the AD-core nonlinear solver."""

from pymrst.modules.ad_core.simulators.nonlinear_solver import (
    NonLinearSolver,
    SimulationReport,
    StepReport,
)

__all__ = ["NonLinearSolver", "SimulationReport", "StepReport"]
