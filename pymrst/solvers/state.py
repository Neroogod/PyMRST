"""Compatibility exports for state initialization helpers."""

from pymrst.solvers.init_res_sol import init_res_sol
from pymrst.solvers.init_state import init_state
from pymrst.solvers.state_types import State, WellSol

__all__ = ["State", "WellSol", "init_res_sol", "init_state"]

