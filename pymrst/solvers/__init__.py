"""Core solver support: transmissibilities and state initialization
(MRST solvers/: computeTrans, initResSol, initState, initWellSol).

Flow and transport solvers live in their MRST modules:
pymrst.modules.incomp (incompTPFA, explicit/implicit transport),
pymrst.modules.ad_core, pymrst.modules.ad_blackoil.
"""

from pymrst.solvers.compute_trans import compute_trans
from pymrst.solvers.get_face_transmissibility import get_face_transmissibility
from pymrst.solvers.init_res_sol import init_res_sol
from pymrst.solvers.init_state import init_state
from pymrst.solvers.init_well_sol import init_well_sol
from pymrst.solvers.accumulate_cartesian_multipliers import (
    accumulate_cartesian_multipliers,
)
from pymrst.solvers.state_types import State, WellSol

__all__ = [
    "State",
    "WellSol",
    "accumulate_cartesian_multipliers",
    "compute_trans",
    "get_face_transmissibility",
    "init_res_sol",
    "init_state",
    "init_well_sol",
]
