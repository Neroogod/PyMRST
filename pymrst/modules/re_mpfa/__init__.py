"""Richards equation with MPFA (MRST ``modules/re-mpfa``)."""

from pymrst.modules.re_mpfa.functions import (
    HydraulicHeadMPFAOperators,
    NewtonResult,
    arithmeticMPFA_hydHead,
    arithmetic_mpfa_hyd_head,
    newtonSolver,
    newton_solver,
    timeStepping,
    time_stepping,
    upstreamMPFA_hydHead,
    upstream_mpfa_hyd_head,
)

__all__ = [
    "HydraulicHeadMPFAOperators",
    "NewtonResult",
    "arithmetic_mpfa_hyd_head",
    "arithmeticMPFA_hydHead",
    "newton_solver",
    "newtonSolver",
    "time_stepping",
    "timeStepping",
    "upstream_mpfa_hyd_head",
    "upstreamMPFA_hydHead",
]
