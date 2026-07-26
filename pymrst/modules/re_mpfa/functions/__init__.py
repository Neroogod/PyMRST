"""Numerical functions from MRST ``re-mpfa/functions``."""

from pymrst.modules.re_mpfa.functions.arithmetic_mpfa_hyd_head import (
    arithmeticMPFA_hydHead,
    arithmetic_mpfa_hyd_head,
)
from pymrst.modules.re_mpfa.functions.hydraulic_head_operators import (
    HydraulicHeadMPFAOperators,
)
from pymrst.modules.re_mpfa.functions.newton_solver import (
    NewtonResult,
    newtonSolver,
    newton_solver,
)
from pymrst.modules.re_mpfa.functions.time_stepping import (
    timeStepping,
    time_stepping,
)
from pymrst.modules.re_mpfa.functions.upstream_mpfa_hyd_head import (
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
