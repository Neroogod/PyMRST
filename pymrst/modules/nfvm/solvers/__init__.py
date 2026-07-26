"""NFVM discretization classes and setup functions."""

from pymrst.modules.nfvm.solvers.avg_mpfa import AvgMPFA
from pymrst.modules.nfvm.solvers.hybrid import Hybrid
from pymrst.modules.nfvm.solvers.ntpfa import NTPFA
from pymrst.modules.nfvm.solvers.set_avg_mpfa_discretization import (
    set_avg_mpfa_discretization,
)
from pymrst.modules.nfvm.solvers.set_hybrid_discretization import (
    set_hybrid_discretization,
)
from pymrst.modules.nfvm.solvers.set_ntpfa_discretization import (
    set_ntpfa_discretization,
)

__all__ = [
    "AvgMPFA",
    "Hybrid",
    "NTPFA",
    "set_avg_mpfa_discretization",
    "set_hybrid_discretization",
    "set_ntpfa_discretization",
]
