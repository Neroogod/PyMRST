"""Nonlinear finite-volume methods (MRST ``modules/nfvm``)."""

from pymrst.modules.nfvm.solvers.avg_mpfa import AvgMPFA
from pymrst.modules.nfvm.solvers.hybrid import Hybrid
from pymrst.modules.nfvm.solvers.utils.find_hap import (
    HarmonicAveragingPoints,
    find_hap,
)
from pymrst.modules.nfvm.solvers.ntpfa import NTPFA
from pymrst.modules.nfvm.solvers.utils.compute_flux import compute_flux
from pymrst.modules.nfvm.solvers.utils.correct_hap import correct_hap
from pymrst.modules.nfvm.solvers.utils.find_os_flux import find_os_flux
from pymrst.modules.nfvm.solvers.utils.trans_ntpfa import trans_ntpfa
from pymrst.modules.nfvm.solvers.set_avg_mpfa_discretization import (
    set_avg_mpfa_discretization,
)
from pymrst.modules.nfvm.solvers.set_hybrid_discretization import (
    set_hybrid_discretization,
)
from pymrst.modules.nfvm.solvers.set_ntpfa_discretization import (
    set_ntpfa_discretization,
)

findHAP = find_hap

__all__ = [
    "HarmonicAveragingPoints",
    "AvgMPFA",
    "Hybrid",
    "NTPFA",
    "compute_flux",
    "correct_hap",
    "find_hap",
    "find_os_flux",
    "findHAP",
    "trans_ntpfa",
    "set_avg_mpfa_discretization",
    "set_hybrid_discretization",
    "set_ntpfa_discretization",
]
