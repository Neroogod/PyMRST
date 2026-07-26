"""Multi-point flux approximation (MRST ``modules/mpfa``).

The current vertical slice contains the MPFA-O local transmissibility
assembly and a single-phase Dirichlet pressure solve.
"""

from pymrst.modules.mpfa.compute_multi_point_trans import computeMultiPointTrans
from pymrst.modules.mpfa.compute_multipoint_trans import compute_multipoint_trans
from pymrst.modules.mpfa.incomp_mpfa import MPFAResult, incompMPFA, incomp_mpfa
from pymrst.modules.mpfa.private.incomp_mpfa_legacy import incomp_mpfa_legacy

__all__ = [
    "compute_multipoint_trans",
    "incomp_mpfa",
    "MPFAResult",
    "incomp_mpfa_legacy",
    "computeMultiPointTrans",
    "incompMPFA",
]
