"""Private MPFA implementation helpers, mirroring MRST ``mpfa/private``."""

from pymrst.modules.mpfa.private.compute_multi_point_trans_tensor_assembly import (
    compute_multi_point_trans_tensor_assembly,
)
from pymrst.modules.mpfa.private.incomp_mpfa_legacy import incomp_mpfa_legacy
from pymrst.modules.mpfa.private.incomp_mpfa_tensor_assembly import (
    incomp_mpfa_tensor_assembly,
)

__all__ = [
    "compute_multi_point_trans_tensor_assembly",
    "incomp_mpfa_legacy",
    "incomp_mpfa_tensor_assembly",
]
