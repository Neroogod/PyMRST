"""Mimetic finite differences (MRST ``modules/mimetic``)."""

from pymrst.modules.mimetic.compute_mimetic_ip import (
    MimeticSystem,
    computeMimeticIP,
    compute_mimetic_ip,
)
from pymrst.modules.mimetic.incomp_mimetic import (
    MimeticResult,
    incompMimetic,
    incomp_mimetic,
)
from pymrst.modules.mimetic.solve_incomp_flow import (
    solveIncompFlow,
    solve_incomp_flow,
)
from pymrst.modules.mimetic.utils.hybrid_solvers import (
    mixed_symm,
    schur_complement_symm,
    schur_complement_symm_fault,
    tpf_symm,
)

__all__ = [
    "MimeticResult",
    "MimeticSystem",
    "compute_mimetic_ip",
    "computeMimeticIP",
    "incomp_mimetic",
    "incompMimetic",
    "solve_incomp_flow",
    "solveIncompFlow",
    "mixed_symm",
    "schur_complement_symm",
    "schur_complement_symm_fault",
    "tpf_symm",
]
