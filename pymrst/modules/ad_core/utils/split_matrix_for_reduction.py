"""Split a linear system into Schur-complement blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class ReductionSystem:
    B: Any = None
    C: Any = None
    D: Any = None
    E: Any = None
    f: Any = None
    h: Any = None
    E_L: Any = None
    E_U: Any = None
    factor: Any = None


def split_matrix_for_reduction(
    A, b, n, strategy: str = "subset",
    do_factor: bool = True, full_factor: bool = False,
) -> ReductionSystem:
    """Port MRST ``splitMatrixForReduction``."""
    del full_factor
    b = np.asarray(b)
    if n is None or int(n) >= b.shape[0] or A is None:
        return ReductionSystem()
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if strategy.lower() not in {"subset", "sparse"}:
        raise ValueError(f"unknown reduction strategy {strategy!r}")
    matrix = sp.csr_matrix(A) if sp.issparse(A) or strategy.lower() == "sparse" else np.asarray(A)
    system = ReductionSystem(
        B=matrix[:n, :n], C=matrix[:n, n:],
        D=matrix[n:, :n], E=matrix[n:, n:],
        f=b[:n].copy(), h=b[n:].copy(),
    )
    if do_factor and system.E.shape[0]:
        if sp.issparse(system.E):
            factor = spla.splu(sp.csc_matrix(system.E))
            system.factor, system.E_L, system.E_U = factor, factor.L, factor.U
        else:
            import scipy.linalg as la
            _, lower, upper = la.lu(system.E)
            system.E_L, system.E_U = lower, upper
    return system


splitMatrixForReduction = split_matrix_for_reduction

__all__ = ["ReductionSystem", "split_matrix_for_reduction", "splitMatrixForReduction"]
