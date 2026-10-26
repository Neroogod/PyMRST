"""MRST ``linearsolvers/amgcl/utils/resetAMGCL.m``: drop the solvers the AMGCL
gateway keeps for reuse (`solver_id` 1000)."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from pymrst.modules.linearsolvers.amgcl.utils.amgcl_matlab import amgcl_matlab
from pymrst.modules.linearsolvers.amgcl.utils.get_amgcl_mex_struct import get_amgcl_mex_struct


def reset_amgcl(**options: Any) -> None:
    """MRST `resetAMGCL(...)`."""
    amg_opt = get_amgcl_mex_struct(**options)
    amgcl_matlab(sp.csc_matrix((0, 0)), np.zeros(0), amg_opt, np.nan, np.nan, 1000)


resetAMGCL = reset_amgcl
__all__ = ["resetAMGCL", "reset_amgcl"]


if __name__ == "__main__":
    def test_reset_runs():
        reset_amgcl()
        print("ok test_reset_runs")

    test_reset_runs()
    print("\nAll tests passed!")
