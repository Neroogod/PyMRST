"""MRST ``VEM_divTEDFM.m`` TEDFM VEM divergence dispatcher."""

import scipy.sparse as sp

from pymrst.modules.vemmech.utils.vem2d_div import vem2d_div
from pymrst.modules.vemmech.utils.vem3d_div_tedfm import vem3d_div_tedfm


def _matrix_grid(g):
    return getattr(g, "Matrix", g)


def vem_div_tedfm(g) -> sp.csr_array:
    """Dispatch TEDFM divergence through ``G.Matrix`` for mechanics grids."""
    mg = _matrix_grid(g)
    return vem3d_div_tedfm(g) if mg.griddim == 3 else vem2d_div(mg)

__all__ = ["vem_div_tedfm"]
