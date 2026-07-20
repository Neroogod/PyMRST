"""MRST ``VEM_div.m`` dimensional VEM divergence dispatcher."""

import scipy.sparse as sp

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.vemmech.utils.vem2d_div import vem2d_div
from pymrst.modules.vemmech.utils.vem3d_div import vem3d_div


def vem_div(g: Grid) -> sp.csr_array:
    """Dispatch to the 2D or 3D VEM divergence implementation."""
    return vem3d_div(g) if g.griddim == 3 else vem2d_div(g)

__all__ = ["vem_div"]
