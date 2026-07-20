"""MRST ``VEM3D_divTEDFM.m`` TEDFM 3D VEM divergence."""

import scipy.sparse as sp

from pymrst.modules.vemmech.utils.vem3d_div import vem3d_div


def _matrix_grid(g):
    return getattr(g, "Matrix", g)


def vem3d_div_tedfm(g) -> sp.csr_array:
    """Return 3D VEM divergence for the mechanical ``G.Matrix`` grid."""
    return vem3d_div(_matrix_grid(g))

__all__ = ["vem3d_div_tedfm"]
